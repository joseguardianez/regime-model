"""
Model Lab: the research behind the "Lab" tab of the website.

* Options-expiration calendar (monthly options, quad witching, VIX) and how stocks behave
  around those dates
* Dealer gamma (GEX) and the dark-pool index (DIX) from SqueezeMetrics
* A top-down macro quad from real GDP and CPI (FRED), lagged to when the data came out
* Version 2 of the regime model: buckets ranked by relative strength, ETF buckets, and
  buckets re-picked every January with getETF using only past data (walk-forward)
* An optimizer that picks settings on 2010-2018 and grades them on 2019 onward

Everything here is plain pandas/numpy (no Streamlit), so it can also run in Colab.
"""
import io
import itertools
import os
import re

import numpy as np
import pandas as pd
import requests
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay,
                                    USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
                                    USThanksgivingDay, nearest_workday, sunday_to_monday)

REGIMES = ["Disinflation", "Reflation", "Stagflation", "Deflation"]

# A common quad playbook in ETFs (growth / inflation direction in brackets)
ETF_BUCKETS = {
    "Disinflation": ["XLK", "XLY", "SMH", "IWF"],   # growth up, inflation down: tech, consumer, growth
    "Reflation":    ["XLE", "XLB", "XLI", "XLF"],   # growth up, inflation up: energy, materials, industrials, banks
    "Stagflation":  ["GLD", "XLU", "TIP", "DBC"],   # growth down, inflation up: gold, utilities, TIPS, commodities
    "Deflation":    ["TLT", "XLP", "XLV", "UUP"],   # growth down, inflation down: long bonds, staples, health care, dollar
}
BENCHMARKS = ["QQQ", "SPY"]
GEX_URL = "https://squeezemetrics.com/monitor/static/DIX.csv"
FRED_URLS = ["https://fred.stlouisfed.org/graph/fredgraph.csv?id={}",
             "https://fred.stlouisfed.org/data/{}.txt"]
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


# ----------------------------------------------------------------------------------
# Options expiration calendar
# ----------------------------------------------------------------------------------
class NYSEHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def third_friday(year, month):
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def expiration_calendar(start_year, end_year):
    """Monthly stock/index options expiration (third Friday, or the business day before it if
    that's a holiday), quad witching (the March/June/September/December ones) and VIX
    expiration (the Wednesday 30 days before the third Friday of the next month; if that
    Wednesday or that Friday is a holiday, the business day before the Wednesday)."""
    hol = set(NYSEHolidays().holidays(f"{start_year - 1}-01-01", f"{end_year + 1}-12-31"))

    def is_bd(d):
        return d.weekday() < 5 and d not in hol

    def prev_bd(d):
        d -= pd.Timedelta(days=1)
        while not is_bd(d):
            d -= pd.Timedelta(days=1)
        return d

    rows = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            f = third_friday(y, m)
            rows.append({"Date": f if is_bd(f) else prev_bd(f),
                         "Event": "Quad witching" if m in (3, 6, 9, 12)
                         else "Monthly options expiration"})
            nxt = third_friday(y + (m == 12), m % 12 + 1)
            wed = nxt - pd.Timedelta(days=30)
            rows.append({"Date": wed if is_bd(wed) and is_bd(nxt) else prev_bd(wed),
                         "Event": "VIX expiration"})
    return pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)


def expiration_flags(index, cal):
    """Per trading day: offset from the nearest options expiration, and flags for the week
    before/after it and VIX expiration days."""
    idx = pd.DatetimeIndex(index)
    pos = np.arange(len(idx))
    opex = cal[cal["Event"] != "VIX expiration"]
    quad = set(cal.loc[cal["Event"] == "Quad witching", "Date"])
    opex_pos = pos[idx.isin(opex["Date"])]
    out = pd.DataFrame(index=idx)
    if len(opex_pos) == 0:
        out["offset"] = np.nan
    else:
        nearest = opex_pos[np.abs(pos[:, None] - opex_pos[None, :]).argmin(axis=1)]
        out["offset"] = pos - nearest
    out["opex_week"] = out["offset"].between(-4, 0)
    out["week_after"] = out["offset"].between(1, 5)
    quad_pos = pos[idx.isin(list(quad))]
    out["quad_week"] = False
    for p in quad_pos:
        out.iloc[max(p - 4, 0):p + 1, out.columns.get_loc("quad_week")] = True
    out["vix_exp"] = idx.isin(cal.loc[cal["Event"] == "VIX expiration", "Date"])
    return out


def expiration_effects(ret, cal):
    """How an asset's daily returns behave around expirations."""
    ret = ret.dropna()
    f = expiration_flags(ret.index, cal)
    groups = {"Week of monthly expiration": f["opex_week"],
              "Week after expiration": f["week_after"],
              "Quad witching week": f["quad_week"],
              "VIX expiration day": f["vix_exp"],
              "All other days": ~(f["opex_week"] | f["week_after"] | f["vix_exp"])}
    rows = []
    for name, m in groups.items():
        r = ret[m]
        rows.append({"Days": name, "Count": int(m.sum()),
                     "Avg daily return": r.mean(), "% of days up": (r > 0).mean(),
                     "Volatility (yearly)": r.std() * np.sqrt(252)})
    by_offset = ret.groupby(f["offset"]).mean().loc[-10:10]
    return pd.DataFrame(rows), by_offset


# ----------------------------------------------------------------------------------
# Gamma (GEX) and DIX from SqueezeMetrics
# ----------------------------------------------------------------------------------
def fetch_gex():
    """Daily SPX price, DIX and GEX (dealer gamma, $ per 1% move) since May 2011."""
    r = requests.get(GEX_URL, headers=UA, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def gex_table(gex, asset, days=20):
    """What the asset did over the next `days` trading days, by gamma level."""
    g = gex.reindex(asset.index).ffill()
    fwd = asset.shift(-days) / asset - 1
    vol = asset.pct_change().rolling(days).std().shift(-days) * np.sqrt(252)
    df = pd.DataFrame({"gex": g, "fwd": fwd, "vol": vol}).dropna()
    if df.empty:
        return pd.DataFrame()
    df["Gamma level"] = pd.qcut(df["gex"], 5, labels=["1 Lowest", "2", "3", "4", "5 Highest"])
    rows = []
    for name, sub in list(df.groupby("Gamma level", observed=True)) + \
            [("Negative gamma (below 0)", df[df["gex"] < 0]),
             ("Positive gamma (above 0)", df[df["gex"] >= 0])]:
        if len(sub):
            rows.append({"Gamma level": str(name), "Days": len(sub),
                         "GEX range ($bn)": f"{sub['gex'].min() / 1e9:.1f} to "
                                            f"{sub['gex'].max() / 1e9:.1f}",
                         f"Avg next {days}d return": sub["fwd"].mean(),
                         "% of times up": (sub["fwd"] > 0).mean(),
                         f"Volatility next {days}d": sub["vol"].mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------
# Top-down macro quad from FRED (real GDP and CPI)
# ----------------------------------------------------------------------------------
def parse_fred(text):
    """FRED data as a Series, from either the CSV download or the plain-text page."""
    rows = re.findall(r"^(\d{4}-\d{2}-\d{2})[,\s]+(-?\d+(?:\.\d+)?)\s*$", text, flags=re.M)
    s = pd.Series([float(v) for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
    return s[~s.index.duplicated()].sort_index()


def fetch_fred(series_id):
    """Download a FRED series, trying two addresses. Raises if both fail."""
    errors = []
    for url in FRED_URLS:
        try:
            r = requests.get(url.format(series_id), headers=UA, timeout=20)
            r.raise_for_status()
            s = parse_fred(r.text)
            if len(s) > 20:
                return s
            errors.append(f"{url.split('/')[3]}: no data")
        except Exception as err:
            errors.append(f"{type(err).__name__}")
    raise RuntimeError(f"FRED {series_id}: " + "; ".join(errors))


SAVED_MACRO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_data.csv")


def saved_macro():
    """Saved copy of GDPC1 and CPIAUCSL from FRED (macro_data.csv in the repo), used when FRED
    can't be reached from the server."""
    df = pd.read_csv(SAVED_MACRO, parse_dates=["date"])
    return tuple(df[df["series"] == sid].set_index("date")["value"].sort_index()
                 for sid in ("GDPC1", "CPIAUCSL"))


def _asof(s, index):
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(s.index.union(index)).ffill().reindex(index)


def macro_quads(gdp, cpi, index):
    """Quad from the direction of growth and inflation, using only what was published:
    real GDP year-over-year growth vs the quarter before (released about a month after the
    quarter ends) and CPI year-over-year inflation vs 3 months before (released about two
    weeks after the month ends). Uses today's revised GDP data, not the first release."""
    # Put both on a regular calendar first, filling single missing readings (FRED has no
    # October 2025 CPI because of the government shutdown) so "a year ago" is really 12 months
    gdp = gdp.asfreq("QS").interpolate(limit_area="inside")
    cpi = cpi.asfreq("MS").interpolate(limit_area="inside")
    g = gdp.pct_change(4, fill_method=None)
    g_up = (g.diff() > 0).where(g.diff().notna())
    g_up.index = g_up.index + pd.DateOffset(months=3) + pd.Timedelta(days=30)
    c = cpi.pct_change(12, fill_method=None)
    c_up = ((c - c.shift(3)) > 0).where((c - c.shift(3)).notna())
    c_up.index = c_up.index + pd.DateOffset(months=1) + pd.Timedelta(days=15)
    gu, cu = _asof(g_up.dropna().astype(float), index), _asof(c_up.dropna().astype(float), index)
    quad = pd.Series(np.select([(gu == 1) & (cu == 0), (gu == 1) & (cu == 1),
                                (gu == 0) & (cu == 1), (gu == 0) & (cu == 0)],
                               [1, 2, 3, 4], default=0), index=index).replace(0, np.nan)
    detail = pd.DataFrame({"GDP growth (YoY)": _asof(g.dropna(), index),
                           "CPI inflation (YoY)": _asof(c.dropna(), index),
                           "Growth speeding up": gu, "Inflation speeding up": cu,
                           "Macro quad": quad})
    return quad, detail


# ----------------------------------------------------------------------------------
# Buckets: fixed, ETF, or re-picked every year (walk-forward getETF)
# ----------------------------------------------------------------------------------
def walk_forward_buckets(prices, quad, years, k=5, lookback_years=5, min_days=500):
    """Each January, rank every candidate the getETF way (average daily return in each
    quad over the previous `lookback_years`) and keep the k most quad-specific names per
    quad (biggest gap between their best quad and their average, scaled by how much the
    name swings day to day). Only past data is used."""
    rets = prices.pct_change(fill_method=None)
    out = {}
    for y in years:
        win = rets.loc[f"{y - lookback_years}-01-01":f"{y - 1}-12-31"]
        ok = win.columns[win.notna().sum() >= min_days]
        if len(ok) == 0:
            continue
        q = quad.reindex(win.index)
        avg = win[ok].groupby(q).mean()
        avg = avg.reindex([1, 2, 3, 4]).dropna(axis=1, how="any")
        best = avg.idxmax()
        # how much better its best quad is than its average, per unit of its own daily swings,
        # so wild small caps don't win just by being volatile
        edge = (avg.max() - avg.mean()) / win[avg.columns].std()
        out[y] = {r: list(edge[best == i + 1].sort_values(ascending=False).index[:k])
                  for i, r in enumerate(REGIMES)}
    return out


def members_on(source, year):
    """Bucket members for a year. `source` is either {regime: [...]} or {year: {regime: [...]}}."""
    if set(source) <= set(REGIMES):
        return source
    years = sorted(y for y in source if y <= year)
    return source[years[-1]] if years else source[min(source)]


def bucket_matrix(values, source, index):
    """Average of each bucket's members (dates x regimes), with yearly membership if given."""
    out = pd.DataFrame(np.nan, index=index, columns=REGIMES)
    years = sorted(set(index.year))
    for y in years:
        mask = index.year == y
        mem = members_on(source, y)
        for r in REGIMES:
            cols = [t for t in mem.get(r, []) if t in values.columns]
            if cols:
                out.loc[mask, r] = values.loc[mask, cols].mean(axis=1).to_numpy()
    return out


def score_matrix(method, votes, prices, source, index, lookback=20, smooth=5):
    """votes: share of the bucket voting BUY (the original model).
    rs: bucket's `lookback`-day return minus the average of all four buckets (relative strength)."""
    if method == "votes":
        raw = bucket_matrix(votes, source, index)
    else:
        mom = prices / prices.shift(lookback) - 1
        b = bucket_matrix(mom, source, index)
        raw = b.sub(b.mean(axis=1), axis=0)
    return raw.rolling(smooth, min_periods=1).mean() if smooth > 1 else raw


# ----------------------------------------------------------------------------------
# Rotation backtest and stats
# ----------------------------------------------------------------------------------
def top_weights(score, top_n=1):
    """Equal weights on the top_n buckets each day (ties go to the earlier regime)."""
    s = score.to_numpy(dtype=float)
    valid = ~np.isnan(s).all(axis=1)
    filled = np.where(np.isnan(s), -np.inf, s)
    order = np.argsort(-filled, axis=1, kind="stable")[:, :top_n]
    w = np.zeros_like(s)
    w[np.arange(len(s))[:, None], order] = 1.0 / top_n
    w[~valid] = np.nan
    return pd.DataFrame(w, index=score.index, columns=score.columns)


def rotation_returns(weights, bucket_ret, cash_ret, exposure=None, delay=1, cost=0.0005):
    """Daily returns of holding the chosen buckets (decided at a close, traded `delay` days
    later). exposure (0 to 1) scales how much is invested; the rest earns cash."""
    W = weights.shift(delay)
    gross = (W * bucket_ret.fillna(0)).sum(axis=1, min_count=1)
    e = pd.Series(1.0, index=W.index) if exposure is None else \
        exposure.reindex(W.index).shift(delay).fillna(1.0)
    turnover = (W.diff().abs().sum(axis=1) * e + e.diff().abs()).fillna(0)
    ret = e * gross + (1 - e) * cash_ret.reindex(W.index).fillna(0) - turnover * cost
    return ret.where(W.notna().any(axis=1))


def stats(ret, cash_ret):
    ret = ret.dropna()
    if len(ret) < 60:
        return {"Per year": np.nan, "Sharpe": np.nan, "Worst drop": np.nan, "Volatility": np.nan}
    years = len(ret) / 252
    growth = (1 + ret).cumprod()
    sd = ret.std()
    ex = ret - cash_ret.reindex(ret.index).fillna(0)
    return {"Per year": growth.iloc[-1] ** (1 / years) - 1,
            "Sharpe": ex.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
            "Worst drop": (growth / growth.cummax() - 1).min(),
            "Volatility": sd * np.sqrt(252)}


def spearman(a, b):
    m = a.notna() & b.notna()
    return a[m].rank().corr(b[m].rank()) if m.sum() > 2 else np.nan


# ----------------------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------------------
OVERLAYS = {
    "Gamma cut": "Half invested when dealer gamma (GEX) was negative the day before",
    "Expiration cut": "Half invested in the week after monthly options expiration",
    "Macro check": "Half invested when the GDP/CPI quad isn't one of the buckets held",
    "Fear buy": "Fully invested whenever VIX is above VIX3M (5-day average), overriding cuts",
}


def exposure_series(index, weights, on, extras):
    e = pd.Series(1.0, index=index)
    if "Gamma cut" in on and extras.get("gex") is not None:
        g = extras["gex"].reindex(index).ffill().shift(1)
        e = e.where(~(g < 0), e * 0.5)
    if "Expiration cut" in on and extras.get("week_after") is not None:
        wa = extras["week_after"].reindex(index).fillna(False).astype(bool).shift(-1)
        e = e.where(~wa.fillna(False).astype(bool), e * 0.5)
    if "Macro check" in on and extras.get("macro") is not None:
        q = extras["macro"].reindex(index).to_numpy(dtype=float)
        w = weights.reindex(index).to_numpy(dtype=float)
        has_q = ~np.isnan(q) & ~np.isnan(w).all(axis=1)
        held = np.zeros(len(index), dtype=bool)
        rows = np.flatnonzero(has_q)
        held[rows] = w[rows, q[rows].astype(int) - 1] > 0
        e = e.where(~pd.Series(has_q & ~held, index=index), e * 0.5)
    if "Fear buy" in on and extras.get("vix_ratio") is not None:
        fear = extras["vix_ratio"].reindex(index).rolling(5).mean() > 1
        e = e.where(~fear, 1.0)
    return e


HINDSIGHT_SOURCE = "Your 20 stocks"


def run_optimizer(prices, votes, quad, own_buckets, cash_ret, extras, start="2010-01-01",
                  train_end="2018-12-31", delay=1, cost=0.0005, seed=0, allow_hindsight=False):
    """Stage 1: try every core setup (bucket source x scoring x settings x how many buckets)
    and rank them on 2010-2018 only. Stage 2: add every combination of the extra filters to
    the best one, again judged on 2010-2018 only. Then report 2019-onward results, which the
    optimizer never saw. Unless allow_hindsight, the winner can't come from your 20 stocks,
    because they were picked with data that includes the test years."""
    index = prices.loc[start:].index
    full_index = prices.index
    last_year = index[-1].year
    candidates = sorted(set(t for v in own_buckets.values() for t in v) |
                        set(t for v in ETF_BUCKETS.values() for t in v))
    candidates = [t for t in candidates if t in prices.columns]
    wf = walk_forward_buckets(prices[candidates], quad, range(int(start[:4]), last_year + 1))
    sources = {"Your 20 stocks": own_buckets, "Sector ETFs": ETF_BUCKETS,
               "Walk-forward picks": wf}
    rets_all = prices.pct_change(fill_method=None)
    bucket_rets = {n: bucket_matrix(rets_all, s, full_index) for n, s in sources.items()}
    cash = cash_ret.reindex(full_index).fillna(0)
    train = slice(start, train_end)
    test_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test = slice(test_start, None)

    # macro quad as a "score": 1 for the macro quad's bucket
    macro = extras.get("macro")
    rows, series = [], {}
    configs = []
    for src in sources:
        for top_n in (1, 2):
            for smooth in (1, 5, 10, 20):
                configs.append((src, "votes", 0, smooth, top_n))
                for lb in (10, 20, 40, 60):
                    configs.append((src, "rs", lb, smooth, top_n))
        if macro is not None:
            configs.append((src, "macro", 0, 1, 1))
    mom_cache = {}
    for src, method, lb, smooth, top_n in configs:
        if method == "macro":
            m = macro.reindex(full_index)
            score = pd.DataFrame({r: (m == i + 1).astype(float).where(m.notna())
                                  for i, r in enumerate(REGIMES)}, index=full_index)
        else:
            key = (src, method, lb)
            if key not in mom_cache:
                mom_cache[key] = score_matrix(method, votes, prices, sources[src], full_index,
                                              lb, 1)
            raw = mom_cache[key]
            score = raw.rolling(smooth, min_periods=1).mean() if smooth > 1 else raw
        w = top_weights(score, top_n)
        r = rotation_returns(w, bucket_rets[src], cash, None, delay, cost).loc[start:]
        tr, te = stats(r.loc[train], cash), stats(r.loc[test], cash)
        name = (f"{src} · " + {"votes": "BUY/SELL votes", "rs": f"relative strength {lb}d",
                               "macro": "GDP/CPI quad"}[method]
                + (f" · smooth {smooth}d" if method != "macro" else "") + f" · top {top_n}")
        rows.append({"Setup": name, "Source": src, "Method": method, "Lookback": lb,
                     "Smoothing": smooth, "Buckets held": top_n,
                     "Train Sharpe": tr["Sharpe"], "Train per year": tr["Per year"],
                     "Test Sharpe": te["Sharpe"], "Test per year": te["Per year"],
                     "Test worst drop": te["Worst drop"]})
        series[name] = (w, r)
    core = pd.DataFrame(rows).sort_values("Train Sharpe", ascending=False).reset_index(drop=True)
    rho = spearman(core["Train Sharpe"], core["Test Sharpe"])
    eligible = core if allow_hindsight else core[core["Source"] != HINDSIGHT_SOURCE]
    best = eligible.iloc[0]
    w_best, _ = series[best["Setup"]]
    src_best = best["Source"]

    # Stage 2: extra filters
    available = [o for o in OVERLAYS
                 if extras.get({"Gamma cut": "gex", "Expiration cut": "week_after",
                                "Macro check": "macro", "Fear buy": "vix_ratio"}[o]) is not None]
    orows, oser = [], {}
    for k in range(0, len(available) + 1):
        for combo in itertools.combinations(available, k):
            e = exposure_series(full_index, w_best, set(combo), extras)
            r = rotation_returns(w_best, bucket_rets[src_best], cash, e, delay, cost).loc[start:]
            tr, te = stats(r.loc[train], cash), stats(r.loc[test], cash)
            label = " + ".join(combo) if combo else "None"
            orows.append({"Extra filters": label, "Train Sharpe": tr["Sharpe"],
                          "Train per year": tr["Per year"], "Test Sharpe": te["Sharpe"],
                          "Test per year": te["Per year"], "Test worst drop": te["Worst drop"],
                          "Average invested": e.loc[start:].mean()})
            oser[label] = (e, r)
    over = pd.DataFrame(orows).sort_values("Train Sharpe", ascending=False).reset_index(drop=True)
    pick = over.iloc[0]
    e_pick, r_pick = oser[pick["Extra filters"]]

    # Benchmarks over the same dates
    bench = {}
    for b in BENCHMARKS:
        if b in prices.columns:
            bench[f"Buy & hold {b}"] = rets_all[b].loc[start:]
    bench[f"All 4 buckets equally ({src_best})"] = bucket_rets[src_best].mean(axis=1).loc[start:]
    final_name = f"Optimized: {best['Setup']}" + \
        (f" + {pick['Extra filters']}" if pick["Extra filters"] != "None" else "")
    curves = pd.DataFrame({final_name: r_pick, **bench}).loc[start:]
    summary = []
    for col in curves:
        tr, te = stats(curves[col].loc[train], cash), stats(curves[col].loc[test], cash)
        summary.append({"Strategy": col, "2010-2018 per year": tr["Per year"],
                        "2010-2018 Sharpe": tr["Sharpe"], "2019+ per year": te["Per year"],
                        "2019+ Sharpe": te["Sharpe"], "2019+ worst drop": te["Worst drop"]})

    # How each bucket source did overall, against buy & hold QQQ on the unseen years
    qqq_test = stats(rets_all["QQQ"].loc[test], cash)["Sharpe"] if "QQQ" in prices else np.nan
    by_source = []
    for src_name, grp in core.groupby("Source", sort=False):
        by_source.append({"Bucket source": src_name + (" (picked with hindsight)"
                                                       if src_name == HINDSIGHT_SOURCE else ""),
                          "Versions": len(grp),
                          "Median 2010-2018 Sharpe": grp["Train Sharpe"].median(),
                          "Median 2019+ Sharpe": grp["Test Sharpe"].median(),
                          "Best 2019+ Sharpe": grp["Test Sharpe"].max(),
                          "Beat QQQ's 2019+ Sharpe": (grp["Test Sharpe"] > qqq_test).mean()})
    by_source = pd.DataFrame(by_source)

    # Luck test on the unseen years: same picks, shifted to random dates
    rng = np.random.default_rng(seed)
    wt = w_best.loc[test_start:].dropna(how="all")
    br, ct = bucket_rets[src_best].loc[wt.index], cash.loc[wt.index]
    et = e_pick.reindex(wt.index)
    real = stats(rotation_returns(wt, br, ct, et, delay, cost), ct)["Sharpe"]
    sims = []
    vals = wt.to_numpy()
    for k in rng.integers(126, max(127, len(vals) - 126), 500):
        ws = pd.DataFrame(np.roll(vals, k, axis=0), index=wt.index, columns=wt.columns)
        sims.append(stats(rotation_returns(ws, br, ct, et, delay, cost), ct)["Sharpe"])
    sims = np.array(sims, dtype=float)
    luck = {"real": real, "avg_random": np.nanmean(sims), "share_as_good": float(np.mean(sims >= real))}

    # Today's pick
    last_w = w_best.dropna(how="all").iloc[-1]
    held = [r for r in REGIMES if last_w[r] > 0]
    mem = members_on(sources[src_best], full_index[-1].year)
    today = {"date": w_best.dropna(how="all").index[-1], "buckets": held,
             "names": {r: mem.get(r, []) for r in held},
             "exposure": float(e_pick.dropna().iloc[-1]), "setup": final_name}
    return {"core": core, "rho": rho, "overlays": over, "summary": pd.DataFrame(summary),
            "curves": curves, "test_start": test_start, "luck": luck, "today": today,
            "walk_forward": wf, "best_source": src_best, "by_source": by_source,
            "qqq_test_sharpe": qqq_test}
