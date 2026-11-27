"""
Bottoms-Up Regime Segmentation: interactive web app
===================================================

Same model as regime_model.py, as a website built with Streamlit, plus a
"Warning signals" tab with market indicators that may move before stocks do.

Run it on your computer:     streamlit run streamlit_app.py
Put it online for free:      push this file + requirements.txt to GitHub, then deploy
                             it on Streamlit Community Cloud (share.streamlit.io).
                             Every time you push a change to GitHub, the site updates.

Prices come from Yahoo Finance and refresh every few hours (or with the Refresh button).
"""

import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import lab

# ----------------------------------------------------------------------------------
# Default settings: edit these here (or change them in the sidebar while the app runs)
# ----------------------------------------------------------------------------------
DEFAULT_BUCKETS = {
    "Disinflation": ["TSLA", "PANW", "META", "AVGO", "AMZN"],   # Quad 1
    "Reflation":    ["XOM", "PWR", "PG", "CVX", "JNJ"],         # Quad 2
    "Stagflation":  ["YTRA", "INTG", "SPWH", "JILL", "DOGZ"],   # Quad 3
    "Deflation":    ["HSY", "ELA", "SMMT", "CCLD", "LNTH"],     # Quad 4
}
DEFAULT_OUTSIDE = ["QQQ", "AMD"]
DEFAULT_START = "2018-10-12"
REGIMES = list(DEFAULT_BUCKETS)        # order matters for ties (same as the IF chain in AJ)
COLORS = {"Disinflation": "#2a7de1", "Reflation": "#2ca02c",
          "Stagflation": "#d62728", "Deflation": "#7f7f7f"}

# Macro quad calendar from ETF_Bucketing.xlsm: "month the quad starts:quad", in order.
# Used by the Add tickers tab to rank a stock the same way getETF does.
QUAD_RUNS = (
    "2005-01:4;2005-02:3;2005-05:4;2005-06:1;2005-07:2;2005-08:3;2005-10:1;2006-01:2;2006-02:"
    "4;2006-04:3;2006-07:4;2006-08:1;2006-11:2;2007-01:1;2007-02:3;2007-04:1;2007-05:2;2007-0"
    "6:1;2007-07:4;2007-09:3;2007-12:4;2008-01:3;2008-02:4;2008-05:3;2008-08:4;2009-02:3;2009"
    "-03:1;2009-08:2;2010-01:1;2010-03:2;2010-04:4;2010-07:3;2010-08:4;2010-09:1;2010-10:2;20"
    "10-11:1;2010-12:2;2011-03:3;2011-10:1;2012-04:4;2012-08:2;2012-11:4;2013-01:1;2013-02:2;"
    "2013-03:1;2013-05:2;2013-08:1;2013-09:4;2013-11:2;2014-01:3;2014-02:1;2014-03:2;2014-06:"
    "1;2015-01:4;2015-02:3;2015-04:4;2015-05:3;2015-09:4;2015-10:3;2016-01:2;2016-02:1;2016-0"
    "3:2;2016-05:1;2016-06:2;2016-07:4;2016-08:3;2016-09:2;2016-11:1;2016-12:2;2017-03:4;2017"
    "-06:1;2017-07:2;2017-10:1;2017-11:2;2017-12:1;2018-02:2;2018-04:3;2018-06:2;2018-07:3;20"
    "18-08:1;2018-09:4;2018-10:3;2018-11:4;2019-03:3;2019-05:4;2019-07:3;2019-08:4;2019-10:2;"
    "2020-02:4;2020-05:1;2020-06:2;2020-10:1;2020-12:2;2021-05:3;2021-07:4;2021-09:3;2022-04:"
    "4;2022-05:3;2022-07:4;2023-01:1;2023-02:4;2023-05:1;2023-07:2;2023-09:4;2023-12:2;2024-0"
    "1:1;2024-02:2;2024-04:4;2024-09:1;2024-10:2;2025-02:4;2025-05:3;2025-06:2;2025-09:3;2025"
    "-10:4;2026-01:1;2026-02:2;2026-03:4;2026-04:1;2026-05:2;2026-07:1;2026-08:2;2026-11:1;20"
    "27-01:2"
)
QUAD_START = "2005-01-03"

# Indicator windows (same as the spreadsheet)
HV_DAYS = 5                    # D: HV-5
VRP_AVG_DAYS = 5               # G: VRP 5DMA
MACD_LONG = (12, 26, 9)        # I, J, L  (calm market)
MACD_SHORT = (6, 15, 4)        # N, O, Q  (stressed market; the sheet's O formula averages 15 days)
RSI_LONG = 21                  # W..AA    (calm market)
RSI_SHORT = 14                 # AE..AI   (stressed market)

# Warning signals. "ratio" signals warn when the ratio is lower than it was `lookback` days ago.
SIGNALS = {
    "Contango (VIX / VIX3M)": dict(
        kind="vixterm", tickers=["^VIX", "^VIX3M"],
        about="Short-term fear vs 3-month fear. Above 1 (backwardation) means traders are paying "
              "up for protection right now. Warning when the 5-day average is above 1."),
    "Lumber vs gold (WOOD / GLD)": dict(
        kind="ratio", tickers=["WOOD", "GLD"],
        about="Growth-sensitive timber vs safe-haven gold (Michael Gayed's lumber/gold idea, using "
              "ETFs). Warning when timber has lagged gold over the lookback."),
    "Credit (HYG / IEF)": dict(
        kind="ratio", tickers=["HYG", "IEF"],
        about="Junk bonds vs Treasuries. Warning when junk bonds lag, a sign investors are "
              "pulling back from risk."),
    "Copper vs gold (CPER / GLD)": dict(
        kind="ratio", tickers=["CPER", "GLD"],
        about="Industrial copper vs gold. Warning when copper lags, a sign of slowing growth."),
    "Small vs large caps (IWM / SPY)": dict(
        kind="ratio", tickers=["IWM", "SPY"],
        about="Russell 2000 vs S&P 500. Warning when small caps lag, a sign of falling risk appetite."),
    "Breadth (RSP / SPY)": dict(
        kind="ratio", tickers=["RSP", "SPY"],
        about="Equal-weight vs regular S&P 500. Warning when the average stock lags the index, "
              "meaning only a few big stocks are holding the market up."),
    "Yield curve (10y - 3m)": dict(
        kind="curve", tickers=["^TNX", "^IRX"],
        about="10-year Treasury yield minus 3-month T-bill yield. Warning when negative (inverted), "
              "a classic recession signal that tends to come months early."),
}


# ----------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner="Downloading prices from Yahoo Finance...")
def load_prices(tickers, start):
    """Adj Close for every ticker, plus the VIX close. Cached for 6 hours."""
    data = yf.download(list(tickers) + ["^VIX"], start=start, auto_adjust=False, progress=False)
    adj = data["Adj Close"].drop(columns="^VIX", errors="ignore")
    vix = data["Close"]["^VIX"]
    adj = adj.dropna(how="all")                      # keep stock-market days only
    vix = vix.reindex(adj.index).ffill()
    return adj, vix


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Downloading prices from Yahoo Finance...")
def load_asset(ticker, start):
    """Adj Close for one comparison ticker: any Yahoo symbol, such as a stock, an ETF,
    an index (^GSPC) or crypto (BTC-USD). Empty if Yahoo has nothing for it."""
    for symbol in dict.fromkeys([ticker, ticker.replace(".", "-")]):   # BRK.B -> BRK-B
        data = yf.download(symbol, start=start, auto_adjust=False, progress=False)
        if data is None or data.empty:
            continue
        col = "Adj Close" if "Adj Close" in data.columns.get_level_values(0) else "Close"
        s = data[col]
        s = (s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s).dropna()
        if len(s):
            return s.rename(ticker)
    return pd.Series(dtype=float, name=ticker)


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Downloading warning-signal data...")
def load_signal_prices(tickers, start):
    """Closing prices for the warning-signal tickers. Cached for 6 hours."""
    data = yf.download(list(tickers), start=start, auto_adjust=False, progress=False)
    return data["Adj Close"].dropna(how="all")


# ----------------------------------------------------------------------------------
# Model (same steps as the stock sheets and the VOTE sheet)
# ----------------------------------------------------------------------------------
def rsi(close, days):
    change = close.diff()                             # T / AB
    gain = change.clip(lower=0)                       # U / AC
    loss = (-change).clip(lower=0)                    # V / AD
    rs = gain.rolling(days).mean() / loss.rolling(days).mean()   # W/X, AE/AF
    return 100 - 100 / (1 + rs)                       # Z / AH


def macd_rule(close, fast, slow, signal):
    line = close.rolling(fast).mean() - close.rolling(slow).mean()   # K / P
    return (line.rolling(signal).mean().round(10) > 0).astype(int)   # M / R


def stock_vote(close, vix):
    """1 = BUY, 0 = SELL for one stock, every day (AL column)."""
    hv5 = close.pct_change().rolling(HV_DAYS).std() * np.sqrt(252) * 100       # D
    vrp_5dma = (vix - hv5).rolling(VRP_AVG_DAYS).mean()                         # F, G
    calm = vrp_5dma.round(10) > 0                                               # H
    macd = np.where(calm, macd_rule(close, *MACD_LONG), macd_rule(close, *MACD_SHORT))    # S
    rsi_ok = np.where(calm, rsi(close, RSI_LONG).round(10) > 50,
                      rsi(close, RSI_SHORT).round(10) > 50)                     # AJ
    vote = pd.Series(((macd == 1) & rsi_ok).astype(float), index=close.index)
    first = close.first_valid_index()
    if first is None:
        return vote * np.nan
    warmup = close.index.get_loc(first) + max(MACD_LONG[1] + MACD_LONG[2], RSI_LONG + 1)
    vote.iloc[:warmup] = np.nan                        # not enough history yet
    return vote


def build_regimes(prices, vix, buckets, smooth_days):
    votes = pd.DataFrame({t: stock_vote(prices[t], vix) for t in prices})
    raw = pd.DataFrame({r: votes[buckets[r]].mean(axis=1) for r in REGIMES})   # G, N, U, AB
    score = raw.rolling(smooth_days).mean()                                     # H, O, V, AC
    ready = score.dropna().round(10)
    winner = ready.idxmax(axis=1)                      # ties go to the first regime, like AJ
    winner_num = winner.map({r: i + 1 for i, r in enumerate(REGIMES)}).reindex(score.index)
    return votes, raw, score, winner_num


def lag_test(score, asset_price, window=20, max_lag=15):
    """Correlation of a regime score with the asset's 20-day return ending `lag` days ago."""
    past_return = asset_price / asset_price.shift(window) - 1
    return pd.Series({lag: score.corr(past_return.shift(lag)) for lag in range(max_lag + 1)})


# ----------------------------------------------------------------------------------
# Warning signals
# ----------------------------------------------------------------------------------
def yield_pct(s):
    """Yahoo has quoted some yield indices x10 in the past; bring them back to percent."""
    return s / 10 if s.dropna().median() > 20 else s


def compute_signal(spec, px, lookback):
    """Returns a table with the signal's level, a risk-on measure (higher = calmer) and a
    0/1 warning flag (1 = risk-off)."""
    a, b = spec["tickers"]
    if a not in px or b not in px or px[a].isna().all() or px[b].isna().all():
        return None
    out = pd.DataFrame(index=px.index)
    if spec["kind"] == "ratio":
        level = px[a] / px[b]
        change = level / level.shift(lookback) - 1
        out["level"], out["risk_on"] = level, change
        out["warning"] = (change < 0).astype(float).where(change.notna())
    elif spec["kind"] == "vixterm":
        level = px[a] / px[b]
        smooth = level.rolling(5).mean()
        out["level"], out["risk_on"] = level, 1 - smooth
        out["warning"] = (smooth > 1).astype(float).where(smooth.notna())
    else:  # yield curve
        level = yield_pct(px[a]) - yield_pct(px[b])
        out["level"], out["risk_on"] = level, level
        out["warning"] = (level < 0).astype(float).where(level.notna())
    return out


def lead_lag(risk_on, asset, window=20, max_shift=30):
    """Correlation of a risk-on measure with the asset's 20-day return, for windows ending
    k trading days from today. Positive k = a future window (the signal leads)."""
    return pd.Series({k: risk_on.corr(asset.shift(-k) / asset.shift(-k + window) - 1)
                      for k in range(-max_shift, max_shift + 1)})


def lag_verdict(corr):
    best = int(corr.idxmax())
    if corr.max() < 0.1:
        return best, "No clear link"
    if best > 2:
        return best, f"Leads by ~{best} days"
    if best < -2:
        return best, f"Lags by ~{-best} days"
    return best, "Moves at the same time"


def forward_returns(flag, asset, days=20):
    """Average asset return over the next `days` trading days, split by the flag."""
    fwd = asset.shift(-days) / asset - 1
    df = pd.DataFrame({"flag": flag, "fwd": fwd}).dropna()
    on, off = df[df.flag == 1].fwd, df[df.flag == 0].fwd
    return (on.mean() if len(on) else np.nan, off.mean() if len(off) else np.nan,
            len(on) / len(df) if len(df) else np.nan)


# ----------------------------------------------------------------------------------
# Backtesting
# ----------------------------------------------------------------------------------
def winner_from(raw, smooth):
    """Regime Winner and loser (1-4) for a given smoothing, with the sheet's tie rule."""
    sc = raw.rolling(smooth).mean()
    ready = sc.dropna().round(10)
    num = {r: i + 1 for i, r in enumerate(REGIMES)}
    win = ready.idxmax(axis=1).map(num).reindex(sc.index)
    lose = ready.idxmin(axis=1).map(num).reindex(sc.index)
    return win, lose


def hold(pos, asset_ret, cash_ret, delay, cost):
    """Daily returns of keeping `pos` (0 to 1) in the asset and the rest in cash. The position
    is decided at a close and traded `delay` days later; `cost` is charged per 100% traded."""
    held = pos.shift(delay)
    return held * asset_ret + (1 - held) * cash_ret - held.diff().abs().fillna(0) * cost


def rotate(choice, bucket_ret, cash_ret, delay, cost):
    """Hold the bucket picked each day (1-4) at equal weight. Switching buckets means selling
    everything and buying the new bucket, so it costs 2x."""
    held = choice.shift(delay)
    vals, idx = bucket_ret.to_numpy(), held.to_numpy()
    out = np.full(len(held), np.nan)
    ok = ~np.isnan(idx)
    out[ok] = vals[np.flatnonzero(ok), idx[ok].astype(int) - 1]
    ret = pd.Series(out, index=held.index).fillna(cash_ret)
    switched = (held != held.shift()) & held.notna() & held.shift().notna()
    return ret - switched.astype(float) * 2 * cost


def perf(ret, cash_ret, pos=None):
    """Summary stats for a daily return series."""
    ret = ret.dropna()
    if len(ret) < 60:
        return {}
    years = len(ret) / 252
    growth = (1 + ret).cumprod()
    excess = ret - cash_ret.reindex(ret.index).fillna(0)
    sd = ret.std()
    out = {"Total return": growth.iloc[-1] - 1,
           "Per year (CAGR)": growth.iloc[-1] ** (1 / years) - 1,
           "Volatility": sd * np.sqrt(252),
           "Sharpe": excess.mean() / sd * np.sqrt(252) if sd > 0 else np.nan,
           "Worst drop": (growth / growth.cummax() - 1).min()}
    if pos is not None:
        p = pos.reindex(ret.index)
        out["Time invested"] = p.mean()
        out["Trades per year"] = (p.diff().abs() > 0).sum() / years
    return out


def sharpe_of(ret, cash_ret):
    sd = ret.std()
    return (ret - cash_ret).mean() / sd * np.sqrt(252) if sd > 0 else np.nan


def fmt_perf(rows):
    """Format a {name: perf dict} table for display."""
    df = pd.DataFrame(rows).T
    for col in df.columns:
        if col in ("Sharpe", "Trades per year"):
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}" if col == "Sharpe"
                                  else f"{x:.1f}")
        else:
            df[col] = df[col].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}"
                                  if col in ("Total return", "Per year (CAGR)", "Worst drop")
                                  else f"{x:.0%}")
    return df.rename_axis("Strategy").reset_index()


def signal_flags(names, lookback):
    """Warning flags for the chosen signals (same rules as the Warning signals tab)."""
    tickers = sorted({t for n in names for t in SIGNALS[n]["tickers"]})
    if not tickers:
        return {}
    px = load_signal_prices(tuple(tickers), str(start)).reindex(prices.index).ffill()
    out = {}
    for n in names:
        result = compute_signal(SIGNALS[n], px, lookback)
        if result is not None:
            out[n] = result
    return out


def luck_test(ret_fn, series, n=1000, seed=0):
    """Shift the strategy's timing by a random amount (keeps how often and how long it is in
    each state) and see how often random timing does as well. Returns the share of random
    versions with a Sharpe at least as high as the real one."""
    rng = np.random.default_rng(seed)
    real = ret_fn(series)
    vals = series.to_numpy()
    sims = []
    for k in rng.integers(126, max(127, len(vals) - 126), n):
        sims.append(ret_fn(pd.Series(np.roll(vals, k), index=series.index)))
    sims = np.array(sims)
    return real, sims, float((sims >= real).mean())

# ----------------------------------------------------------------------------------
# Lab: expirations, gamma, macro quad, model v2 and the optimizer (research in lab.py)
# ----------------------------------------------------------------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_gex():
    return lab.fetch_gex()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_macro_inputs():
    return lab.fetch_fred("GDPC1"), lab.fetch_fred("CPIAUCSL")


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Downloading prices since 2005 for the Lab...")
def load_lab_prices(tickers):
    data = yf.download(list(tickers) + ["^VIX", "^VIX3M", "^IRX"], start="2005-01-01",
                       auto_adjust=False, progress=False)
    return data["Adj Close"]


@st.cache_data(ttl=6 * 60 * 60, show_spinner="Running the optimizer: about a minute the first "
                                              "time, instant after that...")
def run_lab(own_items):
    """Everything the optimizer needs, from 2005 on, then lab.run_optimizer."""
    own = {r: list(t) for r, t in own_items}
    tickers = sorted(set(t for v in own.values() for t in v) |
                     set(t for v in lab.ETF_BUCKETS.values() for t in v) | set(lab.BENCHMARKS))
    px = load_lab_prices(tuple(tickers))
    prices = px[[t for t in tickers if t in px and px[t].notna().any()]].dropna(how="all")
    vix = px["^VIX"].reindex(prices.index).ffill()
    votes = pd.DataFrame({t: stock_vote(prices[t], vix) for t in prices})
    quad = quad_for_dates(prices.index)
    cash = (yield_pct(px["^IRX"]).reindex(prices.index).ffill() / 100 / 252).fillna(0)
    cal = lab.expiration_calendar(2005, prices.index[-1].year + 1)
    extras = {"week_after": lab.expiration_flags(prices.index, cal)["week_after"],
              "vix_ratio": vix / px["^VIX3M"].reindex(prices.index).ffill()}
    notes = []
    try:
        extras["gex"] = load_gex()["gex"]
    except Exception:
        notes.append("gamma (SqueezeMetrics didn't respond)")
    try:
        gdp, cpi = load_macro_inputs()
        extras["macro"], _ = lab.macro_quads(gdp, cpi, prices.index)
    except Exception:
        notes.append("the GDP/CPI quad (FRED didn't respond)")
    result = lab.run_optimizer(prices, votes, quad, own, cash, extras)
    result["missing"] = notes
    return result


def pct_cols(df, cols, plus=True):
    """Format chosen columns as percents for display."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else (f"{x:+.1%}" if plus else f"{x:.0%}"))
    return out


def num_cols(df, cols, digits=2):
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    return out

# ----------------------------------------------------------------------------------
# Adding tickers: getETF-style ranking, saved buckets
# ----------------------------------------------------------------------------------
def yahoo_url(ticker):
    """Yahoo Finance page for a ticker (Yahoo writes BRK.B as BRK-B)."""
    return "https://finance.yahoo.com/quote/" + ticker.replace(".", "-") if ticker else None


def yahoo_column(label):
    """Table column that shows the ticker and opens its Yahoo Finance page when clicked."""
    return st.column_config.LinkColumn(label, display_text=r"https://finance\.yahoo\.com/quote/(.+)",
                                       help="Click a ticker to open it on Yahoo Finance")


def parse_tickers(text):
    """'tsla, panw  meta' -> ['TSLA', 'PANW', 'META'] (commas or spaces, no repeats)."""
    out = []
    for t in text.replace(",", " ").split():
        t = t.strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def quad_for_dates(index):
    """Macro quad (1-4) for each date, from the ETF_Bucketing calendar."""
    runs = [s.split(":") for s in QUAD_RUNS.split(";")]
    starts = np.array([m for m, _ in runs])
    quads = np.array([int(q) for _, q in runs])
    months = np.array(pd.DatetimeIndex(index).strftime("%Y-%m"))
    pos = np.searchsorted(starts, months, side="right") - 1
    return pd.Series(np.where(pos >= 0, quads[pos.clip(0)], 0), index=index)


@st.cache_data(ttl=24 * 60 * 60, show_spinner="Ranking tickers by quad (getETF method)...")
def rank_tickers(tickers):
    """Average daily return in each macro quad since 2005 (or the listing date), like the
    getETF / PerformanceRank macro. Returns one row per ticker."""
    data = yf.download(list(tickers), start=QUAD_START, auto_adjust=False, progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    if not isinstance(data.columns, pd.MultiIndex):
        close.columns = list(tickers)[:1]
    rows = []
    for t in tickers:
        s = close[t].dropna() if t in close else pd.Series(dtype=float)
        if len(s) < 30:
            rows.append({"Ticker": t, "Best quad": None, "Note": "No data from Yahoo"})
            continue
        ret = s.pct_change().dropna()
        quad = quad_for_dates(ret.index)
        avg = ret.groupby(quad).mean()
        avg = {q: avg.get(q, np.nan) for q in (1, 2, 3, 4)}
        order = sorted((q for q in avg if pd.notna(avg[q])), key=lambda q: -avg[q])
        first = s.index[0]
        note = "" if first <= pd.Timestamp(DEFAULT_START) else \
            f"Listed after {pd.Timestamp(DEFAULT_START):%b %d, %Y}: shorter history"
        rows.append({
            "Ticker": t, "Best quad": order[0] if order else None,
            "Bucket": REGIMES[order[0] - 1] if order else None,
            "getETF rank order": "-".join(f"Q{q}" for q in order),
            **{f"Q{q} avg daily %": round(avg[q] * 100, 3) if pd.notna(avg[q]) else None
               for q in (1, 2, 3, 4)},
            "Data since": f"{first:%Y-%m-%d}", "Note": note,
        })
    df = pd.DataFrame(rows)
    df["Best quad"] = df["Best quad"].astype("Int64")
    return df


def add_to_buckets(assignments):
    """Button callback: put each ticker in its bucket (and take it out of any other)."""
    for ticker, regime in assignments.items():
        for r in REGIMES:
            key = f"bucket_{r}"
            names = [t for t in parse_tickers(st.session_state[key]) if t != ticker]
            if r == regime:
                names.append(ticker)
            st.session_state[key] = ", ".join(names)


def add_compare(key):
    """Callback for the Compare against boxes: clean up a typed ticker and, if Yahoo has
    data for it, save it to Outside assets so it stays in the list (and in the page link)."""
    t = (st.session_state.get(key) or "").strip().upper()
    if not t:
        return
    start = str(st.session_state.get("start", DEFAULT_START))
    saved = parse_tickers(st.session_state["outside_text"])
    if load_asset(t, start).empty:
        st.session_state["compare_error"] = t
        st.session_state[key] = saved[0] if saved else None
        return
    st.session_state[key] = t
    if t not in saved:
        st.session_state["outside_text"] = ", ".join(saved + [t])


def reset_buckets():
    """Button callback: forget edits and go back to the default buckets."""
    for r in REGIMES:
        st.session_state.pop(f"bucket_{r}", None)
    st.session_state.pop("outside_text", None)
    st.session_state.pop("buckets_ready", None)
    st.query_params.clear()


# ----------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Bottoms-Up Regime Segmentation", layout="wide")
st.title("Bottoms-Up Regime Segmentation")

# Buckets come from the page link (?q1=...&q2=...) if it has them, otherwise the defaults.
# That way changes stick when you reload, and you can bookmark or share the link.
default_buckets, default_outside = DEFAULT_BUCKETS, DEFAULT_OUTSIDE
if "buckets_ready" not in st.session_state:
    qp = st.query_params
    for i, r in enumerate(REGIMES):
        st.session_state[f"bucket_{r}"] = ", ".join(
            parse_tickers(qp.get(f"q{i + 1}", "")) or default_buckets[r])
    st.session_state["outside_text"] = ", ".join(
        parse_tickers(qp.get("out", "")) or default_outside)
    st.session_state["buckets_ready"] = True

with st.sidebar:
    st.header("Settings")
    start = st.date_input("Start date", pd.Timestamp(DEFAULT_START), key="start")
    smooth_days = st.slider("Smoothing (days)", 5, 40, 19,
                            help="The spreadsheet uses a 19-day average of the raw votes.")
    st.subheader("Buckets")
    buckets = {}
    for r in REGIMES:
        st.text_input(r, key=f"bucket_{r}")
        buckets[r] = parse_tickers(st.session_state[f"bucket_{r}"])
    st.text_input("Outside assets", key="outside_text",
                  help="Tickers to compare the model against. Any Yahoo Finance symbol works. "
                       "You can also type a new one straight into a Compare against box.")
    outside_assets = parse_tickers(st.session_state["outside_text"])
    st.button("Reset buckets to defaults", on_click=reset_buckets)
    st.subheader("Warning signals")
    lookback = st.slider("Ratio lookback (days)", 20, 130, 65,
                         help="A ratio signal warns when the ratio is lower than it was this many "
                              "trading days ago. 65 days is about 13 weeks.")
    chosen = st.multiselect("Signals to include", list(SIGNALS), default=list(SIGNALS))
    if st.button("Refresh data"):
        st.cache_data.clear()

# Keep the page link in step with the buckets, so a reload or bookmark keeps them
user_buckets = {r: list(buckets[r]) for r in REGIMES}
link_params = {f"q{i + 1}": ",".join(buckets[r]) for i, r in enumerate(REGIMES)}
link_params["out"] = ",".join(outside_assets)
if all(buckets[r] == default_buckets[r] for r in REGIMES) and outside_assets == default_outside:
    if len(st.query_params):
        st.query_params.clear()
elif {k: st.query_params.get(k) for k in link_params} != link_params:
    st.query_params.from_dict(link_params)

stocks = sorted({t for names in buckets.values() for t in names})
if any(len(buckets[r]) == 0 for r in REGIMES):
    st.error("Every bucket needs at least one ticker.")
    st.stop()

prices_all, vix = load_prices(tuple(stocks), str(start))
outside_data = {t: load_asset(t, str(start)) for t in outside_assets}

# Tickers Yahoo didn't return
missing = [t for t in stocks if t not in prices_all.columns or prices_all[t].isna().all()]
missing += [t for t in outside_assets if outside_data[t].empty]
if missing:
    st.warning(f"No data from Yahoo for: {', '.join(missing)}. Check the ticker spelling.")
buckets = {r: [t for t in names if t not in missing] for r, names in buckets.items()}
outside_assets = [t for t in outside_assets if t not in missing]
if any(len(buckets[r]) == 0 for r in REGIMES):
    st.error("A bucket has no tickers with data. Fix the tickers in the sidebar.")
    st.stop()

stocks = sorted({t for names in buckets.values() for t in names})
prices = prices_all[stocks].dropna(how="all")
outside = pd.DataFrame({t: outside_data[t] for t in outside_assets}, index=prices.index)
votes, raw, score, winner_num = build_regimes(prices, vix.reindex(prices.index), buckets, smooth_days)
last = winner_num.last_valid_index()
st.caption(f"Data through {last.date()} · Yahoo Finance · refreshes every 6 hours")
if st.session_state.get("compare_error"):
    st.warning(f"No data from Yahoo for {st.session_state.pop('compare_error')}. "
               "Check the spelling (indexes start with ^, like ^GSPC; crypto looks like BTC-USD).")


def compare_box(label, key):
    """Pick a saved outside asset or type any ticker. Returns (ticker, its prices)."""
    t = st.selectbox(label, outside_assets, index=0 if outside_assets else None, key=key,
                     accept_new_options=True, on_change=add_compare, args=(key,),
                     placeholder="Type any ticker, e.g. SPY",
                     help="Pick one from the list or type any Yahoo Finance symbol: a stock, an "
                          "ETF, an index like ^GSPC, or crypto like BTC-USD. New tickers are "
                          "added to Outside assets in the sidebar.")
    if not t:
        return None, None
    px = outside[t] if t in outside else load_asset(t, str(start)).reindex(prices.index)
    if px.dropna().empty:
        st.warning(f"No data from Yahoo for {t}.")
        return None, None
    st.caption(f"[Open {t} on Yahoo Finance ↗]({yahoo_url(t)})")
    return t, px

tab_model, tab_signals, tab_add, tab_bt, tab_lab = st.tabs(
    ["Regime model", "Warning signals", "Add tickers", "Backtest", "Lab"])

# ==================================================================================
# Tab 1: the regime model
# ==================================================================================
with tab_model:
    # Headline: current regime and how each score moved over the last week
    cols = st.columns(5)
    cols[0].metric("Current regime", REGIMES[int(winner_num[last]) - 1])
    week_ago = score.index[max(score.index.get_loc(last) - 5, 0)]
    for col, r in zip(cols[1:], REGIMES):
        col.metric(r, f"{score.loc[last, r]:.2f}",
                   f"{score.loc[last, r] - score.loc[week_ago, r]:+.2f} vs 1 wk ago")

    # Chart 1: regime scores with an outside asset on the right axis
    asset, asset_px = compare_box("Compare against (pick one or type any ticker)", "cmp_model")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for r in REGIMES:
        fig.add_trace(go.Scatter(x=score.index, y=score[r], name=r,
                                 line=dict(color=COLORS[r], width=1.5)), secondary_y=False)
    if asset:
        fig.add_trace(go.Scatter(x=asset_px.index, y=asset_px, name=asset,
                                 line=dict(color="black", width=2)), secondary_y=True)
    fig.update_yaxes(title_text="Regime score (share of stocks voting buy)", range=[0, 1],
                     secondary_y=False)
    fig.update_yaxes(title_text=f"{asset} price" if asset else "", secondary_y=True)
    fig.update_layout(height=520, hovermode="x unified", legend=dict(orientation="h", y=1.08),
                      xaxis=dict(rangeslider=dict(visible=True)), margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")

    # Chart 2: Regime Winner
    fig2 = go.Figure(go.Scatter(x=winner_num.index, y=winner_num, line_shape="hv",
                                line=dict(color="black", width=1), name="Regime Winner"))
    fig2.update_layout(height=220, title="Regime Winner", margin=dict(t=40),
                       yaxis=dict(tickvals=[1, 2, 3, 4],
                                  ticktext=[f"{i + 1} {r}" for i, r in enumerate(REGIMES)]))
    st.plotly_chart(fig2, width="stretch")

    left, right = st.columns(2)

    # Lag test
    with left:
        st.subheader("Lag test")
        if asset:
            regime_for_lag = st.selectbox("Regime score", REGIMES, index=0)
            corr = lag_test(score[regime_for_lag], asset_px)
            best = int(corr.idxmax())
            st.write(f"**{regime_for_lag}** matches **{asset}**'s 20-day return best "
                     f"**{best} trading days back** (correlation {corr[best]:.2f}, "
                     f"vs {corr[0]:.2f} today).")
            bar = go.Figure(go.Bar(x=corr.index, y=corr.values,
                                   marker_color=["#2a7de1" if i == best else "#bbbbbb"
                                                 for i in corr.index]))
            bar.update_layout(height=280, xaxis_title="Days back", yaxis_title="Correlation",
                              margin=dict(t=10))
            st.plotly_chart(bar, width="stretch")
        else:
            st.info("Pick or type a ticker in Compare against to run the lag test.")

    # Latest votes
    with right:
        st.subheader(f"Votes on {last.date()}")
        latest = pd.DataFrame(
            [(r, t, "BUY" if votes.loc[last, t] == 1 else "SELL")
             for r in REGIMES for t in buckets[r]],
            columns=["Regime", "Stock", "Vote"])
        latest["Stock"] = latest["Stock"].map(yahoo_url)
        st.dataframe(latest, hide_index=True, width="stretch", height=280,
                     column_config={"Stock": yahoo_column("Stock")})
        st.caption("Click a stock to open its Yahoo Finance page.")

    # Full data download
    table = votes.copy()
    for r in REGIMES:
        table[f"{r} (Raw Votes%)"] = raw[r]
        table[f"{r} score"] = score[r]
    table["Regime Winner"] = winner_num
    # "price" in the name keeps a compared stock (e.g. NVDA) from clashing with its vote column
    table = table.join(outside.add_suffix(" price"))
    with st.expander("All data"):
        st.dataframe(table.dropna(subset=["Regime Winner"]).iloc[::-1], width="stretch")
    st.download_button("Download CSV", table.to_csv(index_label="Date"), "regime_output.csv",
                       "text/csv")

# ==================================================================================
# Tab 3: add tickers (written before tab 2's code so tab 2 can stop early safely)
# ==================================================================================
with tab_add:
    st.markdown(
        "Type ticker symbols and the site ranks each one with the **getETF method** from "
        "ETF_Bucketing.xlsm: its average daily return in each macro quad since 2005 (or since it "
        "started trading). The stock belongs in the bucket where it ranks first.")
    add_text = st.text_area("Tickers (separate with commas or spaces, up to 100)",
                            key="add_text", placeholder="e.g. NVDA, KO, NEM, ULTA")
    if st.button("Rank tickers", type="primary"):
        wanted = parse_tickers(add_text)[:100]
        if wanted:
            st.session_state["ranked"] = rank_tickers(tuple(wanted))
        else:
            st.warning("Type at least one ticker first.")
    ranked = st.session_state.get("ranked")
    if ranked is not None and len(ranked):
        st.dataframe(ranked.assign(Ticker=ranked["Ticker"].map(yahoo_url)), hide_index=True,
                     width="stretch", column_config={"Ticker": yahoo_column("Ticker")})
        valid = ranked.dropna(subset=["Best quad"]).set_index("Ticker")
        if len(valid):
            pick_add = st.multiselect("Add these to their best bucket", list(valid.index),
                                      default=list(valid.index), key="pick_add")
            st.button(f"Add {len(pick_add)} ticker(s) to the model", type="primary",
                      on_click=add_to_buckets, args=({t: valid.loc[t, "Bucket"] for t in pick_add},),
                      disabled=not pick_add)

    st.divider()
    st.subheader("Current buckets")
    longest = max(len(v) for v in user_buckets.values())
    st.dataframe(pd.DataFrame({r: [yahoo_url(t) for t in user_buckets[r]]
                               + [None] * (longest - len(user_buckets[r])) for r in REGIMES}),
                 hide_index=True, width="stretch",
                 column_config={r: yahoo_column(r) for r in REGIMES})
    st.caption("Click a ticker to open its Yahoo Finance page. "
               "Your changes are saved in this page's link, so they stay when you reload. "
               "Bookmark or share the link to keep them. To remove a ticker, delete it from its "
               "box in the sidebar. **Reset buckets to defaults** in the sidebar undoes everything.")

# ==================================================================================
# Tab 4: backtest (written before tab 2's code so tab 2 can stop early safely)
# ==================================================================================
with tab_bt:
    st.markdown(
        "Does the model actually work? This tab checks (1) whether it picks the same quad as the "
        "ETF_Bucketing quad calendar, and (2) whether trading on it would have beaten just "
        "holding. **No peeking:** a signal at one day's close is traded at the close a day or "
        "more later, and every strategy is tested over the same dates.")
    c1, c2 = st.columns(2)
    with c1:
        bt_asset, bt_px = compare_box("Backtest on (pick one or type any ticker)", "bt_asset")
    with c2:
        invest_in = st.multiselect("Regime strategy: be invested during these regimes (cash "
                                   "otherwise)", REGIMES, default=["Disinflation", "Reflation"],
                                   key="bt_regimes")
    c3, c4, c5, c6 = st.columns(4)
    delay = c3.slider("Trade delay (days)", 1, 5, 1, key="bt_delay",
                      help="1 = trade at the next day's close after the signal.")
    cost = c4.slider("Cost per trade (basis points)", 0, 50, 5, key="bt_cost",
                     help="1 basis point = 0.01% of the amount traded.") / 10000
    warn_n = c5.slider("Warning strategy: go to cash at this many warnings", 1, 7, 4,
                       key="bt_warn")
    use_tbill = c6.checkbox("Cash earns the T-bill rate", True, key="bt_tbill",
                            help="Uses the 3-month Treasury yield (^IRX). Off = cash earns 0%.")

    if bt_asset is None or not invest_in:
        st.info("Pick a ticker and at least one regime to run the backtest.")
    else:
        # ---- Inputs -------------------------------------------------------------
        asset_ret = bt_px.pct_change()
        cash_ret = pd.Series(0.0, index=prices.index)
        if use_tbill:
            irx = load_signal_prices(("^IRX",), str(start))
            if "^IRX" in irx:
                cash_ret = (yield_pct(irx["^IRX"]).reindex(prices.index).ffill() / 100 / 252
                            ).fillna(0)
        stock_ret = prices.pct_change()
        bucket_ret = pd.DataFrame({r: stock_ret[buckets[r]].mean(axis=1) for r in REGIMES})
        all_ret = stock_ret[stocks].mean(axis=1)
        _, loser = winner_from(raw, smooth_days)
        nums = [REGIMES.index(r) + 1 for r in invest_in]
        sigs = signal_flags(chosen, lookback) if chosen else {}
        warn_cnt = (pd.DataFrame({n: s["warning"] for n, s in sigs.items()})
                    .sum(axis=1, min_count=1) if sigs else None)

        valid = winner_num.notna() & bt_px.notna()
        if warn_cnt is not None:
            valid &= warn_cnt.notna()
        if valid.sum() < 260:
            st.warning("Less than a year of overlapping data. Pick an earlier start date.")
        else:
            first = valid.idxmax()
            window = prices.index[prices.index.get_loc(first) + delay:]
            asset_ret, cash_w = asset_ret.fillna(0), cash_ret

            # ---- Strategies -----------------------------------------------------------
            pos = {}
            pos[f"Buy & hold {bt_asset}"] = pd.Series(1.0, index=prices.index)
            regime_on = winner_num.isin(nums).astype(float).where(winner_num.notna())
            label_rf = f"Regime strategy ({' + '.join(invest_in)})"
            pos[label_rf] = regime_on
            pos["Opposite of regime strategy"] = 1 - regime_on
            if warn_cnt is not None:
                pos[f"Cash at {warn_n}+ warnings"] = (warn_cnt < warn_n).astype(float).where(
                    warn_cnt.notna())
            rets = {k: hold(p, asset_ret, cash_ret, delay, cost) for k, p in pos.items()}
            rets["Winning bucket's stocks"] = rotate(winner_num, bucket_ret, cash_ret, delay, cost)
            rets["Losing bucket's stocks"] = rotate(loser, bucket_ret, cash_ret, delay, cost)
            rets["All bucket stocks (equal weight)"] = all_ret
            rets = pd.DataFrame(rets).loc[window]
            if rets.isna().any().any():
                rets = rets.fillna(0)

            st.subheader("1. Strategy results")
            st.caption(f"{window[0]:%b %d, %Y} to {window[-1]:%b %d, %Y} "
                       f"({len(window) / 252:.1f} years). Sharpe = return above cash per unit "
                       "of risk; above 1 is very good, below 0.5 is weak.")
            rows = {k: perf(rets[k], cash_w, pos.get(k)) for k in rets}
            st.dataframe(fmt_perf(rows), hide_index=True, width="stretch")

            growth = 10000 * (1 + rets).cumprod()
            fig_g = go.Figure([go.Scatter(x=growth.index, y=growth[k], name=k,
                                          line=dict(width=2 if i < 2 else 1.2))
                               for i, k in enumerate(growth)])
            fig_g.update_layout(height=430, title="Growth of $10,000 (log scale)",
                                yaxis_type="log", hovermode="x unified",
                                legend=dict(orientation="h", y=-0.15), margin=dict(t=40))
            st.plotly_chart(fig_g, width="stretch")
            dd = growth / growth.cummax() - 1
            fig_d = go.Figure([go.Scatter(x=dd.index, y=dd[k], name=k, line=dict(width=1.2))
                               for k in list(dd)[:4]])
            fig_d.update_layout(height=260, title="Drawdowns (drop from the last high)",
                                yaxis_tickformat=".0%", hovermode="x unified",
                                legend=dict(orientation="h", y=-0.25), margin=dict(t=40))
            st.plotly_chart(fig_d, width="stretch")

            # ---- Accuracy vs the quad calendar ---------------------------------------
            st.subheader("2. Does it pick the right quad?")
            cal = quad_for_dates(prices.index)
            both = pd.DataFrame({"Calendar": cal, "Model": winner_num}).loc[window].dropna()
            hit = (both["Calendar"] == both["Model"]).mean()
            p_cal = both["Calendar"].value_counts(normalize=True)
            p_mod = both["Model"].value_counts(normalize=True)
            chance = sum(p_cal.get(q, 0) * p_mod.get(q, 0) for q in (1, 2, 3, 4))
            shifts = range(-120, 121, 5)
            agree = pd.Series({k: (winner_num == cal.shift(-k))[winner_num.notna()
                                                                & cal.shift(-k).notna()].mean()
                               for k in shifts})
            best_k = int(agree.idxmax())
            a1, a2, a3 = st.columns(3)
            a1.metric("Days it matched the calendar", f"{hit:.0%}")
            a2.metric("Matching by pure chance", f"{chance:.0%}")
            when = (f"{abs(best_k)} days ahead" if best_k > 0 else
                    f"{abs(best_k)} days behind" if best_k < 0 else "Same day")
            a3.metric("Best match when shifted", when)
            how = (f"the quad {abs(best_k)} trading days later, so it tends to call quads early"
                   if best_k > 0 else
                   f"the quad {abs(best_k)} trading days earlier, so it tends to call quads late"
                   if best_k < 0 else "the same day's quad")
            st.caption(f"The model's calls line up best with {how} ({agree.max():.0%} of days "
                       "matched). The calendar is the macro quad from ETF_Bucketing.xlsm, based "
                       "on growth and inflation data that only comes out weeks later.")
            names = {i + 1: f"{i + 1} {r}" for i, r in enumerate(REGIMES)}
            conf = pd.crosstab(both["Calendar"].map(names), both["Model"].map(names),
                               normalize="index").reindex(index=names.values(),
                                                          columns=names.values()).fillna(0)
            conf = conf.map(lambda x: f"{x:.0%}").rename_axis("Real quad \\ Model said")
            l, r_ = st.columns(2)
            with l:
                st.write("**When the real quad was … the model said …** (each row adds to 100%)")
                st.dataframe(conf.reset_index(), hide_index=True, width="stretch")
            with r_:
                by_year = (both["Calendar"] == both["Model"]).groupby(both.index.year).mean()
                fig_y = go.Figure(go.Bar(x=by_year.index.astype(str), y=by_year.values,
                                         marker_color="#2a7de1"))
                fig_y.add_hline(y=chance, line_dash="dash", line_color="gray",
                                annotation_text="chance")
                fig_y.update_layout(height=260, title="Match rate by year",
                                    yaxis_tickformat=".0%", margin=dict(t=40))
                st.plotly_chart(fig_y, width="stretch")

            # ---- Robustness -----------------------------------------------------------
            st.subheader("3. Is it robust, or just luck?")
            bh = rets[f"Buy & hold {bt_asset}"]
            bh_stats = perf(bh, cash_w)
            st.write(f"Buy & hold {bt_asset} for comparison: "
                     f"**{bh_stats['Per year (CAGR)']:+.1%} per year, Sharpe "
                     f"{bh_stats['Sharpe']:.2f}**.")

            # Settings grid
            smooths, delays = [5, 10, 15, 19, 25, 30, 40], [1, 2, 3, 5]
            grid_start = prices.index[min(prices.index.get_loc(first) + 40,
                                          len(prices.index) - 1)]
            g_cagr, g_sharpe = {}, {}
            for sd in smooths:
                w_sd, _ = winner_from(raw, sd)
                on = w_sd.isin(nums).astype(float).where(w_sd.notna())
                for d in delays:
                    rr = hold(on, asset_ret, cash_ret, d, cost).loc[grid_start:].fillna(0)
                    st_ = perf(rr, cash_w)
                    g_cagr.setdefault(f"{d}-day delay", {})[f"{sd} days"] = st_["Per year (CAGR)"]
                    g_sharpe.setdefault(f"{d}-day delay", {})[f"{sd} days"] = st_["Sharpe"]
            gl, gr = st.columns(2)
            with gl:
                st.write(f"**Regime strategy, return per year** by smoothing (rows) and trade delay")
                st.dataframe(pd.DataFrame(g_cagr).map(lambda x: f"{x:+.1%}")
                             .rename_axis("Smoothing").reset_index(), hide_index=True,
                             width="stretch")
            with gr:
                st.write("**Regime strategy, Sharpe** by smoothing (rows) and trade delay")
                st.dataframe(pd.DataFrame(g_sharpe).map(lambda x: f"{x:.2f}")
                             .rename_axis("Smoothing").reset_index(), hide_index=True,
                             width="stretch")
            st.caption("If only one or two cells beat buy & hold, the result is fragile. A real "
                       "edge should hold up across most settings.")

            # Every regime combination
            combos = {}
            for k in range(1, 5):
                for combo in itertools.combinations(REGIMES, k):
                    cn = [REGIMES.index(r) + 1 for r in combo]
                    on = winner_num.isin(cn).astype(float).where(winner_num.notna())
                    rr = hold(on, asset_ret, cash_ret, delay, cost).loc[window].fillna(0)
                    combos[" + ".join(combo)] = perf(rr, cash_w, on)
            combo_df = fmt_perf(combos).rename(columns={"Strategy": "Invested during"})
            combo_df["_s"] = [combos[c]["Sharpe"] for c in combo_df["Invested during"]]
            combo_df = combo_df.sort_values("_s", ascending=False).drop(columns="_s")
            st.write("**Every combination of regimes to be invested in** (sorted by Sharpe)")
            st.dataframe(combo_df[["Invested during", "Per year (CAGR)", "Sharpe", "Worst drop",
                                   "Time invested", "Trades per year"]],
                         hide_index=True, width="stretch")
            st.caption("Trying 15 versions and keeping the best is how backtests fool people. "
                       "The best row here is probably luckier than it will be in the future.")

            # Year by year
            yearly = (1 + rets).groupby(rets.index.year).prod() - 1
            st.write("**Year by year**")
            st.dataframe(yearly.map(lambda x: f"{x:+.1%}").rename_axis("Year").reset_index()
                         .astype({"Year": str}), hide_index=True, width="stretch")

            # Luck test
            held_rf = regime_on.shift(delay).loc[window].fillna(0)
            a_w, c_w = asset_ret.loc[window], cash_ret.loc[window]

            def rf_sharpe(h):
                r_ = h * a_w + (1 - h) * c_w - h.diff().abs().fillna(0) * cost
                return sharpe_of(r_, c_w)

            br_w = bucket_ret.loc[window].to_numpy()
            held_bk = winner_num.shift(delay).loc[window].ffill().bfill()

            def bk_sharpe(h):
                r_ = br_w[np.arange(len(h)), h.to_numpy().astype(int) - 1]
                r_ = pd.Series(r_, index=h.index).fillna(c_w)
                sw = (h != h.shift()).astype(float)
                sw.iloc[0] = 0
                return sharpe_of(r_ - sw * 2 * cost, c_w)

            real1, sims1, p1 = luck_test(rf_sharpe, held_rf)
            real2, sims2, p2 = luck_test(bk_sharpe, held_bk)
            st.write("**Luck test:** the same strategy with its timing shifted to 1,000 random "
                     "dates (same amount of time in and out, just at the wrong times). Real "
                     "skill should beat at least 95% of them.")
            lt = pd.DataFrame({
                "Strategy": [label_rf, "Winning bucket's stocks"],
                "Real Sharpe": [f"{real1:.2f}", f"{real2:.2f}"],
                "Average random Sharpe": [f"{np.nanmean(sims1):.2f}", f"{np.nanmean(sims2):.2f}"],
                "Random versions that did as well": [f"{p1:.0%}", f"{p2:.0%}"],
                "Verdict": ["Likely skill" if p1 <= 0.05 else "Could be luck",
                            "Likely skill" if p2 <= 0.05 else "Could be luck"]})
            st.dataframe(lt, hide_index=True, width="stretch")

            st.warning(
                "**Read before trusting any of this.** (1) The 20 stocks were picked with "
                "getETF using data through 2026, which includes these same years, so the bucket "
                "strategies get a head start they wouldn't have had in real time. (2) The quad "
                "calendar is known with hindsight. (3) This is one stretch of history with many "
                "versions tested, so the best numbers are likely too good. (4) Costs, taxes and "
                "slippage are rough guesses. Research only, not financial advice.")
            out = rets.copy()
            out.columns = [f"{c} daily return" for c in out.columns]
            out["Regime Winner"] = winner_num.loc[window]
            st.download_button("Download backtest CSV", out.to_csv(index_label="Date"),
                               "backtest.csv", "text/csv")

# ==================================================================================
# Tab 5: Lab (written before tab 2's code so tab 2 can stop early safely)
# ==================================================================================
with tab_lab:
    st.markdown(
        "Extra research to make the model better: options **expiration dates**, dealer "
        "**gamma**, a **GDP/CPI quad** from real economic data, and an **optimizer** that "
        "tries 120+ versions of the model (relative strength, sector ETFs, buckets re-picked "
        "every year) and the extra filters. It picks the best version using **2010-2018 only**, "
        "then grades it on **2019 onward**, years it never saw.")
    qqq_long = load_asset("QQQ", "2005-01-01")
    now = pd.Timestamp.today().normalize()
    exp_cal = lab.expiration_calendar(2005, now.year + 1)

    # ---- 1. Expirations ----------------------------------------------------------------
    st.subheader("1. Options expirations")
    upcoming = exp_cal[exp_cal["Date"] >= now].head(8).copy()
    upcoming["Days away"] = (upcoming["Date"] - now).dt.days
    upcoming["Date"] = upcoming["Date"].dt.strftime("%a %b %d, %Y")
    e1, e2 = st.columns([2, 3])
    with e1:
        st.write("**Coming up**")
        st.dataframe(upcoming, hide_index=True, width="stretch")
        st.caption("Monthly stock and index options expire on the third Friday (the day before "
                   "if it's a holiday). Quad witching is the March, June, September and December "
                   "one. VIX options and futures expire on a Wednesday, 30 days before the next "
                   "month's third Friday. Holidays here follow the usual NYSE schedule.")
    if len(qqq_long.dropna()) > 300:
        effects, by_off = lab.expiration_effects(qqq_long.pct_change(), exp_cal)
        with e2:
            st.write(f"**What QQQ did around expirations** "
                     f"({qqq_long.index[0]:%Y}-{qqq_long.index[-1]:%Y})")
            show = pct_cols(effects, ["Avg daily return"])
            show = pct_cols(show, ["% of days up", "Volatility (yearly)"], plus=False)
            show["Avg daily return"] = effects["Avg daily return"].map(lambda x: f"{x:+.3%}")
            st.dataframe(show, hide_index=True, width="stretch")
        fig_ox = go.Figure(go.Bar(x=by_off.index.astype(int), y=by_off.values,
                                  marker_color=["#d62728" if v < 0 else "#2a7de1"
                                                for v in by_off.values]))
        fig_ox.update_layout(height=260, margin=dict(t=40), yaxis_tickformat=".2%",
                             title="QQQ average daily return by trading day around monthly "
                                   "expiration (0 = expiration day)",
                             xaxis_title="Trading days from expiration")
        st.plotly_chart(fig_ox, width="stretch")

    # ---- 2. Gamma ------------------------------------------------------------------------
    st.subheader("2. Dealer gamma (GEX) and dark-pool buying (DIX)")
    try:
        gx = load_gex()
        latest = gx.iloc[-1]
        g1, g2, g3 = st.columns(3)
        g1.metric(f"GEX on {gx.index[-1]:%b %d}", f"${latest['gex'] / 1e9:.2f} bn",
                  f"{(latest['gex'] - gx['gex'].iloc[-6]) / 1e9:+.2f} bn vs 1 wk ago")
        g2.metric("Gamma is", "Positive (calmer)" if latest["gex"] >= 0 else
                  "Negative (bigger swings)")
        g3.metric("DIX", f"{latest['dix']:.1%}",
                  f"{(latest['dix'] - gx['dix'].iloc[-6]) * 100:+.1f} pts vs 1 wk ago")
        st.caption("GEX estimates how much options dealers must buy or sell when the S&P 500 "
                   "moves 1%. Positive: dealers trade against the move, which tends to calm the "
                   "market. Negative: they trade with it, which tends to make swings bigger. DIX "
                   "is the share of dark-pool volume that is buying. Data: SqueezeMetrics, "
                   "since May 2011.")
        fig_gx = make_subplots(specs=[[{"secondary_y": True}]])
        fig_gx.add_trace(go.Bar(x=gx.index, y=gx["gex"] / 1e9, name="GEX ($bn)",
                                marker_color=np.where(gx["gex"] < 0, "#d62728", "#2a7de1")),
                         secondary_y=False)
        fig_gx.add_trace(go.Scatter(x=gx.index, y=gx["price"], name="S&P 500",
                                    line=dict(color="black", width=1.5)), secondary_y=True)
        fig_gx.update_layout(height=360, hovermode="x unified", margin=dict(t=30),
                             legend=dict(orientation="h", y=1.1),
                             xaxis=dict(rangeslider=dict(visible=True)))
        fig_gx.update_yaxes(title_text="GEX ($bn)", secondary_y=False)
        fig_gx.update_yaxes(title_text="S&P 500", secondary_y=True)
        st.plotly_chart(fig_gx, width="stretch")
        gt = lab.gex_table(gx["gex"], qqq_long)
        if len(gt):
            cols_ret = [c for c in gt.columns if "return" in c]
            cols_pct = [c for c in gt.columns if c.startswith("%") or "Volatility" in c]
            st.write("**What QQQ did over the next 20 trading days, by gamma level**")
            st.dataframe(pct_cols(pct_cols(gt, cols_ret), cols_pct, plus=False),
                         hide_index=True, width="stretch")
    except Exception:
        st.info("Couldn't reach SqueezeMetrics for gamma data right now. Try again later.")

    # ---- 3. Macro quad --------------------------------------------------------------------
    st.subheader("3. The quad from real economic data (GDP and CPI)")
    try:
        gdp, cpi = load_macro_inputs()
        lab_index = qqq_long.dropna().index
        mq, mdetail = lab.macro_quads(gdp, cpi, lab_index)
        last_q = mdetail.dropna(subset=["Macro quad"]).iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("GDP/CPI quad today", f"{int(last_q['Macro quad'])} "
                  f"{REGIMES[int(last_q['Macro quad']) - 1]}")
        m2.metric("Real GDP growth (YoY)", f"{last_q['GDP growth (YoY)']:.1%}",
                  "speeding up" if last_q["Growth speeding up"] == 1 else "slowing",
                  delta_color="normal" if last_q["Growth speeding up"] == 1 else "inverse")
        m3.metric("CPI inflation (YoY)", f"{last_q['CPI inflation (YoY)']:.1%}",
                  "speeding up" if last_q["Inflation speeding up"] == 1 else "slowing",
                  delta_color="inverse" if last_q["Inflation speeding up"] == 1 else "normal")
        cal_q = quad_for_dates(lab_index)
        both_c = pd.DataFrame({"m": mq, "c": cal_q}).dropna()
        both_m = pd.DataFrame({"m": mq, "w": winner_num}).dropna()
        st.caption(
            f"Growth or inflation 'speeding up' means its year-over-year rate is higher than "
            f"the last reading. Each number counts only from the day it was published (GDP about "
            f"a month after the quarter, CPI about two weeks after the month), using today's "
            f"revised figures. This quad matched the ETF_Bucketing calendar on "
            f"{(both_c['m'] == both_c['c']).mean():.0%} of days and your stock model's Regime "
            f"Winner on {(both_m['m'] == both_m['w']).mean():.0%} of days.")
        fig_mq = go.Figure()
        fig_mq.add_trace(go.Scatter(x=mq.index, y=mq, name="GDP/CPI quad", line_shape="hv",
                                    line=dict(color="#2a7de1", width=1.5)))
        fig_mq.add_trace(go.Scatter(x=cal_q.index, y=cal_q, name="ETF_Bucketing calendar",
                                    line_shape="hv", line=dict(color="gray", width=1, dash="dot")))
        fig_mq.update_layout(height=240, margin=dict(t=20), legend=dict(orientation="h", y=1.15),
                             yaxis=dict(tickvals=[1, 2, 3, 4],
                                        ticktext=[f"{i + 1} {r}" for i, r in enumerate(REGIMES)]))
        st.plotly_chart(fig_mq, width="stretch")
    except Exception:
        st.info("Couldn't reach FRED for GDP and CPI data right now. Try again later.")

    # ---- 4. Optimizer ---------------------------------------------------------------------
    st.subheader("4. Optimizer")
    st.markdown(
        "**Version 2 of the model**, tested every way:\n"
        "- **Bucket sources:** your 20 stocks, sector ETFs (XLK, XLE, GLD, TLT and so on), or "
        "**walk-forward picks**: every January, getETF re-ranks your stocks and the ETFs using "
        "only the previous 5 years and keeps the 5 names that fit each quad best.\n"
        "- **Scoring:** the original BUY/SELL votes, **relative strength** (each bucket's return "
        "over 10, 20, 40 or 60 days minus the average of all four), or the GDP/CPI quad.\n"
        "- **Smoothing** of 1, 5, 10 or 20 days, holding the **top 1 or top 2** buckets.\n"
        "- **Extra filters:** " + "; ".join(f"*{k}*: {v.lower()}" for k, v in
                                           lab.OVERLAYS.items()) + ".\n\n"
        "Trades happen the day after each signal and pay 5 basis points; cash earns the T-bill "
        "rate. The bucket sources include your current 20 stocks, so that row has a hindsight "
        "head start; the ETF and walk-forward versions don't.")
    if st.button("Run the optimizer", type="primary") or st.session_state.get("lab_ran"):
        st.session_state["lab_ran"] = True
        res = run_lab(tuple((r, tuple(buckets[r])) for r in REGIMES))
        if res["missing"]:
            st.warning("Ran without " + " and ".join(res["missing"]) + ".")
        td = res["today"]
        names = "; ".join(
            f"**{r}**: " + ", ".join(f"[{t}]({yahoo_url(t)})" for t in td["names"][r])
            for r in td["buckets"])
        st.success(f"**Today's pick** (signal from {td['date']:%b %d, %Y}): {names}. "
                   f"Invested: **{td['exposure']:.0%}** (the rest in cash).  \n"
                   f"Setup: {td['setup']}")

        st.write("**The winner vs the benchmarks.** 2010-2018 is where it was picked; 2019+ "
                 "is the real test.")
        summ = res["summary"]
        summ = pct_cols(summ, ["2010-2018 per year", "2019+ per year", "2019+ worst drop"])
        summ = num_cols(summ, ["2010-2018 Sharpe", "2019+ Sharpe"])
        st.dataframe(summ, hide_index=True, width="stretch")
        growth = 10000 * (1 + res["curves"].fillna(0)).cumprod()
        fig_o = go.Figure([go.Scatter(x=growth.index, y=growth[c], name=c,
                                      line=dict(width=2.2 if i == 0 else 1.3))
                           for i, c in enumerate(growth)])
        fig_o.add_vrect(x0=pd.Timestamp(res["test_start"]), x1=growth.index[-1], fillcolor="#2ca02c",
                        opacity=0.07, line_width=0, annotation_text="Unseen test years",
                        annotation_position="top left")
        fig_o.update_layout(height=420, yaxis_type="log", hovermode="x unified",
                            title="Growth of $10,000 (log scale)", margin=dict(t=40),
                            legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_o, width="stretch")

        lk = res["luck"]
        st.write(f"**Luck test (2019+):** real Sharpe **{lk['real']:.2f}** vs "
                 f"{lk['avg_random']:.2f} on average when the same picks are moved to random "
                 f"dates; **{lk['share_as_good']:.0%}** of random versions did as well. "
                 + ("That's under 5%, so it looks like skill." if lk["share_as_good"] <= 0.05
                    else "That's over 5%, so luck can't be ruled out."))

        core = res["core"]
        st.write(f"**All {len(core)} versions, best on 2010-2018 first.** Does doing well "
                 f"before 2019 predict doing well after? Rank correlation: "
                 f"**{res['rho']:.2f}** (1 = perfectly, 0 = not at all, negative = the "
                 f"opposite).")
        core_show = core[["Setup", "Train Sharpe", "Train per year", "Test Sharpe",
                          "Test per year", "Test worst drop"]]
        core_show = num_cols(pct_cols(core_show, ["Train per year", "Test per year",
                                                  "Test worst drop"]),
                             ["Train Sharpe", "Test Sharpe"])
        st.dataframe(core_show, hide_index=True, width="stretch", height=380)

        st.write("**Extra filters added to the best version** (picked on 2010-2018 again)")
        ov = res["overlays"]
        ov_show = num_cols(pct_cols(pct_cols(ov, ["Train per year", "Test per year",
                                                  "Test worst drop"]),
                                    ["Average invested"], plus=False),
                           ["Train Sharpe", "Test Sharpe"])
        st.dataframe(ov_show, hide_index=True, width="stretch")

        wf = res["walk_forward"]
        if wf:
            yr = max(wf)
            st.write(f"**Walk-forward buckets for {yr}** (picked using {yr - 5}-{yr - 1} only)")
            st.dataframe(pd.DataFrame({r: pd.Series(wf[yr][r], dtype=str) for r in REGIMES})
                         .fillna(""), hide_index=True, width="stretch")
            with st.expander("Walk-forward buckets for every year"):
                st.dataframe(pd.DataFrame({y: {r: ", ".join(m[r]) for r in REGIMES}
                                           for y, m in sorted(wf.items())}).T
                             .rename_axis("Year").reset_index().astype({"Year": str}),
                             hide_index=True, width="stretch")
        st.warning(
            "**Read before trusting it.** Picking the best of 120+ versions makes the winner "
            "look better than it is, which is why only the 2019+ numbers count. GDP data is "
            "today's revised version, not what was known at the time. Gamma data starts in "
            "May 2011. Research only, not financial advice.")
    else:
        st.caption("Takes about a minute the first time (it downloads 20 years of prices for "
                   "about 40 tickers), then it's saved for 6 hours.")

# ==================================================================================
# Tab 2: warning signals
# ==================================================================================
with tab_signals:
    st.markdown(
        "Indicators that are often said to move **before** stocks do. Each one is turned into "
        "a simple risk-off warning (1) or all-clear (0), tested for whether it leads or lags the "
        "market, and checked against what the market did next. None of them is reliable on its "
        "own; the point is to test them.")
    sig_asset, asset_px = compare_box("Test against (pick one or type any ticker)", "sig_asset")
    if sig_asset is None:
        st.info("Pick or type a ticker in Test against to test the signals.")
        st.stop()
    if not chosen:
        st.info("Pick at least one signal in the sidebar.")
        st.stop()

    sig_tickers = sorted({t for name in chosen for t in SIGNALS[name]["tickers"]})
    sig_px = load_signal_prices(tuple(sig_tickers), str(start)).reindex(prices.index).ffill()

    signals = {}
    for name in chosen:
        result = compute_signal(SIGNALS[name], sig_px, lookback)
        if result is None:
            st.warning(f"No data from Yahoo for {name}; skipping it.")
        else:
            signals[name] = result

    # The regime model itself, as a baseline to compare against
    regime_risk_on = (score["Disinflation"] + score["Reflation"]
                      - score["Stagflation"] - score["Deflation"])
    regime_warning = winner_num.isin([3, 4]).astype(float).where(winner_num.notna())

    # Combined warning score: how many signals are flashing risk-off each day
    flags = pd.DataFrame({name: s["warning"] for name, s in signals.items()})
    warn_score = flags.sum(axis=1, min_count=1)
    n_live = int(flags.loc[last].notna().sum()) if len(flags.columns) else 0
    today_score = int(warn_score.loc[last]) if pd.notna(warn_score.loc[last]) else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Signals warning today", f"{today_score} of {n_live}")
    c2.metric("Regime model today", REGIMES[int(winner_num[last]) - 1])
    if n_live:
        level = today_score / n_live
        mood = "Risk-off" if level >= 0.6 else "Mixed" if level >= 0.4 else "Risk-on"
        c3.metric("Overall read", mood)

    # Status table
    rows = []
    for name, s in list(signals.items()) + [("Regime model (baseline)", None)]:
        if s is None:
            risk_on, warning, value = regime_risk_on, regime_warning, None
        else:
            risk_on, warning, value = s["risk_on"], s["warning"], s["level"].loc[last]
        corr = lead_lag(risk_on, asset_px)
        best, verdict = lag_verdict(corr)
        avg_warn, avg_clear, share = forward_returns(warning, asset_px)
        flag_now = warning.loc[last]
        rows.append({
            "Signal": name,
            "Latest value": "" if value is None or pd.isna(value) else f"{value:.3f}",
            "Today": "" if pd.isna(flag_now) else ("WARNING" if flag_now == 1 else "All clear"),
            "Lead / lag vs " + sig_asset: verdict,
            "Corr. with next 20 days": round(corr[20], 2),
            f"{sig_asset} next 20d after warning": f"{avg_warn:+.1%}" if pd.notna(avg_warn) else "",
            f"{sig_asset} next 20d otherwise": f"{avg_clear:+.1%}" if pd.notna(avg_clear) else "",
            "% of days warning": f"{share:.0%}" if pd.notna(share) else "",
        })
    st.subheader("Signal scorecard")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Lead / lag: the shift where the signal best matches the asset's 20-day return. "
        "'Leads' means it matched returns that hadn't happened yet. A useful warning signal has "
        "a lower 'next 20d after warning' than 'otherwise'.")

    # Warning score over time
    st.subheader("Combined warning score")
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3.add_trace(go.Scatter(x=warn_score.index, y=warn_score, name="Signals warning",
                              line=dict(color="#d62728", width=1), fill="tozeroy",
                              fillcolor="rgba(214,39,40,0.15)"), secondary_y=False)
    fig3.add_trace(go.Scatter(x=asset_px.index, y=asset_px, name=sig_asset,
                              line=dict(color="black", width=2)), secondary_y=True)
    fig3.update_yaxes(title_text="Signals warning", range=[0, max(len(signals), 1)],
                      secondary_y=False)
    fig3.update_yaxes(title_text=f"{sig_asset} price", secondary_y=True)
    fig3.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h", y=1.1),
                       xaxis=dict(rangeslider=dict(visible=True)), margin=dict(t=40))
    st.plotly_chart(fig3, width="stretch")

    # What happened next, by warning score
    fwd = asset_px.shift(-20) / asset_px - 1
    by_score = (pd.DataFrame({"score": warn_score, "fwd": fwd}).dropna()
                .groupby("score")["fwd"].agg(days="size", avg="mean",
                                             up=lambda s: (s > 0).mean()))
    by_score.index = by_score.index.astype(int)
    by_score = by_score.rename_axis("Signals warning").reset_index()
    by_score["avg"] = by_score["avg"].map(lambda x: f"{x:+.1%}")
    by_score["up"] = by_score["up"].map(lambda x: f"{x:.0%}")
    by_score.columns = ["Signals warning", "Days", f"Avg {sig_asset} next 20 days",
                        f"% of times {sig_asset} rose"]
    st.write(f"**What {sig_asset} did over the next 20 trading days, by warning score**")
    st.dataframe(by_score, hide_index=True, width="stretch")

    # One signal in detail
    st.subheader("Look at one signal")
    pick = st.selectbox("Signal", list(signals), key="sig_pick")
    s = signals[pick]
    st.caption(SIGNALS[pick]["about"] + "  \nOn Yahoo Finance: " + " · ".join(
        f"[{t.replace('^', chr(92) + '^')}]({yahoo_url(t)})"      # \^ so ^VIX isn't read as a footnote
        for t in SIGNALS[pick]["tickers"]))
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Scatter(x=s.index, y=s["level"], name=pick,
                              line=dict(color="#2a7de1", width=1.5)), secondary_y=False)
    warn_days = s["level"].where(s["warning"] == 1)
    fig4.add_trace(go.Scatter(x=s.index, y=warn_days, name="Warning", mode="markers",
                              marker=dict(color="#d62728", size=3)), secondary_y=False)
    fig4.add_trace(go.Scatter(x=asset_px.index, y=asset_px, name=sig_asset,
                              line=dict(color="black", width=1.5)), secondary_y=True)
    kind = SIGNALS[pick]["kind"]
    if kind in ("vixterm", "curve"):
        fig4.add_hline(y=1 if kind == "vixterm" else 0, line_dash="dash", line_color="gray",
                       secondary_y=False)
    fig4.update_layout(height=420, hovermode="x unified", legend=dict(orientation="h", y=1.1),
                       xaxis=dict(rangeslider=dict(visible=True)), margin=dict(t=40))
    st.plotly_chart(fig4, width="stretch")

    corr = lead_lag(s["risk_on"], asset_px)
    best, verdict = lag_verdict(corr)
    bar = go.Figure(go.Bar(x=corr.index, y=corr.values,
                           marker_color=["#2a7de1" if i == best else "#bbbbbb" for i in corr.index]))
    bar.add_vline(x=0, line_dash="dash", line_color="gray")
    bar.update_layout(height=280, margin=dict(t=30),
                      title=f"{verdict}: correlation with {sig_asset}'s 20-day return",
                      xaxis_title="Window ends N trading days from today "
                                  "(negative = past, positive = future)",
                      yaxis_title="Correlation")
    st.plotly_chart(bar, width="stretch")

    export = pd.DataFrame({f"{name} {col}": s[col]
                           for name, s in signals.items() for col in s.columns})
    export["Signals warning"] = warn_score
    export[sig_asset] = asset_px
    st.download_button("Download signals CSV", export.to_csv(index_label="Date"),
                       "warning_signals.csv", "text/csv")
