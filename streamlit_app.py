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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

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
# Adding tickers: getETF-style ranking, saved buckets
# ----------------------------------------------------------------------------------
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
    return t, px

tab_model, tab_signals, tab_add = st.tabs(["Regime model", "Warning signals", "Add tickers"])

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
        st.dataframe(latest, hide_index=True, width="stretch", height=280)

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
        st.dataframe(ranked, hide_index=True, width="stretch")
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
    st.dataframe(pd.DataFrame({r: user_buckets[r] + [""] * (longest - len(user_buckets[r]))
                               for r in REGIMES}), hide_index=True, width="stretch")
    st.caption("Your changes are saved in this page's link, so they stay when you reload. "
               "Bookmark or share the link to keep them. To remove a ticker, delete it from its "
               "box in the sidebar. **Reset buckets to defaults** in the sidebar undoes everything.")

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
    st.caption(SIGNALS[pick]["about"])
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
