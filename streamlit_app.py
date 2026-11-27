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
# Page
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Bottoms-Up Regime Segmentation", layout="wide")
st.title("Bottoms-Up Regime Segmentation")

with st.sidebar:
    st.header("Settings")
    start = st.date_input("Start date", pd.Timestamp(DEFAULT_START))
    smooth_days = st.slider("Smoothing (days)", 5, 40, 19,
                            help="The spreadsheet uses a 19-day average of the raw votes.")
    st.subheader("Buckets")
    buckets = {}
    for r in REGIMES:
        text = st.text_input(r, ", ".join(DEFAULT_BUCKETS[r]))
        buckets[r] = [t.strip().upper() for t in text.split(",") if t.strip()]
    outside_text = st.text_input("Outside assets", ", ".join(DEFAULT_OUTSIDE))
    outside_assets = [t.strip().upper() for t in outside_text.split(",") if t.strip()]
    st.subheader("Warning signals")
    lookback = st.slider("Ratio lookback (days)", 20, 130, 65,
                         help="A ratio signal warns when the ratio is lower than it was this many "
                              "trading days ago. 65 days is about 13 weeks.")
    chosen = st.multiselect("Signals to include", list(SIGNALS), default=list(SIGNALS))
    if st.button("Refresh data"):
        st.cache_data.clear()

stocks = sorted({t for names in buckets.values() for t in names})
if any(len(buckets[r]) == 0 for r in REGIMES):
    st.error("Every bucket needs at least one ticker.")
    st.stop()

prices_all, vix = load_prices(tuple(stocks + outside_assets), str(start))

# Tickers Yahoo didn't return
missing = [t for t in stocks + outside_assets
           if t not in prices_all.columns or prices_all[t].isna().all()]
if missing:
    st.warning(f"No data from Yahoo for: {', '.join(missing)}. Check the ticker spelling.")
buckets = {r: [t for t in names if t not in missing] for r, names in buckets.items()}
outside_assets = [t for t in outside_assets if t not in missing]
if any(len(buckets[r]) == 0 for r in REGIMES):
    st.error("A bucket has no tickers with data. Fix the tickers in the sidebar.")
    st.stop()

stocks = sorted({t for names in buckets.values() for t in names})
prices = prices_all[stocks].dropna(how="all")
outside = prices_all.reindex(prices.index)[outside_assets]
votes, raw, score, winner_num = build_regimes(prices, vix.reindex(prices.index), buckets, smooth_days)
last = winner_num.last_valid_index()
st.caption(f"Data through {last.date()} · Yahoo Finance · refreshes every 6 hours")

tab_model, tab_signals = st.tabs(["Regime model", "Warning signals"])

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
    asset = st.selectbox("Compare against", outside_assets) if outside_assets else None
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for r in REGIMES:
        fig.add_trace(go.Scatter(x=score.index, y=score[r], name=r,
                                 line=dict(color=COLORS[r], width=1.5)), secondary_y=False)
    if asset:
        fig.add_trace(go.Scatter(x=outside.index, y=outside[asset], name=asset,
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
            corr = lag_test(score[regime_for_lag], outside[asset])
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
            st.info("Add an outside asset in the sidebar to run the lag test.")

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
    table = table.join(outside)
    with st.expander("All data"):
        st.dataframe(table.dropna(subset=["Regime Winner"]).iloc[::-1], width="stretch")
    st.download_button("Download CSV", table.to_csv(index_label="Date"), "regime_output.csv",
                       "text/csv")

# ==================================================================================
# Tab 2: warning signals
# ==================================================================================
with tab_signals:
    st.markdown(
        "Indicators that are often said to move **before** stocks do. Each one is turned into "
        "a simple risk-off warning (1) or all-clear (0), tested for whether it leads or lags the "
        "market, and checked against what the market did next. None of them is reliable on its "
        "own; the point is to test them.")
    if not outside_assets:
        st.info("Add an outside asset in the sidebar (for example QQQ) to test the signals.")
        st.stop()
    if not chosen:
        st.info("Pick at least one signal in the sidebar.")
        st.stop()

    sig_asset = st.selectbox("Test against", outside_assets, key="sig_asset")
    asset_px = outside[sig_asset]

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
