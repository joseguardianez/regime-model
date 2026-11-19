"""
Bottoms-Up Regime Segmentation: interactive web app
===================================================

Same model as regime_model.py, as a website built with Streamlit.

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

# Headline: current regime and how each score moved over the last week
last = winner_num.last_valid_index()
st.caption(f"Data through {last.date()} · Yahoo Finance · refreshes every 6 hours")
cols = st.columns(5)
cols[0].metric("Current regime", REGIMES[int(winner_num[last]) - 1])
week_ago = score.index[max(score.index.get_loc(last) - 5, 0)]
for col, r in zip(cols[1:], REGIMES):
    col.metric(r, f"{score.loc[last, r]:.2f}", f"{score.loc[last, r] - score.loc[week_ago, r]:+.2f} vs 1 wk ago")

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
                               marker_color=["#2a7de1" if i == best else "#bbbbbb" for i in corr.index]))
        bar.update_layout(height=280, xaxis_title="Days back", yaxis_title="Correlation",
                          margin=dict(t=10))
        st.plotly_chart(bar, width="stretch")
    else:
        st.info("Add an outside asset in the sidebar to run the lag test.")

# Latest votes
with right:
    st.subheader(f"Votes on {last.date()}")
    latest = pd.DataFrame(
        [(r, t, "BUY" if votes.loc[last, t] == 1 else "SELL") for r in REGIMES for t in buckets[r]],
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
st.download_button("Download CSV", table.to_csv(index_label="Date"), "regime_output.csv", "text/csv")
