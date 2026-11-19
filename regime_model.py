"""
Bottoms-Up Regime Segmentation, Python version
==============================================

This script does the same thing as the Google Sheets model:

  1. Get daily prices for 20 stocks (5 per regime) plus the VIX.
  2. For each stock, turn price into a daily BUY (1) / SELL (0) vote
     (same rules as columns C through AL on each stock's sheet).
  3. For each regime, average its 5 votes   -> "Raw Votes %"  (VOTE sheet G, N, U, AB)
  4. Smooth each regime with a 19-day average -> regime score (VOTE sheet H, O, V, AC)
  5. Regime Winner = the regime with the highest score        (VOTE sheet AJ)
  6. Compare against outside assets (QQQ, AMD): charts + lag test.

How to run
----------
    pip install pandas numpy matplotlib openpyxl yfinance

    # Download fresh data from Yahoo Finance (default)
    python regime_model.py

    # Or use the prices already in your spreadsheet
    python regime_model.py --source excel --file "guardian_jose_bottoms-Up_Regime_SegmentationFall_2026.xlsx"

Outputs (in the --out folder, default "output"):
    regime_output.csv     every date: each stock's vote, raw %, smoothed score, winner
    regimes_<ASSET>.png   regime scores with QQQ / AMD on the right axis
    regime_winner.png     the Regime Winner over time
    lag test results are printed to the screen

The "# %%" lines split the file into cells, so you can also run it step by step
in VS Code or Jupyter.
"""

# %% Settings: change these to try different stocks or windows
import argparse
import os

import numpy as np
import pandas as pd

BUCKETS = {
    "Disinflation": ["TSLA", "PANW", "META", "AVGO", "AMZN"],   # Quad 1
    "Reflation":    ["XOM", "PWR", "PG", "CVX", "JNJ"],         # Quad 2
    "Stagflation":  ["YTRA", "INTG", "SPWH", "JILL", "DOGZ"],   # Quad 3
    "Deflation":    ["HSY", "ELA", "SMMT", "CCLD", "LNTH"],     # Quad 4
}
REGIMES = list(BUCKETS)                    # order matters for ties (same as the IF chain in AJ)
OUTSIDE_ASSETS = ["QQQ", "AMD"]            # "just for charting" assets
START = "2018-10-12"

# Indicator windows (same as the spreadsheet)
HV_DAYS = 5                    # D: HV-5
VRP_AVG_DAYS = 5               # G: VRP 5DMA
MACD_LONG = (12, 26, 9)        # I, J, L  (calm market)
MACD_SHORT = (6, 15, 4)        # N, O, Q  (stressed market). The sheet's header says 13, but
                               # the O formula averages 15 days (B46:B60), so 15 matches the sheet
RSI_LONG = 21                  # W..AA    (calm market)
RSI_SHORT = 14                 # AE..AI   (stressed market)
SMOOTH_DAYS = 19               # VOTE sheet H = average(G2:G20)


# %% Step 1: load prices
def load_yahoo(start=START, end=None):
    """Adj Close for every stock + outside asset, and the VIX close."""
    import yfinance as yf

    stocks = [t for names in BUCKETS.values() for t in names]
    tickers = stocks + OUTSIDE_ASSETS + ["^VIX"]
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    adj = data["Adj Close"]
    vix = data["Close"]["^VIX"]

    # Keep only days when the stock market was open (VIX can trade on a few extra days)
    adj = adj.dropna(subset=stocks, how="all")
    prices = adj[stocks]
    outside = adj[OUTSIDE_ASSETS]
    vix = vix.reindex(prices.index).ffill()
    vix_by_stock = pd.DataFrame({t: vix for t in stocks})  # same VIX for every stock
    return prices, vix_by_stock, outside


def load_excel(path, fix_vix=False):
    """Read the prices already in the workbook: column A (date), B (Adj Close), E (VIX)."""
    stocks = [t for names in BUCKETS.values() for t in names]
    book = pd.ExcelFile(path)                      # open the workbook once (much faster)
    prices, vix = {}, {}
    for t in stocks:
        df = book.parse(t, usecols="A,B,E", header=0)
        df.columns = ["Date", "Close", "VIX"]
        df = df.dropna(subset=["Date"]).set_index("Date")
        prices[t], vix[t] = df["Close"], df["VIX"]
    prices = pd.DataFrame(prices)
    vix = pd.DataFrame(vix)

    # Warn about missing or frozen VIX values, since they change the calm/stressed switch
    for t in stocks:
        missing = vix[t].isna().sum()
        if missing:
            first = vix[t][vix[t].isna()].index[0].date()
            print(f"  WARNING: {t} has {missing} days with no VIX (from {first}).")
    common = vix.median(axis=1)
    frozen = (common == common.iloc[-1])[::-1].cumprod().sum()
    if frozen > 20:
        print(f"  WARNING: VIX is the same value ({common.iloc[-1]}) for the last {frozen} days. "
              "Re-download the VIX to fix this.")

    if fix_vix:
        vix = vix.apply(lambda col: col.fillna(common))
    else:
        vix = vix.fillna(0)        # Google Sheets treats an empty cell as 0

    # Outside assets live on the VOTE sheet, one row per date (same rows as the stock sheets)
    vote = book.parse("VOTE", header=0)
    outside_cols = [c for c in vote.columns if "just for charting" in str(c)]
    outside = vote[outside_cols].iloc[: len(prices)].copy()
    outside.columns = [str(c).split(" (")[0] for c in outside_cols]
    outside.index = prices.index
    return prices, vix, outside


# %% Step 2: one stock's buy/sell vote
def rsi(close, days):
    """RSI the way the sheet does it: simple averages of gains and losses (W..Z, AE..AH)."""
    change = close.diff()                           # T / AB: B - previous B
    gain = change.clip(lower=0)                     # U / AC
    loss = (-change).clip(lower=0)                  # V / AD
    avg_gain = gain.rolling(days).mean()            # W / AE
    avg_loss = loss.rolling(days).mean()            # X / AF
    rs = avg_gain / avg_loss                        # Y / AG
    return 100 - 100 / (1 + rs)                     # Z / AH (avg_loss = 0 gives RSI 100)


def macd_rule(close, fast, slow, signal):
    """1 when the signal average of (fast SMA - slow SMA) is above zero (M / R)."""
    line = close.rolling(fast).mean() - close.rolling(slow).mean()    # K / P
    sig = line.rolling(signal).mean().round(10)   # L / Q (round off tiny float noise so an
    return (sig > 0).astype(int)                  # exact 0 counts as 0, like in Sheets)


def stock_signals(close, vix):
    """Every column of one stock sheet, as a table."""
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["daily_ret"] = close.pct_change()                                        # C
    out["hv5"] = out["daily_ret"].rolling(HV_DAYS).std() * np.sqrt(252) * 100   # D
    out["vix"] = vix                                                             # E
    out["vrp"] = out["vix"] - out["hv5"]                                         # F
    out["vrp_5dma"] = out["vrp"].rolling(VRP_AVG_DAYS).mean()                    # G
    out["calm"] = (out["vrp_5dma"].round(10) > 0).astype(int)                    # H

    out["macd_long"] = macd_rule(close, *MACD_LONG)                              # M
    out["macd_short"] = macd_rule(close, *MACD_SHORT)                            # R
    out["macd"] = np.where(out["calm"] == 1, out["macd_long"], out["macd_short"])  # S

    out["rsi_long"] = (rsi(close, RSI_LONG).round(10) > 50).astype(int)          # AA
    out["rsi_short"] = (rsi(close, RSI_SHORT).round(10) > 50).astype(int)        # AI
    out["rsi"] = np.where(out["calm"] == 1, out["rsi_long"], out["rsi_short"])     # AJ

    out["vote"] = ((out["macd"] == 1) & (out["rsi"] == 1)).astype(int)           # AK / AL

    # Blank out the warm-up days, before every window has enough history
    warmup = max(MACD_LONG[1] + MACD_LONG[2], RSI_LONG + 1, HV_DAYS + VRP_AVG_DAYS)
    out.loc[out.index[:warmup], "vote"] = np.nan
    return out


# %% Step 3-5: regimes
def build_regimes(prices, vix):
    votes = pd.DataFrame({t: stock_signals(prices[t], vix[t])["vote"] for t in prices})

    raw = pd.DataFrame({r: votes[BUCKETS[r]].mean(axis=1) for r in REGIMES})       # G, N, U, AB
    score = raw.rolling(SMOOTH_DAYS).mean()                                          # H, O, V, AC
    ready = score.dropna().round(10)           # skip warm-up days; round so exact ties stay ties
    winner = ready.idxmax(axis=1)              # first max wins ties, like the IF chain in AJ
    winner_num = winner.map({r: i + 1 for i, r in enumerate(REGIMES)}).reindex(score.index)
    return votes, raw, score, winner_num


# %% Step 6a: lag test (same idea as the CORREL / OFFSET table)
def lag_test(score, asset_price, window=20, max_lag=15):
    """Correlation between a regime score and the asset's return over the past `window`
    days, ending `lag` days ago. The lag with the highest correlation = how far behind."""
    past_return = asset_price / asset_price.shift(window) - 1
    results = {}
    for lag in range(0, max_lag + 1):
        results[lag] = score.corr(past_return.shift(lag))
    results = pd.Series(results, name="correlation")
    return results, int(results.idxmax())


# %% Step 6b: charts
def make_charts(score, winner_num, outside, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Disinflation": "#2a7de1", "Reflation": "#2ca02c",
              "Stagflation": "#d62728", "Deflation": "#7f7f7f"}

    for asset in outside.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        for r in REGIMES:
            ax.plot(score.index, score[r], label=r, color=colors[r], linewidth=1.2)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Regime score (share of stocks voting buy, 19-day avg)")
        ax2 = ax.twinx()
        ax2.plot(outside.index, outside[asset], color="black", linewidth=1.5, label=asset)
        ax2.set_ylabel(f"{asset} price")
        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        ax.legend(lines, REGIMES + [asset], loc="upper left", ncol=5)
        ax.set_title(f"Regime scores vs {asset}")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"regimes_{asset}.png"), dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.step(winner_num.index, winner_num, where="post", color="black", linewidth=1)
    ax.set_yticks([1, 2, 3, 4], [f"{i + 1} {r}" for i, r in enumerate(REGIMES)])
    ax.set_title("Regime Winner")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "regime_winner.png"), dpi=130)
    plt.close(fig)


# %% Run everything
def main():
    parser = argparse.ArgumentParser(description="Bottoms-Up Regime Segmentation")
    parser.add_argument("--source", choices=["yahoo", "excel"], default="yahoo")
    parser.add_argument("--file", help="path to the .xlsx when --source excel")
    parser.add_argument("--fix-vix", action="store_true",
                        help="excel only: fill missing VIX cells instead of treating them as 0")
    parser.add_argument("--start", default=START)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading prices...")
    if args.source == "excel":
        prices, vix, outside = load_excel(args.file, fix_vix=args.fix_vix)
    else:
        prices, vix, outside = load_yahoo(start=args.start)
    print(f"  {len(prices)} trading days, {prices.index[0].date()} to {prices.index[-1].date()}")

    votes, raw, score, winner_num = build_regimes(prices, vix)

    table = votes.copy()
    for r in REGIMES:
        table[f"{r} (Raw Votes%)"] = raw[r]
        table[f"{r} score"] = score[r]
    table["Regime Winner"] = winner_num
    table = table.join(outside)
    table.to_csv(os.path.join(args.out, "regime_output.csv"), index_label="Date")

    counts = winner_num.value_counts().sort_index()
    print("\nRegime Winner, number of days:")
    for num, n in counts.items():
        print(f"  {int(num)} {REGIMES[int(num) - 1]:<13} {n}")
    last = winner_num.dropna().index[-1]
    print(f"\nLatest ({last.date()}): {REGIMES[int(winner_num[last]) - 1]}")

    print("\nLag test (Disinflation score vs each asset's 20-day return):")
    for asset in outside.columns:
        corr, best = lag_test(score["Disinflation"], outside[asset])
        print(f"  {asset}: best match {best} trading days back "
              f"(corr {corr[best]:.2f}, vs {corr[0]:.2f} at 0 days)")

    make_charts(score, winner_num, outside, args.out)
    print(f"\nSaved CSV and charts to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
