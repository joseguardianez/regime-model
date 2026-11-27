# Bottoms-Up Regime Segmentation

**Live website: https://jose-regime-model.streamlit.app**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jose-regime-model.streamlit.app)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joseguardianez/regime-model/blob/main/colab_quickstart.ipynb)

This project estimates which economic regime the market is pricing in, from the bottom up. Twenty stocks vote every trading day, five for each regime: Disinflation, Reflation, Stagflation and Deflation. Whichever regime has the most stocks giving buy signals is the regime the market is currently rewarding. The model started as a Google Sheets project for a Fall 2026 class (Mid-Term 3: Bottoms-Up Regime Segmentation). It was then rebuilt in Python and published as an interactive website that updates itself with new market data.

---

## How to use the website (step by step)

1. **Open the link.** Go to **https://jose-regime-model.streamlit.app** in any browser, on a computer or a phone. You don't need an account.
2. **Wake it up if it's asleep.** Free Streamlit apps go to sleep after 12 hours with no visitors. If you see *"This app has gone to sleep due to inactivity"*, click **"Yes, get this app back up!"** and wait about a minute while it starts. Anyone can do this, not just the owner.
3. **Wait for the prices to download.** On first load the app downloads fresh prices from Yahoo Finance, which can take 20–30 seconds. You'll see a spinner that says *"Downloading prices from Yahoo Finance..."*.
4. **Read the current regime.** The top row shows the **Current regime** and each regime's score from 0 to 1. A score is the share of that regime's stocks voting buy, averaged over 19 days. The small green or red number shows how much it changed over the past week.
5. **Explore the main chart.**
   - Hover over any date to see all four regime scores and the outside asset's price.
   - Drag the slider under the chart, or click and drag on the chart, to zoom into a period such as the COVID crash in 2020. Double-click to zoom back out.
   - Click a name in the legend to hide or show that line.
   - Use **Compare against** to switch the outside asset. Pick one from the list (QQQ, AMD) or type **any ticker**: a stock, an ETF, an index like `^GSPC`, or crypto like `BTC-USD`. A ticker you type is added to your list so you can switch back to it.
6. **Check the Regime Winner chart.** It shows which regime was on top each day, numbered 1 to 4.
7. **Look at the lag test.** It shows how many trading days the regime score trails the outside asset. The blue bar marks the best match.
8. **See today's votes.** The **Votes on …** table lists every stock and whether it's voting BUY or SELL on the latest day.
9. **Change the settings (optional).** Open the sidebar to edit the settings. On a phone, tap the **>>** arrow at the top left.
   - **Start date:** how far back to go.
   - **Smoothing (days):** 19 matches the spreadsheet. Smaller numbers react faster but jump around more.
   - **Buckets:** type tickers separated by commas to try different stocks in each regime.
   - **Outside assets:** your saved list of tickers to compare against. Tickers you type into **Compare against** or **Test against** are added here automatically.
   - **Refresh data:** downloads the newest prices right away.

   These changes are just for you. Your buckets and outside assets are saved in the page link, so they stay when you reload. Bookmark the link to keep them.
10. **Download the data (optional).** Click **Download CSV** at the bottom for every date's votes, scores and Regime Winner. Open **All data** to browse it on the page.
11. **Check the Warning signals tab.** Click **Warning signals** at the top of the page to see market indicators that may move before stocks do (details below).
12. **Add your own tickers.** Click **Add tickers**, type one or more symbols, and click **Rank tickers**. The site ranks each one the same way the getETF macro does and puts it in the bucket where it ranks first (details below).

---

## Add tickers tab

1. Type ticker symbols, separated by commas or spaces (up to 100 at once), for example `NVDA, KO, NEM`.
2. Click **Rank tickers**. For each stock, the site downloads daily prices since Jan 3, 2005 (or since it started trading) and averages its daily return in each macro quad, using the same quad calendar and method as `ETF_Bucketing.xlsm`. The table shows the rank order, the average daily return in each quad, and the bucket where it ranks first.
3. Choose which tickers to keep and click **Add … ticker(s) to the model**. Each one goes into its best bucket, and it's removed from any other bucket it was in.

Your buckets are saved in the page link, so they stay when you reload. Bookmark or share that link to keep them. To remove a ticker, delete it from its box in the sidebar. **Reset buckets to defaults** in the sidebar undoes all changes.

---

## Warning signals tab

The regime model reacts to price, so it trails the market by about a week. This tab tests indicators that are often said to move *first*. Each one is turned into a simple risk-off **warning (1)** or **all clear (0)**:

| Signal | Calculation | Warning when |
|---|---|---|
| Contango | VIX / VIX3M | 5-day average above 1 (backwardation) |
| Lumber vs gold | WOOD / GLD | Ratio lower than it was 65 trading days ago |
| Credit | HYG / IEF | Ratio lower than it was 65 trading days ago |
| Copper vs gold | CPER / GLD | Ratio lower than it was 65 trading days ago |
| Small vs large caps | IWM / SPY | Ratio lower than it was 65 trading days ago |
| Breadth | RSP / SPY | Ratio lower than it was 65 trading days ago |
| Yield curve | 10-year yield − 3-month yield | Below 0 (inverted) |

What the tab shows:
- **Signals warning today:** how many signals flag risk-off right now, plus an overall read of Risk-on, Mixed or Risk-off.
- **Signal scorecard:** each signal's latest value and status, whether it **leads or lags** the asset you pick in **Test against** (QQQ by default, or type any ticker), its correlation with the *next* 20 days, and that asset's average next-20-day return after a warning compared with all other days. The regime model is included as a baseline.
- **Combined warning score:** the number of signals warning each day, charted against the chosen asset, with a table of what it did next at each score.
- **Look at one signal:** a chart of any single signal with its warning days marked, plus its lead/lag profile.

In the sidebar you can change the ratio lookback (65 days is about 13 weeks) and choose which signals to include. No indicator times the market reliably, so treat this tab as a way to test ideas, not as trading advice.

---

## How the model works

### 1. Bucketing the stocks
Each stock was placed with the **getETF / PerformanceRank** macro in `ETF_Bucketing.xlsm`. The macro labels every trading day since Jan 3, 2005 with that month's macro quad and averages the stock's daily return within each quad. Stocks listed after 2005 use their history from the listing date. A stock goes in the quad where its average return is highest. All 20 stocks rank first in their assigned quad.

| Regime (Quad) | Growth / Inflation | Stocks |
|---|---|---|
| Disinflation (Q1) | Growth up, inflation down | TSLA, PANW, META, AVGO, AMZN |
| Reflation (Q2) | Growth up, inflation up | XOM, PWR, PG, CVX, JNJ |
| Stagflation (Q3) | Growth down, inflation up | YTRA, INTG, SPWH, JILL, DOGZ |
| Deflation (Q4) | Growth down, inflation down | HSY, ELA, SMMT, CCLD, LNTH |

Most of the Quad 3 and Quad 4 names came from screening hundreds of consumer discretionary and healthcare stocks listed before Oct 12, 2018, so every stock has the full history the model needs.

### 2. Each stock's daily buy/sell vote
1. **Volatility risk premium:** VIX minus the stock's 5-day realized volatility, averaged over 5 days.
2. **Calm vs. stressed:** if that premium is positive, the market is calm and the model uses the slower rules: MACD 12/26/9 and a 21-day RSI. If it's negative, the market is stressed and the model uses the faster rules: MACD 6/15/4 and a 14-day RSI.
3. **Vote:** a stock votes **BUY (1)** only when its MACD signal is above zero *and* its RSI is above 50. Otherwise it votes **SELL (0)**.

### 3. From votes to a regime
- **Raw score:** the share of a regime's stocks voting buy that day.
- **Regime score:** the raw score averaged over the last 19 days.
- **Regime Winner:** the regime with the highest score. Ties go to the earlier regime, Q1 before Q2 and so on, the same as the spreadsheet.

---

## Findings
These results come from the spreadsheet version, Oct 12, 2018 – Nov 6, 2026.

- **The model lags the market.** Against QQQ, the regime scores match QQQ's return from about **7 trading days earlier** best (correlation 0.80, versus 0.60 for the same day). Against AMD the lag is about 8 trading days. The lag comes from the method itself: MACD and RSI are built from past prices, and the 19-day average slows changes further.
- **Early warning before COVID.** The model switched to Deflation on **Feb 12, 2020**, four trading days before QQQ peaked on Feb 19, and stayed there through the crash.
- **Late in 2025.** During the April 2025 tariff selloff, the model only switched to Deflation on **Apr 22**, after QQQ had already bottomed on Apr 8.
- **Stagflation as a rebound signal.** After the 18 times Stagflation took the lead, QQQ rose an average of **4.3%** over the next 20 trading days, compared with 1.7% normally.
- **Markets, not macro data.** The Regime Winner matches the official macro quads on only about 25% of days. The model reflects what the market is pricing in more than it forecasts the next quad.

### Ideas to improve it
- Compare each stock's vote to its own normal buy rate, because Disinflation stocks vote buy more often (about 52% of days versus 39% for Stagflation).
- Shorten the 19-day average, or use an exponential average, to cut the lag.
- Re-run getETF every year on a rolling 5-year window.
- Add more stocks or sector ETFs to each bucket so no single stock can swing a regime.
- Add macro data such as CPI and the yield curve, and backtest a trading rule.

---

## Files in this repo

| File | What it is |
|---|---|
| `streamlit_app.py` | The website. Streamlit runs this file. |
| `regime_model.py` | The same model as a script for Colab or your computer. It saves a CSV and charts and prints the lag test. |
| `colab_quickstart.ipynb` | A Google Colab notebook that downloads `regime_model.py` from this repo and runs it. |
| `requirements.txt` | Python packages the website installs. |

## Running it yourself

**Google Colab (easiest):** click the *Open in Colab* badge at the top, then run the cells in order (Runtime → Run all).

**On your computer:**
```bash
pip install -r requirements.txt openpyxl matplotlib
streamlit run streamlit_app.py      # the website, at http://localhost:8501
python regime_model.py              # the script version; results go to ./output
```
To use the prices from the original spreadsheet instead of Yahoo:
```bash
python regime_model.py --source excel --file "your_workbook.xlsx"
```

## Updating the website
1. Open `streamlit_app.py` on GitHub and click the **pencil** icon.
2. Make your change, for example the tickers in `DEFAULT_BUCKETS` near the top.
3. Click **Commit changes**. The website updates within a few seconds. If it doesn't, open the app, click **Manage app → Reboot app**.

## Notes and known issues
- **The Python version matches the spreadsheet exactly.** On the spreadsheet's own prices, the Python code reproduces every stock's vote, all four regime scores and the Regime Winner on every day from Dec 31, 2018 onward.
- **The spreadsheet's VIX is frozen.** In the class template the VIX column stays at 17.44 from Oct 31, 2025 onward, and the SPWH sheet has no VIX from Apr 6, 2026. The website uses the real VIX from Yahoo, so its last year of results differs from the spreadsheet.
- **The short MACD window is 15 days.** The spreadsheet's header says 13, but its formula averages 15 days, so the code uses 15 to match.

## Disclaimer
This is an educational project. It is not investment advice.

## Credits
Built by [@joseguardianez](https://github.com/joseguardianez). The regime framework follows the growth/inflation "quad" approach used in the course's `ETF_Bucketing.xlsm` file. Python code and website built with help from Claude (Anthropic). Price data comes from Yahoo Finance through the `yfinance` package.
