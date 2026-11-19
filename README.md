# Bottoms-Up Regime Segmentation

Interactive version of my regime model. Twenty stocks (five per regime) vote buy or sell each day using MACD and RSI rules that switch with the VIX volatility risk premium. Each regime's score is the share of its stocks voting buy, smoothed over 19 days, and the highest score is the Regime Winner.

Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`
