import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.title("FX Analytics Dashboard")

st.markdown("""
Analyse foreign exchange rates: trend, volatility, and return distribution.
""")

# ----- USER INPUTS -----
st.sidebar.header("Settings")

# Simple list of popular FX pairs (Yahoo Finance tickers)
default_pairs = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "EURGBP=X",
    "EURJPY=X",
]

ticker = st.sidebar.selectbox("Select FX pair (Yahoo ticker):", default_pairs)
period = st.sidebar.selectbox("History period:", ["6mo", "1y", "2y", "5y"], index=1)

if st.sidebar.button("Run analysis"):
    # ----- 1. DATA COLLECTION -----
    data = yf.download(ticker, period=period)

    if data.empty:
        st.error("No data found for this ticker/period.")
    else:
        # Keep only Close price
        close = data["Close"].copy()
        close.name = ticker

        # ----- 2. CORE ANALYTICS -----
        # Daily returns
        returns = close.pct_change()

        # Rolling vol (30d & 90d), annualized
        vol30 = returns.rolling(30, min_periods=1).std() * np.sqrt(252)
        vol90 = returns.rolling(90, min_periods=1).std() * np.sqrt(252)

        # SMA 20 & 50
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()

        # ----- 3. VISUALISATIONS -----

        # A) Price + SMA
        st.subheader("Price with 20d & 50d Simple Moving Averages")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(close.index, close, label="Close", linewidth=1)
        ax1.plot(sma20.index, sma20, label="SMA 20", linewidth=1)
        ax1.plot(sma50.index, sma50, label="SMA 50", linewidth=1)
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.set_title(f"{ticker} - Price & Moving Averages")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # B) Rolling volatility
        st.subheader("Rolling Annualized Volatility (30d & 90d)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(vol30.index, vol30, label="30-day Vol", linewidth=1)
        ax2.plot(vol90.index, vol90, label="90-day Vol", linewidth=1)
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Volatility")
        ax2.set_title(f"{ticker} - Rolling Annualized Volatility")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # C) Histogram of daily returns
        st.subheader("Distribution of Daily Returns")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.hist(returns.dropna(), bins=40, alpha=0.7)
        ax3.set_xlabel("Daily Return")
        ax3.set_ylabel("Frequency")
        ax3.set_title(f"{ticker} - Histogram of Daily Returns")
        st.pyplot(fig3)

        # Optional: small stats
        st.subheader("Summary Statistics")
        stats = pd.DataFrame({
            "Mean daily return": [returns.mean()],
            "Std daily return": [returns.std()],
            "Annualized vol (last 30d)": [vol30.iloc[-1]],
            "Annualized vol (last 90d)": [vol90.iloc[-1]],
        }).T
        stats.columns = ["Value"]
        st.table(stats)
else:
    st.info("Choose a pair and click 'Run analysis' in the sidebar.")
