import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG & GLOBAL STYLE
# -------------------------
st.set_page_config(
    page_title="FX Analytics Dashboard",
    page_icon="💱",
    layout="wide"
)

# Optional: hide Streamlit menu & footer inside the app
HIDE_STREAMLIT_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

# -------------------------
# HEADER
# -------------------------
st.title("💱 FX Analytics Dashboard")

st.markdown(
    """
    Analyse foreign exchange rates with **trend**, **volatility**, and **return distribution** metrics.
    
    Select a currency pair and period in the sidebar, then click **Run analysis**.
    """
)

# -------------------------
# SIDEBAR: USER INPUTS
# -------------------------
st.sidebar.header("Settings")

# Popular FX pairs (Yahoo Finance tickers)
default_pairs = [
    "EURUSD=X",
    "GBPUSD=X",
    "USDJPY=X",
    "EURGBP=X",
    "EURJPY=X",
]

ticker = st.sidebar.selectbox("Select FX pair (Yahoo Finance ticker):", default_pairs)
period = st.sidebar.selectbox("History period:", ["6mo", "1y", "2y", "5y"], index=1)

st.sidebar.markdown(
    """
    _Tip_: You can search the ticker on Yahoo Finance and copy it here
    if you want to extend the list later.
    """
)

# -------------------------
# HELPER
# -------------------------
def safe_last(series: pd.Series) -> float:
    """Return last non-NaN value of a Series, or NaN if empty."""
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan

# -------------------------
# MAIN LOGIC
# -------------------------
if st.sidebar.button("Run analysis"):

    with st.spinner("Downloading data from Yahoo Finance..."):
        try:
            data = yf.download(ticker, period=period, progress=False)
        except Exception as e:
            st.error(f"Error while downloading data: {e}")
            data = pd.DataFrame()

    if data.empty:
        st.error("No data found for this ticker/period. Try another combination.")
    else:
        # -------------------------
        # 1. CLOSE PRICE SERIES
        # -------------------------
        close = data["Close"].copy()

        # In some versions, this can be a DataFrame (e.g. MultiIndex) → take first column
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close.name = ticker

        # -------------------------
        # 2. CORE ANALYTICS
        # -------------------------
        returns = close.pct_change()

        # Rolling vol (30d & 90d), annualized
        vol30 = returns.rolling(30, min_periods=1).std() * np.sqrt(252)
        vol90 = returns.rolling(90, min_periods=1).std() * np.sqrt(252)

        # SMA 20 & 50
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()

        # Latest metrics
        last_price = safe_last(close)
        last_vol30 = safe_last(vol30)
        last_vol90 = safe_last(vol90)

        # -------------------------
        # OVERVIEW + METRICS
        # -------------------------
        st.markdown(f"### {ticker} overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Last price", f"{last_price:.4f}" if not np.isnan(last_price) else "-")
        with col2:
            st.metric(
                "Ann. vol (30d)",
                f"{last_vol30:.2%}" if not np.isnan(last_vol30) else "-"
            )
        with col3:
            st.metric(
                "Ann. vol (90d)",
                f"{last_vol90:.2%}" if not np.isnan(last_vol90) else "-"
            )

        # -------------------------
        # TABS: PLOTS & STATS
        # -------------------------
        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Price & MAs", "📉 Volatility", "📊 Returns distribution", "📌 Stats table"]
        )

        # ---- TAB 1: PRICE & MAs ----
        with tab1:
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

        # ---- TAB 2: ROLLING VOLATILITY ----
        with tab2:
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

        # ---- TAB 3: HISTOGRAM OF RETURNS ----
        with tab3:
            st.subheader("Distribution of Daily Returns")

            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.hist(returns.dropna(), bins=40, alpha=0.7)
            ax3.set_xlabel("Daily Return")
            ax3.set_ylabel("Frequency")
            ax3.set_title(f"{ticker} - Histogram of Daily Returns")
            st.pyplot(fig3)

        # ---- TAB 4: SUMMARY STATISTICS ----
        with tab4:
            st.subheader("Summary Statistics")

            stats = pd.DataFrame({
                "Mean daily return": [returns.mean()],
                "Std daily return": [returns.std()],
                "Annualized vol (last 30d)": [last_vol30],
                "Annualized vol (last 90d)": [last_vol90],
            }).T
            stats.columns = ["Value"]
            st.table(stats)

else:
    st.info("Use the sidebar to select a pair and click **Run analysis** to start.")
