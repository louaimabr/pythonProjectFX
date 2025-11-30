import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
    Analyse foreign exchange rates with **trend**, **volatility**, **momentum**, 
    **risk metrics**, and a simple **forecasting model**.
    
    Select a currency pair and period in the sidebar, then click **Run analysis**.
    """
)

# -------------------------
# SIDEBAR: USER INPUTS
# -------------------------
st.sidebar.header("Settings")

# Map nice labels → Yahoo Finance tickers
FX_PAIRS = {
    "EUR / USD": "EURUSD=X",
    "EUR / GBP": "EURGBP=X",
    "EUR / JPY": "EURJPY=X",
    "EUR / CHF": "EURCHF=X",
    "EUR / CAD": "EURCAD=X",
    "EUR / AUD": "EURAUD=X",
    "EUR / NZD": "EURNZD=X",
    "EUR / SEK": "EURSEK=X",
    "EUR / NOK": "EURNOK=X",
    "EUR / DKK": "EURDKK=X",
    "EUR / PLN": "EURPLN=X",
    "EUR / TRY": "EURTRY=X",
    "EUR / ZAR": "EURZAR=X",
    "EUR / BRL (Brazilian real)": "EURBRL=X",
    "EUR / MXN (Mexican peso)": "EURMXN=X",
    "EUR / MAD (Moroccan dirham)": "EURMAD=X",
    "EUR / VND (Vietnamese dong)": "EURVND=X",
    "USD / JPY": "USDJPY=X",
    "USD / MXN (Mexican peso)": "USDMXN=X",
    "USD / BRL (Brazilian real)": "USDBRL=X",
    "USD / ZAR (South African rand)": "USDZAR=X",
}

pair_label = st.sidebar.selectbox(
    "Select FX pair:",
    list(FX_PAIRS.keys()),
    index=0
)
ticker = FX_PAIRS[pair_label]  # used for yfinance

period = st.sidebar.selectbox("History period:", ["6mo", "1y", "2y", "5y"], index=1)

st.sidebar.markdown(
    """
    _Tip_: These are common FX crosses.  
    You can also look up extra tickers on Yahoo Finance and add them to this list.
    """
)

# -------------------------
# HELPERS
# -------------------------
def safe_last(series: pd.Series) -> float:
    """Return last non-NaN value of a Series, or NaN if empty."""
    s = series.dropna()
    return s.iloc[-1] if not s.empty else np.nan

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute a simple RSI with given window length."""
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -------------------------
# MAIN LOGIC
# -------------------------
if st.sidebar.button("Run analysis"):

    with st.spinner(f"Downloading data for {pair_label} from Yahoo Finance..."):
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

        # Sometimes this is a DataFrame (MultiIndex) → take first column
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close.name = pair_label

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

        # Momentum (RSI 14d)
        rsi14 = compute_rsi(close, window=14)

        # Risk metrics: VaR & ES at 95%
        VaR_95 = returns.quantile(0.05)
        ES_95 = returns[returns <= VaR_95].mean()

        # Latest metrics
        last_price = safe_last(close)
        last_vol30 = safe_last(vol30)
        last_vol90 = safe_last(vol90)

        # -------------------------
        # OVERVIEW + METRICS
        # -------------------------
        st.markdown(f"### {pair_label} overview")

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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📈 Price & MAs",
                "📉 Volatility",
                "📊 Returns distribution",
                "📐 Momentum (RSI)",
                "📌 Stats table",
                "🔮 Forecasting",
            ]
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
            ax1.set_title(f"{pair_label} - Price & Moving Averages")
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
            ax2.set_title(f"{pair_label} - Rolling Annualized Volatility")
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
            ax3.set_title(f"{pair_label} - Histogram of Daily Returns")
            st.pyplot(fig3)

        # ---- TAB 4: MOMENTUM (RSI) ----
        with tab4:
            st.subheader("14-day Relative Strength Index (RSI)")

            fig4, ax4 = plt.subplots(figsize=(10, 3))
            ax4.plot(rsi14.index, rsi14, linewidth=1)
            ax4.axhline(70, linestyle="--")
            ax4.axhline(30, linestyle="--")
            ax4.set_ylim(0, 100)
            ax4.set_xlabel("Date")
            ax4.set_ylabel("RSI")
            ax4.set_title(f"{pair_label} - RSI (14d)")
            st.pyplot(fig4)

            st.markdown(
                """
                - RSI above **70** is often interpreted as **overbought** (strong recent appreciation).
                - RSI below **30** is often interpreted as **oversold** (strong recent depreciation).
                """
            )

        # ---- TAB 5: SUMMARY STATISTICS ----
        with tab5:
            st.subheader("Summary Statistics")

            stats = pd.DataFrame({
                "Mean daily return": [returns.mean()],
                "Std daily return": [returns.std()],
                "Annualized vol (last 30d)": [last_vol30],
                "Annualized vol (last 90d)": [last_vol90],
                "VaR 95% (daily)": [VaR_95],
                "Expected Shortfall 95% (daily)": [ES_95],
            }).T
            stats.columns = ["Value"]
            st.table(stats)

        # ---- TAB 6: FORECASTING ----
        with tab6:
            st.subheader("30-day Linear Trend Forecast (using last 120 days)")

            series = close.dropna()
            if len(series) < 40:
                st.warning("Not enough data to build a meaningful forecast (need at least 40 observations).")
            else:
                horizon = 30
                lookback = min(120, len(series))  # use last 120 days (or fewer if not available)

                recent = series.iloc[-lookback:]
                X = np.arange(len(recent)).reshape(-1, 1)
                y = recent.values

                model = LinearRegression()
                model.fit(X, y)

                X_future = np.arange(len(recent), len(recent) + horizon).reshape(-1, 1)
                y_future = model.predict(X_future)

                last_date = recent.index[-1]
                future_index = pd.date_range(last_date + pd.Timedelta(days=1),
                                             periods=horizon, freq="D")

                forecast_series = pd.Series(y_future, index=future_index, name="Forecast")

                fig6, ax6 = plt.subplots(figsize=(10, 4))
                ax6.plot(recent.index, recent, label="Historical (recent window)", linewidth=1)
                ax6.plot(
                    forecast_series.index,
                    forecast_series,
                    label="Forecast (linear regression)",
                    linestyle="--",
                    linewidth=2,
                )
                ax6.set_xlabel("Date")
                ax6.set_ylabel("Price")
                ax6.set_title(
                    f"{pair_label} - {horizon}-day Linear Trend Forecast "
                    f"(using last {lookback} days)"
                )
                ax6.legend()
                ax6.grid(True)
                st.pyplot(fig6)

                st.markdown(
                    """
                    This forecast uses a **linear regression** fitted on the last 120 days 
                    of data (or fewer if not available) and extends that trend 30 days 
                    into the future.

                    In practice, FX rates often behave close to a random walk, so this is mainly an 
                    **illustrative forecasting exercise**, not a trading signal.
                    """
                )

else:
    st.info("Use the sidebar to select a pair and click **Run analysis** to start.")
