# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import math
import numpy as np
import plotly.express as px
from scipy.stats import norm, probplot, jarque_bera, skew, kurtosis

# -- Page configuration ----------------------------------
# st.set_page_config must be the FIRST Streamlit command in the script.
# If you add any other st.* calls above this line, you'll get an error.
st.set_page_config(page_title="Stock Analyzer", layout="wide")
st.title("Stock Analysis Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prices",
    "Returns",
    "Risk",
    "Correlation",
    "About"
])

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Settings")

tickers_input = st.sidebar.text_input(
    "Enter 2 to 5 Stock Tickers (comma separated)",
    value="AAPL,MSFT"
).upper()

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# Default date range: one year back from today
default_start = date.today() - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970,1,1) )
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970,1,1))

# Validate that the date range makes sense
if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

# Validate ticker count
if len(tickers) < 2 or len(tickers) > 5:
    st.sidebar.error("Please enter between 2 and 5 stock tickers.")
    st.stop()

# Enforce minimum 1-year range
if (end_date - start_date).days < 365:
    st.sidebar.error("Date range must be at least 1 year.")
    st.stop()

# -- Data download ----------------------------------------
# We wrap the download in st.cache_data so repeated runs with
# the same inputs don't re-download every time. The ttl (time-to-live)
# ensures the cache expires after one hour so data stays fresh.
@st.cache_data(show_spinner="Fetching data...", ttl=3600)
def load_data(tickers: list[str], start: date, end: date):
    all_tickers = tickers + ["^GSPC"]
    price_dict = {}
    bad_tickers = []

    for t in all_tickers:
        try:
            df = yf.download(
                t,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                threads=False
            )

            if df.empty:
                bad_tickers.append(t)
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if "Adj Close" not in df.columns:
                bad_tickers.append(t)
                continue

            price_dict[t] = df["Adj Close"]

        except Exception:
            bad_tickers.append(t)

    if not price_dict:
        return pd.DataFrame(), bad_tickers

    prices = pd.DataFrame(price_dict)

    return prices, bad_tickers

@st.cache_data(ttl=3600)
def summary_stats(returns_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = pd.DataFrame(index=returns_df.columns)

    stats_df["Annualized Mean Return"] = returns_df.mean() * 252
    stats_df["Annualized Volatility"] = returns_df.std() * np.sqrt(252)
    stats_df["Skewness"] = returns_df.apply(skew)
    stats_df["Kurtosis"] = returns_df.apply(kurtosis)
    stats_df["Min Daily Return"] = returns_df.min()
    stats_df["Max Daily Return"] = returns_df.max()

    return stats_df

# -- Main logic -------------------------------------------
if tickers:
    try:
        prices, bad_tickers = load_data(tickers, start_date, end_date)
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()   

    if df.empty:
        st.error(
            f"No data found for **{tickers[0]}**. "
            "Check the ticker symbol and try again."
        )
        st.stop()

    # Flatten any multi-level columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -- Compute a derived column -------------------------
    df["Daily Return"] = df["Close"].pct_change()
    returns = df["Close"].pct_change().dropna()

    # -- Key metrics --------------------------------------
    latest_close = float(df["Close"].iloc[-1])
    total_return = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1)
    volatility = float(df["Daily Return"].std())
    ann_volatility = volatility * math.sqrt(252)  # Annualize: daily sigma * sqrt(trading days)
    max_close = float(df["Close"].max())
    min_close = float(df["Close"].min())

    with tab1:

        st.subheader(f"{tickers[0]} — Key Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Latest Close", f"${latest_close:,.2f}")
        col2.metric("1-Year Return", f"{total_return:.2%}")
        col3.metric("Annualized Volatility (sigma)", f"{ann_volatility:.2%}")

        col4, col5, _ = st.columns(3)
        col4.metric("Period High", f"${max_close:,.2f}")
        col5.metric("Period Low", f"${min_close:,.2f}")

        st.divider()

    # -- Price chart --------------------------------------
        st.subheader("Closing Price")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["Close"],
                mode="lines", name="Close Price",
                line=dict(width=1.5)
            )
        )
        fig.update_layout(
            yaxis_title="Price (USD)", xaxis_title="Date",
            template="plotly_white", height=450
        )
        st.plotly_chart(fig, width="stretch")
    with tab2:

            st.subheader("Summary Statistics")

            stats_table = summary_stats(returns.to_frame(name=tickers[0]))

            st.dataframe(
                stats_table.style.format({
                    "Annualized Mean Return": "{:.2%}",
                    "Annualized Volatility": "{:.2%}",
                    "Skewness": "{:.3f}",
                    "Kurtosis": "{:.3f}",
                    "Min Daily Return": "{:.2%}",
                    "Max Daily Return": "{:.2%}",
                }),
                width="stretch"
            )
else: 
    st.info("Enter 2 to 5 stock tickers in the sidebar to get started.")