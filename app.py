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

            if "Adj Close" in df.columns:
                price_dict[t] = df["Adj Close"]
            elif "Close" in df.columns:
                price_dict[t] = df["Close"]
            else:
                bad_tickers.append(t)
                continue

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
        df = pd.DataFrame({"Close": prices[tickers[0]]})
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

        selected_chart_tickers = [t for t in tickers if t in prices.columns]

        if "^GSPC" in prices.columns:
            selected_chart_tickers.append("^GSPC")

        fig = go.Figure()

        for t in selected_chart_tickers:
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices[t],
                    mode="lines",
                    name=t,
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
        st.subheader("Growth of $10,000")

        wealth_df = pd.DataFrame(index=prices.index)

        for col in prices.columns:
            wealth_df[col] = 10000 * (prices[col] / prices[col].iloc[0])

        fig_wealth = go.Figure()

        for col in wealth_df.columns:
            fig_wealth.add_trace(
                go.Scatter(
                    x=wealth_df.index,
                    y=wealth_df[col],
                    mode="lines",
                    name=col
                )
            )

        fig_wealth.update_layout(
            title="Cumulative Wealth (Normalized to $10,000)",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_wealth, width="stretch")

    with tab3:
        #ROLL VOL ---------------------------------
        st.subheader("Rolling Volatility (30-Day)")
        returns_df = prices.pct_change().dropna()

        rolling_vol = returns_df.rolling(30).std() * np.sqrt(252)

        fig_vol = go.Figure()

        for col in rolling_vol.columns:
            fig_vol.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol[col],
                    mode="lines",
                    name=col
                )
            )

        fig_vol.update_layout(
            title="Rolling 30-Day Annualized Volatility",
            xaxis_title="Date",
            yaxis_title="Volatility",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_vol, width="stretch")

        st.subheader("Return Distribution")

        selected_stock = st.selectbox(
            "Select a stock for distribution analysis",
            options=[t for t in prices.columns if t != "^GSPC"]
        )

        #HIST -------------------------------------------------
        r = prices[selected_stock].pct_change().dropna()

        # Fit normal distribution
        mu, sigma = norm.fit(r)

        x_vals = np.linspace(r.min(), r.max(), 300)
        y_vals = norm.pdf(x_vals, mu, sigma)

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=r,
                histnorm="probability density",
                name="Returns",
                opacity=0.6
            )
        )

        fig_hist.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name="Normal Distribution Fit"
            )
        )

        fig_hist.update_layout(
            title=f"{selected_stock} Return Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Density",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_hist, width="stretch")

        r = prices[selected_stock].pct_change().dropna()
        jb_stat, jb_p = jarque_bera(r)

        st.write(f"**Jarque-Bera Statistic:** {jb_stat:.4f}")
        st.write(f"**p-value:** {jb_p:.6f}")

        if jb_p < 0.05:
            st.error("Rejects normality (p < 0.05)")
        else:
            st.success("Fails to reject normality (p >= 0.05)")
        # QQ plot ---------------------------------------------    
        st.subheader("Q-Q Plot")

        qq = probplot(r, dist="norm")
        theoretical = qq[0][0]
        ordered = qq[0][1]

        fig_qq = go.Figure()

        fig_qq.add_trace(
            go.Scatter(
                x=theoretical,
                y=ordered,
                mode="markers",
                name="Q-Q Points"
            )
        )

        fig_qq.add_trace(
            go.Scatter(
                x=theoretical,
                y=theoretical,
                mode="lines",
                name="45° Line"
            )
        )

        fig_qq.update_layout(
            title=f"{selected_stock} Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_qq, width="stretch")
    with tab4:
        st.subheader("Correlation Matrix")

        returns_df = prices.pct_change().dropna()

        corr_matrix = returns_df.corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation")
            )
        )

        fig_corr.update_layout(
            title="Correlation Heatmap of Returns",
            height=500
        )

        st.plotly_chart(fig_corr, width="stretch")
else: 
    st.info("Enter 2 to 5 stock tickers in the sidebar to get started.")