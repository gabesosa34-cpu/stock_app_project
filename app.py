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
st.markdown("""
    <h1 style='text-align: center; color: #0A84FF;'>
        Stock Performance & Risk Dashboard
    </h1>
    <p style='text-align: center; color: gray;'>
        Compare stocks, analyze risk, and benchmark against the S&P 500
    </p>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market View",
    "Performance",
    "Risk Analysis",
    "Relationships",
    "About"
])

# -- Sidebar: user inputs --------------------------------
st.sidebar.header("Dashboard Controls")
st.sidebar.caption("Choose stocks and a date range to update the analysis.")

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
    main_ticker = tickers[0]
    main_prices = prices[main_ticker].dropna()

    latest_close = float(main_prices.iloc[-1])
    total_return = float((main_prices.iloc[-1] / main_prices.iloc[0]) - 1)

    volatility = float(main_prices.pct_change().dropna().std())
    ann_volatility = volatility * np.sqrt(252)

    max_close = float(main_prices.max())
    min_close = float(main_prices.min())
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

        selected_stocks = st.multiselect(
            "Select stocks to display",
            options=prices.columns.tolist(),
            default=prices.columns.tolist()
        )

        fig = go.Figure()

        for t in selected_stocks:
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

        stats_returns = prices.pct_change().dropna()
        stats_table = summary_stats(stats_returns)

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
            wealth_df[col] = 10000 * (prices[col] / prices[col].dropna().iloc[0])

        portfolio_returns = prices[[t for t in prices.columns if t != "^GSPC"]].pct_change().dropna().mean(axis=1)
        wealth_df["Equal-Weight Portfolio"] = 10000 * (1 + portfolio_returns).cumprod()

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
        vol_window = st.selectbox("Rolling volatility window", [30, 60, 90], index=0)

        st.subheader(f"Rolling Volatility ({vol_window}-Day)")
        returns_df = prices.pct_change().dropna()

        rolling_vol = returns_df.rolling(vol_window).std() * np.sqrt(252)

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
            title=f"Rolling {vol_window}-Day Annualized Volatility",
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
        #BOX -----------------------------------
        st.subheader("Box Plot of Daily Returns")

        box_returns = prices[[t for t in prices.columns if t != "^GSPC"]].pct_change().dropna()

        fig_box = go.Figure()

        for col in box_returns.columns:
            fig_box.add_trace(
                go.Box(
                    y=box_returns[col],
                    name=col,
                    boxmean=True
                )
            )

        fig_box.update_layout(
            title="Daily Return Distributions",
            yaxis_title="Daily Return",
            xaxis_title="Stocks",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_box, width="stretch")

    with tab4:
        #COR ----------------------------------
        st.subheader("Correlation Matrix")

        returns_df = prices.pct_change().dropna()

        corr_matrix = returns_df.corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1
        )

        fig_corr.update_layout(
            title="Correlation Heatmap of Returns",
            height=500
        )
        st.plotly_chart(fig_corr, width="stretch")
        #Scat-plot ------------------------------
        st.subheader("Return Scatter Plot")

        stock_x = st.selectbox("Select X-axis stock", prices.columns, index=0)
        stock_y = st.selectbox("Select Y-axis stock", prices.columns, index=1)

        scatter_df = prices[[stock_x, stock_y]].pct_change().dropna()

        fig_scatter = go.Figure()

        fig_scatter.add_trace(
            go.Scatter(
                x=scatter_df[stock_x],
                y=scatter_df[stock_y],
                mode="markers"
            )
        )

        fig_scatter.update_layout(
            title=f"{stock_x} vs {stock_y} Daily Returns",
            xaxis_title=stock_x,
            yaxis_title=stock_y,
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_scatter, width="stretch")
        #roll-corr ---------------------------------
        st.subheader("Rolling Correlation")

        corr_stock_1 = st.selectbox("Select first stock", prices.columns, index=0, key="corr1")
        corr_stock_2 = st.selectbox("Select second stock", prices.columns, index=1, key="corr2")
        corr_window = st.selectbox("Rolling correlation window", [30, 60, 90], index=0)

        rolling_corr = (
            prices[[corr_stock_1, corr_stock_2]]
            .pct_change()
            .dropna()[corr_stock_1]
            .rolling(corr_window)
            .corr(
                prices[[corr_stock_1, corr_stock_2]]
                .pct_change()
                .dropna()[corr_stock_2]
            )
        )

        fig_rollcorr = go.Figure()

        fig_rollcorr.add_trace(
            go.Scatter(
                x=rolling_corr.index,
                y=rolling_corr,
                mode="lines",
                name="Rolling Correlation"
            )
        )

        fig_rollcorr.update_layout(
            title=f"Rolling {corr_window}-Day Correlation: {corr_stock_1} vs {corr_stock_2}",
            xaxis_title="Date",
            yaxis_title="Correlation",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_rollcorr, width="stretch")
    with tab5:
        st.subheader("About This App")

        st.write("""
        This dashboard analyzes stock performance using historical data from Yahoo Finance.

        It allows users to compare multiple stocks against the S&P 500 and evaluate:
        - Price performance
        - Returns and cumulative growth
        - Risk (volatility and distribution)
        - Correlation between assets

        The analysis includes statistical measures and visualizations to better understand risk and return behavior.
        """)
else: 
    st.info("Enter 2 to 5 stock tickers in the sidebar to get started.")