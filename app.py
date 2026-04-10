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
    <h1 style='text-align: center; color: #FF3B30;'>
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
@st.cache_data(show_spinner="Fetching data...", ttl=300)
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

    except Exception as e:
        st.error(f"Failed to download data: {e}")
        st.stop()

    valid_tickers = [t for t in tickers if t in prices.columns]

    missing_pct = prices.isna().mean()
    partial_cols = [col for col in prices.columns if 0 < missing_pct[col] <= 0.05 and col != "^GSPC"]
    drop_cols = [col for col in prices.columns if missing_pct[col] > 0.05 and col != "^GSPC"]

    if partial_cols:
        st.info("Some tickers had partial data. Calculations use the overlapping available date range.")

    if drop_cols:
        st.warning(f"These tickers had too much missing data and were dropped: {', '.join(drop_cols)}")
        prices = prices.drop(columns=drop_cols)

    valid_tickers = [t for t in tickers if t in prices.columns]

    if len(valid_tickers) < 2:
        st.error("Please enter at least 2 valid stock tickers.")
        st.stop()
        
    if bad_tickers:
        st.warning(f"These tickers could not be downloaded or had insufficient data: {', '.join(bad_tickers)}")

    df = pd.DataFrame({"Close": prices[valid_tickers[0]]})

    if df.empty:
        st.error(f"No data found for {valid_tickers[0]}. Check the ticker symbol and try again.")
        st.stop()

    # Flatten any multi-level columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # -- Compute a derived column -------------------------
    df["Daily Return"] = df["Close"].pct_change()
    returns = df["Close"].pct_change().dropna()

    # -- Key metrics --------------------------------------
    main_ticker = valid_tickers[0]
    main_prices = prices[main_ticker].dropna()

    latest_close = float(main_prices.iloc[-1])
    total_return = float((main_prices.iloc[-1] / main_prices.iloc[0]) - 1)

    volatility = float(main_prices.pct_change().dropna().std())
    ann_volatility = volatility * np.sqrt(252)

    max_close = float(main_prices.max())
    min_close = float(main_prices.min())
    
    with tab1:
        
        st.subheader(f"{valid_tickers[0]} — Key Metrics")

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
        colors = ["#3A86FF", "#1E3A8A", "#FF3B30", "#F97316", "#9CA3AF"]
        for i, t in enumerate(selected_stocks):
            fig.add_trace(
                go.Scatter(
                    x=prices.index,
                    y=prices[t],
                    mode="lines",
                    name=t,
                    line=dict(color=colors[i % len(colors)], width=2)
                )
            )
        fig.update_layout(
            yaxis_title="Price (USD)", xaxis_title="Date",
            template="plotly_dark", height=450
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

        dist_view = st.radio("Select view", ["Histogram", "Q-Q Plot"], horizontal=True)

        r = prices[selected_stock].pct_change().dropna()

        jb_stat, jb_p = jarque_bera(r)

        st.write(f"**Jarque-Bera Statistic:** {jb_stat:.4f}")
        st.write(f"**p-value:** {jb_p:.6f}")

        if jb_p < 0.05:
            st.error("Rejects normality (p < 0.05)")
        else:
            st.success("Fails to reject normality (p >= 0.05)")

        if dist_view == "Histogram":
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

        if dist_view == "Q-Q Plot":
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
        #Heat -------------------------
        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 14}
            )
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
        #TWO Asset -----------------------------------
        st.subheader("Two-Asset Portfolio Explorer")

        stock_a = st.selectbox("Select Stock A", valid_tickers, key="port_a")
        stock_b = st.selectbox("Select Stock B", valid_tickers, index=1, key="port_b")

        weight = st.slider("Weight on Stock A (%)", 0, 100, 50) / 100

        returns_df = prices[[stock_a, stock_b]].pct_change().dropna()

        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252

        w = weight

        portfolio_return = w * mean_returns[stock_a] + (1 - w) * mean_returns[stock_b]

        portfolio_vol = np.sqrt(
            (w**2) * cov_matrix.loc[stock_a, stock_a] +
            ((1 - w)**2) * cov_matrix.loc[stock_b, stock_b] +
            2 * w * (1 - w) * cov_matrix.loc[stock_a, stock_b]
        )
        st.write(f"Annualized Return: {portfolio_return:.2%}")
        st.write(f"Annualized Volatility: {portfolio_vol:.2%}")
        weights = np.linspace(0, 1, 100)
        vols = []

        for w in weights:
            vol = np.sqrt(
                (w**2) * cov_matrix.loc[stock_a, stock_a] +
                ((1 - w)**2) * cov_matrix.loc[stock_b, stock_b] +
                2 * w * (1 - w) * cov_matrix.loc[stock_a, stock_b]
            )
            vols.append(vol)

        fig_port = go.Figure()

        fig_port.add_trace(
            go.Scatter(
                x=weights,
                y=vols,
                mode="lines",
                name="Portfolio Volatility"
            )
        )

        fig_port.add_trace(
            go.Scatter(
                x=[weight],
                y=[portfolio_vol],
                mode="markers",
                marker=dict(size=10),
                name="Current Allocation"
            )
        )

        fig_port.update_layout(
            title="Portfolio Volatility vs Weight",
            xaxis_title="Weight on Stock A",
            yaxis_title="Volatility",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig_port, width="stretch")
        st.caption(
            "This curve demonstrates diversification. When two stocks are not perfectly correlated, "
            "combining them can reduce overall portfolio risk. The lower the correlation, the greater the benefit."
        )
    with tab5:
        st.subheader("About This App")

        st.write("""
        This dashboard analyzes stock performance using historical adjusted closing price data from Yahoo Finance (via yfinance).

        Key assumptions and methodology:

        - Returns are computed as **simple (arithmetic) daily returns** using percentage change.
        - Annualized return is calculated as mean daily return × 252 trading days.
        - Annualized volatility is calculated as standard deviation × √252.
        - The S&P 500 (^GSPC) is used as a benchmark for comparison.
        - Portfolio calculations assume equal weighting unless otherwise specified.

        This dashboard provides insights into price performance, risk, return distributions, and diversification across multiple assets.
        """)
else: 
    st.info("Enter 2 to 5 stock tickers in the sidebar to get started.")