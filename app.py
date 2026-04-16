# app.py
# -------------------------------------------------------
# A simple Streamlit stock analysis dashboard.
# Run with:  uv run streamlit run app.py
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import numpy as np
from scipy.stats import norm, probplot, jarque_bera, skew, kurtosis

st.set_page_config(page_title="PulseQuant Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp{background:radial-gradient(circle at top left,rgba(79,209,197,.14),transparent 28%),radial-gradient(circle at top right,rgba(255,107,107,.12),transparent 24%),linear-gradient(180deg,#030b16 0%,#07111f 55%,#0b1627 100%);color:#e9f1ff}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,rgba(7,17,31,.98),rgba(9,20,36,.98));border-right:1px solid rgba(143,173,217,.18)}
    .hero,.section-card,.mini-card{border:1px solid rgba(143,173,217,.18);background:linear-gradient(180deg,rgba(12,28,47,.95),rgba(7,18,34,.88))}
    .hero{padding:1.35rem 1.6rem;border-radius:24px;box-shadow:0 18px 50px rgba(0,0,0,.25);margin-bottom:1rem}
    .section-card{padding:1rem 1.1rem;border-radius:20px;margin-bottom:1rem}
    .mini-card{padding:.9rem 1rem;border-radius:18px;min-height:110px}
    .kicker{color:#4fd1c5;text-transform:uppercase;letter-spacing:.18em;font-size:.75rem;font-weight:700}
    .title{font-size:2.2rem;font-weight:800;margin:.2rem 0;color:#e9f1ff}
    .copy,.mini-note{color:#8ea4c7}
    .mini-label{font-size:.78rem;color:#8ea4c7;text-transform:uppercase;letter-spacing:.1em}
    .mini-value{font-size:1.8rem;font-weight:800;color:#e9f1ff}
    .insight{border-left:4px solid #4fd1c5;background:rgba(79,209,197,.08);padding:.85rem 1rem;border-radius:14px;color:#e9f1ff;margin-bottom:1rem}
    .stMetric{background:linear-gradient(180deg,rgba(11,22,38,.9),rgba(8,16,29,.95));border:1px solid rgba(143,173,217,.18);padding:.7rem;border-radius:16px}
    .stButton>button{background:linear-gradient(135deg,#18a999,#0e7b6c);color:#fff;border-radius:999px;border:none}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="kicker">PulseQuant Terminal</div>
      <div class="title">Stock Performance and Risk Dashboard</div>
      <div class="copy">Benchmark stocks against the S&amp;P 500, inspect return behavior, and surface risk with a more polished trading-desk feel.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market View", "Performance", "Risk Analysis", "Relationships", "About"])

st.sidebar.header("Control Center")
tickers_input = st.sidebar.text_input("Enter 2 to 5 stock tickers", value="AAPL,MSFT,NVDA").upper()
tickers = list(dict.fromkeys([t.strip() for t in tickers_input.split(",") if t.strip()]))
default_start = date.today() - timedelta(days=365 * 2)
start_date = st.sidebar.date_input("Start Date", value=default_start, min_value=date(1970, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today(), min_value=date(1970, 1, 1))
ma_short = st.sidebar.selectbox("Short moving average", [20, 50, 100], index=1)
ma_long = st.sidebar.selectbox("Long moving average", [100, 150, 200], index=2)
show_benchmark = st.sidebar.toggle("Show S&P 500 benchmark", value=True)

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()
if len(tickers) < 2 or len(tickers) > 5:
    st.sidebar.error("Please enter between 2 and 5 stock tickers.")
    st.stop()
if (end_date - start_date).days < 365:
    st.sidebar.error("Date range must be at least 1 year.")
    st.stop()

@st.cache_data(show_spinner="Fetching market data...", ttl=300)
def load_data(tickers_list: list[str], start: date, end: date):
    price_dict, bad_tickers = {}, []
    for ticker in tickers_list + ["^GSPC"]:
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, threads=False, auto_adjust=False)
            if df.empty:
                bad_tickers.append(ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if price_col not in df.columns:
                bad_tickers.append(ticker)
                continue
            price_dict[ticker] = df[price_col].rename(ticker)
        except Exception:
            bad_tickers.append(ticker)
    return (pd.DataFrame(price_dict).sort_index(), bad_tickers) if price_dict else (pd.DataFrame(), bad_tickers)

@st.cache_data(ttl=3600)
def summary_stats(returns_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = pd.DataFrame(index=returns_df.columns)
    stats_df["Annualized Mean Return"] = returns_df.mean() * 252
    stats_df["Annualized Volatility"] = returns_df.std() * np.sqrt(252)
    stats_df["Sharpe Ratio"] = np.where(stats_df["Annualized Volatility"] > 0, stats_df["Annualized Mean Return"] / stats_df["Annualized Volatility"], np.nan)
    stats_df["Skewness"] = returns_df.apply(skew)
    stats_df["Kurtosis"] = returns_df.apply(kurtosis)
    stats_df["Min Daily Return"] = returns_df.min()
    stats_df["Max Daily Return"] = returns_df.max()
    return stats_df

def compute_drawdown(series: pd.Series) -> pd.Series:
    return series / series.cummax() - 1

def beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    joined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if joined.empty or joined.iloc[:, 1].var() == 0:
        return np.nan
    return joined.iloc[:, 0].cov(joined.iloc[:, 1]) / joined.iloc[:, 1].var()

def style_figure(fig: go.Figure, height: int = 500):
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(7,17,31,.25)", margin=dict(l=30, r=20, t=50, b=30), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.08)")

prices, bad_tickers = load_data(tickers, start_date, end_date)
if prices.empty:
    st.error("No market data could be downloaded for the current selection.")
    st.stop()

missing_pct = prices.isna().mean()
drop_cols = [c for c in prices.columns if missing_pct[c] > 0.05 and c != "^GSPC"]
if drop_cols:
    st.warning(f"These tickers had too much missing data and were dropped: {', '.join(drop_cols)}")
    prices = prices.drop(columns=drop_cols)

valid_tickers = [t for t in tickers if t in prices.columns]
if len(valid_tickers) < 2:
    st.error("Please enter at least 2 valid stock tickers with enough historical data.")
    st.stop()
if bad_tickers:
    bad_user_tickers = [t for t in bad_tickers if t != "^GSPC"]
    if bad_user_tickers:
        st.warning(f"These tickers could not be downloaded: {', '.join(bad_user_tickers)}")

benchmark_col = "^GSPC" if "^GSPC" in prices.columns else None
display_columns = valid_tickers + ([benchmark_col] if benchmark_col and show_benchmark else [])
main_ticker = valid_tickers[0]
main_prices = prices[main_ticker].dropna()
main_returns = main_prices.pct_change().dropna()
benchmark_returns = prices[benchmark_col].pct_change().dropna() if benchmark_col else pd.Series(dtype=float)

latest_close = float(main_prices.iloc[-1])
period_return = float(main_prices.iloc[-1] / main_prices.iloc[0] - 1)
ann_volatility = float(main_returns.std() * np.sqrt(252))
main_drawdown = compute_drawdown(main_prices)
beta_value = beta(main_returns, benchmark_returns) if benchmark_col else np.nan

leaderboard = []
for ticker in valid_tickers:
    s = prices[ticker].dropna()
    r = s.pct_change().dropna()
    leaderboard.append({
        "Ticker": ticker,
        "Return": s.iloc[-1] / s.iloc[0] - 1,
        "Volatility": r.std() * np.sqrt(252),
        "Sharpe": ((r.mean() * 252) / (r.std() * np.sqrt(252))) if r.std() > 0 else np.nan,
        "Max Drawdown": compute_drawdown(s).min(),
        "Beta": beta(r, benchmark_returns) if benchmark_col else np.nan,
    })
leaderboard_df = pd.DataFrame(leaderboard).sort_values("Return", ascending=False)

trend_note = (
    f"{main_ticker} is trading above both the {ma_short}-day and {ma_long}-day moving averages."
    if main_prices.iloc[-1] > main_prices.rolling(ma_short).mean().iloc[-1] and main_prices.iloc[-1] > main_prices.rolling(ma_long).mean().iloc[-1]
    else f"{main_ticker} is below at least one key moving average, suggesting weaker momentum."
)

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(f"{main_ticker} Snapshot")
    cols = st.columns(5)
    cols[0].metric("Latest Close", f"${latest_close:,.2f}")
    cols[1].metric("Period Return", f"{period_return:.2%}")
    cols[2].metric("Annualized Volatility", f"{ann_volatility:.2%}")
    cols[3].metric("Max Drawdown", f"{float(main_drawdown.min()):.2%}")
    cols[4].metric("Beta vs S&P 500", "N/A" if np.isnan(beta_value) else f"{beta_value:.2f}")
    st.markdown(f'<div class="insight"><strong>Market read:</strong> {trend_note}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    selected_stocks = st.multiselect("Select stocks to display", options=display_columns, default=display_columns)
    if selected_stocks:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28], subplot_titles=("Price and Trend", "Drawdown"))
        palette = ["#4fd1c5", "#ff6b6b", "#ffd166", "#5aa9ff", "#c084fc", "#9be564"]
        for i, ticker in enumerate(selected_stocks):
            color = palette[i % len(palette)]
            s = prices[ticker].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=ticker, line=dict(color=color, width=2.4)), row=1, col=1)
            dd = compute_drawdown(s)
            fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=f"{ticker} Drawdown", line=dict(color=color, width=1.5), showlegend=False, fill="tozeroy"), row=2, col=1)
            if ticker == main_ticker:
                fig.add_trace(go.Scatter(x=s.index, y=s.rolling(ma_short).mean(), mode="lines", name=f"{ticker} {ma_short}D MA", line=dict(color="#8ea4c7", width=1.8, dash="dot")), row=1, col=1)
                fig.add_trace(go.Scatter(x=s.index, y=s.rolling(ma_long).mean(), mode="lines", name=f"{ticker} {ma_long}D MA", line=dict(color="#ffd166", width=1.8, dash="dash")), row=1, col=1)
        fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        style_figure(fig, 720)
        st.plotly_chart(fig, width="stretch")

        norm_df = prices[selected_stocks].apply(lambda col: col / col.dropna().iloc[0] if col.dropna().size else col)
        fig_norm = go.Figure()
        for i, ticker in enumerate(norm_df.columns):
            fig_norm.add_trace(go.Scatter(x=norm_df.index, y=norm_df[ticker], mode="lines", name=ticker, line=dict(color=palette[i % len(palette)], width=2.2)))
        fig_norm.update_layout(title="Normalized Performance (Start = 1.0)", xaxis_title="Date", yaxis_title="Growth Multiple")
        style_figure(fig_norm, 460)
        st.plotly_chart(fig_norm, width="stretch")
    else:
        st.warning("Select at least one ticker to render the market view charts.")

with tab2:
    st.subheader("Performance Leaderboard")
    st.dataframe(leaderboard_df.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}", "Beta": "{:.2f}"}), width="stretch")
    st.subheader("Summary Statistics")
    stats_returns = prices[display_columns].pct_change().dropna()
    st.dataframe(summary_stats(stats_returns).style.format({"Annualized Mean Return": "{:.2%}", "Annualized Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}", "Skewness": "{:.3f}", "Kurtosis": "{:.3f}", "Min Daily Return": "{:.2%}", "Max Daily Return": "{:.2%}"}), width="stretch")
    st.subheader("Growth of $10,000")
    wealth_df = pd.DataFrame(index=prices.index)
    for col in display_columns:
        wealth_df[col] = 10000 * (prices[col] / prices[col].dropna().iloc[0])
    wealth_df["Equal-Weight Portfolio"] = 10000 * (1 + prices[valid_tickers].pct_change().dropna().mean(axis=1)).cumprod()
    fig_wealth = go.Figure()
    for col in wealth_df.columns:
        fig_wealth.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df[col], mode="lines", name=col, line=dict(width=2.3 if col == "Equal-Weight Portfolio" else 1.9)))
    fig_wealth.update_layout(title="Cumulative Wealth (Normalized to $10,000)", xaxis_title="Date", yaxis_title="Portfolio Value ($)")
    style_figure(fig_wealth, 520)
    st.plotly_chart(fig_wealth, width="stretch")
    c1, c2 = st.columns(2)
    c1.markdown(f'<div class="mini-card"><div class="mini-label">Top Performer</div><div class="mini-value">{leaderboard_df.iloc[0]["Ticker"]}</div><div class="mini-note">Return over period: {leaderboard_df.iloc[0]["Return"]:.2%}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="mini-card"><div class="mini-label">Lagging Name</div><div class="mini-value">{leaderboard_df.iloc[-1]["Ticker"]}</div><div class="mini-note">Return over period: {leaderboard_df.iloc[-1]["Return"]:.2%}</div></div>', unsafe_allow_html=True)

with tab3:
    risk_stocks = st.multiselect("Select stocks to display in Risk Analysis", options=valid_tickers, default=valid_tickers)
    if risk_stocks:
        vol_window = st.selectbox("Rolling volatility window", [30, 60, 90], index=0)
        risk_df = prices[risk_stocks + ([benchmark_col] if benchmark_col and show_benchmark else [])].pct_change().dropna()
        rolling_vol = risk_df.rolling(vol_window).std() * np.sqrt(252)
        fig_vol = go.Figure()
        for col in rolling_vol.columns:
            fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col))
        fig_vol.update_layout(title=f"Rolling {vol_window}-Day Annualized Volatility", xaxis_title="Date", yaxis_title="Volatility")
        style_figure(fig_vol, 500)
        st.plotly_chart(fig_vol, width="stretch")

        selected_stock = st.selectbox("Select a stock for distribution analysis", options=risk_stocks)
        r = prices[selected_stock].pct_change().dropna()
        jb_stat, jb_p = jarque_bera(r)
        a, b, c = st.columns(3)
        a.metric("Jarque-Bera", f"{jb_stat:.2f}")
        b.metric("p-value", f"{jb_p:.4f}")
        c.metric("Skewness", f"{skew(r):.2f}")
        st.error("Daily returns deviate meaningfully from a normal distribution.") if jb_p < 0.05 else st.success("The normality test does not reject a normal distribution at the 5% level.")

        mu, sigma = norm.fit(r)
        x_vals = np.linspace(r.min(), r.max(), 300)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=r, histnorm="probability density", name="Returns", opacity=.62, marker_color="#4fd1c5"))
        fig_hist.add_trace(go.Scatter(x=x_vals, y=norm.pdf(x_vals, mu, sigma), mode="lines", name="Normal Fit", line=dict(color="#ff6b6b", width=2.5)))
        fig_hist.update_layout(title=f"{selected_stock} Return Distribution", xaxis_title="Daily Return", yaxis_title="Density", barmode="overlay")
        style_figure(fig_hist, 500)
        st.plotly_chart(fig_hist, width="stretch")

        qq = probplot(r, dist="norm")
        theoretical, ordered = qq[0][0], qq[0][1]
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical, y=ordered, mode="markers", name="Q-Q Points", marker=dict(color="#5aa9ff", size=7)))
        fig_qq.add_trace(go.Scatter(x=theoretical, y=theoretical, mode="lines", name="45 Degree Line", line=dict(color="#ffd166", dash="dash")))
        fig_qq.update_layout(title=f"{selected_stock} Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
        style_figure(fig_qq, 500)
        st.plotly_chart(fig_qq, width="stretch")

        fig_box = go.Figure()
        for col in prices[risk_stocks].pct_change().dropna().columns:
            fig_box.add_trace(go.Box(y=prices[risk_stocks].pct_change().dropna()[col], name=col, boxmean=True))
        fig_box.update_layout(title="Daily Return Distributions", yaxis_title="Daily Return", xaxis_title="Stocks")
        style_figure(fig_box, 500)
        st.plotly_chart(fig_box, width="stretch")
    else:
        st.warning("Please select at least one stock.")

with tab4:
    st.subheader("Correlation Matrix")
    corr_df = prices[display_columns].pct_change().dropna()
    corr_matrix = corr_df.corr()
    fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale="RdBu", zmin=-1, zmax=1, colorbar=dict(title="Correlation"), text=corr_matrix.round(2).values, texttemplate="%{text}", textfont={"size": 13}))
    fig_corr.update_layout(title="Correlation Heatmap of Daily Returns")
    style_figure(fig_corr, 500)
    st.plotly_chart(fig_corr, width="stretch")

    stock_x = st.selectbox("Select X-axis stock", display_columns, index=0)
    stock_y = st.selectbox("Select Y-axis stock", display_columns, index=1)
    if stock_x != stock_y:
        scatter_df = prices[[stock_x, stock_y]].pct_change().dropna()
        fit = np.polyfit(scatter_df[stock_x], scatter_df[stock_y], 1)
        trend_x = np.linspace(scatter_df[stock_x].min(), scatter_df[stock_x].max(), 100)
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(x=scatter_df[stock_x], y=scatter_df[stock_y], mode="markers", name="Daily Returns", marker=dict(color="#4fd1c5", size=8, opacity=.65)))
        fig_scatter.add_trace(go.Scatter(x=trend_x, y=fit[0] * trend_x + fit[1], mode="lines", name="Trend Line", line=dict(color="#ff6b6b", width=2)))
        fig_scatter.update_layout(title=f"{stock_x} vs {stock_y} Daily Returns", xaxis_title=stock_x, yaxis_title=stock_y)
        style_figure(fig_scatter, 500)
        st.plotly_chart(fig_scatter, width="stretch")
    else:
        st.warning("Please select two different stocks for the scatter plot.")

    corr_stock_1 = st.selectbox("Select first stock", display_columns, index=0, key="corr1")
    corr_stock_2 = st.selectbox("Select second stock", display_columns, index=1, key="corr2")
    corr_window = st.selectbox("Rolling correlation window", [30, 60, 90], index=0)
    if corr_stock_1 != corr_stock_2:
        pair_returns = prices[[corr_stock_1, corr_stock_2]].pct_change().dropna()
        rolling_corr = pair_returns[corr_stock_1].rolling(corr_window).corr(pair_returns[corr_stock_2])
        fig_rollcorr = go.Figure()
        fig_rollcorr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name="Rolling Correlation", line=dict(color="#ffd166", width=2.3)))
        fig_rollcorr.update_layout(title=f"Rolling {corr_window}-Day Correlation: {corr_stock_1} vs {corr_stock_2}", xaxis_title="Date", yaxis_title="Correlation")
        style_figure(fig_rollcorr, 500)
        st.plotly_chart(fig_rollcorr, width="stretch")
    else:
        st.warning("Please select two different stocks for rolling correlation.")

    stock_a = st.selectbox("Select Stock A", valid_tickers, key="port_a")
    stock_b = st.selectbox("Select Stock B", valid_tickers, index=1 if len(valid_tickers) > 1 else 0, key="port_b")
    if stock_a != stock_b:
        weight = st.slider("Weight on Stock A (%)", 0, 100, 50) / 100
        pair = prices[[stock_a, stock_b]].pct_change().dropna()
        mean_returns = pair.mean() * 252
        cov_matrix = pair.cov() * 252
        portfolio_return = weight * mean_returns[stock_a] + (1 - weight) * mean_returns[stock_b]
        portfolio_vol = np.sqrt((weight ** 2) * cov_matrix.loc[stock_a, stock_a] + ((1 - weight) ** 2) * cov_matrix.loc[stock_b, stock_b] + 2 * weight * (1 - weight) * cov_matrix.loc[stock_a, stock_b])
        c1, c2, c3 = st.columns(3)
        c1.metric("Annualized Return", f"{portfolio_return:.2%}")
        c2.metric("Annualized Volatility", f"{portfolio_vol:.2%}")
        c3.metric("Pair Correlation", f"{pair.corr().iloc[0, 1]:.2f}")
        weights = np.linspace(0, 1, 100)
        vols, rets = [], []
        for w in weights:
            rets.append(w * mean_returns[stock_a] + (1 - w) * mean_returns[stock_b])
            vols.append(np.sqrt((w ** 2) * cov_matrix.loc[stock_a, stock_a] + ((1 - w) ** 2) * cov_matrix.loc[stock_b, stock_b] + 2 * w * (1 - w) * cov_matrix.loc[stock_a, stock_b]))
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=vols, y=rets, mode="lines", name="Portfolio Curve", line=dict(color="#4fd1c5", width=2.5)))
        fig_port.add_trace(go.Scatter(x=[portfolio_vol], y=[portfolio_return], mode="markers", marker=dict(size=12, color="#ff6b6b"), name="Current Allocation"))
        fig_port.update_layout(title="Risk / Return Path for Two-Asset Mix", xaxis_title="Annualized Volatility", yaxis_title="Annualized Return")
        style_figure(fig_port, 520)
        st.plotly_chart(fig_port, width="stretch")
    else:
        st.warning("Please select two different stocks for the portfolio explorer.")

with tab5:
    st.subheader("About This App")
    st.write(
        """
        PulseQuant compares stocks, benchmarks them against the S&P 500, and explores volatility, drawdowns, return distribution, and diversification.

        Methodology:
        - Returns use simple daily percentage change.
        - Annualized return uses mean daily return times 252 trading days.
        - Annualized volatility uses daily standard deviation times sqrt(252).
        - Beta is estimated from daily return covariance relative to the S&P 500.
        - Drawdown measures the percentage drop from the running peak.
        """
    )
    st.markdown('<div class="section-card"><strong>Next upgrade ideas</strong><br><br>Add earnings dates, analyst targets, saved watchlists, sector rotation heatmaps, and AI-generated "what changed today" summaries for each ticker.</div>', unsafe_allow_html=True)
