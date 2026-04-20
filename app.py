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
from scipy.stats import norm, jarque_bera, skew, kurtosis

st.set_page_config(page_title="PulseQuant Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stApp{background:radial-gradient(circle at top left,rgba(34,197,94,.09),transparent 22%),radial-gradient(circle at top right,rgba(59,130,246,.10),transparent 24%),linear-gradient(180deg,#07111b 0%,#0b1724 52%,#0e1c2d 100%);color:#e6eef8}
    [data-testid="stHeader"]{background:rgba(0,0,0,0)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#0b1622,#0f1c2c);border-right:1px solid rgba(125,151,179,.16)}
    [data-testid="stSidebar"] *{color:#d9e5f5 !important}
    .terminal-bar,.hero,.section-card,.mini-card,.side-card,.market-row{border:1px solid rgba(125,151,179,.14);background:linear-gradient(180deg,rgba(14,24,37,.96),rgba(10,19,31,.98))}
    .terminal-bar{padding:.8rem 1rem;border-radius:18px;margin-bottom:1rem;box-shadow:0 16px 40px rgba(0,0,0,.22)}
    .terminal-brand{font-size:1.05rem;font-weight:800;letter-spacing:.05em;color:#f8fbff}
    .terminal-copy{color:#8ea4c1;font-size:.9rem}
    .hero{padding:1.5rem 1.7rem;border-radius:26px;box-shadow:0 24px 60px rgba(0,0,0,.22);margin-bottom:1.1rem;position:relative;overflow:hidden}
    .hero:before{content:"";position:absolute;inset:auto -8% -45% auto;width:280px;height:280px;background:radial-gradient(circle,rgba(59,130,246,.18),transparent 66%)}
    .section-card{padding:1rem 1.1rem;border-radius:22px;margin-bottom:1rem;box-shadow:0 12px 32px rgba(0,0,0,.16)}
    .mini-card{padding:1rem 1.05rem;border-radius:18px;min-height:110px;box-shadow:0 10px 24px rgba(0,0,0,.14)}
    .side-card{padding:1rem;border-radius:20px;margin-bottom:1rem}
    .kicker{color:#4ade80;text-transform:uppercase;letter-spacing:.18em;font-size:.72rem;font-weight:800}
    .title{font-size:2.4rem;font-weight:900;margin:.25rem 0;color:#f8fbff;line-height:1.03}
    .copy,.mini-note,.panel-copy{color:#8ea4c1}
    .mini-label,.panel-kicker{font-size:.75rem;color:#7dd3fc;text-transform:uppercase;letter-spacing:.11em}
    .mini-value{font-size:1.8rem;font-weight:900;color:#f8fbff}
    .insight{border-left:4px solid #22c55e;background:linear-gradient(90deg,rgba(34,197,94,.12),rgba(34,197,94,.04));padding:.95rem 1rem;border-radius:16px;color:#e6eef8;margin-bottom:1rem}
    .section-title{font-size:1.25rem;font-weight:800;color:#f8fbff;margin-bottom:.2rem}
    .stMetric{background:linear-gradient(180deg,rgba(16,27,42,.98),rgba(12,21,33,.98));border:1px solid rgba(125,151,179,.14);padding:.75rem;border-radius:18px;box-shadow:0 10px 28px rgba(0,0,0,.15)}
    .stMetric label,.stMetric [data-testid="stMetricLabel"],.stMetric [data-testid="stMetricValue"]{color:#f8fbff !important}
    .stButton>button{background:linear-gradient(135deg,#2563eb,#1d4ed8);color:#fff;border-radius:999px;border:none;box-shadow:0 10px 24px rgba(37,99,235,.22)}
    .stTabs [data-baseweb="tab-list"]{gap:.45rem;border-bottom:1px solid rgba(125,151,179,.18);padding-bottom:.35rem}
    .stTabs [data-baseweb="tab"]{background:rgba(16,27,42,.85);border:1px solid rgba(125,151,179,.16);border-radius:14px;color:#dbeafe !important;font-weight:700;padding:.4rem .9rem}
    .stTabs [aria-selected="true"]{background:rgba(30,41,59,.98) !important;color:#ffffff !important;border-color:rgba(96,165,250,.45);box-shadow:0 10px 24px rgba(0,0,0,.18)}
    .stTabs [data-baseweb="tab-highlight"]{display:none !important}
    .stSelectbox label,.stMultiSelect label,.stDateInput label,.stTextInput label,.stSlider label,.stToggle label{color:#cfe0f5 !important;font-weight:700}
    .stCaption,.stMarkdown,.stText{color:#dbeafe}
    p, label, span, div[data-testid="stMarkdownContainer"]{color:#dbeafe}
    [data-baseweb="select"] > div,[data-baseweb="input"] > div,.stDateInput > div > div{background:#101b2a !important;border:1px solid rgba(125,151,179,.16) !important;color:#f8fbff !important;border-radius:14px !important;box-shadow:none}
    [data-baseweb="tag"]{background:#15314d !important;color:#dff7ff !important;border:1px solid rgba(125,151,179,.18)}
    .market-row{display:flex;justify-content:space-between;align-items:flex-start;padding:.75rem .85rem;border-radius:14px;margin-bottom:.6rem}
    .ticker-name{font-size:.95rem;font-weight:700;color:#f8fbff}
    .ticker-meta{font-size:.78rem;color:#7f93ae}
    .ticker-price{font-size:1rem;font-weight:800;color:#f8fbff;text-align:right}
    .badge-up,.badge-down,.badge-flat{display:inline-block;padding:.22rem .5rem;border-radius:999px;font-size:.76rem;font-weight:700}
    .badge-up{background:rgba(34,197,94,.16);color:#7ef0a1}
    .badge-down{background:rgba(239,68,68,.16);color:#ff9999}
    .badge-flat{background:rgba(148,163,184,.16);color:#d6e1ee}
    .help-note{padding:.9rem 1rem;border-radius:16px;background:rgba(37,99,235,.10);border:1px solid rgba(59,130,246,.18);color:#dbeafe}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="terminal-bar">
      <div class="terminal-brand">Stock Dashboard</div>
      <div class="terminal-copy">A rebuilt market workspace with chart-first layout, performance board, and relationship panels.</div>
    </div>
    <div class="hero">
      <div class="kicker">Market Workspace</div>
      <div class="title">Chart-first stock analysis, rebuilt.</div>
      <div class="copy">Inspired by trading terminals and market maps, this layout puts the main chart, movers, risk, and relationships on one screen instead of hiding them behind tabs.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Performance", "Risk", "Relationships", "Method"])

st.sidebar.header("Control Center")
tickers_input = st.sidebar.text_input("Enter 2 to 5 stock tickers", value="", placeholder="AAPL, MSFT, NVDA").upper()
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
if not tickers:
    st.info("Enter 2 to 5 stock tickers in the sidebar to get started.")
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
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=30, r=20, t=70, b=30),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
        ),
        font=dict(color="#18324f"),
        title_x=0.02,
        title_xanchor="left",
        title_y=0.97,
        title_yanchor="top",
        title_font=dict(size=18, color="#18324f"),
    )
    fig.update_xaxes(showgrid=False, color="#48627e")
    fig.update_yaxes(gridcolor="rgba(24,50,79,.10)", color="#48627e")

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
display_name_map = {"^GSPC": "S&P 500", "Equal-Weight Portfolio": "Equal-Weight"}
chart_palette = ["#0f766e", "#2563eb", "#f97316", "#7c3aed", "#dc2626", "#0891b2"]
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

latest_returns = prices[valid_tickers].pct_change().dropna().iloc[-1].sort_values(ascending=False)
stats_returns = prices[display_columns].pct_change().dropna()
wealth_df = pd.DataFrame(index=prices.index)
for col in display_columns:
    wealth_df[col] = 10000 * (prices[col] / prices[col].dropna().iloc[0])
wealth_df["Equal-Weight Portfolio"] = 10000 * (1 + prices[valid_tickers].pct_change().dropna().mean(axis=1)).cumprod()
corr_df = prices[display_columns].pct_change().dropna()
corr_matrix = corr_df.corr()

with tab1:
    selected_stocks = st.multiselect(
        "Chart symbols",
        options=display_columns,
        default=display_columns,
        help="Choose which symbols appear in the overview chart.",
        key="overview_symbols",
    )
    top_left, top_right = st.columns([2.3, 1], gap="large")
    with top_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Market Focus</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight"><strong>{main_ticker} read:</strong> {trend_note}</div>', unsafe_allow_html=True)
        metric_cols = st.columns(5)
        metric_cols[0].metric("Latest Close", f"${latest_close:,.2f}")
        metric_cols[1].metric("Period Return", f"{period_return:.2%}")
        metric_cols[2].metric("Ann. Volatility", f"{ann_volatility:.2%}")
        metric_cols[3].metric("Max Drawdown", f"{float(main_drawdown.min()):.2%}")
        metric_cols[4].metric("Beta vs S&P 500", "N/A" if np.isnan(beta_value) else f"{beta_value:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

        if selected_stocks:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.72, 0.28])
            for i, ticker in enumerate(selected_stocks):
                color = chart_palette[i % len(chart_palette)]
                s = prices[ticker].dropna()
                fig.add_trace(go.Scatter(x=s.index, y=s, mode="lines", name=ticker, line=dict(color=color, width=2.4)), row=1, col=1)
                dd = compute_drawdown(s)
                fig.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=f"{ticker} Drawdown", line=dict(color=color, width=1.5), showlegend=False, fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.22)"), row=2, col=1)
                if ticker == main_ticker:
                    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(ma_short).mean(), mode="lines", name=f"{ticker} {ma_short}D MA", line=dict(color="#94a3b8", width=1.8, dash="dot")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(ma_long).mean(), mode="lines", name=f"{ticker} {ma_long}D MA", line=dict(color="#f59e0b", width=1.8, dash="dash")), row=1, col=1)
            fig.update_layout(title="Price Action and Drawdown")
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            style_figure(fig, 760)
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("Select at least one symbol to render the main chart panel.")

    with top_right:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-kicker">Live Board</div><div class="section-title">Market Movers</div><div class="panel-copy">Latest daily move across the current watchlist.</div>', unsafe_allow_html=True)
        for ticker in latest_returns.index:
            latest_price = prices[ticker].dropna().iloc[-1]
            move = latest_returns[ticker]
            badge_class = "badge-up" if move > 0 else "badge-down" if move < 0 else "badge-flat"
            badge_text = f"{move:+.2%}"
            st.markdown(
                f"""
                <div class="market-row">
                  <div>
                    <div class="ticker-name">{ticker}</div>
                    <div class="ticker-meta">{'Benchmark' if ticker == '^GSPC' else 'Tracked symbol'}</div>
                  </div>
                  <div style="text-align:right">
                    <div class="ticker-price">${latest_price:,.2f}</div>
                    <span class="{badge_class}">{badge_text}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="mini-card"><div class="mini-label">Top Performer</div><div class="mini-value">{leaderboard_df.iloc[0]["Ticker"]}</div><div class="mini-note">Return over period: {leaderboard_df.iloc[0]["Return"]:.2%}</div></div>', unsafe_allow_html=True)
        st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mini-card"><div class="mini-label">Weakest Performer</div><div class="mini-value">{leaderboard_df.iloc[-1]["Ticker"]}</div><div class="mini-note">Return over period: {leaderboard_df.iloc[-1]["Return"]:.2%}</div></div>', unsafe_allow_html=True)

with tab2:
    perf_symbols = st.multiselect(
        "Performance symbols",
        options=display_columns,
        default=display_columns,
        help="Choose which symbols appear in the performance charts.",
        key="performance_symbols",
    )
    perf_left, perf_right = st.columns(2, gap="large")
    with perf_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        norm_df = prices[perf_symbols].apply(lambda col: col / col.dropna().iloc[0] if col.dropna().size else col) if perf_symbols else pd.DataFrame()
        fig_norm = go.Figure()
        for i, ticker in enumerate(norm_df.columns):
            fig_norm.add_trace(go.Scatter(x=norm_df.index, y=norm_df[ticker], mode="lines", name=ticker, line=dict(color=chart_palette[i % len(chart_palette)], width=2.4)))
        fig_norm.update_layout(title="Relative Performance", xaxis_title="Date", yaxis_title="Growth Multiple")
        fig_norm.for_each_trace(lambda trace: trace.update(name=display_name_map.get(trace.name, trace.name)))
        style_figure(fig_norm, 430)
        st.plotly_chart(fig_norm, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with perf_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_wealth = go.Figure()
        for i, col in enumerate(wealth_df.columns):
            width = 2.6 if col == "Equal-Weight Portfolio" else 2.0
            color = "#f97316" if col == "Equal-Weight Portfolio" else chart_palette[i % len(chart_palette)]
            fig_wealth.add_trace(go.Scatter(x=wealth_df.index, y=wealth_df[col], mode="lines", name=col, line=dict(color=color, width=width)))
        fig_wealth.update_layout(title="Growth of $10,000", xaxis_title="Date", yaxis_title="Portfolio Value ($)")
        fig_wealth.for_each_trace(lambda trace: trace.update(name=display_name_map.get(trace.name, trace.name)))
        style_figure(fig_wealth, 430)
        st.plotly_chart(fig_wealth, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Performance Tables</div>', unsafe_allow_html=True)
    st.dataframe(
        leaderboard_df.style.format({"Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.2%}", "Beta": "{:.2f}"}),
        width="stretch",
    )
    st.dataframe(
        summary_stats(stats_returns).style.format({"Annualized Mean Return": "{:.2%}", "Annualized Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}", "Skewness": "{:.3f}", "Kurtosis": "{:.3f}", "Min Daily Return": "{:.2%}", "Max Daily Return": "{:.2%}"}),
        width="stretch",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    risk_stocks = st.multiselect(
        "Risk symbols",
        options=valid_tickers,
        default=valid_tickers,
        help="Choose which stocks to include in volatility analysis.",
        key="risk_symbols",
    )
    vol_window = st.selectbox("Volatility window", [30, 60, 90], index=0, key="risk_window")
    selected_stock = st.selectbox("Distribution focus", options=risk_stocks if risk_stocks else valid_tickers, key="risk_focus")
    risk_left, risk_right = st.columns([1.6, 1], gap="large")
    with risk_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        risk_df = prices[risk_stocks + ([benchmark_col] if benchmark_col and show_benchmark and risk_stocks else [])].pct_change().dropna() if risk_stocks else pd.DataFrame()
        rolling_vol = risk_df.rolling(vol_window).std() * np.sqrt(252) if not risk_df.empty else pd.DataFrame()
        fig_vol = go.Figure()
        for i, col in enumerate(rolling_vol.columns):
            fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[col], mode="lines", name=col, line=dict(color=chart_palette[i % len(chart_palette)], width=2.2)))
        fig_vol.update_layout(title=f"Rolling Volatility ({vol_window}D)", xaxis_title="Date", yaxis_title="Volatility")
        fig_vol.for_each_trace(lambda trace: trace.update(name=display_name_map.get(trace.name, trace.name)))
        style_figure(fig_vol, 460)
        st.plotly_chart(fig_vol, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with risk_right:
        st.markdown('<div class="side-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-kicker">Risk Lens</div><div class="section-title">Return Distribution</div>', unsafe_allow_html=True)
        r = prices[selected_stock].pct_change().dropna()
        jb_stat, jb_p = jarque_bera(r)
        side_metrics = st.columns(3)
        side_metrics[0].metric("JB", f"{jb_stat:.2f}")
        side_metrics[1].metric("p", f"{jb_p:.3f}")
        side_metrics[2].metric("Skew", f"{skew(r):.2f}")
        if jb_p < 0.05:
            st.markdown('<div class="help-note">This return series is not especially bell-curve shaped, so extreme moves matter more than a basic normal assumption suggests.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="help-note">This return series stays relatively close to a bell-curve pattern over the selected range.</div>', unsafe_allow_html=True)
        mu, sigma = norm.fit(r)
        x_vals = np.linspace(r.min(), r.max(), 300)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=r, histnorm="probability density", name="Returns", opacity=.72, marker_color="#22c55e"))
        fig_hist.add_trace(go.Scatter(x=x_vals, y=norm.pdf(x_vals, mu, sigma), mode="lines", name="Normal Fit", line=dict(color="#60a5fa", width=2.4)))
        fig_hist.update_layout(title=f"{selected_stock} Daily Return Shape", xaxis_title="Daily Return", yaxis_title="Density", barmode="overlay")
        style_figure(fig_hist, 460)
        st.plotly_chart(fig_hist, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    stock_x = st.selectbox("Scatter X", display_columns, index=0, key="rel_scatter_x")
    stock_y = st.selectbox("Scatter Y", display_columns, index=1, key="rel_scatter_y")
    corr_stock_1 = st.selectbox("Rolling corr first", display_columns, index=0, key="corr1")
    corr_stock_2 = st.selectbox("Rolling corr second", display_columns, index=1, key="corr2")
    corr_window = st.selectbox("Rolling corr window", [30, 60, 90], index=0, key="corr_window")
    stock_a = st.selectbox("Portfolio stock A", valid_tickers, key="port_a")
    stock_b = st.selectbox("Portfolio stock B", valid_tickers, index=1 if len(valid_tickers) > 1 else 0, key="port_b")
    weight = st.slider("Weight on stock A (%)", 0, 100, 50, key="portfolio_weight") / 100
    rel_top_left, rel_top_right = st.columns(2, gap="large")
    with rel_top_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale="Tealgrn", zmin=-1, zmax=1, colorbar=dict(title="Correlation"), text=corr_matrix.round(2).values, texttemplate="%{text}", textfont={"size": 13}))
        fig_corr.update_layout(title="Correlation Map")
        style_figure(fig_corr, 430)
        st.plotly_chart(fig_corr, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    with rel_top_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if stock_x != stock_y:
            scatter_df = prices[[stock_x, stock_y]].pct_change().dropna()
            fit = np.polyfit(scatter_df[stock_x], scatter_df[stock_y], 1)
            trend_x = np.linspace(scatter_df[stock_x].min(), scatter_df[stock_x].max(), 100)
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(x=scatter_df[stock_x], y=scatter_df[stock_y], mode="markers", name="Daily Returns", marker=dict(color="#22c55e", size=8, opacity=.6)))
            fig_scatter.add_trace(go.Scatter(x=trend_x, y=fit[0] * trend_x + fit[1], mode="lines", name="Trend Line", line=dict(color="#60a5fa", width=2.3)))
            fig_scatter.update_layout(title=f"{stock_x} vs {stock_y}", xaxis_title=stock_x, yaxis_title=stock_y)
            style_figure(fig_scatter, 430)
            st.plotly_chart(fig_scatter, width="stretch")
        else:
            st.warning("Pick two different names for the relationship scatter.")
        st.markdown("</div>", unsafe_allow_html=True)

    rel_bottom_left, rel_bottom_right = st.columns(2, gap="large")
    with rel_bottom_left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if corr_stock_1 != corr_stock_2:
            pair_returns = prices[[corr_stock_1, corr_stock_2]].pct_change().dropna()
            rolling_corr = pair_returns[corr_stock_1].rolling(corr_window).corr(pair_returns[corr_stock_2])
            fig_rollcorr = go.Figure()
            fig_rollcorr.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, mode="lines", name="Rolling Correlation", line=dict(color="#a855f7", width=2.4)))
            fig_rollcorr.update_layout(title=f"Rolling Correlation: {corr_stock_1} vs {corr_stock_2}", xaxis_title="Date", yaxis_title="Correlation")
            style_figure(fig_rollcorr, 430)
            st.plotly_chart(fig_rollcorr, width="stretch")
        else:
            st.warning("Pick two different names for rolling correlation.")
        st.markdown("</div>", unsafe_allow_html=True)
    with rel_bottom_right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        if stock_a != stock_b:
            pair = prices[[stock_a, stock_b]].pct_change().dropna()
            mean_returns = pair.mean() * 252
            cov_matrix = pair.cov() * 252
            portfolio_return = weight * mean_returns[stock_a] + (1 - weight) * mean_returns[stock_b]
            portfolio_vol = np.sqrt((weight ** 2) * cov_matrix.loc[stock_a, stock_a] + ((1 - weight) ** 2) * cov_matrix.loc[stock_b, stock_b] + 2 * weight * (1 - weight) * cov_matrix.loc[stock_a, stock_b])
            pcols = st.columns(2)
            pcols[0].metric("Return", f"{portfolio_return:.2%}")
            pcols[1].metric("Volatility", f"{portfolio_vol:.2%}")
            weights = np.linspace(0, 1, 100)
            vols, rets = [], []
            for w in weights:
                rets.append(w * mean_returns[stock_a] + (1 - w) * mean_returns[stock_b])
                vols.append(np.sqrt((w ** 2) * cov_matrix.loc[stock_a, stock_a] + ((1 - w) ** 2) * cov_matrix.loc[stock_b, stock_b] + 2 * w * (1 - w) * cov_matrix.loc[stock_a, stock_b]))
            fig_port = go.Figure()
            fig_port.add_trace(go.Scatter(x=vols, y=rets, mode="lines", name="Portfolio Curve", line=dict(color="#22c55e", width=2.5)))
            fig_port.add_trace(go.Scatter(x=[portfolio_vol], y=[portfolio_return], mode="markers", marker=dict(size=12, color="#f97316"), name="Current Mix"))
            fig_port.update_layout(title="Risk / Return Path", xaxis_title="Annualized Volatility", yaxis_title="Annualized Return")
            style_figure(fig_port, 430)
            st.plotly_chart(fig_port, width="stretch")
        else:
            st.warning("Pick two different names for the portfolio mix.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-kicker">Reference Deck</div><div class="section-title">Method and Notes</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="help-note">
        Returns use simple daily percentage changes. Annualized return is mean daily return multiplied by 252 trading days.
        Annualized volatility is daily standard deviation multiplied by sqrt(252). Beta is estimated versus the S&amp;P 500 benchmark.
        Drawdown tracks the percentage drop from each running peak.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="help-note" style="margin-top:1rem">
        This version keeps the cleaner modern styling from the redesign, but reorganizes the experience back into focused tabs so each area is easier to scan.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


