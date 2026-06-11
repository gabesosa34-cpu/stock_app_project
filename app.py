from __future__ import annotations

import re
from datetime import date, timedelta
from email.utils import parsedate_to_datetime
from html import escape
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots


APP_NAME = "PulseQuant Investor Lab"
DEFAULT_TICKERS = "SPY, QQQ, AAPL, MSFT, NVDA"
PALETTE = ["#0f766e", "#2563eb", "#f97316", "#7c3aed", "#dc2626", "#0891b2", "#475569"]


st.set_page_config(page_title=APP_NAME, layout="wide")

st.markdown(
    """
    <style>
    :root {
        --bg: #f5f7fb;
        --surface: #ffffff;
        --surface-2: #f8fafc;
        --border: #dbe3ef;
        --text: #172033;
        --muted: #667085;
        --brand: #0f766e;
        --blue: #2563eb;
        --amber: #f97316;
        --red: #dc2626;
        --green: #0f766e;
        --navy: #111827;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 247, 251, 0.86);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(219, 227, 239, 0.78);
    }

    [data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid #1f2937;
    }

    [data-testid="stSidebar"] * {
        color: #f9fafb !important;
    }

    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] .stDateInput input {
        color: #111827 !important;
    }

    .block-container {
        padding-top: 1.35rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }

    h1, h2, h3, p, label, span, div {
        letter-spacing: 0 !important;
    }

    .app-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 1rem;
        padding: 1rem 0 0.85rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1rem;
    }

    .app-title {
        font-size: 1.95rem;
        line-height: 1.12;
        font-weight: 800;
        color: var(--text);
        margin: 0;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-top: 0.35rem;
        max-width: 780px;
    }

    .source-pill {
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--muted);
        padding: 0.45rem 0.65rem;
        border-radius: 8px;
        white-space: nowrap;
        font-size: 0.82rem;
    }

    .metric-card, .panel, .news-card, .lesson-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
    }

    .metric-card {
        padding: 0.9rem 0.95rem;
        min-height: 116px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        font-weight: 700;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.55rem;
        line-height: 1.15;
        font-weight: 800;
        margin-top: 0.4rem;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: var(--muted);
        margin-top: 0.35rem;
        font-size: 0.86rem;
    }

    .delta-up { color: var(--green); font-weight: 750; }
    .delta-down { color: var(--red); font-weight: 750; }
    .delta-flat { color: var(--muted); font-weight: 750; }

    .panel {
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .panel-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0.2rem;
    }

    .panel-copy {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
    }

    .signal-list {
        display: grid;
        gap: 0.5rem;
        margin-top: 0.75rem;
    }

    .signal-row {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 0.75rem;
        border-top: 1px solid #eef2f7;
        padding-top: 0.55rem;
        font-size: 0.9rem;
    }

    .signal-name {
        color: var(--muted);
    }

    .signal-value {
        color: var(--text);
        font-weight: 750;
        text-align: right;
    }

    .badge {
        display: inline-block;
        border-radius: 8px;
        padding: 0.25rem 0.45rem;
        font-size: 0.78rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .badge-up {
        background: #ecfdf5;
        color: #047857;
        border-color: #a7f3d0;
    }

    .badge-down {
        background: #fef2f2;
        color: #b91c1c;
        border-color: #fecaca;
    }

    .badge-mixed {
        background: #fff7ed;
        color: #c2410c;
        border-color: #fed7aa;
    }

    .badge-neutral {
        background: #eff6ff;
        color: #1d4ed8;
        border-color: #bfdbfe;
    }

    .news-card {
        padding: 0.85rem 0.9rem;
        margin-bottom: 0.75rem;
    }

    .news-meta {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 0.25rem;
    }

    .news-title {
        color: var(--text);
        font-weight: 760;
        font-size: 0.98rem;
        line-height: 1.35;
    }

    .news-title a {
        color: var(--text);
        text-decoration: none;
    }

    .news-title a:hover {
        color: var(--blue);
        text-decoration: underline;
    }

    .lesson-card {
        padding: 1rem;
        height: 100%;
    }

    .lesson-card ul {
        padding-left: 1.1rem;
        margin-bottom: 0;
    }

    .lesson-card li {
        margin-bottom: 0.45rem;
        color: #344054;
    }

    .fine-print {
        color: var(--muted);
        font-size: 0.8rem;
        border-top: 1px solid var(--border);
        padding-top: 0.8rem;
        margin-top: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        border: 1px solid transparent;
        color: #344054;
        font-weight: 750;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff;
        border-color: var(--border);
        border-bottom-color: #ffffff;
        color: var(--text) !important;
    }

    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid #0f766e;
        background: #0f766e;
        color: #ffffff;
        font-weight: 760;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_tickers(raw: str) -> list[str]:
    symbols: list[str] = []
    for item in re.split(r"[\s,;]+", raw.upper().replace("$", "")):
        symbol = item.strip()
        if not symbol:
            continue
        if re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", symbol) and symbol not in symbols:
            symbols.append(symbol)
    return symbols[:10]


def format_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.2f}"


def format_pct(value: float | int | None, digits: int = 2, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    sign = "+" if signed else ""
    return f"{float(value):{sign}.{digits}%}"


def format_number(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def delta_class(value: float | int | None) -> str:
    if value is None or pd.isna(value) or abs(float(value)) < 1e-12:
        return "delta-flat"
    return "delta-up" if float(value) > 0 else "delta-down"


def metric_card(label: str, value: str, note: str = "", delta: float | None = None) -> None:
    note_class = delta_class(delta) if delta is not None else "metric-note"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="{note_class if delta is not None else 'metric-note'}">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_figure(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        margin=dict(l=35, r=25, t=58, b=34),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        font=dict(color="#172033", size=12),
        title=dict(font=dict(size=17, color="#172033"), x=0.0, xanchor="left"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, color="#475467")
    fig.update_yaxes(gridcolor="#eef2f7", zerolinecolor="#eef2f7", color="#475467")
    return fig


@st.cache_data(show_spinner="Fetching market data...", ttl=300)
def load_market_data(symbols: tuple[str, ...], start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    prices: dict[str, pd.Series] = {}
    volumes: dict[str, pd.Series] = {}
    errors: dict[str, str] = {}
    end_exclusive = end + timedelta(days=1)

    for symbol in symbols:
        try:
            frame = yf.download(
                symbol,
                start=start,
                end=end_exclusive,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            errors[symbol] = str(exc)
            continue

        if frame.empty:
            errors[symbol] = "No price history returned."
            continue

        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)

        if "Close" not in frame:
            errors[symbol] = "Close column missing."
            continue

        close = frame["Close"].dropna()
        if close.empty:
            errors[symbol] = "Close column empty."
            continue

        prices[symbol] = close.rename(symbol)
        if "Volume" in frame:
            volumes[symbol] = frame["Volume"].reindex(close.index).rename(symbol)

    price_df = pd.DataFrame(prices).sort_index()
    volume_df = pd.DataFrame(volumes).sort_index()
    return price_df, volume_df, errors


@st.cache_data(show_spinner=False, ttl=900)
def fetch_news(symbols: tuple[str, ...], limit: int = 18) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    seen: set[str] = set()

    for symbol in symbols:
        query = urlencode({"s": symbol, "region": "US", "lang": "en-US"})
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?{query}"
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=8) as response:
                root = ET.fromstring(response.read())
        except Exception:
            continue

        for node in root.findall(".//item"):
            title = (node.findtext("title") or "").strip()
            link = (node.findtext("link") or "").strip()
            published_raw = (node.findtext("pubDate") or "").strip()
            key = link or title
            if not title or key in seen:
                continue

            published = published_raw
            if published_raw:
                try:
                    published = parsedate_to_datetime(published_raw).strftime("%b %-d, %Y %I:%M %p")
                except Exception:
                    published = published_raw

            seen.add(key)
            items.append(
                {
                    "symbol": symbol,
                    "title": title,
                    "link": link,
                    "published": published,
                }
            )
            if len(items) >= limit:
                return items

    return items[:limit]


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def compute_drawdown(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    return clean / clean.cummax() - 1


def calc_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def beta_vs_benchmark(asset_returns: pd.Series, benchmark_returns: pd.Series | None) -> float:
    if benchmark_returns is None or benchmark_returns.empty:
        return np.nan
    joined = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if len(joined) < 20:
        return np.nan
    benchmark_var = joined.iloc[:, 1].var()
    if benchmark_var == 0 or pd.isna(benchmark_var):
        return np.nan
    return float(joined.iloc[:, 0].cov(joined.iloc[:, 1]) / benchmark_var)


def latest_value(series: pd.Series) -> float:
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else np.nan


def trend_profile(series: pd.Series) -> dict[str, object]:
    clean = series.dropna()
    if len(clean) < 35:
        return {
            "status": "Building data",
            "badge": "neutral",
            "summary": "A longer history is needed for moving-average trend context.",
            "ma20": np.nan,
            "ma50": np.nan,
            "ma200": np.nan,
            "rsi": np.nan,
            "distance_high": np.nan,
            "distance_low": np.nan,
            "slope_50": np.nan,
        }

    price = float(clean.iloc[-1])
    ma20 = latest_value(clean.rolling(20).mean())
    ma50_series = clean.rolling(50).mean()
    ma50 = latest_value(ma50_series)
    ma200 = latest_value(clean.rolling(200).mean())
    rsi_value = latest_value(calc_rsi(clean))
    trailing_year = clean.tail(min(252, len(clean)))
    high_52 = float(trailing_year.max())
    low_52 = float(trailing_year.min())
    distance_high = price / high_52 - 1 if high_52 else np.nan
    distance_low = price / low_52 - 1 if low_52 else np.nan
    slope_50 = np.nan

    ma50_clean = ma50_series.dropna()
    if len(ma50_clean) > 20 and ma50_clean.iloc[-21] != 0:
        slope_50 = float(ma50_clean.iloc[-1] / ma50_clean.iloc[-21] - 1)

    above_20 = not pd.isna(ma20) and price > ma20
    above_50 = not pd.isna(ma50) and price > ma50
    above_200 = not pd.isna(ma200) and price > ma200
    ma_stack = not pd.isna(ma50) and not pd.isna(ma200) and ma50 > ma200
    rising_50 = not pd.isna(slope_50) and slope_50 > 0

    if above_50 and above_200 and ma_stack and rising_50:
        status = "Uptrend"
        badge = "up"
        summary = "Price is above key averages, and the medium-term average is rising."
    elif above_200 and not above_50:
        status = "Pullback"
        badge = "mixed"
        summary = "Long-term trend is intact, but price is below the 50-day average."
    elif not above_200 and not rising_50:
        status = "Downtrend"
        badge = "down"
        summary = "Price is below the 200-day average and medium-term momentum is weak."
    elif not above_20 and above_50:
        status = "Cooling"
        badge = "mixed"
        summary = "Short-term momentum has cooled while the broader setup is still holding."
    else:
        status = "Mixed"
        badge = "neutral"
        summary = "Signals are not aligned enough to call a clear regime."

    if not pd.isna(rsi_value) and rsi_value >= 72 and status == "Uptrend":
        status = "Extended uptrend"
        badge = "mixed"
        summary = "Trend is positive, but RSI is stretched and pullback risk is elevated."

    return {
        "status": status,
        "badge": badge,
        "summary": summary,
        "ma20": ma20,
        "ma50": ma50,
        "ma200": ma200,
        "rsi": rsi_value,
        "distance_high": distance_high,
        "distance_low": distance_low,
        "slope_50": slope_50,
    }


def build_metrics(
    prices: pd.DataFrame,
    symbols: list[str],
    benchmark_symbol: str,
    risk_free_rate: float,
) -> pd.DataFrame:
    returns = daily_returns(prices)
    benchmark_returns = returns[benchmark_symbol].dropna() if benchmark_symbol in returns else None
    rows: list[dict[str, object]] = []

    for symbol in symbols:
        if symbol not in prices:
            continue
        series = prices[symbol].dropna()
        if len(series) < 2:
            continue

        ret = series.pct_change(fill_method=None).dropna()
        total_return = float(series.iloc[-1] / series.iloc[0] - 1)
        ann_return = np.nan
        if len(ret) > 0 and total_return > -1:
            ann_return = float((1 + total_return) ** (252 / len(ret)) - 1)
        ann_vol = float(ret.std() * np.sqrt(252)) if len(ret) > 1 else np.nan
        sharpe = float((ann_return - risk_free_rate) / ann_vol) if ann_vol and not pd.isna(ann_vol) else np.nan
        downside = ret[ret < 0].std() * np.sqrt(252)
        sortino = float((ann_return - risk_free_rate) / downside) if downside and not pd.isna(downside) else np.nan
        max_dd = float(compute_drawdown(series).min())
        one_day = float(series.iloc[-1] / series.iloc[-2] - 1)
        var_95 = float(ret.quantile(0.05)) if not ret.empty else np.nan
        profile = trend_profile(series)

        rows.append(
            {
                "Symbol": symbol,
                "Latest Price": float(series.iloc[-1]),
                "1D Move": one_day,
                "Total Return": total_return,
                "Annual Return": ann_return,
                "Volatility": ann_vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Max Drawdown": max_dd,
                "Beta": beta_vs_benchmark(ret, benchmark_returns),
                "Daily VaR 95": var_95,
                "RSI": profile["rsi"],
                "Trend": profile["status"],
                "Trend Badge": profile["badge"],
            }
        )

    return pd.DataFrame(rows).set_index("Symbol") if rows else pd.DataFrame()


def display_metric_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics

    table = metrics[
        [
            "Latest Price",
            "1D Move",
            "Total Return",
            "Annual Return",
            "Volatility",
            "Sharpe",
            "Max Drawdown",
            "Beta",
            "RSI",
            "Trend",
        ]
    ].copy()
    table["Latest Price"] = table["Latest Price"].map(format_money)
    for col in ["1D Move", "Total Return", "Annual Return", "Volatility", "Max Drawdown"]:
        table[col] = table[col].map(lambda value: format_pct(value, signed=col == "1D Move"))
    for col in ["Sharpe", "Beta", "RSI"]:
        table[col] = table[col].map(lambda value: format_number(value, 2))
    return table


def normalized_prices(prices: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    base = prices[symbols].dropna(how="all").ffill()
    normalized = pd.DataFrame(index=base.index)
    for symbol in symbols:
        clean = base[symbol].dropna()
        if clean.empty:
            continue
        normalized[symbol] = base[symbol] / clean.iloc[0] * 100
    return normalized


def price_volume_chart(symbol: str, prices: pd.DataFrame, volumes: pd.DataFrame) -> go.Figure:
    series = prices[symbol].dropna()
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=(f"{symbol} price trend", "Volume"),
    )
    fig.add_trace(
        go.Scatter(x=series.index, y=series, mode="lines", name="Close", line=dict(color="#0f766e", width=2.6)),
        row=1,
        col=1,
    )
    for window, color, dash in [(20, "#2563eb", "dot"), (50, "#f97316", "dash"), (200, "#475569", "solid")]:
        average = series.rolling(window).mean().dropna()
        if not average.empty:
            fig.add_trace(
                go.Scatter(
                    x=average.index,
                    y=average,
                    mode="lines",
                    name=f"{window}D MA",
                    line=dict(color=color, width=1.8, dash=dash),
                ),
                row=1,
                col=1,
            )

    if symbol in volumes and not volumes[symbol].dropna().empty:
        volume = volumes[symbol].reindex(series.index)
        fig.add_trace(
            go.Bar(x=volume.index, y=volume, name="Volume", marker_color="#cbd5e1", opacity=0.75),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Shares", row=2, col=1)
    return style_figure(fig, 590)


def relative_performance_chart(prices: pd.DataFrame, symbols: list[str]) -> go.Figure:
    norm = normalized_prices(prices, symbols)
    fig = go.Figure()
    for index, symbol in enumerate(norm.columns):
        fig.add_trace(
            go.Scatter(
                x=norm.index,
                y=norm[symbol],
                mode="lines",
                name=symbol,
                line=dict(color=PALETTE[index % len(PALETTE)], width=2.4),
            )
        )
    fig.update_layout(title="Relative performance", yaxis_title="Start = 100")
    return style_figure(fig, 440)


def rolling_volatility_chart(prices: pd.DataFrame, symbols: list[str], window: int) -> go.Figure:
    returns = daily_returns(prices[symbols]).dropna(how="all")
    rolling = returns.rolling(window).std() * np.sqrt(252)
    fig = go.Figure()
    for index, symbol in enumerate(rolling.columns):
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling[symbol],
                mode="lines",
                name=symbol,
                line=dict(color=PALETTE[index % len(PALETTE)], width=2.3),
            )
        )
    fig.update_layout(title=f"Rolling realized volatility ({window} trading days)", yaxis_title="Annualized volatility")
    fig.update_yaxes(tickformat=".0%")
    return style_figure(fig, 430)


def drawdown_chart(prices: pd.DataFrame, symbols: list[str]) -> go.Figure:
    fig = go.Figure()
    for index, symbol in enumerate(symbols):
        dd = compute_drawdown(prices[symbol])
        if dd.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd,
                mode="lines",
                name=symbol,
                line=dict(color=PALETTE[index % len(PALETTE)], width=2.2),
                fill="tozeroy",
            )
        )
    fig.update_layout(title="Drawdown from prior highs", yaxis_title="Drawdown")
    fig.update_yaxes(tickformat=".0%")
    return style_figure(fig, 430)


def correlation_heatmap(prices: pd.DataFrame, symbols: list[str]) -> go.Figure:
    corr = daily_returns(prices[symbols]).dropna(how="all").corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0.0, "#b91c1c"],
                [0.5, "#f8fafc"],
                [1.0, "#0f766e"],
            ],
            text=corr.round(2).values,
            texttemplate="%{text}",
            colorbar=dict(title="Corr."),
        )
    )
    fig.update_layout(title="Return correlation map")
    return style_figure(fig, 430)


def risk_return_chart(metrics: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if metrics.empty:
        return style_figure(fig, 420)

    fig.add_trace(
        go.Scatter(
            x=metrics["Volatility"],
            y=metrics["Annual Return"],
            mode="markers+text",
            text=metrics.index,
            textposition="top center",
            marker=dict(
                size=np.clip((metrics["Sharpe"].fillna(0) + 1.5) * 12, 10, 34),
                color=metrics["Sharpe"],
                colorscale="Tealgrn",
                colorbar=dict(title="Sharpe"),
                line=dict(color="#ffffff", width=1),
            ),
            name="Symbols",
        )
    )
    fig.update_layout(title="Risk and return map", xaxis_title="Annualized volatility", yaxis_title="Annualized return")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return style_figure(fig, 430)


def trend_signal_panel(symbol: str, profile: dict[str, object]) -> None:
    badge = str(profile["badge"])
    badge_class = {
        "up": "badge-up",
        "down": "badge-down",
        "mixed": "badge-mixed",
        "neutral": "badge-neutral",
    }.get(badge, "badge-neutral")
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{escape(symbol)} trend read</div>
            <div class="panel-copy">{escape(str(profile["summary"]))}</div>
            <span class="badge {badge_class}">{escape(str(profile["status"]))}</span>
            <div class="signal-list">
                <div class="signal-row"><div class="signal-name">20-day average</div><div class="signal-value">{escape(format_money(profile["ma20"]))}</div></div>
                <div class="signal-row"><div class="signal-name">50-day average</div><div class="signal-value">{escape(format_money(profile["ma50"]))}</div></div>
                <div class="signal-row"><div class="signal-name">200-day average</div><div class="signal-value">{escape(format_money(profile["ma200"]))}</div></div>
                <div class="signal-row"><div class="signal-name">RSI 14</div><div class="signal-value">{escape(format_number(profile["rsi"]))}</div></div>
                <div class="signal-row"><div class="signal-name">From 52-week high</div><div class="signal-value">{escape(format_pct(profile["distance_high"]))}</div></div>
                <div class="signal-row"><div class="signal-name">Above 52-week low</div><div class="signal-value">{escape(format_pct(profile["distance_low"]))}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.sidebar.header("Control Center")
raw_tickers = st.sidebar.text_input("Watchlist", value=DEFAULT_TICKERS, help="Use commas or spaces between tickers.")
tickers = parse_tickers(raw_tickers)
if not tickers:
    st.info("Add at least one ticker in the sidebar to start the analysis.")
    st.stop()

benchmark_options = ["^GSPC", "SPY", "QQQ", "DIA", "IWM"]
benchmark = st.sidebar.selectbox("Benchmark", benchmark_options, index=0)
range_label = st.sidebar.selectbox("Date range", ["1Y", "2Y", "5Y", "10Y", "Custom"], index=1)

today = date.today()
if range_label == "Custom":
    start_date = st.sidebar.date_input("Start date", value=today - timedelta(days=365 * 2), max_value=today - timedelta(days=30))
    end_date = st.sidebar.date_input("End date", value=today, max_value=today)
else:
    years = int(range_label.replace("Y", ""))
    start_date = today - timedelta(days=365 * years)
    end_date = today

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

risk_free_rate = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=20.0, value=4.50, step=0.25) / 100
scenario_amount = st.sidebar.number_input("Scenario amount ($)", min_value=100, max_value=1_000_000, value=10_000, step=500)
rolling_window = st.sidebar.selectbox("Volatility window", [20, 30, 60, 90], index=1)

symbols_to_download = tuple(dict.fromkeys(tickers + [benchmark]))
prices, volumes, errors = load_market_data(symbols_to_download, start_date, end_date)

valid_tickers = [symbol for symbol in tickers if symbol in prices and not prices[symbol].dropna().empty]
if not valid_tickers:
    st.error("No usable market data was returned for the watchlist.")
    if errors:
        st.caption("; ".join(f"{symbol}: {message}" for symbol, message in errors.items()))
    st.stop()

if benchmark not in prices or prices[benchmark].dropna().empty:
    benchmark = valid_tickers[0]

focus_symbol = st.sidebar.selectbox("Focus symbol", valid_tickers, index=0)
chart_symbols = st.sidebar.multiselect("Chart symbols", valid_tickers, default=valid_tickers[: min(5, len(valid_tickers))])
if not chart_symbols:
    chart_symbols = [focus_symbol]

metrics = build_metrics(prices, valid_tickers, benchmark, risk_free_rate)
focus_profile = trend_profile(prices[focus_symbol])
focus_metrics = metrics.loc[focus_symbol]

st.markdown(
    f"""
    <div class="app-header">
        <div>
            <h1 class="app-title">{APP_NAME}</h1>
            <div class="app-subtitle">A clean starter workspace for stock analysis, market news, trend reading, and volatility awareness.</div>
        </div>
        <div class="source-pill">Data: Yahoo Finance | Through {escape(str(end_date))}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if errors:
    missing = [symbol for symbol in tickers if symbol in errors]
    if missing:
        st.warning(f"Some symbols could not be loaded: {', '.join(missing)}")

top_cols = st.columns(5)
with top_cols[0]:
    metric_card("Focus", focus_symbol, str(focus_metrics["Trend"]))
with top_cols[1]:
    metric_card("Latest Price", format_money(focus_metrics["Latest Price"]), f"{format_pct(focus_metrics['1D Move'], signed=True)} today", focus_metrics["1D Move"])
with top_cols[2]:
    metric_card("Total Return", format_pct(focus_metrics["Total Return"]), f"Since {start_date}", focus_metrics["Total Return"])
with top_cols[3]:
    metric_card("Realized Volatility", format_pct(focus_metrics["Volatility"]), f"{rolling_window}D view in Risk tab")
with top_cols[4]:
    scenario_value = scenario_amount * (1 + float(focus_metrics["Total Return"]))
    metric_card("Scenario Value", format_money(scenario_value), f"{format_money(scenario_amount)} invested")

tab_dashboard, tab_trends, tab_risk, tab_news, tab_learn = st.tabs(
    ["Dashboard", "Trends", "Volatility", "News", "Learn"]
)

with tab_dashboard:
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Price, moving averages, and volume</div>', unsafe_allow_html=True)
        st.plotly_chart(price_volume_chart(focus_symbol, prices, volumes), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        trend_signal_panel(focus_symbol, focus_profile)
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Watchlist leaders</div>
                <div class="panel-copy">Ranked by total return over the selected date range.</div>
            """,
            unsafe_allow_html=True,
        )
        leaders = metrics.sort_values("Total Return", ascending=False)
        for symbol, row in leaders.head(5).iterrows():
            badge_class = "badge-up" if row["Total Return"] >= 0 else "badge-down"
            st.markdown(
                f"""
                <div class="signal-row">
                    <div class="signal-name">{escape(symbol)}</div>
                    <div class="signal-value"><span class="badge {badge_class}">{escape(format_pct(row["Total Return"], signed=True))}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Watchlist snapshot</div>', unsafe_allow_html=True)
    st.dataframe(display_metric_table(metrics), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_trends:
    trend_left, trend_right = st.columns([1.45, 1], gap="large")
    with trend_left:
        st.markdown('<div class="panel"><div class="panel-title">Relative performance</div><div class="panel-copy">Each line starts at 100 for easy comparison.</div>', unsafe_allow_html=True)
        st.plotly_chart(relative_performance_chart(prices, chart_symbols + ([benchmark] if benchmark not in chart_symbols else [])), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with trend_right:
        st.markdown('<div class="panel"><div class="panel-title">Risk and return map</div><div class="panel-copy">Higher return is up; higher volatility is right.</div>', unsafe_allow_html=True)
        st.plotly_chart(risk_return_chart(metrics), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Trend signal board</div>', unsafe_allow_html=True)
    signal_rows = []
    for symbol in valid_tickers:
        profile = trend_profile(prices[symbol])
        signal_rows.append(
            {
                "Symbol": symbol,
                "Trend": profile["status"],
                "RSI": format_number(profile["rsi"], 2),
                "50D MA": format_money(profile["ma50"]),
                "200D MA": format_money(profile["ma200"]),
                "50D Slope": format_pct(profile["slope_50"]),
                "From 52W High": format_pct(profile["distance_high"]),
            }
        )
    st.dataframe(pd.DataFrame(signal_rows), hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_risk:
    risk_symbols = st.multiselect("Risk symbols", valid_tickers, default=chart_symbols, key="risk_symbols")
    if not risk_symbols:
        risk_symbols = [focus_symbol]

    risk_left, risk_right = st.columns(2, gap="large")
    with risk_left:
        st.markdown('<div class="panel"><div class="panel-title">Rolling volatility</div>', unsafe_allow_html=True)
        st.plotly_chart(rolling_volatility_chart(prices, risk_symbols, rolling_window), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with risk_right:
        st.markdown('<div class="panel"><div class="panel-title">Drawdown history</div>', unsafe_allow_html=True)
        st.plotly_chart(drawdown_chart(prices, risk_symbols), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    corr_left, dist_right = st.columns([1, 1], gap="large")
    with corr_left:
        st.markdown('<div class="panel"><div class="panel-title">Correlation map</div>', unsafe_allow_html=True)
        st.plotly_chart(correlation_heatmap(prices, risk_symbols + ([benchmark] if benchmark not in risk_symbols else [])), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with dist_right:
        returns = prices[focus_symbol].pct_change(fill_method=None).dropna()
        var_95 = focus_metrics["Daily VaR 95"]
        st.markdown('<div class="panel"><div class="panel-title">Daily return distribution</div>', unsafe_allow_html=True)
        dist_cols = st.columns(3)
        with dist_cols[0]:
            metric_card("Daily VaR 95", format_pct(var_95), "Historical 5th percentile", var_95)
        with dist_cols[1]:
            metric_card("Worst Day", format_pct(returns.min()), "Selected range", returns.min())
        with dist_cols[2]:
            metric_card("Best Day", format_pct(returns.max()), "Selected range", returns.max())

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=returns, nbinsx=45, marker_color="#0f766e", opacity=0.76, name="Daily returns"))
        fig_hist.add_vline(x=var_95, line_color="#dc2626", line_width=2, line_dash="dash", annotation_text="VaR 95")
        fig_hist.update_layout(title=f"{focus_symbol} daily returns", xaxis_title="Daily return", yaxis_title="Trading days")
        fig_hist.update_xaxes(tickformat=".0%")
        st.plotly_chart(style_figure(fig_hist, 320), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_news:
    news_symbols = tuple(dict.fromkeys([focus_symbol] + valid_tickers[:5]))
    news_items = fetch_news(news_symbols)
    news_left, news_right = st.columns([1.45, 1], gap="large")
    with news_left:
        st.markdown('<div class="panel"><div class="panel-title">Latest market headlines</div><div class="panel-copy">Headlines are pulled from Yahoo Finance RSS feeds for the current watchlist.</div>', unsafe_allow_html=True)
        if news_items:
            for item in news_items:
                link = escape(item["link"])
                title = escape(item["title"])
                meta = escape(f'{item["symbol"]} | {item["published"] or "Recent"}')
                st.markdown(
                    f"""
                    <div class="news-card">
                        <div class="news-meta">{meta}</div>
                        <div class="news-title"><a href="{link}" target="_blank" rel="noopener noreferrer">{title}</a></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("News is unavailable right now. The rest of the dashboard will continue to work.")
        st.markdown("</div>", unsafe_allow_html=True)

    with news_right:
        st.markdown(
            """
            <div class="lesson-card">
                <div class="panel-title">Headline checklist</div>
                <ul>
                    <li>Separate company-specific news from broad market moves.</li>
                    <li>Check whether the story changes revenue, margins, cash flow, or valuation.</li>
                    <li>Compare the price reaction with volume and prior trend context.</li>
                    <li>Watch follow-through. One strong day does not confirm a durable trend.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_learn:
    learn_cols = st.columns(3, gap="large")
    with learn_cols[0]:
        st.markdown(
            f"""
            <div class="lesson-card">
                <div class="panel-title">Reading {escape(focus_symbol)} right now</div>
                <ul>
                    <li>Trend regime: <strong>{escape(str(focus_profile["status"]))}</strong>.</li>
                    <li>RSI: <strong>{escape(format_number(focus_profile["rsi"]))}</strong>. Around 70 can mean stretched; around 30 can mean washed out.</li>
                    <li>Annualized volatility: <strong>{escape(format_pct(focus_metrics["Volatility"]))}</strong>. This estimates how wide moves have been historically.</li>
                    <li>Max drawdown: <strong>{escape(format_pct(focus_metrics["Max Drawdown"]))}</strong>. This shows the largest peak-to-trough drop in the range.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with learn_cols[1]:
        st.markdown(
            """
            <div class="lesson-card">
                <div class="panel-title">Trend terms</div>
                <ul>
                    <li><strong>Moving average:</strong> a smoothed price line used to filter daily noise.</li>
                    <li><strong>50D vs 200D:</strong> a simple medium-term and long-term trend comparison.</li>
                    <li><strong>RSI:</strong> a momentum gauge. It is useful, but it is not a buy or sell signal by itself.</li>
                    <li><strong>Relative performance:</strong> shows which asset is leading after normalizing the starting point.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with learn_cols[2]:
        st.markdown(
            """
            <div class="lesson-card">
                <div class="panel-title">Risk terms</div>
                <ul>
                    <li><strong>Volatility:</strong> the size of historical price swings, annualized from daily returns.</li>
                    <li><strong>Sharpe:</strong> return per unit of volatility after a risk-free-rate assumption.</li>
                    <li><strong>Beta:</strong> sensitivity to the selected benchmark.</li>
                    <li><strong>VaR 95:</strong> the historical daily loss level exceeded roughly 5% of the time.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Good next additions</div>
            <div class="panel-copy">This starter app is built so you can layer on more serious investor workflow over time.</div>
            <ul>
                <li>Portfolio upload and position-level risk.</li>
                <li>Earnings calendar, analyst estimate changes, and dividend history.</li>
                <li>Options implied volatility and unusual volume screens.</li>
                <li>Saved watchlists, notes, alerts, and a trade journal.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="fine-print">
        Educational tool only. Market data can be delayed, revised, or unavailable. This app does not provide investment advice.
    </div>
    """,
    unsafe_allow_html=True,
)
