# PulseQuant Investor Lab

A professional Streamlit starter app for stock analysis, market news, trend reading, and volatility awareness.

## Features

- Watchlist dashboard with price, return, volatility, drawdown, beta, Sharpe, RSI, and trend regime.
- Moving-average trend charts with 20-day, 50-day, and 200-day averages.
- Relative performance, risk-return, rolling volatility, drawdown, correlation, and return distribution charts.
- Yahoo Finance RSS headlines for the current watchlist.
- Learning panels that explain trend and risk terms in plain language.

## Run Locally

```powershell
uv run streamlit run app.py
```

Or, if you are using the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Streamlit Community Cloud

Use these settings when deploying from GitHub:

- Repository: `gabesosa34-cpu/stock_app_project`
- Branch: `main`
- Main file path: `app.py`

The app uses `requirements.txt`, so Streamlit Community Cloud can install dependencies without extra setup.
