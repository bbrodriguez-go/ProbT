# probt — Engine

Python XAUUSD Probability Engine + FastAPI gateway.

## Quick start

```bash
cd engine
uv venv .venv
uv pip install -r requirements.txt
cp .env.example .env   # add NEWSAPI_KEY + GEMINI_API_KEY (optional)

# Run the full pipeline (one time)
uv run python data_collector.py      # download 10y data
uv run python feature_engineer.py    # build feature matrix
uv run python feature_diagnostics.py # check multicollinearity
uv run python labeler.py             # triple-barrier labels
uv run python model_trainer.py       # train + calibrate (Brier Score)
uv run python live_engine.py         # real-time readout

# Start the API gateway
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Architecture

| Layer | File | Purpose |
|---|---|---|
| 0 | `indicators.py` | RSI, EMA, ATR, MACD, trend_bias, MTF |
| 0 | `smc.py` | Swings, BOS structure, order blocks, FVGs |
| 0 | `support_resistance.py` | Zone clustering from swings |
| 0 | `news_sentiment.py` | NewsAPI headlines + Gemini sentiment |
| 1 | `data_collector.py` | yfinance → CSV (daily + hourly) |
| 2a | `feature_engineer.py` | 17 stationary features |
| 2b | `volume_profile.py` | CME volume POC + value area |
| 2c | `feature_diagnostics.py` | VIF + correlation check |
| 3 | `labeler.py` | Triple-Barrier (TP=2·ATR, SL=1·ATR, 10d) |
| 4 | `model_trainer.py` | LogReg L1 + Platt + TimeSeriesSplit |
| 5 | `live_engine.py` | Real-time Tier A/B/News/Sizing |
| 6 | `trade_logger.py` | Forward-test log |
| — | `api/main.py` | FastAPI REST + WebSocket |

## API endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | Engine status |
| GET | `/api/reading` | Live Tier A/B/News/Sizing |
| GET | `/api/backtest` | Equity curve + metrics |
| GET | `/api/kpis` | 10 hero KPIs |
| GET | `/api/probability-dist` | Probability histogram |
| GET | `/api/trades` | Trade history |
| GET | `/api/models` | ML model cards |
| GET | `/api/market` | Multi-asset prices |
| GET | `/api/heatmap` | Feature correlation matrix |
| GET | `/api/confidence` | Gauge metrics |
| GET | `/api/insights` | AI insight cards |
| GET | `/api/features` | Feature importance |
| WS | `/ws/stream` | Real-time reading push |

## Current model

- **Algorithm:** LogisticRegression L1 (C=0.1) + Platt calibration
- **Features:** 17 (dropped `bb_pct_b_1d` for |corr|=0.89 with `rsi_1d`)
- **Data:** 592 rows (2024-02 → 2026-06, limited by hourly history)
- **Brier Score (CV):** 0.266 — honest, no statistical edge yet (needs more data)
- **Label balance:** 48.6% / 51.4%

## Notes

- Gold ticker fallback: `XAUUSD=X` → `GC=F` (XAUUSD=X delisted on yfinance)
- Sentiment: Gemini via ADC (Cloud Run) or API key (local). Skips gracefully if unavailable.
- NewsAPI: optional, skips gracefully without key.
