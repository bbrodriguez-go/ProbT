# probt

> **probt v2** (motor condicionado a zonas: modelos bounce/break sobre eventos de toque
> de zona SMC, con reporte honesto de resultados) está documentado en
> [README.v2.md](README.v2.md) (inglés) / [README.v2.es.md](README.v2.es.md) (español).
> Este documento describe la plataforma v1, que sigue operativa — ambos motores coexisten.

Plataforma profesional de análisis cuantitativo y trading asistido por IA para **XAU/USD** (oro).

Motor de probabilidad estadística de dos niveles (Tier A régimen diario ML-calibrado + Tier B timing horario SMC) con dashboard enterprise para visualización en tiempo real.

---

## URLs en producción

| Servicio | URL |
|---|---|
| **Dashboard** | https://probt-web-354072478643.us-central1.run.app |
| **API Engine** | https://probt-api-354072478643.us-central1.run.app |
| **API Docs** | https://probt-api-354072478643.us-central1.run.app/docs |
| **GCP Project** | `probt-cloud` (us-central1) |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                    Cloud Run (probt-cloud)               │
│                                                          │
│  ┌──────────────────┐    ┌───────────────────────────┐  │
│  │  probt-web       │    │  probt-api                │  │
│  │  Next.js 16      │───►│  FastAPI + WebSocket      │  │
│  │  React 19        │    │  Engine Python (Layers 0-6)│  │
│  │  ECharts         │    │  Data baked in image       │  │
│  │  Tailwind v4     │    │  Reading refresh: 60s     │  │
│  │  shadcn/ui       │    │  Port 8080                │  │
│  └──────────────────┘    └───────────────────────────┘  │
│         Port 3000                    Port 8080          │
└─────────────────────────────────────────────────────────┘
         ▲                                        ▲
         │ HTTPS                                   │ HTTPS + WSS
         │                                        │
    🧑 Usuario                         yfinance + NewsAPI + Gemini
```

**Sin base de datos.** Los datos (CSVs + modelo .pkl) están baked en la imagen Docker del engine. El FastAPI gateway refresca el reading en memoria cada 60s vía background task.

---

## Estructura del monorepo

```
probt/
├── engine/                     # Python XAUUSD Probability Engine
│   ├── indicators.py           # Layer 0: RSI, EMA, ATR, MACD, trend_bias
│   ├── smc.py                  # Layer 0: swings, BOS, order blocks, FVGs
│   ├── support_resistance.py   # Layer 0: zone clustering
│   ├── news_sentiment.py       # Layer 0: NewsAPI + Gemini sentiment
│   ├── data_collector.py       # Layer 1: yfinance → CSV
│   ├── volume_profile.py       # Layer 2b: CME volume POC/VA
│   ├── feature_engineer.py     # Layer 2a: 17 stationary features
│   ├── feature_diagnostics.py  # Layer 2c: VIF + correlation check
│   ├── labeler.py              # Layer 3: Triple-Barrier labeling
│   ├── model_trainer.py        # Layer 4: LogReg L1 + Platt calibration
│   ├── live_engine.py          # Layer 5: real-time Tier A/B/News/Sizing
│   ├── trade_logger.py         # Layer 6: forward-test log
│   ├── api/
│   │   ├── main.py             # FastAPI app (12 REST + 1 WS)
│   │   └── services.py         # data derivation (backtest, KPIs, etc.)
│   ├── data/                   # generated CSVs + model (gitignored)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .env.example
│
├── web/                        # Next.js Dashboard
│   ├── src/
│   │   ├── app/                # layout.tsx + page.tsx + globals.css
│   │   ├── components/
│   │   │   ├── layout/         # header, sidebar, footer, command-palette
│   │   │   ├── cards/          # kpi-card, model-card, market-tile
│   │   │   ├── charts/         # echarts-base, cumulative, probability, etc.
│   │   │   ├── widgets/        # ai-status-panel
│   │   │   ├── tables/         # live-trades-table
│   │   │   └── ui/             # shadcn/ui primitives (20 components)
│   │   ├── hooks/              # use-api (React Query), use-websocket
│   │   ├── lib/                # constants, format, chart-colors, utils
│   │   ├── types/              # TypeScript domain types
│   │   └── components/providers/
│   ├── Dockerfile
│   ├── cloudbuild.yaml
│   └── package.json
│
├── infra/                      # Deployment scripts
│   ├── deploy.ps1              # Full deploy (engine + web)
│   └── README.md
│
├── .gitignore
└── README.md                   # (this file)
```

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| **Engine** | Python 3.12 · pandas · scikit-learn · yfinance · joblib · statsmodels |
| **API** | FastAPI · WebSocket · uvicorn · Pydantic |
| **Web** | Next.js 16 (App Router) · React 19 · TypeScript 5 |
| **UI** | Tailwind CSS v4 · shadcn/ui (Base UI) · ECharts · Framer Motion · Lucide |
| **Data** | React Query (TanStack) · clsx + tailwind-merge · next-themes |
| **Cloud** | Google Cloud Run · Artifact Registry · Cloud Build |

---

## Modelo (Tier A)

| Atributo | Valor |
|---|---|
| **Algoritmo** | LogisticRegression L1 (C=0.1) + Platt calibration |
| **Features** | 17 stationary (drop `bb_pct_b_1d` por \|corr\|=0.89 con `rsi_1d`) |
| **Labeling** | Triple-Barrier (TP=2·ATR, SL=1·ATR, horizonte=10 días) |
| **Validación** | TimeSeriesSplit(5) — sin random splits en series temporales |
| **Datos** | 592 filas (2024-02 → 2026-06, limitado por historia horaria de yfinance) |
| **Brier Score (CV)** | 0.266 — honesto, sin edge estadístico aún |
| **Label balance** | 48.6% / 51.4% |
| **Métrica primaria** | Brier Score (no accuracy, no P&L en backtest) |

> **Nota:** El Brier Score de 0.266 (ligeramente peor que random 0.25) refleja honestamente que con 592 filas y 2.4 años de datos, el modelo aún no tiene edge estadístico robusto. El engine funciona correctamente; el edge mejorará al acumular más historia.

### Features canónicas (17)

```
rsi_1d, macd_pct_1d, ema_cross_1d, atr_pct_1d,
rsi_1h, ema_cross_1h, rsi_4h, ema_cross_4h, macd_pct_4h,
smc_bias_1d, zone_dist_atr, at_zone, poc_dist_atr, in_value_area,
dxy_return_1d, vix_level, tnx_level
```

---

## Dashboard (16 secciones)

1. **Header fijo** — logo, LIVE status, UTC clock, market/timeframe selectors, search, theme toggle, profile
2. **Sidebar colapsable** — 4 secciones de navegación, animated active indicator
3. **10 KPIs hero** — Total Profit, Win Rate, Current Position, AI Confidence, Prediction, Today's Profit, Sharpe, Max DD, EV, Probability Score (cada uno con sparkline + tendencia)
4. **Main chart** — curva acumulativa equity + probabilidad + predicción futura + banda confianza + crosshair + zoom slider
5. **AI Status panel** — signal, confidence, risk, expected ROI, leverage, modelo activo, latency, server health
6. **Probability Distribution** — histograma + acumulada + percentiles + current marker
7. **AI Prediction** — 7 insight cards en lenguaje natural (volatilidad, momentum, macro, SMC, liquidez, sentimiento, decision driver)
8. **Market Overview** — 7 assets reales (BTC, ETH, NASDAQ, S&P500, DXY, Gold, Oil) via yfinance
9. **Live Trades** — tabla con hover, status badges (TP/SL), confidence, PnL
10. **ML Models** — 6 tarjetas (LogReg-L1 active + XGBoost/RF/Bayesian benchmarks + Transformer/LSTM queued)
11. **Heatmap** — matriz correlación 17×17 de features
12. **Confidence Analysis** — 6 gauges (Sharpe, Sortino, Calmar, Probability, EV, Risk)
13. **Feature Importance** — diverging bar chart de coeficientes L1
14. **Footer status** — API, latency, CPU, mem, GPU, workers, streaming, logs
15. **Command Palette** — Ctrl+K para búsqueda y acciones rápidas
16. **Dark mode** — toggle persistente con paleta neutral derivada (Vercel/Linear style)

---

## API Reference

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/api/health` | Estado del engine |
| GET | `/api/reading` | Tier A/B/News/Sizing en vivo |
| GET | `/api/backtest` | Equity curve + métricas (142 trades) |
| GET | `/api/kpis` | 10 KPIs hero |
| GET | `/api/probability-dist` | Histograma de probabilidades del modelo |
| GET | `/api/trades` | Historial de trades |
| GET | `/api/models` | Tarjetas de modelos ML |
| GET | `/api/market` | Precios multi-asset (yfinance) |
| GET | `/api/heatmap` | Matriz correlación features |
| GET | `/api/confidence` | Gauges de métricas risk-adjusted |
| GET | `/api/insights` | AI insight cards |
| GET | `/api/features` | Feature importance (L1 coeficientes) |
| WS | `/ws/stream` | Push de reading cada 30s |

Docs interactivas: https://probt-api-354072478643.us-central1.run.app/docs

---

## Desarrollo local

### Prerrequisitos

- Python 3.10+ (vía `py`)
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes Python)
- Node.js 22+
- npm 10+

### Engine

```bash
cd engine
uv venv .venv
uv pip install -r requirements.txt
cp .env.example .env   # opcional: añadir NEWSAPI_KEY + GEMINI_API_KEY

# Pipeline completo (primera vez)
uv run python data_collector.py
uv run python feature_engineer.py
uv run python feature_diagnostics.py
uv run python labeler.py
uv run python model_trainer.py
uv run python live_engine.py

# API gateway
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Dashboard

```bash
cd web
npm install
cp .env.example .env.local   # ya configurado para localhost:8000
npm run dev
```

Abrir http://127.0.0.1:3000

---

## Despliegue a Google Cloud

### Prerrequisitos

- gcloud CLI autenticado
- Billing account activa

### Deploy rápido

```powershell
cd infra
.\deploy.ps1
```

### Deploy manual paso a paso

```bash
# 1. Variables
PROJECT=probt-cloud
REGION=us-central1
REPO=us-central1-docker.pkg.dev/$PROJECT/probt

# 2. Build engine
gcloud builds submit ./engine --tag=$REPO/engine:latest --project=$PROJECT

# 3. Deploy engine
gcloud run deploy probt-api \
  --image=$REPO/engine:latest \
  --region=$REGION --port=8080 --allow-unauthenticated \
  --memory=1Gi --cpu=1 --min-instances=0 --max-instances=3 \
  --timeout=300 --project=$PROJECT

# 4. Capturar URL del engine
API_URL=$(gcloud run services describe probt-api --region=$REGION \
  --format="value(status.url)" --project=$PROJECT)

# 5. Build web (con engine URL)
# Editar web/cloudbuild.yaml con los valores de $API_URL
gcloud builds submit ./web --config=web/cloudbuild.yaml --project=$PROJECT

# 6. Deploy web
gcloud run deploy probt-web \
  --image=$REPO/web:latest \
  --region=$REGION --port=3000 --allow-unauthenticated \
  --memory=512Mi --cpu=1 --min-instances=0 --max-instances=3 \
  --timeout=60 --cpu-boost --project=$PROJECT
```

### Re-entrenar el modelo y redeployar

```bash
cd engine
uv run python data_collector.py       # datos frescos
uv run python feature_engineer.py
uv run python labeler.py
uv run python model_trainer.py        # nuevo Brier Score

# Redeployar engine con datos actualizados
cd ..
gcloud builds submit ./engine --tag=$REPO/engine:latest --project=$PROJECT
gcloud run deploy probt-api --image=$REPO/engine:latest --region=$REGION --project=$PROJECT
```

---

## Configuración

### Engine (.env)

```bash
NEWSAPI_KEY=           # opcional: newsapi.org key para headlines
GEMINI_API_KEY=        # opcional: aistudio.google.com/apikey para sentiment
GCLOUD_PROJECT=        # opcional: para Vertex AI ADC en Cloud Run
GCLOUD_LOCATION=us-central1
```

Sin keys, el engine funciona pero omite el news overlay (degradación graceful).

### Web (.env.local)

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
NEXT_PUBLIC_WS_URL=ws://127.0.0.1:8000/ws/stream
```

---

## Decisiones de diseño y limitaciones

### Honestidad estadística
- El Brier Score (0.266) se muestra sin maquillaje. No hay edge todavía con 592 filas.
- El documento original previene contra el "confluence score trap" (suma manual de indicadores). Este sistema usa regresión logística L1 — los pesos se aprenden de datos, no se adivinan.

### Limitaciones de datos
- **Historia horaria limitada:** yfinance da ~730 días de datos 1H. Las features sub-diarias (RSI 1H/4H, VP) solo cubren ~2.4 años.
- **Ticker fallback:** `XAUUSD=X` está delisted en yfinance → se usa `GC=F` (futuros CME, volumen real).
- **Tick-volume evitado:** el engine usa volumen real de CME (GC=F), no tick-volume de broker.

### Arquitectura sin DB
- Sin Firestore, sin BigQuery, sin Cloud SQL. Datos baked en imagen Docker.
- El reading se refresca en memoria del FastAPI cada 60s (background task).
- Para persistir históricos de readings, añadir GCS bucket + worker job (futuro).

### Cloud Run
- **Min-instances=0:** escala a cero cuando no hay tráfico (costo ~$0 en idle).
- **Cold start:** ~3-5s para engine, ~2s para web.
- **WebSocket:** soportado en Cloud Run. El server pusha cada 30s; conexiones se reconectan tras timeout (60 min max).
- **Costo estimado:** $0-5/mes para tráfico de demo. Sube con tráfico real.

---

## Flujo de uso del trader

```
1. Abrir dashboard → leer Tier A (¿probabilidad > 55%?)
2. Leer Tier B (¿sesgo SMC coincide? ¿at_zone = true?)
3. Leer News overlay (¿contradice la dirección?)
4. Si Tier A ≥ 60% + Tier B at_zone + bias coincide + news no contradice
   → ejecutar entrada manual con size sugerido (Half-Kelly, cap 2%)
5. Registrar en trade_logger.py después de la entrada
```

**Esto NO es un bot de trading.** Es un sistema de soporte a la decisión manual.

---

## Documentación técnica

Para el documento técnico completo del motor (fundamentos matemáticos, Triple-Barrier, multicolinealidad, calibración, Kelly), ver el documento de especificación original del XAUUSD Probability Engine v2.0.

---

## Licencia

Propietario. Uso interno.
