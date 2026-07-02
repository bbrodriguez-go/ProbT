"""api/main.py — FastAPI gateway for the probt multi-asset engine.

REST endpoints accept `symbol` and `timeframe` query params and serve
dashboard-ready data derived from the trained model bundle for that pair.
WebSocket /ws/stream pushes the live reading every WS_PUSH_INTERVAL seconds
(default 10s) for the requested pair.

Run:  uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager

# UTF-8 stdout on Windows containers
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Make the engine root importable (so `import live_engine`, `from api import services`)
_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from symbols import (  # noqa: E402
    normalize_symbol, normalize_timeframe, DEFAULT_SYMBOL, DEFAULT_TIMEFRAME, SYMBOLS, TIMEFRAMES,
    available_pairs, pair_exists,
)
from . import services  # noqa: E402

# ─── In-memory caches keyed by (symbol, timeframe) ────────────────
_CACHE: dict = {
    "readings": {},        # (s, tf) -> live reading dict
    "reading_ts": {},      # (s, tf) -> last refresh epoch
    "market": None,
    "market_ts": 0,
    "started": time.time(),
    "visited": set(),      # pairs touched since boot (drives bg refresh)
}
_DERIVED: dict = {}  # (s, tf) -> {backtest, kpis, probability_dist, ...}
_DERIVED_TTL = 600   # 10 min
_CHART_CACHE: dict = {}  # (s, tf, bars) -> {data, ts}
_CHART_TTL = 120          # 2 min

# Cadence (s) at which the WebSocket pushes a fresh reading to subscribed
# clients. Keep in sync with `LIVE_INTERVAL_MS` on the web side.
WS_PUSH_INTERVAL = 10
# How often the background loop refreshes cached readings for visited pairs.
READING_REFRESH_INTERVAL = 30


def _key(symbol: str, timeframe: str) -> tuple[str, str]:
    return (normalize_symbol(symbol), normalize_timeframe(timeframe))


def _refresh_derived_for(symbol: str, timeframe: str) -> dict:
    """Compute (or recompute) heavy derived datasets for one pair."""
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    if not pair_exists(symbol, timeframe):
        raise FileNotFoundError(f"No trained model for {symbol} {timeframe}")
    bucket = {
        "backtest": services.backtest(symbol, timeframe),
        "probability_dist": services.probability_dist(symbol, timeframe),
        "trades": services.trades(symbol, timeframe),
        "models": services.models(symbol, timeframe),
        "heatmap": services.heatmap(symbol, timeframe),
        "confidence": services.confidence(symbol, timeframe),
        "insights": services.insights(symbol, timeframe),
        "features": services.features(symbol, timeframe),
        "kpis": services.kpis(symbol, timeframe),
        "ts": time.time(),
    }
    _DERIVED[_key(symbol, timeframe)] = bucket
    return bucket


def _derived(symbol: str, timeframe: str) -> dict:
    """Get cached derived dataset, refresh if missing or stale."""
    k = _key(symbol, timeframe)
    b = _DERIVED.get(k)
    if b is None or (time.time() - b.get("ts", 0)) > _DERIVED_TTL:
        b = _refresh_derived_for(symbol, timeframe)
    return b


def _refresh_reading(symbol: str, timeframe: str):
    """Pull a fresh live reading for one pair."""
    try:
        import live_engine
        _CACHE["readings"][_key(symbol, timeframe)] = live_engine.compute_reading(symbol, timeframe)
        _CACHE["reading_ts"][_key(symbol, timeframe)] = time.time()
        _CACHE["visited"].add(_key(symbol, timeframe))
    except Exception as e:
        print(f"[api] reading refresh failed for {symbol} {timeframe}: {e}", flush=True)


def _refresh_market():
    if time.time() - _CACHE["market_ts"] < 300 and _CACHE["market"]:
        return
    try:
        _CACHE["market"] = services.market()
        _CACHE["market_ts"] = time.time()
    except Exception as e:
        print(f"[api] market refresh failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] warming up default pair XAUUSD 1D...", flush=True)
    try:
        _refresh_derived_for(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME)
        _refresh_reading(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME)
    except Exception as e:
        print(f"[api] startup warmup failed: {e}", flush=True)
    _refresh_market()
    print(f"[api] ready -> http://0.0.0.0:{os.environ.get('PORT', 8000)} "
          f"| {len(available_pairs())} trained pairs", flush=True)
    task = asyncio.create_task(_bg_loop())
    yield
    task.cancel()


async def _bg_loop():
    """Refresh readings of recently-visited pairs every READING_REFRESH_INTERVAL seconds; market every 5 min."""
    while True:
        await asyncio.sleep(READING_REFRESH_INTERVAL)
        for k in list(_CACHE["visited"]):
            s, t = k
            await asyncio.to_thread(_refresh_reading, s, t)
        await asyncio.to_thread(_refresh_market)


app = FastAPI(title="probt-engine", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── helpers for query params ──────────────────────────────────────
def _pair_params(
    symbol: str | None = None,
    timeframe: str | None = None,
) -> tuple[str, str]:
    return normalize_symbol(symbol), normalize_timeframe(timeframe)


# ─── REST endpoints ────────────────────────────────────────────────
@app.get("/api/health")
def health():
    default_key = _key(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME)
    return {
        "status": "ok",
        "uptime_s": round(time.time() - _CACHE["started"], 0),
        "pairs_trained": len(available_pairs()),
        "default_warm": pair_exists(DEFAULT_SYMBOL, DEFAULT_TIMEFRAME),
        "reading_ready": bool(_CACHE["readings"].get(default_key)),
    }


@app.get("/api/catalog")
def catalog():
    """List of symbols + timeframes + pairs that have trained models."""
    pairs = available_pairs()
    return {
        "symbols": [
            {"id": s, "name": v["name"], "has_macro": v["has_macro"]}
            for s, v in SYMBOLS.items()
        ],
        "timeframes": list(TIMEFRAMES.keys()),
        "default_symbol": DEFAULT_SYMBOL,
        "default_timeframe": DEFAULT_TIMEFRAME,
        "available_pairs": [{"symbol": s, "timeframe": t} for s, t in pairs],
    }


@app.get("/api/reading")
def reading(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    k = (s, t)
    if k not in _CACHE["readings"]:
        _refresh_reading(s, t)
    return _CACHE["readings"].get(k) or {"error": "reading not ready",
                                          "symbol": s, "timeframe": t}


@app.get("/api/reading_v2")
def reading_v2(symbol: str | None = None, timeframe: str | None = None):
    """Event-driven zone-conditional reading (live_engine_v2). Most calls
    return state="no_signal" by design (§5.4/§9.4 of the v2 spec)."""
    s, t = _pair_params(symbol, timeframe)
    k = (s, t)
    cache = _CACHE.setdefault("readings_v2", {})
    ts = _CACHE.setdefault("readings_v2_ts", {})
    if k not in cache or time.time() - ts.get(k, 0) > READING_REFRESH_INTERVAL:
        try:
            import live_engine_v2
            cache[k] = live_engine_v2.compute_reading_v2(s, t)
            ts[k] = time.time()
        except Exception as e:
            print(f"[api] reading_v2 refresh failed for {s} {t}: {e}", flush=True)
    return cache.get(k) or {"error": "reading_v2 not ready", "symbol": s,
                            "timeframe": t}


@app.get("/api/backtest")
def get_backtest(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["backtest"]


@app.get("/api/kpis")
def get_kpis(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    # KPIs reflect the latest reading's probability if available
    b = _derived(s, t)
    kpis = b["kpis"]
    # Update the "today" KPI with the latest live proba when possible
    k = (s, t)
    r = _CACHE["readings"].get(k)
    if r:
        live_p = r["tier_a"]["probability"]
        kpis[3]["value"] = round(live_p * 100, 1)  # ai_confidence
        kpis[9]["value"] = round(live_p * 100, 1)  # probability_score
    return kpis


@app.get("/api/probability-dist")
def get_probability_dist(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["probability_dist"]


@app.get("/api/trades")
def get_trades(limit: int = 50,
               symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["trades"][:limit]


@app.get("/api/models")
def get_models(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["models"]


@app.get("/api/market")
def get_market():
    if not _CACHE["market"]:
        _refresh_market()
    return _CACHE["market"] or []


@app.get("/api/heatmap")
def get_heatmap(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["heatmap"]


@app.get("/api/confidence")
def get_confidence(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["confidence"]


@app.get("/api/insights")
def get_insights(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["insights"]


@app.get("/api/features")
def get_features(symbol: str | None = None, timeframe: str | None = None):
    s, t = _pair_params(symbol, timeframe)
    return _derived(s, t)["features"]


@app.get("/api/chart")
def get_chart(symbol: str | None = None, timeframe: str | None = None,
              bars: int = 400, swing_length: int = 10, internal_length: int = 4):
    """SMC + Supply/Demand candlestick overlays. No trained model required."""
    s, t = _pair_params(symbol, timeframe)
    k = (s, t, bars)
    cached = _CHART_CACHE.get(k)
    if cached and (time.time() - cached["ts"]) < _CHART_TTL:
        return cached["data"]
    data = services.chart(s, t, bars=bars, swing_length=swing_length,
                          internal_length=internal_length)
    _CHART_CACHE[k] = {"data": data, "ts": time.time()}
    return data


@app.get("/api/system")
def get_system():
    return services.system()


# ─── WebSocket: push live reading every WS_PUSH_INTERVAL seconds ────
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    # Read the first message to know which pair the client wants
    symbol, timeframe = DEFAULT_SYMBOL, DEFAULT_TIMEFRAME
    try:
        first = await asyncio.wait_for(ws.receive_text(), timeout=2.0)
        payload = json.loads(first)
        symbol = normalize_symbol(payload.get("symbol", DEFAULT_SYMBOL))
        timeframe = normalize_timeframe(payload.get("timeframe", DEFAULT_TIMEFRAME))
    except Exception:
        pass
    try:
        while True:
            k = (symbol, timeframe)
            if k not in _CACHE["readings"]:
                await asyncio.to_thread(_refresh_reading, symbol, timeframe)
            payload = _CACHE["readings"].get(k) or {}
            await ws.send_text(json.dumps({"type": "reading", "data": payload}))
            await asyncio.sleep(WS_PUSH_INTERVAL)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
