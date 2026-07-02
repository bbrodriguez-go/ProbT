"""Layer 5 — live_engine.py

Real-time engine for any (symbol, timeframe). Produces the
Tier A / Tier B / News / Sizing readout.

For pairs without a trained model (insufficient data at training time), the
readout degrades gracefully: Tier A falls back to a heuristic RSI-based
probability estimate and is_probability=False is flagged in tier_a.

compute_reading(symbol, timeframe) returns a dict for the API gateway.
__main__ prints a formatted terminal report.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import feature_engineer
import news_sentiment
import smc
import support_resistance
from data_collector import _download_first
from symbols import (
    normalize_symbol, normalize_timeframe, symbol_tickers,
    symbol_has_macro, symbol_name, symbol_news_query,
    tf_interval, tf_period, tf_resample, tf_horizon,
    pair_path, pair_exists,
)

_AT_ZONE_PCT = 0.003


# ─── helpers ───────────────────────────────────────────────────────
def _pull_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    """Live download of recent bars at the requested timeframe."""
    interval = tf_interval(timeframe)
    period = tf_period(timeframe)
    _, df = _download_first(symbol_tickers(symbol), period=period, interval=interval)
    if tf_resample(timeframe) == "4h":
        df = df.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _pull_macro(symbol: str) -> pd.DataFrame | None:
    if not symbol_has_macro(symbol):
        return None
    try:
        import data_collector
        return data_collector._download_macro_daily()
    except Exception:
        return None


def _heuristic_proba(last_row: pd.Series, timeframe: str) -> float:
    """Fallback probability estimate when no ML model exists for the pair.

    RSI-distance from 0.5 midpoint + ema_cross agreement, mapped to [0.4, 0.65].
    Clearly flagged as non-probabilistic via is_probability=False downstream.
    """
    tf = timeframe.lower()
    rsi = float(last_row.get(f"rsi_{tf}", 0.5))
    ema = float(last_row.get(f"ema_cross_{tf}", 0))
    base = 0.5 + (rsi - 0.5) * 0.3 + ema * 0.05
    return float(np.clip(base, 0.40, 0.65))


# ─── main entry point ──────────────────────────────────────────────
def compute_reading(symbol: str, timeframe: str) -> dict:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    horizon = tf_horizon(timeframe)

    # ─── 1. Build live feature row (fast inference: only last row) ──
    bars = _pull_bars(symbol, timeframe)
    macro = _pull_macro(symbol)
    last_row = feature_engineer.build_last_row_features(
        bars, timeframe, macro=macro, symbol=symbol
    )
    if last_row is None:
        raise RuntimeError(f"not enough live bars for {symbol} {timeframe}: {len(bars)}")
    price = float(last_row["close"])
    # Snapshot current bar timestamp from the bars index
    asof = str(bars.index[-1])

    # ─── 2. Tier A: model prediction or heuristic fallback ────────
    is_probability = True
    if pair_exists(symbol, timeframe):
        with open(pair_path(symbol, timeframe, "features.json")) as f:
            feats = json.load(f)
        bundle = joblib.load(pair_path(symbol, timeframe, "model.pkl"))
        # features.json references Series indices that should be in last_row
        missing = [c for c in feats if c not in last_row.index]
        if missing:
            # Pair trained with a feature the live builder didn't produce — fall back
            proba = _heuristic_proba(last_row, timeframe)
            is_probability = False
        else:
            X = pd.DataFrame([last_row[feats].values], columns=feats)
            Xs = bundle["scaler"].transform(X)
            proba = float(bundle["model"].predict_proba(Xs)[0, 1])
    else:
        proba = _heuristic_proba(last_row, timeframe)
        is_probability = False

    # ─── 2b. Conformal prediction interval (90% coverage) ──────────
    # Loads the half-width q_hat saved by model_trainer._fit_conformal. If the
    # file is missing (old model / failed fit), fall back to a symmetric +/-8%.
    proba_lo = proba
    proba_hi = proba
    conformal_path = pair_path(symbol, timeframe, "conformal.pkl")
    if is_probability and os.path.exists(conformal_path):
        try:
            conf = joblib.load(conformal_path)
            q = float(conf.get("q_hat", 0.08))
            proba_lo = max(proba - q, 0.0)
            proba_hi = min(proba + q, 1.0)
        except Exception:
            proba_lo = max(proba - 0.08, 0.0)
            proba_hi = min(proba + 0.08, 1.0)
    elif is_probability:
        proba_lo = max(proba - 0.08, 0.0)
        proba_hi = min(proba + 0.08, 1.0)
    ev_positive = proba_lo > (1.0 / 3.0)  # even the lower bound beats 2:1 TP/SL breakeven

    # ─── 3. Tier B: SMC + nearest S/R zone (last 100 bars) ────────
    last100 = bars.tail(100)
    bias_b = smc.detect_structure(last100)
    zones = support_resistance.detect_zones(last100)
    nz = support_resistance.nearest_zone(price, zones)
    if nz:
        dist_pct = abs(nz["price"] - price) / price if price else 0.0
        at_zone = dist_pct < _AT_ZONE_PCT
        nearest = {"kind": nz["kind"], "price": round(nz["price"], 2),
                   "distance_pct": round(dist_pct * 100, 3), "at_zone": bool(at_zone)}
    else:
        nearest = {"kind": None, "price": None, "distance_pct": None, "at_zone": False}

    # ─── 4. News overlay (gold-heavy; per-symbol query, cached 6h) ──
    news = news_sentiment.get_news_sentiment(query=symbol_news_query(symbol))

    # ─── 5. Sizing: Half-Kelly, cap 2% ────────────────────────────
    p, b = proba, 2.0
    full_kelly = max((p * b - (1 - p)) / b, 0.0)
    half_kelly = 0.5 * full_kelly
    final_size = min(max(half_kelly, 0.0), 0.02)

    # ─── 6. Latest feature values for the API/insights ────────────
    exposed = {c: (float(v) if pd.notna(v) else None)
               for c, v in last_row.items()
               if c not in ("high", "low", "close")}

    return {
        "symbol": symbol,
        "symbol_name": symbol_name(symbol),
        "timeframe": timeframe,
        "asof": asof,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "price": round(price, 2),
        "horizon_bars": horizon,
        "tier_a": {
            "probability": round(proba, 4),
            "probability_lo": round(proba_lo, 4),
            "probability_hi": round(proba_hi, 4),
            "ev_positive": bool(ev_positive),
            "horizon_bars": horizon,
            "tp_sl_ratio": "2:1",
            "is_probability": is_probability,
            "direction": "LONG" if proba >= 0.5 else "SHORT",
        },
        "tier_b": {"bias": bias_b, "nearest_zone": nearest, "is_probability": False},
        "news": news,
        "sizing": {
            "full_kelly_pct": round(full_kelly * 100, 3),
            "half_kelly_pct": round(half_kelly * 100, 3),
            "final_pct": round(final_size * 100, 3),
            "cap_pct": 2.0,
        },
        "features": exposed,
    }


def _print_report(r: dict):
    p = r["tier_a"]["probability"]
    nz = r["tier_b"]["nearest_zone"]
    line = "=" * 60
    print(line)
    print(f"{r['symbol']} {r['timeframe']}  |  {r['asof']}  |  ${r['price']}")
    print(line)
    proba_tag = "(calibrated ML)" if r["tier_a"]["is_probability"] else "(heuristic fallback)"
    print(f"\nTIER A  -  {proba_tag}")
    print(f"  Probabilidad alcista ({r['horizon_bars']} barras): {p:.1%}")
    ta = r["tier_a"]
    print(f"  Intervalo conformal 90%: [{ta['probability_lo']:.1%}, {ta['probability_hi']:.1%}]"
          f"  EV+ (lo>1/3): {'SI' if ta['ev_positive'] else 'NO'}")
    print(f"  Direccion: {ta['direction']}")
    print("\nTIER B  -  Heuristica SMC (no probabilistica)")
    print(f"  Bias: {r['tier_b']['bias'].upper()}")
    if nz["price"]:
        print(f"  Zona: {nz['kind'].upper()} @ ${nz['price']} (dist {nz['distance_pct']:.3f}%)")
        print(f"  En zona? {'SI' if nz['at_zone'] else 'NO'}")
    print("\nNEWS OVERLAY")
    print(f"  Sentimiento: {r['news']['score']:+.2f}  |  {r['news']['summary']}")
    print("\nSIZING")
    print(f"  Half-Kelly cap: {r['sizing']['final_pct']:.3f}%")
    print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        _print_report(compute_reading(args[0], args[1]))
    else:
        _print_report(compute_reading("XAUUSD", "1D"))
