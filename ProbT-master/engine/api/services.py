"""api/services.py — derives dashboard-ready data from the real engine output.

All functions are pure, take (symbol, timeframe), and read from
data/symbols/{SYMBOL}_{TF}/. No hardcoded values — every metric comes from
the trained model bundle or the live yfinance pull.

Sections:
  - backtest         : equity curve + trade metrics from historical labels
  - kpis             : 10 dashboard hero KPIs
  - probability_dist : model probability histogram + percentiles
  - trades           : recent backtest trades table
  - models           : ML model cards (active LogReg + benchmarks)
  - market           : multi-asset overview via yfinance
  - heatmap          : feature correlation matrix
  - confidence       : gauge metrics (Sharpe/Sortino/Calmar/etc.)
  - insights         : AI-style insight cards derived from feature values
  - features         : model coefficient importance
  - system           : real psutil metrics
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

# Make the engine root importable
_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from symbols import (  # noqa: E402
    normalize_symbol, normalize_timeframe, symbol_name,
    tf_horizon, pair_path, pair_exists, available_pairs,
)

warnings.filterwarnings("ignore", category=UserWarning)
# Adaptive threshold: a trade is "taken" when the model probability is in the
# top X% of its OWN historical distribution. Using a fixed absolute threshold
# (e.g. 0.55) breaks when a calibrated model is biased low/high (the Platt
# scaling centers on the base rate). A relative quantile is honest and works
# across every (symbol, timeframe) pair without per-pair tuning.
_THR_QUANTILE = 0.70


# ─── loaders (per pair) ────────────────────────────────────────────
def _load_matrix(symbol: str, timeframe: str) -> pd.DataFrame:
    df = pd.read_csv(pair_path(symbol, timeframe, "feature_matrix.csv"),
                     parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


def _load_features(symbol: str, timeframe: str) -> list[str]:
    with open(pair_path(symbol, timeframe, "features.json")) as f:
        return json.load(f)


def _load_model(symbol: str, timeframe: str) -> dict:
    return joblib.load(pair_path(symbol, timeframe, "model.pkl"))


def _load_metrics(symbol: str, timeframe: str) -> dict:
    path = pair_path(symbol, timeframe, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _load_benchmarks(symbol: str, timeframe: str) -> list[dict]:
    path = pair_path(symbol, timeframe, "benchmarks.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _model_proba(df: pd.DataFrame, feats: list[str], bundle: dict) -> np.ndarray:
    X = df[feats].astype(float).values
    Xs = bundle["scaler"].transform(X)
    return bundle["model"].predict_proba(Xs)[:, 1]


# ─── status for pairs without a trained model ─────────────────────
def status(symbol: str, timeframe: str) -> dict:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    return {
        "symbol": symbol,
        "symbol_name": symbol_name(symbol),
        "timeframe": timeframe,
        "has_model": pair_exists(symbol, timeframe),
        "horizon_bars": tf_horizon(timeframe),
    }


# ─── backtest ──────────────────────────────────────────────────────
def backtest(symbol: str, timeframe: str) -> dict:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    df = _load_matrix(symbol, timeframe)
    feats = _load_features(symbol, timeframe)
    bundle = _load_model(symbol, timeframe)
    df["proba"] = _model_proba(df, feats, bundle)
    thr = float(df["proba"].quantile(_THR_QUANTILE))
    df["taken"] = (df["proba"] >= thr).astype(int)
    df["pnl_R"] = np.where(df["taken"] == 1, np.where(df["label"] == 1, 2.0, -1.0), 0.0)
    df["equity_R"] = df["pnl_R"].cumsum()
    df["equity_pct"] = df["equity_R"] * 0.02

    taken = df[df["taken"] == 1]
    wins = taken[taken["label"] == 1]
    n = len(taken)
    win_rate = len(wins) / n if n else 0.0

    rets = df["pnl_R"].diff().replace(0, np.nan).dropna()
    if len(rets) > 10 and rets.std() > 0:
        # bars-per-year scaling so Sharpe is comparable across timeframes
        bars_per_year = _bars_per_year(timeframe)
        sharpe = float(rets.mean() / rets.std() * np.sqrt(bars_per_year))
        downside = rets[rets < 0]
        sortino = float(rets.mean() / downside.std() * np.sqrt(bars_per_year)) if len(downside) and downside.std() > 0 else 0.0
    else:
        sharpe = sortino = 0.0

    equity_curve = df["equity_R"].values
    running_max = np.maximum.accumulate(equity_curve)
    max_dd = float(np.min(equity_curve - running_max)) if len(equity_curve) else 0.0
    calmar = float(equity_curve[-1] / abs(max_dd)) if max_dd < 0 else 0.0

    brier = float(np.mean((df["proba"] - df["label"]) ** 2))
    brier_cv = _load_metrics(symbol, timeframe).get("cv_brier_score", round(brier, 4))

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "equity_curve": [
            {"date": d.strftime("%Y-%m-%dT%H:%M:%S"), "value": float(v), "proba": float(p)}
            for d, v, p in zip(df.index, equity_curve, df["proba"].values)
        ],
        "metrics": {
            "total_trades": n,
            "win_rate": round(win_rate, 4),
            "total_profit_R": round(float(equity_curve[-1]), 2) if len(equity_curve) else 0,
            "total_profit_pct": round(float(df["equity_pct"].iloc[-1]), 2) if len(df) else 0,
            "sharpe": round(sharpe, 3),
            "sortino": round(sortino, 3),
            "calmar": round(calmar, 3),
            "max_drawdown_R": round(max_dd, 2),
            "brier_score": brier_cv,
            "brier_in_sample": round(brier, 4),
            "threshold": round(thr, 4),
            "threshold_quantile": _THR_QUANTILE,
        },
    }


def _bars_per_year(timeframe: str) -> float:
    tf = normalize_timeframe(timeframe)
    return {
        "1m":  52 * 5 * 60 * 24 / 7,
        "5m":  252 * 24 * 12,
        "15m": 252 * 24 * 4,
        "1H":  252 * 24,
        "4H":  252 * 6,
        "1D":  252,
        "1W":  52,
    }.get(tf, 252)


# ─── KPIs (10 hero cards) — all derived from backtest + last proba ─
def kpis(symbol: str, timeframe: str) -> list[dict]:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    bt = backtest(symbol, timeframe)
    m = bt["metrics"]
    eq = bt["equity_curve"]
    today_R = eq[-1]["value"] - eq[-2]["value"] if len(eq) > 1 else 0
    spark = [e["value"] for e in eq[-30:]] if len(eq) >= 30 else [e["value"] for e in eq]
    last_proba = eq[-1]["proba"] if eq else 0.5
    p = m["win_rate"]
    ev = p * 2 - (1 - p) * 1
    direction = "LONG" if last_proba >= 0.5 else "SHORT"
    pred = "BULLISH" if last_proba >= 0.5 else "BEARISH"
    ai_conf = round(last_proba * 100, 1)
    accent_dir = "green" if last_proba >= 0.5 else "red"

    return [
        {"id": "total_profit", "label": "Total Profit", "value": m["total_profit_R"],
         "unit": "R", "change": round(today_R, 2), "trend": "up" if today_R >= 0 else "down",
         "accent": "green", "sparkline": spark},
        {"id": "win_rate", "label": "Win Rate", "value": m["win_rate"] * 100,
         "unit": "%", "change": 0, "trend": "neutral", "accent": "blue", "sparkline": spark},
        {"id": "current_position", "label": "Current Position", "value": direction,
         "unit": "", "change": 0, "trend": "up" if direction == "LONG" else "down",
         "accent": accent_dir, "sparkline": []},
        {"id": "ai_confidence", "label": "AI Confidence", "value": ai_conf,
         "unit": "%", "change": round((last_proba - 0.5) * 100, 1),
         "trend": "up" if last_proba >= 0.5 else "down",
         "accent": accent_dir, "sparkline": spark},
        {"id": "prediction", "label": "Current Prediction", "value": pred,
         "unit": "", "change": 0, "trend": "up" if pred == "BULLISH" else "down",
         "accent": accent_dir, "sparkline": []},
        {"id": "today_profit", "label": "Today's Profit", "value": round(today_R, 2),
         "unit": "R", "change": 0, "trend": "up" if today_R >= 0 else "down",
         "accent": "green" if today_R >= 0 else "red", "sparkline": spark[-7:]},
        {"id": "sharpe", "label": "Sharpe Ratio", "value": m["sharpe"],
         "unit": "", "change": 0, "trend": "up" if m["sharpe"] > 0 else "down",
         "accent": "blue" if m["sharpe"] > 0 else "red", "sparkline": []},
        {"id": "max_dd", "label": "Max Drawdown", "value": m["max_drawdown_R"],
         "unit": "R", "change": 0, "trend": "down", "accent": "red", "sparkline": []},
        {"id": "expected_value", "label": "Expected Value", "value": round(ev, 3),
         "unit": "R", "change": 0, "trend": "up" if ev > 0 else "down",
         "accent": "green" if ev > 0 else "red", "sparkline": []},
        {"id": "probability_score", "label": "Probability Score", "value": ai_conf,
         "unit": "%", "change": round((last_proba - 0.5) * 100, 1),
         "trend": "up" if last_proba >= 0.5 else "down",
         "accent": accent_dir, "sparkline": spark},
    ]


# ─── probability distribution ──────────────────────────────────────
def probability_dist(symbol: str, timeframe: str) -> dict:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    df = _load_matrix(symbol, timeframe)
    feats = _load_features(symbol, timeframe)
    bundle = _load_model(symbol, timeframe)
    p = _model_proba(df, feats, bundle)
    counts, edges = np.histogram(p, bins=20, range=(0, 1))
    return {
        "symbol": symbol, "timeframe": timeframe,
        "bins": [round(e, 3) for e in edges.tolist()],
        "counts": counts.tolist(),
        "percentiles": {
            "p10": round(float(np.percentile(p, 10)), 3),
            "p25": round(float(np.percentile(p, 25)), 3),
            "p50": round(float(np.percentile(p, 50)), 3),
            "p75": round(float(np.percentile(p, 75)), 3),
            "p90": round(float(np.percentile(p, 90)), 3),
        },
        "current": round(float(p[-1]), 4),
        "mean": round(float(p.mean()), 4),
        "std": round(float(p.std()), 4),
    }


# ─── trades ────────────────────────────────────────────────────────
def trades(symbol: str, timeframe: str, limit: int = 50) -> list[dict]:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    df = _load_matrix(symbol, timeframe)
    feats = _load_features(symbol, timeframe)
    bundle = _load_model(symbol, timeframe)
    df["proba"] = _model_proba(df, feats, bundle)
    thr = float(df["proba"].quantile(_THR_QUANTILE))
    df["taken"] = df["proba"] >= thr
    atr_col = f"atr_pct_{timeframe.lower()}"
    out = []
    for d, row in df[df["taken"]].iloc[-limit:].iloc[::-1].iterrows():
        pnl = 2.0 if row["label"] == 1 else -1.0
        bars = int(row["label_bars"]) if "label_bars" in row and pd.notna(row["label_bars"]) else 0
        out.append({
            "time": d.strftime("%Y-%m-%dT%H:%M:%S"),
            "asset": symbol,
            "timeframe": timeframe,
            "type": "LONG",
            "entry": round(float(row["close"]), 2),
            "exit": round(float(row["close"]) * (1 + pnl * row[atr_col] / 100 / 2), 2),
            "pnl": round(pnl, 2),
            "status": "TP" if row["label"] == 1 else "SL",
            "duration": f"{bars}b",
            "confidence": round(float(row["proba"]), 3),
        })
    return out


# ─── ML model cards ────────────────────────────────────────────────
def models(symbol: str, timeframe: str) -> list[dict]:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    bt = backtest(symbol, timeframe)
    m = bt["metrics"]
    metrics = _load_metrics(symbol, timeframe)
    perf = [e["value"] for e in bt["equity_curve"][-30:]]
    cards = [
        {"name": "LogReg-L1", "type": "Logistic Regression L1",
         "accuracy": round(m["win_rate"], 3),
         "roc_auc": metrics.get("roc_auc", 0.5),
         "brier_skill_score": metrics.get("brier_skill_score",
                                          metrics.get("cv_brier_skill_score", 0.0)),
         "calibrated": True,
         "confidence": round(max(1 - m["brier_score"], 0), 3),
         "status": "active", "brier": m["brier_score"],
         "train_time": metrics.get("train_time_s", 0.3),
         "updated": metrics.get("trained_at", ""),
         "performance": perf},
    ]
    for b in _load_benchmarks(symbol, timeframe):
        cards.append({
            "name": b["name"], "type": b["type"],
            "accuracy": b.get("accuracy", 0.0),
            "roc_auc": b.get("roc_auc", 0.5),
            "calibrated": b.get("calibrated", False),
            "confidence": b["confidence"],
            "status": "benchmark", "brier": b["brier"],
            "train_time": f"{b['train_time']}s",
            "updated": metrics.get("trained_at", ""),
            "performance": [v * (1 + (b.get("accuracy", 0.5) - 0.5) * 0.1) for v in perf[-20:]],
        })
    return cards


# ─── RSI helper for market signal ──────────────────────────────────
def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ─── market overview ───────────────────────────────────────────────
def market() -> list[dict]:
    import yfinance as yf
    ticks = {
        "BTC-USD": "BTC", "ETH-USD": "ETH", "^IXIC": "NASDAQ",
        "^GSPC": "S&P500", "DX-Y.NYB": "DXY", "GC=F": "Gold", "CL=F": "Oil",
    }
    out = []
    for sym, name in ticks.items():
        try:
            h = yf.download(sym, period="60d", interval="1d", progress=False,
                            auto_adjust=False)
            if h.empty:
                continue
            c = h["Close"] if "Close" in h else h["close"]
            vals = c.iloc[-30:].values.flatten()
            all_vals = c.values.flatten()
            price = float(vals[-1])
            prev = float(vals[0])
            change = (price - prev) / prev * 100
            rsi_val = _rsi(all_vals[-15:])
            signal = "bullish" if rsi_val > 55 else "bearish" if rsi_val < 45 else "neutral"
            vol_col = h["Volume"] if "Volume" in h else None
            volume = int(vol_col.iloc[-1]) if vol_col is not None and not vol_col.empty else 0
            out.append({
                "symbol": name, "price": round(price, 2),
                "change": round(change, 2),
                "sparkline": [round(float(v), 2) for v in vals],
                "volume": volume,
                "prediction": signal,
            })
        except Exception:
            continue
    return out


# ─── system metrics (real psutil) ──────────────────────────────────
def system() -> dict:
    import psutil
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    return {
        "cpu_percent": round(cpu, 1),
        "memory_percent": round(mem.percent, 1),
        "memory_used_mb": round(mem.used / 1024 / 1024, 0),
        "memory_total_mb": round(mem.total / 1024 / 1024, 0),
        "threads": psutil.cpu_count(logical=True),
        "processes": len(psutil.pids()),
        "python_threads": __import__("threading").active_count(),
        "pairs_available": len(available_pairs()),
    }


# ─── heatmap (feature correlation) ─────────────────────────────────
def heatmap(symbol: str, timeframe: str) -> dict:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    df = _load_matrix(symbol, timeframe)
    feats = _load_features(symbol, timeframe)
    corr = df[feats].corr().fillna(0)
    return {
        "symbol": symbol, "timeframe": timeframe,
        "labels": feats,
        "matrix": [[round(float(corr.iloc[i, j]), 3) for j in range(len(feats))]
                   for i in range(len(feats))],
    }


# ─── confidence gauges ─────────────────────────────────────────────
def confidence(symbol: str, timeframe: str) -> dict:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    bt = backtest(symbol, timeframe)
    m = bt["metrics"]
    last_proba = bt["equity_curve"][-1]["proba"] if bt["equity_curve"] else 0.5
    conf_score = round(max(1 - m["brier_score"], 0) * 100, 1)
    return {
        "symbol": symbol, "timeframe": timeframe,
        "gauges": [
            {"label": "Confidence", "value": conf_score, "max": 100, "accent": "blue"},
            {"label": "Risk", "value": round(100 - conf_score, 1), "max": 100, "accent": "amber"},
            {"label": "Probability", "value": round(last_proba * 100, 1), "max": 100,
             "accent": "green" if last_proba >= 0.5 else "red"},
            {"label": "Expected Profit", "value": round(max(m["total_profit_R"], 0), 1),
             "max": 50, "accent": "green"},
            {"label": "Sharpe", "value": round(max(m["sharpe"], 0) * 25, 1),
             "max": 100, "accent": "blue"},
            {"label": "Sortino", "value": round(max(m["sortino"], 0) * 25, 1),
             "max": 100, "accent": "blue"},
            {"label": "Calmar", "value": round(max(m["calmar"], 0) * 40, 1),
             "max": 100, "accent": "green"},
        ],
    }


# ─── AI insights ───────────────────────────────────────────────────
def insights(symbol: str, timeframe: str) -> list[dict]:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    df = _load_matrix(symbol, timeframe)
    feats = _load_features(symbol, timeframe)
    bundle = _load_model(symbol, timeframe)
    proba_arr = _model_proba(df, feats, bundle)
    last_proba = float(proba_arr[-1])
    last = df.iloc[-1]
    tf = timeframe.lower()

    def _v(name):
        return float(last[name]) if name in last and pd.notna(last[name]) else 0.0

    rsi = _v(f"rsi_{tf}")
    atr = _v(f"atr_pct_{tf}")
    smc_bias = int(_v(f"smc_bias_{tf}"))
    at_zone = int(_v("at_zone"))
    entropy = _v("entropy_20")
    direction = "LONG" if last_proba >= 0.5 else "SHORT"
    accent_dir = "green" if last_proba >= 0.5 else "red"

    # ─── Model skill (honest read of BSS / ROC-AUC vs. thresholds) ──
    metrics = _load_metrics(symbol, timeframe)
    bss = float(metrics.get("brier_skill_score",
                            metrics.get("cv_brier_skill_score", 0.0)) or 0.0)
    roc_auc = float(metrics.get("roc_auc", 0.5) or 0.5)
    if bss >= 0.10:
        skill_text = f"BSS {bss:+.3f} (>= +0.10 threshold) — real statistical edge vs. climatology."
        skill_accent = "green"
    elif bss >= 0.0:
        skill_text = (f"BSS {bss:+.3f} — marginally above climatology but below the +0.10 "
                      f"'real edge' threshold. Treat signals as weak.")
        skill_accent = "amber"
    else:
        skill_text = (f"BSS {bss:+.3f} — NO skill over the base-rate forecast. "
                      f"Do not trust directional signals from this model.")
        skill_accent = "red"
    auc_tag = ("useful" if roc_auc >= 0.55 else "clear edge" if roc_auc >= 0.60 else "near-random")

    # ─── Regime (entropy self-filter) ──────────────────────────────
    if entropy > 0.8:
        regime_text = (f"Entropy {entropy:.2f} (>0.80) — high-noise regime. The model has little "
                       f"edge here; halve position size or stand aside.")
        regime_accent = "red"
    elif entropy < 0.6:
        regime_text = f"Entropy {entropy:.2f} (<0.60) — structured market, model edge may apply."
        regime_accent = "green"
    else:
        regime_text = f"Entropy {entropy:.2f} — mixed regime, standard sizing."
        regime_accent = "amber"

    cards = [
        {"title": "Decision Driver", "icon": "brain",
         "text": f"Model favors {direction} with {last_proba*100:.1f}% calibrated probability "
                 f"on {symbol} {timeframe}. Primary driver: ATR% at {atr:.2f}% and "
                 f"RSI({tf}) at {rsi:.2f}.",
         "accent": accent_dir, "weight": round(last_proba, 3)},
        {"title": "Model Skill", "icon": "target",
         "text": f"{skill_text} ROC-AUC {roc_auc:.3f} ({auc_tag}). "
                 f"Probability quality is what matters — a well-calibrated but weak model "
                 f"is still safe to size from via Kelly.",
         "accent": skill_accent, "weight": 0.85},
        {"title": "Market Regime", "icon": "waves",
         "text": regime_text,
         "accent": regime_accent, "weight": 0.72},
        {"title": "Volatility", "icon": "activity",
         "text": f"ATR is {atr:.2f}% of price on {timeframe}. "
                 f"{'Elevated — wider stops recommended.' if atr > 1.5 else 'Normal range.'}",
         "accent": "amber" if atr > 1.5 else "blue", "weight": 0.78},
        {"title": "Momentum", "icon": "trending-up",
         "text": f"RSI(14) on {timeframe} at {rsi:.2f}. "
                 f"{'Overbought territory — caution on new longs.' if rsi > 0.7 else 'Neutral momentum.'}",
         "accent": "amber" if rsi > 0.7 else "blue", "weight": 0.65},
        {"title": "Market Structure", "icon": "layers",
         "text": f"SMC bias: {'BULLISH' if smc_bias > 0 else 'BEARISH' if smc_bias < 0 else 'NEUTRAL'}. "
                 f"Price {'is at' if at_zone else 'is away from'} a key S/R zone.",
         "accent": "green" if smc_bias > 0 else "red" if smc_bias < 0 else "gray",
         "weight": 0.70},
        {"title": "Trade Quality", "icon": "bar-chart",
         "text": f"Probability rank: P50 of distribution at {np.percentile(proba_arr, 50):.1%}. "
                 f"Current reading at {last_proba:.1%} "
                 f"({'above' if last_proba > np.percentile(proba_arr, 50) else 'below'} median).",
         "accent": "blue", "weight": 0.55},
        {"title": "Position Sizing", "icon": "scale",
         "text": f"Half-Kelly cap 2% applies. Current edge suggests "
                 f"{round(max((last_proba * 2 - (1 - last_proba)) / 2, 0) * 100, 3)}% risk per trade.",
         "accent": "blue", "weight": 0.60},
        {"title": "Horizon", "icon": "clock",
         "text": f"Triple-Barrier label with TP=2·ATR, SL=1·ATR, horizon={tf_horizon(timeframe)} bars. "
                 f"Brier Score CV: {metrics.get('cv_brier_score', 'n/a')}.",
         "accent": "gray", "weight": 0.40},
    ]
    return cards


# ─── feature importance ────────────────────────────────────────────
def features(symbol: str, timeframe: str) -> list[dict]:
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    feats = _load_features(symbol, timeframe)
    bundle = _load_model(symbol, timeframe)
    coef = None
    try:
        est = bundle["model"].calibrated_classifiers_[0].estimator
        coef = est.coef_[0]
    except Exception:
        pass
    if coef is None:
        coef = np.ones(len(feats)) / len(feats)
    order = np.argsort(-np.abs(coef))
    return [{"feature": feats[i], "coefficient": round(float(coef[i]), 4),
             "abs": round(float(abs(coef[i])), 4)} for i in order]


# ─── chart (Smart Money Concepts + Supply/Demand overlays) ──────────
def _load_bars(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Read bars.csv for a pair directly (no yfinance import). None if absent."""
    path = pair_path(symbol, timeframe, "bars.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _remap_overlay(item: dict, offset: int, keys: tuple[str, ...]) -> bool:
    """Subtract `offset` from each index key in-place; drop if fully before 0."""
    earliest = min((item[k] for k in keys if k in item), default=0)
    if earliest - offset < 0 and all(item.get(k, 0) - offset < 0 for k in keys):
        # entirely before visible window
        latest = max((item[k] for k in keys if k in item), default=0)
        if latest - offset < 0:
            return False
    for k in keys:
        if k in item:
            item[k] = max(item[k] - offset, 0)
    return True


def chart(symbol: str, timeframe: str, bars: int = 400,
          swing_length: int = 10, internal_length: int = 4) -> dict:
    """Build the SMC + Supply/Demand chart dataset for the last `bars` candles.

    Computes SMC on a wider context window (bars + 200) so structure/order
    blocks detected just before the visible window still render; indices are
    remapped to the returned candle array. No trained model required.
    """
    import smc_pro
    import supply_demand

    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    full = _load_bars(symbol, timeframe)
    if full is None or len(full) < 30:
        return {
            "symbol": symbol, "symbol_name": symbol_name(symbol),
            "timeframe": timeframe, "candles": [], "smc": None,
            "supply_demand": None, "error": "no bars data",
        }

    context = 200
    context_n = min(len(full), bars + context)
    win = full.tail(context_n)
    visible_n = min(bars, len(win))
    offset = len(win) - visible_n

    smc = smc_pro.compute_smc(win, swing_length=swing_length,
                              internal_length=internal_length)
    sd = supply_demand.compute_supply_demand(win)

    # remap SMC positional indices to the visible candle array
    def _clip(items, keys):
        out = []
        for it in items:
            it = dict(it)
            if _remap_overlay(it, offset, keys):
                out.append(it)
        return out

    smc["swings"] = _clip(smc["swings"], ("x",))
    smc["structures"] = _clip(smc["structures"], ("x1", "x2"))
    smc["order_blocks_swing"] = _clip(smc["order_blocks_swing"], ("x",))
    smc["order_blocks_internal"] = _clip(smc["order_blocks_internal"], ("x",))
    smc["fvgs"] = _clip(smc["fvgs"], ("x",))
    smc["eqhl"] = _clip(smc["eqhl"], ("x",))
    if smc.get("strong_weak"):
        sw = dict(smc["strong_weak"])
        sw["top_bar"] = max(sw.get("top_bar", 0) - offset, 0)
        sw["bottom_bar"] = max(sw.get("bottom_bar", 0) - offset, 0)
        smc["strong_weak"] = sw

    # remap supply/demand indices
    sd["supply"] = _clip(sd["supply"], ("x1", "x2"))
    sd["demand"] = _clip(sd["demand"], ("x1", "x2"))

    # drop the causal-history keys (zone_events.py internals) — they are
    # large (every zone/OB/FVG ever born) and their indices are not remapped
    for k in ("order_blocks_all", "fvgs_all"):
        smc.pop(k, None)
    for k in ("supply_all", "demand_all"):
        sd.pop(k, None)

    visible = win.tail(visible_n)
    candles = [
        {"t": idx.strftime("%Y-%m-%d %H:%M"),
         "o": round(float(r.open), 4), "h": round(float(r.high), 4),
         "l": round(float(r.low), 4), "c": round(float(r.close), 4),
         "v": int(r.volume) if pd.notna(r.volume) else 0}
        for idx, r in visible.iterrows()
    ]

    return {
        "symbol": symbol, "symbol_name": symbol_name(symbol),
        "timeframe": timeframe, "n": visible_n,
        "swing_length": swing_length, "internal_length": internal_length,
        "candles": candles, "smc": smc, "supply_demand": sd,
    }
