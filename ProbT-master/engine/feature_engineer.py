"""Layer 2a — feature_engineer.py

Builds the canonical stationary feature matrix for a (symbol, timeframe) pair.

Universal feature set (no cross-timeframe, no symbol-specific quirks):

  Technical (9) — all computed on the timeframe's own bars:
    rsi_<tf>       RSI(14) / 100                          [0, 1]
    macd_pct_<tf>  MACD line / close * 100
    ema_cross_<tf> sign(EMA20 - EMA50)                    {-1, 0, 1}
    atr_pct_<tf>   ATR(14) / close * 100
    ret_1          log-return of last bar
    ret_5          5-bar log-return
    smc_bias_<tf>  encoded SMC BOS bias                   {-1, 0, 1}
    zone_dist_atr  distance to nearest S/R zone / ATR
    at_zone        1 if zone_dist_atr < 0.5 else 0

  Macro (3) — only attached when has_macro=True (gold); always daily:
    dxy_return_1d  log-return of DXY
    vix_level      VIX / 50 capped at 1
    tnx_level      TNX / 10

Pure: takes DataFrames in, returns a DataFrame out. Used by both
build(symbol, tf) (offline training) and live_engine (online inference).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import indicators
import smc
import support_resistance
from symbols import (
    normalize_symbol, normalize_timeframe, symbol_has_macro,
    pair_path, data_dir,
)

_SMC_WINDOW = 150


# ─── helpers ───────────────────────────────────────────────────────
def _shannon_entropy(close: pd.Series, window: int = 20, n_bins: int = 10) -> pd.Series:
    """Rolling Shannon entropy of the return distribution, normalized to [0, 1].

    H_norm = 0 -> perfect order (predictable). H_norm -> 1 -> pure noise.
    Computed on log-returns with a `window`-bar trailing window and `n_bins`
    histogram bins. Normalized by log2(n_bins) so the output is comparable
    across configurations. Vectorized via a sliding-window histogram.
    """
    log_returns = np.log(close / close.shift(1))
    vals = log_returns.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    if n <= window:
        return pd.Series(out, index=close.index)
    # Fixed bin edges over the full series so windows are comparable.
    finite = vals[np.isfinite(vals)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(out, index=close.index)
    edges = np.linspace(lo, hi, n_bins + 1)
    h_max = np.log2(n_bins)
    for i in range(window, n):
        w = vals[i - window:i]
        if np.isnan(w).any():
            continue
        counts, _ = np.histogram(w, bins=edges)
        p = counts / counts.sum()
        p = p[p > 0]
        out[i] = -np.sum(p * np.log2(p)) / h_max
    return pd.Series(out, index=close.index)


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.duplicated(keep="last")]
    return df


def _attach_macro(features: pd.DataFrame, macro: pd.DataFrame | None) -> pd.DataFrame:
    if macro is None or macro.empty or "dxy" not in macro.columns:
        return features
    macro = _clean_index(macro)
    # Build a per-day macro frame and merge by normalized day, then ffill gaps.
    daily = macro.groupby(macro.index.normalize()).last().reset_index()
    daily.columns = ["_day", "dxy", "vix", "tnx"]
    features = features.copy()
    features["_day"] = features.index.normalize()
    merged = features.merge(daily, on="_day", how="left")
    merged.index = features.index
    extra = merged[["dxy", "vix", "tnx"]].ffill().bfill()
    if extra["dxy"].isna().all():
        features.drop(columns="_day", inplace=True)
        return features
    features["dxy_return_1d"] = np.log(extra["dxy"] / extra["dxy"].shift(1))
    features["vix_level"] = (extra["vix"] / 50.0).clip(upper=1.0)
    features["tnx_level"] = extra["tnx"] / 10.0
    features.drop(columns="_day", inplace=True)
    return features


def build_last_row_features(
    bars: pd.DataFrame,
    timeframe: str,
    macro: pd.DataFrame | None = None,
    symbol: str | None = None,
    window: int = 200,
) -> pd.Series | None:
    """Fast inference path: compute only the LAST row of features.

    Used by live_engine.compute_reading to avoid re-running the O(N²) SMC loop
    across the entire history. We slice a trailing window of `window` bars
    (>=150 for SMC) and run the same math build_features would on that slice,
    then return the last row as a Series. NaN if not enough bars.
    """
    timeframe = normalize_timeframe(timeframe)
    bars = _clean_index(bars)
    if len(bars) < 60:
        return None
    tail = bars.tail(window).copy()
    fm = build_features(tail, timeframe, macro=macro, symbol=symbol)
    if fm.empty:
        return None
    return fm.iloc[-1]


# ─── core ──────────────────────────────────────────────────────────
def build_features(
    bars: pd.DataFrame,
    timeframe: str,
    macro: pd.DataFrame | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Universal stationary feature matrix for the given bars.

    Pure: no file I/O. NaN rows from indicator warmup are kept so the labeler
    can choose its own horizon; callers should dropna() before training.
    """
    timeframe = normalize_timeframe(timeframe)
    tf_suffix = timeframe.lower()
    bars = _clean_index(bars)
    if len(bars) < 60:
        return pd.DataFrame()

    close = bars["close"]
    out = pd.DataFrame(index=bars.index)

    # ─── 1. Technical on the TF base ──────────────────────────────
    out[f"rsi_{tf_suffix}"] = indicators.rsi(close, 14) / 100.0
    out[f"macd_pct_{tf_suffix}"] = indicators.macd_line(close) / close * 100.0
    ema_f, ema_s = indicators.ema(close, 20), indicators.ema(close, 50)
    out[f"ema_cross_{tf_suffix}"] = np.sign(ema_f - ema_s)
    atr_abs = indicators.atr(bars[["high", "low", "close"]], 14)
    out[f"atr_pct_{tf_suffix}"] = atr_abs / close * 100.0

    log_c = np.log(close)
    out["ret_1"] = log_c.diff(1)
    out["ret_5"] = log_c.diff(5)

    # Shannon entropy regime filter (normalized to [0, 1]).
    # < 0.6 -> structured market, model may have edge; > 0.8 -> noise, down-weight.
    out["entropy_20"] = _shannon_entropy(close, window=20, n_bins=10)

    # ─── 2. SMC bias + S/R zones (trailing window per bar) ────────
    smc_bias, zone_dist, at_zone = [], [], []
    cols = ["open", "high", "low", "close"]
    for i in range(len(bars)):
        if i < 60:
            smc_bias.append(0)
            zone_dist.append(np.nan)
            at_zone.append(0)
            continue
        start = max(0, i - _SMC_WINDOW)
        window_df = bars.iloc[start:i + 1][cols]
        bias = smc.detect_structure(window_df)
        smc_bias.append(indicators.encode_bias(bias))
        zones = support_resistance.detect_zones(window_df)
        price = float(close.iloc[i])
        atr_i = float(atr_abs.iloc[i]) if pd.notna(atr_abs.iloc[i]) else 0.0
        nz = support_resistance.nearest_zone(price, zones)
        if nz and atr_i > 0:
            dist = abs(nz["price"] - price) / atr_i
            zone_dist.append(dist)
            at_zone.append(1 if dist < 0.5 else 0)
        else:
            zone_dist.append(np.nan)
            at_zone.append(0)
    out[f"smc_bias_{tf_suffix}"] = smc_bias
    out["zone_dist_atr"] = zone_dist
    out["at_zone"] = at_zone

    # ─── 3. Macro (gold only) ─────────────────────────────────────
    if symbol is None or symbol_has_macro(symbol):
        out = _attach_macro(out, macro)

    # Keep OHLC for the labeler, then drop warmup NaNs from indicators
    out["high"] = bars["high"]
    out["low"] = bars["low"]
    out["close"] = close
    out = out.dropna()
    out = out[~out.index.duplicated(keep="last")]
    return out


def build(symbol: str, timeframe: str) -> pd.DataFrame:
    """Read bars.csv (+macro.csv if applicable), build features, persist."""
    import data_collector

    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    bars = data_collector.load_bars(symbol, timeframe)
    macro = data_collector.load_macro(symbol, timeframe) if symbol_has_macro(symbol) else None
    out = build_features(bars, timeframe, macro=macro, symbol=symbol)

    path = pair_path(symbol, timeframe, "feature_matrix.csv")
    out.to_csv(path)
    print(f"[feature_engineer] {symbol} {timeframe}: {len(out)} rows | "
          f"{out.index.min().date()} → {out.index.max().date()} | "
          f"{len(out.columns)} cols")
    return out


# ─── legacy entry point (XAUUSD daily) ────────────────────────────
def build_legacy():
    """Backward-compatible build() from v1: writes daily_feature_matrix.csv.

    Uses macro_daily.csv + gold_hourly.csv with cross-timeframe features.
    New code should call build(symbol, timeframe).
    """
    daily_path = os.path.join(data_dir(), "macro_daily.csv")
    hourly_path = os.path.join(data_dir(), "gold_hourly.csv")
    if not os.path.exists(daily_path):
        raise FileNotFoundError("macro_daily.csv missing; run data_collector first")
    daily = pd.read_csv(daily_path, parse_dates=True, index_col=0)
    daily.index = pd.to_datetime(daily.index, utc=True).tz_localize(None)
    out = pd.DataFrame(index=daily.index)
    out["close"] = daily["close"]

    # Cross-timeframe features from hourly
    if os.path.exists(hourly_path):
        hourly = pd.read_csv(hourly_path, parse_dates=True, index_col=0)
        hourly.index = pd.to_datetime(hourly.index, utc=True).tz_localize(None)
        h1 = hourly
        h4 = hourly.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna()
        for sub, name in [(h1, "1h"), (h4, "4h")]:
            c = sub["close"]
            rsi = indicators.rsi(c, 14) / 100.0
            ema_f2, ema_s2 = indicators.ema(c, 20), indicators.ema(c, 50)
            cross = np.sign(ema_f2 - ema_s2)
            per_day = rsi.groupby(rsi.index.normalize()).last()
            cross_day = pd.Series(cross.values, index=c.index).groupby(
                pd.Series(cross.values, index=c.index).index.normalize()
            ).last()
            out[f"rsi_{name}"] = per_day.reindex(out.index.normalize().to_pydatetime()).values
            out[f"ema_cross_{name}"] = cross_day.reindex(out.index.normalize().to_pydatetime()).values

    out.to_csv(os.path.join(data_dir(), "daily_feature_matrix_legacy.csv"))
    print("[feature_engineer] legacy cross-TF matrix written (XAUUSD daily only)")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print("Usage: python feature_engineer.py SYMBOL TIMEFRAME")
        sys.exit(0)
    if len(args) == 2:
        build(args[0], args[1])
    else:
        print("Usage: python feature_engineer.py SYMBOL TIMEFRAME")
