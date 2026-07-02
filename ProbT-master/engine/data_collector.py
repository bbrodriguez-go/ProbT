"""Layer 1 — data_collector.py

Downloads OHLCV history per (symbol, timeframe) via yfinance and writes:
  data/symbols/{SYMBOL}_{TF}/bars.csv   : OHLCV at the requested timeframe
  data/symbols/{SYMBOL}_{TF}/macro.csv  : DXY/VIX/TNX (gold only, daily)

Single-symbol legacy collect() is kept as a thin wrapper around XAUUSD/1D for
backward compatibility with old docs, but the engine + API use collect(symbol, tf).
"""
from __future__ import annotations

import os

import pandas as pd
import yfinance as yf

import symbols
from symbols import (
    normalize_symbol, normalize_timeframe,
    symbol_tickers, symbol_has_macro, tf_interval, tf_period, tf_resample,
    pair_path, data_dir,
)

_DAYS_FOR_MACRO = 800


# ─── helpers ───────────────────────────────────────────────────────
def _download_first(tickers: list[str], **kw):
    """Try each ticker in order; return (used_ticker, df) for the first non-empty."""
    for t in tickers:
        try:
            df = yf.download(t, **kw, progress=False, auto_adjust=False)
        except Exception:
            df = None
        if df is not None and not df.empty:
            return t, _flat(df)
    raise RuntimeError(f"All tickers failed: {tickers}")


def _flat(df):
    """Rename columns to lowercase and flatten any MultiIndex from yfinance."""
    df = df.rename(columns=str.lower)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _resample_4h(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1H bars into 4H bars (volume sum, OHLC proper rollup)."""
    return (
        hourly.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
    )


def _apply_resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = tf_resample(tf)
    if rule is None:
        return df
    if rule == "4h":
        return _resample_4h(df)
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()


# ─── public API ────────────────────────────────────────────────────
def collect(symbol: str, timeframe: str) -> dict:
    """Download bars + macro for (symbol, timeframe) and write to disk.

    Returns metadata dict: {symbol, timeframe, ticker, rows, range, has_macro}.
    """
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    interval = tf_interval(timeframe)
    period = tf_period(timeframe)
    resample = tf_resample(timeframe)

    used, df = _download_first(symbol_tickers(symbol), period=period, interval=interval)
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if resample is not None:
        df = _apply_resample(df, timeframe)

    bars_path = pair_path(symbol, timeframe, "bars.csv")
    df.to_csv(bars_path)

    meta = {
        "symbol": symbol, "timeframe": timeframe, "ticker": used,
        "rows": int(len(df)),
        "range": [str(df.index.min()), str(df.index.max())],
        "has_macro": False,
    }

    # ─── Macro (DXY/VIX/TNX) only for XAUUSD, always daily ────────
    if symbol_has_macro(symbol):
        mac = _download_macro_daily()
        if mac is not None and not mac.empty:
            mac.to_csv(pair_path(symbol, timeframe, "macro.csv"))
            meta["has_macro"] = True

    print(f"[data_collector] {symbol} {timeframe} via {used}: "
          f"{meta['rows']:,} rows | {meta['range'][0]} → {meta['range'][1]}")
    return meta


def _download_macro_daily() -> pd.DataFrame | None:
    """Download DXY + VIX + TNX daily history (~3y). Cached to data/macro_daily.csv."""
    cache = os.path.join(data_dir(), "macro_daily.csv")
    try:
        _, dxy = _download_first(["DX-Y.NYB", "UUP"], period=f"{_DAYS_FOR_MACRO}d", interval="1d")
        vix = _flat(yf.download("^VIX", period=f"{_DAYS_FOR_MACRO}d", interval="1d",
                                progress=False, auto_adjust=False))
        tnx = _flat(yf.download("^TNX", period=f"{_DAYS_FOR_MACRO}d", interval="1d",
                                progress=False, auto_adjust=False))
        mac = pd.DataFrame(index=dxy.index)
        mac["dxy"] = dxy["close"]
        mac["vix"] = vix["close"]
        mac["tnx"] = tnx["close"]
        mac = mac.dropna(subset=["dxy"])
        mac[["vix", "tnx"]] = mac[["vix", "tnx"]].ffill().bfill()
        mac.to_csv(cache)
        return mac
    except Exception as e:
        print(f"[data_collector] macro download failed: {e}")
        return None


def load_bars(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load bars.csv for (symbol, timeframe). Raises if missing."""
    path = pair_path(symbol, timeframe, "bars.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def load_macro(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load macro.csv for (symbol, timeframe); None if absent."""
    path = pair_path(symbol, timeframe, "macro.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    return df


# ─── legacy single-symbol entry point (XAUUSD daily) ──────────────
def collect_legacy():
    """Backward-compatible collect() from v1: writes macro_daily.csv + gold_hourly.csv.

    Kept only so any old script or doc reference still works; new code should
    call collect(symbol, timeframe).
    """
    os.makedirs(data_dir(), exist_ok=True)
    meta = collect("XAUUSD", "1D")
    bars = load_bars("XAUUSD", "1D")
    bars.to_csv(os.path.join(data_dir(), "macro_daily.csv"))
    # hourly history used by old feature_engineer.build()
    _, hourly = _download_first(symbol_tickers("XAUUSD"), period="730d", interval="60m")
    hourly = hourly[["open", "high", "low", "close", "volume"]].dropna()
    hourly.to_csv(os.path.join(data_dir(), "gold_hourly.csv"))
    print("[data_collector] legacy: macro_daily.csv + gold_hourly.csv written")
    return meta


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        collect_legacy()
    elif len(args) == 2:
        collect(args[0], args[1])
    else:
        print("Usage: python data_collector.py [SYMBOL TIMEFRAME]")
        print(f"Symbols: {', '.join(symbols.SYMBOLS.keys())}")
        print(f"Timeframes: {', '.join(symbols.TIMEFRAMES.keys())}")
