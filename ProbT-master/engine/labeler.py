"""Layer 3 — labeler.py

Triple-Barrier labeling per (symbol, timeframe).

  atr_abs     = (atr_pct_<tf> / 100) * close
  upper (TP)  = close + 2 * atr_abs
  lower (SL)  = close - 1 * atr_abs
  horizon (T) = tf_horizon(tf) bars forward

  label = 1 if high[t] >= upper BEFORE lower is touched or horizon expires
  label = 0 otherwise
  Tie-break: same bar touches both -> 0 (conservative, intrabar order unknown)

Overwrites feature_matrix.csv with the new label/label_bars columns.
Drops the last `horizon` rows (no complete forward window).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from symbols import (
    normalize_symbol, normalize_timeframe, tf_horizon,
    pair_path,
)


def label_matrix(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Pure: takes a feature matrix, returns it with label + label_bars columns.

    The DF must have a column named `atr_pct_<tf>` and the OHLC columns
    `close`, `high`, `low` (the latter two optional — fall back to close).
    """
    timeframe = normalize_timeframe(timeframe)
    horizon = tf_horizon(timeframe)
    atr_col = f"atr_pct_{timeframe.lower()}"

    df = df.copy()
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float) if "high" in df else close
    low = df["low"].to_numpy(dtype=float) if "low" in df else close
    if atr_col not in df:
        raise KeyError(f"missing required column '{atr_col}' for TF={timeframe}")
    atr_abs = (df[atr_col].to_numpy(dtype=float) / 100.0) * close
    n = len(df)

    labels = np.full(n, np.nan)
    label_bars = np.full(n, np.nan)
    for i in range(n - horizon):
        upper = close[i] + 2.0 * atr_abs[i]
        lower = close[i] - 1.0 * atr_abs[i]
        if not (np.isfinite(upper) and np.isfinite(lower)):
            continue
        for j in range(i + 1, min(i + 1 + horizon, n)):
            tou = high[j] >= upper
            tod = low[j] <= lower
            if tou and tod:
                labels[i] = 0
                label_bars[i] = j - i
                break
            if tou:
                labels[i] = 1
                label_bars[i] = j - i
                break
            if tod:
                labels[i] = 0
                label_bars[i] = j - i
                break
        else:
            labels[i] = 0
            label_bars[i] = horizon

    df["label"] = labels
    df["label_bars"] = label_bars
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["label_bars"] = df["label_bars"].astype(int)
    return df


def label(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load feature_matrix.csv, attach labels, overwrite. Returns the labeled DF."""
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    path = pair_path(symbol, timeframe, "feature_matrix.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = label_matrix(df, timeframe)
    df.to_csv(path)
    counts = df["label"].value_counts(normalize=True).to_dict()
    print(f"[labeler] {symbol} {timeframe}: {len(df)} rows | "
          f"horizon={tf_horizon(timeframe)} bars | "
          f"balance 1={counts.get(1, 0):.1%} 0={counts.get(0, 0):.1%} | "
          f"avg {df['label_bars'].mean():.1f} bars to resolution")
    return df


# ─── legacy entry point (XAUUSD daily) ────────────────────────────
def label_legacy():
    """Backward-compatible label() from v1: reads/writes daily_feature_matrix.csv."""
    from symbols import data_dir
    path = os.path.join(data_dir(), "daily_feature_matrix.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    df = label_matrix(df, "1D")
    df.to_csv(path)
    counts = df["label"].value_counts(normalize=True).to_dict()
    print(f"[labeler] legacy XAUUSD 1D: {len(df)} rows | balance "
          f"1={counts.get(1, 0):.1%} 0={counts.get(0, 0):.1%}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        label_legacy()
    elif len(args) == 2:
        label(args[0], args[1])
    else:
        print("Usage: python labeler.py [SYMBOL TIMEFRAME]")
