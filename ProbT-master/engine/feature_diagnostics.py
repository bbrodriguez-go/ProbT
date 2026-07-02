"""Layer 2c — feature_diagnostics.py

Verifies multicollinearity BEFORE training. Mandatory checkpoint.

Inputs : data/symbols/{SYMBOL}_{TF}/feature_matrix.csv  (or legacy
         data/daily_feature_matrix.csv when called with no args)
Outputs: correlation matrix PNG + VIF report CSV in the pair directory.

Decision rules:
  VIF > 10            -> drop the feature OR document why it is kept
  |correlation| > 0.85 -> drop one of the pair (prefer the more interpretable)

Usage:
  python feature_diagnostics.py XAUUSD 1H    # pair-specific (recommended)
  python feature_diagnostics.py              # legacy daily matrix
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

from symbols import pair_path, pair_dir, data_dir

NON_FEATURES = ["label", "label_bars", "date", "open", "high", "low", "close", "volume"]


def run(matrix_path: str, out_dir: str, label: str = "") -> None:
    df = pd.read_csv(matrix_path, parse_dates=True, index_col=0)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feats].astype(float).dropna()

    # ─── Correlation heatmap ───────────────────────────────────────
    corr = X.corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8})
    plt.title(f"Feature correlation matrix — {label or matrix_path}")
    plt.tight_layout()
    corr_png = os.path.join(out_dir, "correlation_matrix.png")
    plt.savefig(corr_png, dpi=120)
    plt.close()

    # ─── Variance Inflation Factor ─────────────────────────────────
    vif_rows = []
    for i, col in enumerate(X.columns):
        try:
            v = float(variance_inflation_factor(X.values, i))
        except Exception:
            v = np.nan
        vif_rows.append({"feature": col, "vif": v})
    vif = pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)
    vif_csv = os.path.join(out_dir, "vif_report.csv")
    vif.to_csv(vif_csv, index=False)

    # ─── Warnings ──────────────────────────────────────────────────
    print(f"[feature_diagnostics] {label or matrix_path}")
    print(f"  {len(feats)} features, {len(X)} rows")
    print(vif.to_string(index=False))
    high_vif = vif[vif["vif"] > 10]
    for _, r in high_vif.iterrows():
        print(f"  WARNING - alta multicolinealidad: {r['feature']}: VIF={r['vif']:.2f}")

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack().reset_index()
        .rename(columns={"level_0": "a", "level_1": "b", 0: "corr"})
    )
    high_pairs = pairs[pairs["corr"].abs() > 0.85]
    for _, r in high_pairs.iterrows():
        print(f"  WARNING - correlacion alta: {r['a']} <-> {r['b']}: {r['corr']:.2f}")

    print(f"[feature_diagnostics] saved {corr_png} + {vif_csv}")


def main():
    args = sys.argv[1:]
    if len(args) == 2:
        from symbols import normalize_symbol, normalize_timeframe
        symbol = normalize_symbol(args[0])
        timeframe = normalize_timeframe(args[1])
        matrix_path = pair_path(symbol, timeframe, "feature_matrix.csv")
        if not os.path.exists(matrix_path):
            print(f"[feature_diagnostics] {matrix_path} missing — run feature_engineer first")
            sys.exit(1)
        run(matrix_path, pair_dir(symbol, timeframe),
            label=f"{symbol} {timeframe}")
    elif len(args) == 0:
        # Legacy: daily_feature_matrix.csv in data/
        legacy = os.path.join(data_dir(), "daily_feature_matrix.csv")
        if not os.path.exists(legacy):
            print(f"[feature_diagnostics] no matrix at {legacy} — "
                  f"usage: python feature_diagnostics.py SYMBOL TIMEFRAME")
            sys.exit(1)
        run(legacy, data_dir(), label="legacy XAUUSD 1D")
    else:
        print("Usage: python feature_diagnostics.py [SYMBOL TIMEFRAME]")
        sys.exit(1)


if __name__ == "__main__":
    main()
