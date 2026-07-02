"""Layer 4 — model_trainer.py

Trains, calibrates and saves the Tier A model for a (symbol, timeframe) pair.

  - LogisticRegression(penalty='l1', solver='liblinear', C=0.3)
  - StandardScaler fit on train only per fold
  - TimeSeriesSplit(5) — never random splits on time series
  - CalibratedClassifierCV(method='sigmoid') = Platt scaling
  - Primary metrics: Brier Score (0.25 = random, <0.22 = good, <0.18 = edge),
    Brier Skill Score vs. climatology reference, ROC-AUC (threshold-free).

Outputs per pair at data/symbols/{SYMBOL}_{TF}/:
  model.pkl               = {'model': calibrated, 'scaler': StandardScaler}
  conformal.pkl           = split-conformal q_hat for 90% coverage intervals
  features.json           = feature column names in training order
  metrics.json            = CV Brier + BSS + ROC-AUC + metadata
  benchmarks.json         = XGBoost/RF/GradientBoost/AdaBoost/Bayesian comparison
  reliability_diagram.png = calibration curve (BS + BSS annotated)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from symbols import (
    normalize_symbol, normalize_timeframe, tf_min_bars,
    pair_path,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

NON_FEATURES = ["label", "label_bars", "date", "open", "high", "low", "close", "volume"]
C = 0.3


def _l1_logreg() -> LogisticRegression:
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000)


def train(symbol: str, timeframe: str) -> dict | None:
    """Train and persist the model for (symbol, timeframe).

    Returns metrics dict, or None if training was skipped (insufficient data).
    """
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)

    path = pair_path(symbol, timeframe, "feature_matrix.csv")
    if not os.path.exists(path):
        print(f"[model_trainer] {symbol} {timeframe}: feature_matrix.csv missing — skip")
        return None
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    if "label" not in df.columns:
        print(f"[model_trainer] {symbol} {timeframe}: feature_matrix has no label — run labeler first")
        return None

    feats = [c for c in df.columns if c not in NON_FEATURES]
    if len(df) < tf_min_bars(timeframe):
        print(f"[model_trainer] {symbol} {timeframe}: only {len(df)} rows "
              f"(< {tf_min_bars(timeframe)} min) — skip")
        return None

    X = df[feats].astype(float)
    y = df["label"].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    briers, last_coefs = [], None
    # Collect per-fold OOS probabilities assembled by test index so we can score
    # ROC-AUC / BSS on the full out-of-sample prediction vector (TimeSeriesSplit
    # test folds are contiguous and non-overlapping — safe to assemble this way).
    oos_proba = np.full(len(X), np.nan)
    oos_mask = np.zeros(len(X), dtype=bool)
    for k, (tr, te) in enumerate(tscv.split(X), 1):
        sc = StandardScaler().fit(X.iloc[tr])
        Xt = sc.transform(X.iloc[tr])
        Xs = sc.transform(X.iloc[te])
        clf = _l1_logreg()
        clf.fit(Xt, y.iloc[tr])
        proba = clf.predict_proba(Xs)[:, 1]
        oos_proba[te] = proba
        oos_mask[te] = True
        bs = brier_score_loss(y.iloc[te], proba)
        briers.append(bs)
        if k == tscv.n_splits:
            last_coefs = pd.Series(clf.coef_[0], index=feats)
        print(f"  {symbol} {timeframe} fold {k}: train={len(tr):4d} test={len(te):4d} Brier={bs:.4f}")

    avg_bs = float(np.mean(briers))
    # ─── Out-of-sample metrics ─────────────────────────────────────
    # ROC-AUC is threshold-free; BSS uses the climatology reference (base-rate
    # forecast), NOT a hardcoded 0.25 which overstates skill on imbalanced y.
    base_rate = float(y.mean())
    brier_ref = base_rate * (1.0 - base_rate)  # BS of always-predict-base-rate
    bss_cv = 1.0 - avg_bs / brier_ref
    print(f"[model_trainer] {symbol} {timeframe} avg Brier={avg_bs:.4f} "
          f"(ref={brier_ref:.4f}) BSS={bss_cv:+.4f} (>0.10 = real edge)")
    if last_coefs is not None:
        top = last_coefs.reindex(last_coefs.abs().sort_values(ascending=False).index)
        print(top.head(8).to_string())

    # Full OOS probability vector → Brier / BSS / ROC-AUC on the assembled folds
    y_arr = np.asarray(y)
    brier_oos = float(brier_score_loss(y_arr[oos_mask], oos_proba[oos_mask]))
    bss_oos = 1.0 - brier_oos / brier_ref
    roc_auc = float(roc_auc_score(y_arr[oos_mask], oos_proba[oos_mask]))
    y_prob_oos = oos_proba[oos_mask]
    print(f"[model_trainer] {symbol} {timeframe} OOS Brier={brier_oos:.4f} "
          f"BSS={bss_oos:+.4f} ROC-AUC={roc_auc:.4f} (>0.55 = useful, >0.60 = clear edge)")

    # ─── Refit on full dataset ────────────────────────────────────
    final_scaler = StandardScaler().fit(X)
    X_all = final_scaler.transform(X)
    calibrated = CalibratedClassifierCV(
        _l1_logreg(),
        method="sigmoid",
        cv=TimeSeriesSplit(n_splits=5),
    )
    calibrated.fit(X_all, y)

    bundle = {"model": calibrated, "scaler": final_scaler}
    joblib.dump(bundle, pair_path(symbol, timeframe, "model.pkl"))
    with open(pair_path(symbol, timeframe, "features.json"), "w") as f:
        json.dump(feats, f, indent=2)
    metrics = {
        "symbol": symbol,
        "timeframe": timeframe,
        "cv_brier_score": round(avg_bs, 4),
        "cv_brier_skill_score": round(bss_cv, 4),
        "cv_folds": [round(b, 4) for b in briers],
        "brier": round(brier_oos, 4),
        "brier_skill_score": round(bss_oos, 4),
        "roc_auc": round(roc_auc, 4),
        "brier_reference": round(brier_ref, 4),
        "label_rate": round(base_rate, 4),
        "label_balance": {str(k): round(v, 4) for k, v in y.value_counts(normalize=True).items()},
        "n_rows": int(len(y)),
        "n_features": len(feats),
        "features": feats,
        "c": C,
        "penalty": "l1",
        "trained_at": pd.Timestamp.now().isoformat(),
    }
    with open(pair_path(symbol, timeframe, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[model_trainer] {symbol} {timeframe}: saved model.pkl + features.json + metrics.json")

    # Reliability diagram + conformal interval wrapper (best-effort)
    save_reliability_diagram(
        y_arr[oos_mask], y_prob_oos,
        save_path=pair_path(symbol, timeframe, "reliability_diagram.png"),
        model_name="LogReg-L1",
    )
    _fit_conformal(symbol, timeframe, y_arr[oos_mask], y_prob_oos)
    _train_benchmarks(symbol, timeframe, X_all, y, feats)
    return metrics


def save_reliability_diagram(y_true, y_prob, save_path: str, model_name: str = "Model"):
    """Generate a calibration curve (reliability diagram) and save as PNG.

    Annotates Brier Score and Brier Skill Score (vs. climatology reference) so
    the operator can read the model's probabilistic quality at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = float(brier_score_loss(y_true, y_prob))
    ref = float(np.mean(y_true)) * (1.0 - float(np.mean(y_true)))
    bss = 1.0 - bs / ref
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", lw=1)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_pred, frac_pos, "s-", color="#3b82f6", label=f"{model_name}", lw=2)
    ax.set_title(f"Reliability Diagram | BS={bs:.4f} | BSS={bss:+.4f}")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"[model_trainer] reliability diagram saved: {save_path}")


def _fit_conformal(symbol: str, timeframe: str, y, y_prob_oos, alpha: float = 0.10) -> None:
    """Split-conformal probability interval (90% coverage when alpha=0.10).

    Uses the last 20% of the out-of-sample predictions as the calibration set.
    Nonconformity score s_i = |p_i - y_i| (y in {0,1}); the (1-alpha) quantile
    of these scores with the finite-sample correction gives a half-width q_hat
    such that Prob(true_prob in [p - q_hat, p + q_hat]) >= 1 - alpha.

    Saves a tiny dict {q_hat, alpha, n_cal} to conformal.pkl. The live engine
    applies the half-width around its point estimate. This is mathematically a
    conformal prediction — chosen over the mapie library because mapie's
    classification module returns boolean prediction sets (which classes are
    plausible), not the probability bounds needed for the EV/sizing decision.
    """
    try:
        y_arr = np.asarray(y, dtype=float)
        p_arr = np.asarray(y_prob_oos, dtype=float)
        n = len(y_arr)
        cal_start = int(n * 0.80)
        s_cal = np.abs(p_arr[cal_start:] - y_arr[cal_start:])
        n_cal = len(s_cal)
        if n_cal < 20:
            print(f"[model_trainer] {symbol} {timeframe}: calibration set too small "
                  f"({n_cal}) — conformal skipped, live engine uses fallback +/-8%")
            return
        # Finite-sample conformal quantile: k-th order statistic where
        # k = ceil((n+1)(1-alpha)) — the /n variant is a common bug that
        # collapses the rank toward 1 and produces absurdly tight intervals.
        q_level = int(np.ceil((n_cal + 1) * (1.0 - alpha)))
        q_level = min(q_level, n_cal)
        q_hat = float(np.sort(s_cal)[q_level - 1])
        joblib.dump(
            {"q_hat": q_hat, "alpha": alpha, "n_cal": n_cal, "coverage": 1.0 - alpha},
            pair_path(symbol, timeframe, "conformal.pkl"),
        )
        print(f"[model_trainer] {symbol} {timeframe}: conformal interval saved "
              f"(q_hat={q_hat:.4f}, {1-alpha:.0%} coverage, n_cal={n_cal})")
    except Exception as e:
        print(f"[model_trainer] conformal fit failed ({e}) — live engine uses fallback +/-8%")


def _train_benchmarks(symbol: str, timeframe: str, X_all, y, feats):
    """Train benchmark models with CalibratedClassifierCV + the same TimeSeriesSplit protocol.

    Every benchmark is calibrated the same way as the active LogReg (isotonic on
    trees, which are overconfident by default) so their Brier Scores are
    comparable — an uncalibrated tree always looks artificially good on Brier.
    """
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    )
    from sklearn.naive_bayes import GaussianNB
    try:
        from xgboost import XGBClassifier
        xgb_avail = True
    except Exception:
        xgb_avail = False

    tscv = TimeSeriesSplit(n_splits=5)
    estimators = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=1),
        "GradientBoost": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=42),
        "Bayesian": GaussianNB(),
    }
    if xgb_avail:
        estimators = {"XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                               eval_metric="logloss", verbosity=0, n_jobs=1),
                      **estimators}

    results = []
    for name, est in estimators.items():
        briers, aucs = [], []
        for tr, te in tscv.split(X_all):
            sc = StandardScaler().fit(X_all[tr])
            Xt = sc.transform(X_all[tr])
            Xs = sc.transform(X_all[te])
            try:
                cal = CalibratedClassifierCV(est, method="isotonic", cv=TimeSeriesSplit(n_splits=3))
                cal.fit(Xt, y.iloc[tr])
                proba = cal.predict_proba(Xs)[:, 1]
            except Exception:
                est.fit(Xt, y.iloc[tr])
                proba = est.predict_proba(Xs)[:, 1]
            briers.append(float(brier_score_loss(y.iloc[te], proba)))
            try:
                aucs.append(float(roc_auc_score(y.iloc[te], proba)))
            except Exception:
                aucs.append(0.5)
        avg_brier = float(np.mean(briers))
        avg_auc = float(np.mean(aucs))
        sc_full = StandardScaler().fit(X_all)
        X_scaled = sc_full.transform(X_all)
        t0 = time.time()
        try:
            cal_full = CalibratedClassifierCV(est, method="isotonic", cv=TimeSeriesSplit(n_splits=5))
            cal_full.fit(X_scaled, y)
            train_time = time.time() - t0
        except Exception:
            est.fit(X_scaled, y)
            train_time = time.time() - t0
        results.append({
            "name": name, "type": type(est).__name__,
            "roc_auc": round(avg_auc, 4),
            "confidence": round(max(1 - avg_brier, 0), 4),
            "status": "benchmark", "brier": round(avg_brier, 4),
            "calibrated": True,
            "train_time": round(train_time, 2),
        })
        print(f"  {name:14s} Brier={avg_brier:.4f} ROC-AUC={avg_auc:.4f} T={train_time:.1f}s")

    with open(pair_path(symbol, timeframe, "benchmarks.json"), "w") as f:
        json.dump(results, f, indent=2)


# ─── legacy entry point (XAUUSD daily) ────────────────────────────
def train_legacy():
    """Backward-compatible train() from v1 — reads daily_feature_matrix.csv
    and writes gold_prob_model.pkl + model_features.json + metrics.json at data/."""
    from symbols import data_dir
    path = os.path.join(data_dir(), "daily_feature_matrix.csv")
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    feats = [c for c in df.columns if c not in NON_FEATURES]
    X = df[feats].astype(float)
    y = df["label"].astype(int)
    tscv = TimeSeriesSplit(n_splits=5)
    briers = []
    for tr, te in tscv.split(X):
        sc = StandardScaler().fit(X.iloc[tr])
        Xt = sc.transform(X.iloc[tr])
        Xs = sc.transform(X.iloc[te])
        clf = _l1_logreg()
        clf.fit(Xt, y.iloc[tr])
        briers.append(brier_score_loss(y.iloc[te], clf.predict_proba(Xs)[:, 1]))
    avg_bs = float(np.mean(briers))
    base_rate = float(y.mean())
    brier_ref = base_rate * (1.0 - base_rate)
    bss = 1.0 - avg_bs / brier_ref
    final_scaler = StandardScaler().fit(X)
    calibrated = CalibratedClassifierCV(_l1_logreg(), method="sigmoid",
                                        cv=TimeSeriesSplit(n_splits=5))
    calibrated.fit(final_scaler.transform(X), y)
    joblib.dump({"model": calibrated, "scaler": final_scaler},
                os.path.join(data_dir(), "gold_prob_model.pkl"))
    with open(os.path.join(data_dir(), "model_features.json"), "w") as f:
        json.dump(feats, f, indent=2)
    print(f"[model_trainer] legacy XAUUSD 1D: Brier={avg_bs:.4f} BSS={bss:+.4f} "
          f"(ref={brier_ref:.4f})")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        train_legacy()
    elif len(args) == 2:
        train(args[0], args[1])
    else:
        print("Usage: python model_trainer.py [SYMBOL TIMEFRAME]")
