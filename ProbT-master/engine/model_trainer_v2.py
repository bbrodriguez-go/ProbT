"""probt v2 — model_trainer_v2.py: trains M_bounce and M_break separately,
per (symbol, timeframe), on zone-touch events (§5.2, §7, §8).

SKELETON ONLY — signatures + docstrings, no computation. Per §0 ground rule
6, wait for product-owner review before implementing any function body.
Every function currently raises NotImplementedError.

Keeps model_trainer.py's core choice (L1 logistic regression + Platt
calibration; complexity lives in features, not the classifier — §7) but
adds what §7/§8 require for v2: a per-release C sweep over
{0.01, 0.03, 0.1, 0.3, 1.0, 3.0} selected by CV Brier (P4), purged
walk-forward splits with an embargo equal to the Triple-Barrier horizon
(V1), full-OOS assembly for Brier/BSS/ROC-AUC/ECE (V2/V3), per-branch
ablation (V6), per-VIX-regime slicing (V7), cost-sensitivity backtesting
(V8), and economic-sign coefficient sanity flags (§7 "Coefficient sanity").

Both M_bounce and M_break are trained on the SAME event rows (one row per
surviving zone_events.ZoneEvent from bounce_break_labeler.label_events) but
against their own label column, and must independently clear V1-V5 before
either is considered live-eligible — a model that predicts bounce well but
break poorly is only half-shipped (flag in metrics.json, do not silently
promote the other).

Observed ablation effect: TBD — pending Step 7 of the build order (§12);
this module is what PRODUCES the ablation numbers, so this note is
necessarily filled in only after Steps 3-6 (Family P, Family Q,
zone-strength) have run.

OUTPUTS per (symbol, timeframe) under data/symbols/{SYMBOL}_{TF}/ (§10):

  model_bounce.pkl, model_break.pkl     {'model': CalibratedClassifierCV, 'scaler': StandardScaler}
  conformal_bounce.pkl, conformal_break.pkl   split-conformal q_hat, one per model
  features.json                          shared training-order feature column names
  metrics.json                            extended: per-model CV/OOS Brier, BSS, ROC-AUC,
                                          ECE, chosen C, per-VIX-regime slice, cost-adjusted
                                          expectancy/Sharpe
  benchmarks.json                        XGBoost/RF/GradientBoost/AdaBoost/GaussianNB,
                                          calibrated, per model
  reliability_diagram_bounce.png, reliability_diagram_break.png
  ablation_report.json                   baseline / +P / +Q / +Z / cumulative, per model
  regime_slice_report.json               Brier/BSS/ROC-AUC per VIX regime, per model
  cost_sensitivity.json                  expectancy/Sharpe with and without spread+slippage
  coefficient_audit.json                 coefficients per model with economic-sign flags
"""
from __future__ import annotations

from typing import Iterator, Literal

import numpy as np
import pandas as pd

ModelName = Literal["bounce", "break"]

C_GRID: list[float] = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# Expected coefficient sign per feature and per model, where structure gives
# an unambiguous prior. Direction-dependent macro priors (e.g. real yields)
# deliberately absent: bounce/break are zone-relative, not long/short, so a
# macro sign flips with zone_kind and cannot be audited unconditionally.
EXPECTED_COEFFICIENT_SIGNS: dict[str, dict[str, int]] = {
    "bounce": {"htf_alignment": 1},          # HTF agreement helps the hold
    "break": {
        "break_target_dist_atr": -1,          # farther target, fewer hits
        "q6_harmonic_oscillator_energy": 1,   # escaping zone, more breaks
        "zone_width_atr": -1,                 # wider zone, harder traverse
        "htf_alignment": -1,                  # HTF agrees with hold, fewer breaks
    },
}

# Final live feature recipe per model — the Step 3-6 gate outcomes, fixed
# globally (not re-selected per pair, to avoid per-pair snooping):
#   bounce: no branch cleared; hilbert_inst_freq was the only +0.01-dBSS
#           feature (1H) -> included, though the model may still fail
#           eligibility on absolute skill.
#   break:  Z branch cleared decisively; q6 cleared in isolation;
#           break_target_dist_atr is the honest explicit geometry feature.
LIVE_FEATURE_RECIPE: dict[str, list[str]] = {
    "bounce": ["hilbert_inst_freq"],
    "break": ["zone_delta_pct", "zone_age_bars", "zone_touches",
              "zone_width_atr", "zone_confluence_count", "htf_alignment",
              "distance_from_poc_atr", "volume_delta_since_zone",
              "q6_harmonic_oscillator_energy", "break_target_dist_atr"],
}

# Live-eligibility floor (§9.4 honesty: is_probability=True only above it).
# Deliberately mild — it separates "some skill" from "at/below climatology",
# not "good" from "bad"; §13's aspirational targets stay in the writeup.
ELIGIBLE_MIN_BSS = 0.0
ELIGIBLE_MIN_AUC = 0.52

MIN_EVENTS = 300  # ponytail: flat floor; revisit if small pairs matter

# VIX regime boundaries for §8 V7 slicing (calm / normal / stressed).
VIX_REGIME_BOUNDARIES: tuple[float, float] = (15.0, 25.0)

# §8 V8 execution-cost defaults, in ATR units (XAUUSD retail spread ~0.3-0.5
# USD against a 1H ATR of ~8-15 USD).
SPREAD_ATR = 0.04
SLIPPAGE_ATR = 0.02


def purged_walk_forward_splits(
    bar_indices: np.ndarray, n_splits: int = 5, embargo: int = 10,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """§8 V1. TimeSeriesSplit-style contiguous folds over EVENT rows, purged
    by BAR distance: a training event whose Triple-Barrier window
    (bar_index, bar_index + embargo] reaches the first test event's bar is
    dropped, so no training label overlaps a test label in time.
    `bar_indices` maps each event row to its bar position (events are
    irregularly spaced — purging by row count would be wrong); `embargo`
    should equal tf_horizon(timeframe). Since folds are walk-forward, train
    always precedes test, so only the train tail needs purging. Yields
    (train_idx, test_idx) positional arrays into 0..len(bar_indices)-1.
    """
    from sklearn.model_selection import TimeSeriesSplit

    bar_indices = np.asarray(bar_indices)
    for train_idx, test_idx in TimeSeriesSplit(n_splits).split(bar_indices):
        test_start_bar = bar_indices[test_idx[0]]
        keep = bar_indices[train_idx] + embargo < test_start_bar
        yield train_idx[keep], test_idx


def load_event_dataset(symbol: str, timeframe: str) -> dict:
    """Shared loader for every ablation runner (Steps 3-6) and train_dual:
    cleaned bars + macro + v1 feature matrix + baseline column list + labeled
    zone events (built on demand), all on the canonical naive-UTC index.

    Returns {"bars", "macro", "fm", "baseline_cols", "events", "horizon"}.
    """
    import json
    import os

    import feature_engineer
    from symbols import (normalize_symbol, normalize_timeframe, pair_path,
                         tf_horizon)

    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    bars = feature_engineer._clean_index(
        pd.read_csv(pair_path(symbol, timeframe, "bars.csv"), index_col=0))
    macro_path = pair_path(symbol, timeframe, "macro.csv")
    macro = (pd.read_csv(macro_path, index_col=0, parse_dates=True)
             if os.path.exists(macro_path) else None)
    fm = feature_engineer._clean_index(
        pd.read_csv(pair_path(symbol, timeframe, "feature_matrix.csv"), index_col=0))
    baseline_cols = json.load(open(pair_path(symbol, timeframe, "features.json")))

    ev_path = pair_path(symbol, timeframe, "zone_events.csv")
    if not os.path.exists(ev_path):
        import bounce_break_labeler
        bounce_break_labeler.build(symbol, timeframe)
    events = pd.read_csv(ev_path, index_col=0)
    events.index = pd.to_datetime(events.index, utc=True).tz_localize(None)

    return {"bars": bars, "macro": macro, "fm": fm,
            "baseline_cols": baseline_cols, "events": events,
            "horizon": tf_horizon(timeframe)}


def ablation_table(
    events: pd.DataFrame, fm: pd.DataFrame, baseline_cols: list[str],
    extra_features: pd.DataFrame, groups: dict[str, list[str]],
    horizon: int,
) -> None:
    """Print the cumulative per-group ablation table (baseline -> +group1 ->
    +group2 ...) for both labels on ONE shared row set (rows with any NaN in
    any candidate column are dropped once, so every step scores the same
    events)."""
    X_base = fm[baseline_cols].reindex(events.index)
    # extras may already be event-aligned (duplicate timestamps when several
    # zones fire on one bar) — reindex would raise on a duplicated axis
    X_extra = (extra_features if extra_features.index.equals(events.index)
               else extra_features.reindex(events.index))
    full = pd.concat([events[["bar_index", "label_bounce", "label_break"]],
                      X_base, X_extra], axis=1, sort=False).dropna()
    print(f"  events with complete features: {len(full)} / {len(events)}")

    usable = [g for g in groups if set(groups[g]) <= set(full.columns)]
    steps = [("baseline", list(baseline_cols))]
    for g in usable:
        steps.append((f"+{g}", steps[-1][1] + groups[g]))

    bar_idx = full["bar_index"].to_numpy()
    header = f"  {'step':10s} {'cols':>4s}"
    for lab in ("bounce", "break"):
        header += f" | {lab}: {'Brier':>6s} {'BSS':>7s} {'AUC':>5s} {'dBSS':>7s}"
    print(header)
    prev = {"bounce": None, "break": None}
    for name, cols in steps:
        line = f"  {name:10s} {len(cols):4d}"
        for lab in ("bounce", "break"):
            m = evaluate_feature_set(full[cols], full[f"label_{lab}"],
                                     bar_idx, embargo=horizon)
            d = "" if prev[lab] is None else f"{m['bss'] - prev[lab]:+7.4f}"
            prev[lab] = m["bss"]
            line += f" |  {m['brier']:6.4f} {m['bss']:+7.4f} {m['roc_auc']:5.3f} {d:>7s}"
        print(line)


def evaluate_feature_set(
    X: pd.DataFrame, y: pd.Series, bar_indices: np.ndarray,
    n_splits: int = 5, embargo: int = 10, c: float = 0.3,
) -> dict:
    """§8 V2. Purged walk-forward CV of the standard v2 trunk (StandardScaler
    fit on train only + L1 LogisticRegression at fixed `c` +
    Platt calibration) and full-OOS assembly: per-fold test predictions are
    concatenated into one OOS vector, on which Brier, BSS (vs. climatology
    base_rate*(1-base_rate), §8 V3), ROC-AUC and ECE (10 bins) are computed.

    This is the shared measuring stick for every ablation delta (Steps 3-6)
    — same folds, same trunk, only the feature set varies. The C sweep (P4)
    happens once at final training time (sweep_c/train_dual), not per
    ablation step, so deltas reflect features rather than tuning.

    Returns {"brier","bss","roc_auc","ece","base_rate","n_oos","n_train_avg",
             "y_prob" (OOS vector), "oos_idx" (positions into X)}.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    probs, idxs, train_sizes = [], [], []
    for train_idx, test_idx in purged_walk_forward_splits(bar_indices, n_splits, embargo):
        if len(train_idx) < 50 or len(np.unique(y_arr[train_idx])) < 2:
            continue
        scaler = StandardScaler().fit(X_arr[train_idx])
        # l1_ratio=1.0 is the sklearn>=1.8 spelling of penalty="l1"
        clf = CalibratedClassifierCV(
            LogisticRegression(solver="liblinear", C=c, max_iter=5000, l1_ratio=1.0),
            method="sigmoid", cv=TimeSeriesSplit(3),
        )
        clf.fit(scaler.transform(X_arr[train_idx]), y_arr[train_idx])
        probs.append(clf.predict_proba(scaler.transform(X_arr[test_idx]))[:, 1])
        idxs.append(test_idx)
        train_sizes.append(len(train_idx))

    y_prob = np.concatenate(probs)
    oos_idx = np.concatenate(idxs)
    y_true = y_arr[oos_idx]
    base = float(y_true.mean())
    brier = float(np.mean((y_prob - y_true) ** 2))
    ref = base * (1.0 - base)
    bss = float(1.0 - brier / ref) if ref > 0 else float("nan")

    # ECE, 10 equal-width bins
    bins = np.clip((y_prob * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += abs(y_prob[m].mean() - y_true[m].mean()) * m.mean()

    return {
        "brier": brier, "bss": bss,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "ece": float(ece), "base_rate": base,
        "n_oos": int(len(y_true)), "n_train_avg": float(np.mean(train_sizes)),
        "y_prob": y_prob, "oos_idx": oos_idx,
    }


def sweep_c(
    X: pd.DataFrame, y: pd.Series, bar_indices: np.ndarray, embargo: int,
    c_grid: list[float] = C_GRID,
) -> tuple[float, dict[float, float]]:
    """§7, P4. evaluate_feature_set at each C; select by full-OOS Brier.
    The whole sweep is returned, not just the winner (§11 G4)."""
    sweep = {}
    for c in c_grid:
        sweep[c] = evaluate_feature_set(X, y, bar_indices, embargo=embargo,
                                        c=c)["brier"]
    best = min(sweep, key=sweep.get)
    return best, sweep


def coefficient_sanity_check(
    coefficients: pd.Series, expected_signs: dict[str, int],
) -> list[dict]:
    """§7. Flags coefficients whose sign disagrees with the structural
    prior. Zeroed-out (L1-dropped) coefficients are not flagged."""
    flags = []
    for feat, sign in expected_signs.items():
        if feat in coefficients.index:
            c = float(coefficients[feat])
            if c != 0.0 and np.sign(c) != sign:
                flags.append({"feature": feat, "coefficient": c,
                              "expected_sign": sign, "flag": "sign_mismatch"})
    return flags


def build_regime_slice_report(
    y_true: np.ndarray, y_prob: np.ndarray, vix: np.ndarray,
) -> dict:
    """§8 V7. Brier/BSS/ROC-AUC per VIX regime on the full OOS vector."""
    from sklearn.metrics import roc_auc_score

    lo, hi = VIX_REGIME_BOUNDARIES
    out = {}
    slices = {"calm_vix_lt_15": vix < lo,
              "normal_vix_15_25": (vix >= lo) & (vix <= hi),
              "stressed_vix_gt_25": vix > hi}
    for name, m in slices.items():
        n = int(m.sum())
        if n < 30 or len(np.unique(y_true[m])) < 2:
            out[name] = {"n": n, "note": "too few OOS events for a slice"}
            continue
        base = float(y_true[m].mean())
        brier = float(np.mean((y_prob[m] - y_true[m]) ** 2))
        out[name] = {
            "n": n, "base_rate": base, "brier": brier,
            "bss": float(1 - brier / (base * (1 - base))),
            "roc_auc": float(roc_auc_score(y_true[m], y_prob[m])),
        }
    return out


def build_cost_sensitivity_report(
    y_true: np.ndarray, y_prob: np.ndarray, payoff_b: np.ndarray,
    stop_atr: float, spread_atr: float = SPREAD_ATR,
    slippage_atr: float = SLIPPAGE_ATR,
) -> dict:
    """§8 V8. EV-gated backtest on the OOS events: enter when
    p * b - (1 - p) > 0; PnL per trade in R units (R = stop distance): win
    +b, loss -1; costs = (spread + slippage) / stop_atr subtracted from
    every trade. Reported with and without costs. `payoff_b` is per-event
    (fixed 2.0 for bounce; variable target-distance/2 for break)."""
    cost_r = (spread_atr + slippage_atr) / stop_atr
    out = {"spread_atr": spread_atr, "slippage_atr": slippage_atr,
           "stop_atr": stop_atr, "cost_r_per_trade": cost_r}
    for tag, cost in (("no_costs", 0.0), ("with_costs", cost_r)):
        take = y_prob * payoff_b - (1 - y_prob) > 0
        pnl = np.where(y_true == 1, payoff_b, -1.0)[take] - cost
        n = int(take.sum())
        if n < 10:
            out[tag] = {"n_trades": n, "note": "fewer than 10 EV+ trades"}
            continue
        out[tag] = {
            "n_trades": n,
            "hit_rate": float(y_true[take].mean()),
            "expectancy_r": float(pnl.mean()),
            "sharpe": float(pnl.mean() / pnl.std()) if pnl.std() > 0 else None,
        }
    return out


def _fit_conformal_with_check(y_true: np.ndarray, y_prob: np.ndarray,
                              alpha: float = 0.10, n_bins: int = 3,
                              min_bin: int = 40) -> dict:
    """Split-conformal q_hat on the last 20% of OOS predictions (v1
    algorithm), plus an honest coverage check (§8 V5): q_hat fit on the
    60-80% OOS slice is evaluated on the untouched final 20%.

    Also fits a Mondrian (probability-BINNED) variant: the global |p - y|
    score against a binary y forces q_hat ~ max(p, 1-p) (observed 0.65-0.79
    -> near-vacuous bands). Binning the calibration set by predicted p and
    taking a per-bin q_hat keeps validity per bin under exchangeability
    while shrinking the band where the model is confident. The live engine
    prefers the binned q_hat; the global one is kept for comparability with
    the original spec. Bins thinner than `min_bin` fall back to the global
    q_hat (never under-cover for lack of data)."""
    def _qhat(scores: np.ndarray) -> float:
        n_cal = len(scores)
        k = int(np.ceil((n_cal + 1) * (1 - alpha)))
        return float(np.sort(scores)[min(k, n_cal) - 1])

    def _binned(p_cal: np.ndarray, s_cal: np.ndarray, q_global: float):
        edges = np.quantile(p_cal, np.linspace(0, 1, n_bins + 1)[1:-1])
        qs = []
        for b in range(n_bins):
            m = np.digitize(p_cal, edges) == b
            qs.append(_qhat(s_cal[m]) if m.sum() >= min_bin else q_global)
        return list(map(float, edges)), qs

    n = len(y_true)
    scores = np.abs(y_prob - y_true)
    i60, i80 = int(n * 0.6), int(n * 0.8)

    q_check = _qhat(scores[i60:i80])
    coverage = float((scores[i80:] <= q_check).mean())

    # binned coverage check: bins fit on 60-80%, evaluated on the last 20%
    edges_chk, qs_chk = _binned(y_prob[i60:i80], scores[i60:i80], q_check)
    assigned = np.digitize(y_prob[i80:], edges_chk)
    cov_binned = float(np.mean(scores[i80:] <= np.array(qs_chk)[assigned]))

    q_final = _qhat(scores[i80:])
    edges, qs = _binned(y_prob[i80:], scores[i80:], q_final)

    # Calibration bins over the FULL OOS vector: per predicted-probability
    # quintile, the observed hit count. The live engine derives a Wilson 90%
    # CI from these — "when the model said ~p, reality delivered k/n" — which
    # is the uncertainty a trader can actually use. Conformal |p - y| against
    # a binary y (above) can only quantify OUTCOME uncertainty and is
    # near-vacuous by construction (q_hat ~ max(p, 1-p)); both are recorded.
    cal_edges = list(map(float, np.quantile(y_prob, [0.2, 0.4, 0.6, 0.8])))
    cal_bins = []
    assigned_all = np.digitize(y_prob, cal_edges)
    for b in range(5):
        m = assigned_all == b
        cal_bins.append({"n": int(m.sum()), "k": int(y_true[m].sum())})

    return {"q_hat": q_final, "alpha": alpha, "n_calibration": n - i80,
            "binned": {"edges": edges, "q_hats": qs,
                       "empirical_coverage_last20": cov_binned,
                       "passes_88pct": bool(cov_binned >= 0.88)},
            "calibration_bins": {"edges": cal_edges, "bins": cal_bins},
            "coverage_check": {"q_hat_6080": q_check,
                               "empirical_coverage_last20": coverage,
                               "passes_88pct": bool(coverage >= 0.88)}}


def train_single_model(
    X: pd.DataFrame, y: pd.Series, bar_indices: np.ndarray, embargo: int,
    model_name: ModelName,
) -> dict:
    """C sweep + purged-CV OOS metrics + conformal (+coverage check) + a
    final full-data calibrated model for live use + coefficient audit."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    best_c, sweep = sweep_c(X, y, bar_indices, embargo)
    oos = evaluate_feature_set(X, y, bar_indices, embargo=embargo, c=best_c)
    y_true_oos = y.to_numpy(dtype=int)[oos["oos_idx"]]
    conformal = _fit_conformal_with_check(y_true_oos, oos["y_prob"])

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = CalibratedClassifierCV(
        LogisticRegression(solver="liblinear", C=best_c, max_iter=5000,
                           l1_ratio=1.0),
        method="sigmoid", cv=TimeSeriesSplit(5)).fit(Xs, y)
    # audit coefficients from the uncalibrated final fit (Platt rescales
    # probabilities monotonically; it cannot change coefficient signs)
    raw = LogisticRegression(solver="liblinear", C=best_c, max_iter=5000,
                             l1_ratio=1.0).fit(Xs, y)
    coefficients = pd.Series(raw.coef_[0], index=X.columns)

    live_eligible = (oos["bss"] > ELIGIBLE_MIN_BSS
                     and oos["roc_auc"] > ELIGIBLE_MIN_AUC)
    metrics = {k: oos[k] for k in
               ("brier", "bss", "roc_auc", "ece", "base_rate", "n_oos")}
    metrics.update({"chosen_c": best_c,
                    "c_sweep_brier": {str(c): v for c, v in sweep.items()},
                    "n_events": int(len(X)), "live_eligible": bool(live_eligible)})
    return {"model": model, "scaler": scaler, "conformal": conformal,
            "metrics": metrics, "coefficients": coefficients,
            "oos_prob": oos["y_prob"], "oos_idx": oos["oos_idx"],
            "sign_flags": coefficient_sanity_check(
                coefficients, EXPECTED_COEFFICIENT_SIGNS[model_name])}


def build_ablation_report(
    full: pd.DataFrame, baseline_cols: list[str],
    branches: dict[str, list[str]], embargo: int,
) -> dict:
    """§8 V6 final table: baseline -> +branch (cumulative, both labels) on
    ONE shared row set. Every branch is reported whether or not it cleared
    its gate (§11 G4); what ships live is LIVE_FEATURE_RECIPE's decision,
    recorded here under 'retained'."""
    bar_idx = full["bar_index"].to_numpy()
    report: dict = {"n_events": int(len(full)), "steps": {}}
    cols = list(baseline_cols)
    steps = [("baseline", [])] + list(branches.items())
    prev = {}
    for name, extra in steps:
        cols = cols + [c for c in extra if c in full.columns]
        entry = {}
        for lab in ("bounce", "break"):
            m = evaluate_feature_set(full[cols], full[f"label_{lab}"],
                                     bar_idx, embargo=embargo)
            entry[lab] = {"brier": m["brier"], "bss": m["bss"],
                          "roc_auc": m["roc_auc"],
                          "dbss_vs_prev": (None if name == "baseline"
                                           else m["bss"] - prev[lab])}
            prev[lab] = m["bss"]
        report["steps"][name] = entry
    report["retained"] = LIVE_FEATURE_RECIPE
    return report


def _train_benchmarks_v2(
    X: pd.DataFrame, y: pd.Series, bar_indices: np.ndarray, embargo: int,
) -> dict:
    """§7 benchmarks under the identical purged-CV + Platt protocol. Never
    used live unless they beat LogReg OOS Brier by > 0.005 (§7) — the
    comparison is recorded, the decision stays with the reader of
    benchmarks_v2.json."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (AdaBoostClassifier,
                                  GradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler

    zoo: dict[str, object] = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=5,
                                               random_state=0),
        "GradientBoost": GradientBoostingClassifier(random_state=0),
        "AdaBoost": AdaBoostClassifier(random_state=0),
        "GaussianNB": GaussianNB(),
    }
    try:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=3,
                                       learning_rate=0.1, random_state=0,
                                       verbosity=0)
    except ImportError:
        pass

    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    out = {}
    for name, est in zoo.items():
        probs, idxs = [], []
        for tr, te in purged_walk_forward_splits(bar_indices, 5, embargo):
            if len(tr) < 50 or len(np.unique(y_arr[tr])) < 2:
                continue
            sc = StandardScaler().fit(X_arr[tr])
            clf = CalibratedClassifierCV(est, method="sigmoid",
                                         cv=TimeSeriesSplit(3))
            clf.fit(sc.transform(X_arr[tr]), y_arr[tr])
            probs.append(clf.predict_proba(sc.transform(X_arr[te]))[:, 1])
            idxs.append(te)
        p = np.concatenate(probs)
        yt = y_arr[np.concatenate(idxs)]
        base = float(yt.mean())
        brier = float(np.mean((p - yt) ** 2))
        out[name] = {"brier": brier,
                     "bss": float(1 - brier / (base * (1 - base))),
                     "roc_auc": float(roc_auc_score(yt, p))}
    return out


def train_dual(symbol: str, timeframe: str) -> dict | None:
    """Top-level v2 training: both models, all §10 report files, persisted
    under the pair dir. Writes features_v2.json (NOT features.json — that
    file belongs to the v1 bar-level model and the v1 live engine still
    reads it)."""
    import json

    import joblib
    from model_trainer import save_reliability_diagram
    from probability_engine_v2 import build_probability_theory_features
    from quantum_features import (build_quantum_state_features,
                                  build_signal_features,
                                  harmonic_oscillator_energy)
    from symbols import normalize_symbol, normalize_timeframe, pair_path
    from zone_features import build_zone_features_bulk

    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    ds = load_event_dataset(symbol, timeframe)
    events, bars, fm = ds["events"], ds["bars"], ds["fm"]
    baseline_cols, horizon = ds["baseline_cols"], ds["horizon"]
    if len(events) < MIN_EVENTS:
        print(f"[trainer_v2] {symbol} {timeframe}: only {len(events)} events "
              f"(< {MIN_EVENTS}) — skipped")
        return None

    print(f"[trainer_v2] {symbol} {timeframe}: {len(events)} events — "
          f"building all feature branches...")
    idxs = sorted(set(int(i) for i in events["bar_index"]))
    pfeat = build_probability_theory_features(bars, timeframe, ds["macro"], symbol)
    qsig = build_signal_features(bars, at_indices=idxs)
    qst = build_quantum_state_features(bars, fm[baseline_cols], timeframe,
                                       macro=ds["macro"], at_indices=idxs
                                       ).reindex(events.index)
    close = bars["close"].to_numpy(dtype=float)
    qst["q6_harmonic_oscillator_energy"] = [
        harmonic_oscillator_energy(close[int(t)], top, bot)
        for t, top, bot in zip(events["bar_index"], events["zone_top"],
                               events["zone_bottom"])]
    zf = build_zone_features_bulk(events, bars, symbol, timeframe)

    everything = pd.concat(
        [events[["bar_index", "label_bounce", "label_break"]],
         fm[baseline_cols].reindex(events.index),
         pfeat.reindex(events.index), qsig.reindex(events.index), qst, zf],
        axis=1, sort=False)

    # §8 V6 ablation on one shared row set (all branch columns present)
    from probability_engine_v2 import FEATURES as P_FEATURES
    from quantum_features import SIGNAL_FEATURES
    from zone_features import FEATURES as Z_FEATURES
    q_real = [c for c in qst.columns]
    branches = {"P": P_FEATURES, "Q_signal": SIGNAL_FEATURES,
                "Q_real": q_real, "Z": Z_FEATURES}
    abl_rows = everything.dropna()
    print(f"  ablation row set: {len(abl_rows)} events")
    ablation = build_ablation_report(abl_rows, baseline_cols, branches, horizon)

    # ── final per-model training (each on its OWN row set: only its own
    # columns need to be complete, so fewer warmup rows are lost) ────────
    results, regime, costs, coef_audit, benchmarks = {}, {}, {}, {}, {}
    for name in ("bounce", "break"):
        cols = baseline_cols + LIVE_FEATURE_RECIPE[name]
        need = cols + ["bar_index", f"label_{name}"]
        if name == "break":
            need = need + ["break_payoff_b"]
        rows = everything[need].dropna()
        X, y = rows[cols], rows[f"label_{name}"]
        bar_idx = rows["bar_index"].to_numpy()
        print(f"  M_{name}: {len(rows)} events, {len(cols)} features — "
              f"C sweep + train...")
        r = train_single_model(X, y, bar_idx, horizon, name)
        results[name] = r

        y_oos = y.to_numpy(dtype=int)[r["oos_idx"]]
        save_reliability_diagram(
            y_oos, r["oos_prob"],
            pair_path(symbol, timeframe, f"reliability_diagram_{name}.png"),
            model_name=f"M_{name} {symbol} {timeframe}")

        if "vix_level" in rows.columns:
            vix = rows["vix_level"].to_numpy(dtype=float)[r["oos_idx"]] * 50.0
            regime[name] = build_regime_slice_report(y_oos, r["oos_prob"], vix)
        payoff = (rows["break_payoff_b"].to_numpy(dtype=float)[r["oos_idx"]]
                  if name == "break" else np.full(len(y_oos), 2.0))
        stop_atr = 2.0 if name == "break" else 1.0
        costs[name] = build_cost_sensitivity_report(
            y_oos, r["oos_prob"], payoff, stop_atr)
        coef_audit[name] = {
            "coefficients": {k: float(v)
                             for k, v in r["coefficients"].items()},
            "sign_flags": r["sign_flags"]}
        benchmarks[name] = _train_benchmarks_v2(X, y, bar_idx, horizon)
        benchmarks[name]["LogReg_L1"] = {
            k: r["metrics"][k] for k in ("brier", "bss", "roc_auc")}

        joblib.dump({"model": r["model"], "scaler": r["scaler"]},
                    pair_path(symbol, timeframe, f"model_{name}.pkl"))
        joblib.dump(r["conformal"],
                    pair_path(symbol, timeframe, f"conformal_{name}.pkl"))

    feats = {n: baseline_cols + LIVE_FEATURE_RECIPE[n] for n in results}
    metrics = {n: results[n]["metrics"] for n in results}
    for n in results:
        metrics[n]["conformal"] = results[n]["conformal"]

    def _dump(obj, fname):
        with open(pair_path(symbol, timeframe, fname), "w") as f:
            json.dump(obj, f, indent=2, default=float)

    _dump(feats, "features_v2.json")
    _dump(metrics, "metrics_v2.json")
    _dump(ablation, "ablation_report.json")
    _dump(regime, "regime_slice_report.json")
    _dump(costs, "cost_sensitivity.json")
    _dump(coef_audit, "coefficient_audit.json")
    _dump(benchmarks, "benchmarks_v2.json")

    for n in results:
        m = metrics[n]
        cov = results[n]["conformal"]["coverage_check"]
        print(f"  M_{n}: C={m['chosen_c']} OOS Brier {m['brier']:.4f} "
              f"BSS {m['bss']:+.4f} AUC {m['roc_auc']:.3f} ECE {m['ece']:.3f} "
              f"| coverage {cov['empirical_coverage_last20']:.1%} "
              f"(>=88%: {cov['passes_88pct']}) | live_eligible: "
              f"{m['live_eligible']}")
        if results[n]["sign_flags"]:
            print(f"    SIGN FLAGS: {results[n]['sign_flags']}")
    return metrics


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        train_dual(args[0], args[1])
    else:
        print("Usage: python model_trainer_v2.py SYMBOL TIMEFRAME")
