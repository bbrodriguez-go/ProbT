"""probt v2 — Family P: classical probability-theory features.

Hypothesis: XAUUSD forward direction is not fully captured by point-in-time
technical/SMC features (v1). The features below add distributional and
dependence structure that a single-snapshot indicator cannot express —
posterior uncertainty (Beta), regime persistence (Markov), cross-asset tail
co-movement (copulas), and fat-tail risk conditional on volatility (EVT).
All are computed causally (rolling/expanding windows, no future data) and
must survive purged walk-forward CV (§8 V1); the branch ships only if it
clears the ablation gate (+0.01 BSS, no ROC-AUC degradation, §8 V6).

Observed ablation effect (2026-07-01, XAUUSD 1H n=4178 events / 4H n=1028,
purged walk-forward, L1 LogReg C=0.3 + Platt): NO MEASURABLE EDGE ON THIS
DATASET — the branch does NOT clear the +0.01 BSS gate (§8 V6) and is NOT
wired into the live pipeline (§0 ground rule 3). Cumulative dBSS on 1H:
bounce -0.0005/-0.0007/-0.0018/+0.0011 (beta/markov/copula/evt), break
+0.0034/-0.0037/-0.0002/+0.0008. Only beta-on-break was positive (+0.0034,
AUC 0.551->0.561), still below gate. On 4H everything is at or below
climatology. Baseline context: v1 features alone score AUC ~0.50 on bounce
(no signal) and ~0.55 on break. Re-test when more history accumulates or
when the event definition changes. Run `python probability_engine_v2.py
SYMBOL TF` to reproduce.

FEATURES (all stationary / bounded):

  beta_posterior_mean      Beta(1,1)-prior posterior mean of the trailing
                            v1 Triple-Barrier outcome stream. An outcome
                            enters the posterior only once RESOLVED
                            (bar i + label_bars[i] <= t) — no lookahead.
  beta_posterior_var       Posterior variance of the same Beta. High = few
                            resolved outcomes / instability -> weak prior.

  markov_transition_entropy   Mean row entropy (normalized to [0,1]) of the
                            k=3-state return transition matrix estimated on
                            the trailing window. Low = persistent regime.
  markov_stationary_entropy   Normalized entropy of the chain's stationary
                            distribution. Low = time concentrated in few
                            states.
  markov_mixing_time        |second-largest eigenvalue| of the transition
                            matrix. ~1 = slow mixing (trending), ~0 = fast
                            mixing (mean-reverting/noisy).

  copula_dxy_lambda_l      Lower-tail dependence (Clayton, via Kendall-tau
                            inversion) of daily gold returns vs. NEGATED
                            daily DXY returns — i.e. P(gold crashes | dollar
                            spikes). Gold-DXY dependence is negative, and
                            Clayton/Gumbel only express positive dependence,
                            so the DXY leg is rotated by negation.
  copula_dxy_lambda_u      Upper-tail dependence (Gumbel) of gold vs. -DXY:
                            P(gold spikes | dollar dumps).
  copula_vix_lambda_l      Lower-tail dependence of gold vs. VIX changes
                            (unrotated: joint crash).
  copula_vix_lambda_u      Upper-tail dependence of gold vs. VIX changes
                            (joint spike = crisis safe-haven bid).
                            Daily values are assigned to intraday bars from
                            the last COMPLETED day (shift by one day) — no
                            intraday lookahead.

  evt_gpd_scale             Scale parameter (method-of-moments) of a
                            Generalized Pareto fit to loss exceedances,
                            conditional on the current volatility bucket.
                            NOTE: the master prompt says vol DECILES; with a
                            500-bar window a decile holds ~50 bars and its
                            tail ~5 points — pure noise — so the default is
                            QUINTILES (~100 bars, ~20 tail points). The
                            bucket count stays a parameter.

Gold-vs-macro copula features require macro data; build skips them when
symbols.symbol_has_macro(symbol) is False (mirrors feature_engineer's
_attach_macro gating).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "beta_posterior_mean",
    "beta_posterior_var",
    "markov_transition_entropy",
    "markov_stationary_entropy",
    "markov_mixing_time",
    "copula_dxy_lambda_l",
    "copula_dxy_lambda_u",
    "copula_vix_lambda_l",
    "copula_vix_lambda_u",
    "evt_gpd_scale",
]

# Feature groups in build-order §12 Step 3 sequence — the ablation runner
# adds them cumulatively in this order.
FEATURE_GROUPS: dict[str, list[str]] = {
    "beta": ["beta_posterior_mean", "beta_posterior_var"],
    "markov": ["markov_transition_entropy", "markov_stationary_entropy",
               "markov_mixing_time"],
    "copula": ["copula_dxy_lambda_l", "copula_dxy_lambda_u",
               "copula_vix_lambda_l", "copula_vix_lambda_u"],
    "evt": ["evt_gpd_scale"],
}

BETA_WINDOW = 500          # bars of resolved outcomes in the posterior
MARKOV_WINDOW = 250        # bars per transition-matrix estimate
MARKOV_STATES = 3          # terciles
COPULA_WINDOW = 120        # DAYS per copula fit
COPULA_MIN_DAYS = 60
EVT_WINDOW = 500           # bars pooled per GPD fit
EVT_VOL_BUCKETS = 5        # quintiles (see docstring note on deciles)
EVT_TAIL_PCT = 80          # losses above this in-bucket percentile = tail
EVT_MIN_EXCEEDANCES = 8


# ─── Bayesian: rolling Beta posterior over resolved TB outcomes ──────
def rolling_beta_posterior(
    outcomes: pd.Series,
    resolution_bars: pd.Series,
    window: int = BETA_WINDOW,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> pd.DataFrame:
    """Rolling Beta-Bernoulli posterior over a binary outcome stream.

    outcomes[i] (0/1) is the Triple-Barrier label of bar i, which becomes
    KNOWN at bar i + resolution_bars[i]. The posterior at bar t therefore
    includes exactly the outcomes resolved in (t - window, t]. Returns
    columns beta_posterior_mean / beta_posterior_var indexed like outcomes.
    """
    n = len(outcomes)
    y = outcomes.to_numpy(dtype=float)
    res = resolution_bars.to_numpy(dtype=float)

    # resolution bar of each outcome (NaN outcome -> never resolves)
    events = []  # (resolution_bar, outcome)
    for i in range(n):
        if np.isfinite(y[i]) and np.isfinite(res[i]):
            events.append((i + int(res[i]), y[i]))
    events.sort()
    ev_bar = np.array([e[0] for e in events], dtype=int)
    ev_y = np.array([e[1] for e in events], dtype=float)

    mean = np.full(n, np.nan)
    var = np.full(n, np.nan)
    lo = hi = 0  # events with resolution bar in (t - window, t]
    for t in range(n):
        while hi < len(ev_bar) and ev_bar[hi] <= t:
            hi += 1
        while lo < hi and ev_bar[lo] <= t - window:
            lo += 1
        k = hi - lo
        s = float(ev_y[lo:hi].sum())
        a = prior_alpha + s
        b = prior_beta + k - s
        mean[t] = a / (a + b)
        var[t] = a * b / ((a + b) ** 2 * (a + b + 1.0))
    return pd.DataFrame(
        {"beta_posterior_mean": mean, "beta_posterior_var": var},
        index=outcomes.index)


# ─── Markov chain over discretized returns ───────────────────────────
def markov_chain_features(
    returns: pd.Series,
    window: int = MARKOV_WINDOW,
    n_states: int = MARKOV_STATES,
) -> pd.DataFrame:
    """Trailing-window k-state transition matrix of discretized returns.

    State edges are the terciles of the SAME trailing window (data <= t
    only). Laplace smoothing (+1) keeps rows valid when a state was never
    visited. Stationary distribution = left eigenvector of eigenvalue 1;
    mixing feature = |second-largest eigenvalue| (Frobenius mixing rate).
    """
    r = returns.to_numpy(dtype=float)
    n = len(r)
    out = np.full((n, 3), np.nan)
    logk = np.log(n_states)
    for t in range(window, n):
        w = r[t - window + 1: t + 1]
        if not np.all(np.isfinite(w)):
            continue
        edges = np.quantile(w, np.linspace(0, 1, n_states + 1)[1:-1])
        states = np.digitize(w, edges)
        tm = np.ones((n_states, n_states))  # Laplace prior
        for a, b in zip(states[:-1], states[1:]):
            tm[a, b] += 1.0
        tm /= tm.sum(axis=1, keepdims=True)

        row_ent = -np.sum(tm * np.log(tm), axis=1) / logk
        out[t, 0] = row_ent.mean()

        eigval, eigvec = np.linalg.eig(tm.T)
        order = np.argsort(-np.abs(eigval))
        pi = np.real(eigvec[:, order[0]])
        pi = np.abs(pi) / np.abs(pi).sum()
        out[t, 1] = float(-np.sum(pi * np.log(np.clip(pi, 1e-12, None))) / logk)
        out[t, 2] = float(np.abs(eigval[order[1]]))
    return pd.DataFrame(out, index=returns.index, columns=[
        "markov_transition_entropy", "markov_stationary_entropy",
        "markov_mixing_time"])


# ─── Copulas: rolling tail dependence via Kendall-tau inversion ──────
def copula_tail_dependence(
    x: pd.Series, y: pd.Series,
    window: int = COPULA_WINDOW, min_periods: int = COPULA_MIN_DAYS,
) -> pd.DataFrame:
    """Rolling lower/upper tail dependence between two return series.

    Moment-matching fit: Kendall's tau on the trailing window, inverted to
    the one-parameter Clayton (theta = 2*tau/(1-tau), lambda_L = 2^(-1/theta))
    and Gumbel (theta = 1/(1-tau), lambda_U = 2 - 2^(1/theta)) copulas.
    tau <= 0 gives no positive tail dependence -> both lambdas 0 (rotate the
    inputs upstream if the economic relation is inverse, as with gold/DXY).
    Columns: lambda_l, lambda_u — caller renames per asset pair.
    """
    from scipy.stats import kendalltau

    df = pd.concat({"x": x, "y": y}, axis=1, sort=False).dropna()
    n = len(df)
    xv = df["x"].to_numpy(dtype=float)
    yv = df["y"].to_numpy(dtype=float)
    ll = np.full(n, np.nan)
    lu = np.full(n, np.nan)
    for t in range(min_periods - 1, n):
        s = max(0, t - window + 1)
        tau = kendalltau(xv[s:t + 1], yv[s:t + 1]).statistic
        if not np.isfinite(tau) or tau <= 0:
            ll[t] = 0.0
            lu[t] = 0.0
            continue
        tau = min(tau, 0.99)
        theta_c = 2.0 * tau / (1.0 - tau)
        ll[t] = float(2.0 ** (-1.0 / theta_c))
        theta_g = 1.0 / (1.0 - tau)
        lu[t] = float(2.0 - 2.0 ** (1.0 / theta_g))
    return pd.DataFrame({"lambda_l": ll, "lambda_u": lu}, index=df.index)


# ─── EVT: GPD tail scale conditional on volatility bucket ────────────
def evt_tail_scale(
    returns: pd.Series,
    vol: pd.Series,
    window: int = EVT_WINDOW,
    n_vol_buckets: int = EVT_VOL_BUCKETS,
) -> pd.Series:
    """Rolling GPD scale of the LOSS tail, conditional on the current
    volatility bucket.

    At bar t: bucket the trailing `window` bars by `vol` quantiles, keep the
    bars sharing t's bucket, take their losses (-returns > 0), threshold at
    the EVT_TAIL_PCT percentile, and fit GPD scale to the exceedances by
    method of moments: with exceedance mean m and variance v,
    xi = (1 - m^2/v)/2, sigma = m*(m^2/v + 1)/2. NaN when fewer than
    EVT_MIN_EXCEEDANCES exceedances (thin bucket) or v == 0.
    """
    r = returns.to_numpy(dtype=float)
    vv = vol.to_numpy(dtype=float)
    n = len(r)
    out = np.full(n, np.nan)
    qs = np.linspace(0, 1, n_vol_buckets + 1)[1:-1]
    for t in range(window, n):
        w_r = r[t - window + 1: t + 1]
        w_v = vv[t - window + 1: t + 1]
        m = np.isfinite(w_r) & np.isfinite(w_v)
        if m.sum() < window // 2 or not np.isfinite(vv[t]):
            continue
        edges = np.quantile(w_v[m], qs)
        bucket = np.digitize([vv[t]], edges)[0]
        in_bucket = m & (np.digitize(w_v, edges) == bucket)
        losses = -w_r[in_bucket]
        losses = losses[losses > 0]
        if len(losses) < EVT_MIN_EXCEEDANCES:
            continue
        thr = np.percentile(losses, EVT_TAIL_PCT)
        exc = losses[losses > thr] - thr
        if len(exc) < EVT_MIN_EXCEEDANCES:
            continue
        em, ev = float(exc.mean()), float(exc.var())
        if ev <= 0:
            continue
        out[t] = 0.5 * em * (em * em / ev + 1.0)
    return pd.Series(out, index=returns.index, name="evt_gpd_scale")


# ─── assembler ───────────────────────────────────────────────────────
def build_probability_theory_features(
    bars: pd.DataFrame,
    timeframe: str,
    macro: pd.DataFrame | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Assemble all Family P features aligned to bars.index (pure, no file
    I/O; NaN warmup rows kept for the caller to drop — mirrors
    feature_engineer.build_features). Copula columns are skipped when the
    symbol has no macro data."""
    import indicators
    from symbols import normalize_timeframe, symbol_has_macro, tf_horizon

    import feature_engineer

    timeframe = normalize_timeframe(timeframe)
    bars = feature_engineer._clean_index(bars)  # naive-UTC, deduped — the
    # canonical index treatment shared with the v1 feature matrix
    close = bars["close"].astype(float)
    ret_1 = np.log(close).diff()
    out = pd.DataFrame(index=bars.index)

    # Beta posterior over v1-style Triple-Barrier outcomes (recomputed here
    # from bars so this module never reads feature_matrix.csv — same
    # barriers as labeler.label_matrix: +2A / -1A, horizon = tf_horizon)
    atr_abs = indicators.atr(bars[["high", "low", "close"]], 14)
    labels, res = _triple_barrier_outcomes(bars, atr_abs, tf_horizon(timeframe))
    beta = rolling_beta_posterior(labels, res)
    out[beta.columns] = beta

    mk = markov_chain_features(ret_1)
    out[mk.columns] = mk

    if macro is not None and not macro.empty and (
            symbol is None or symbol_has_macro(symbol)):
        daily_gold = np.log(close.resample("D").last().dropna()).diff()
        daily_gold.index = daily_gold.index.normalize()
        macro_idx = pd.to_datetime(macro.index)
        for pair_name, series, rotate in (
            ("dxy", macro.get("dxy"), True),
            ("vix", macro.get("vix"), False),
        ):
            if series is None:
                continue
            m_ret = np.log(series.astype(float)).diff()
            m_ret.index = macro_idx.normalize()
            leg = -m_ret if rotate else m_ret
            td = copula_tail_dependence(daily_gold, leg)
            # assign to bars from the last COMPLETED day (no intraday lookahead)
            td_shifted = td.shift(1)
            bar_days = bars.index.normalize()
            mapped = td_shifted.reindex(bar_days.unique().sort_values(), method="ffill")
            out[f"copula_{pair_name}_lambda_l"] = mapped["lambda_l"].loc[bar_days].to_numpy()
            out[f"copula_{pair_name}_lambda_u"] = mapped["lambda_u"].loc[bar_days].to_numpy()

    vol = (atr_abs / close) * 100.0
    out["evt_gpd_scale"] = evt_tail_scale(ret_1, vol)
    return out


def _triple_barrier_outcomes(
    bars: pd.DataFrame, atr_abs: pd.Series, horizon: int,
) -> tuple[pd.Series, pd.Series]:
    """v1 labeler barriers (+2A before -1A within horizon) recomputed from
    raw bars; returns (label, bars_to_resolution) with NaN in the trailing
    `horizon` rows. Feeds the Beta posterior outcome stream."""
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    a = atr_abs.to_numpy(dtype=float)
    n = len(bars)
    lab = np.full(n, np.nan)
    res = np.full(n, np.nan)
    for i in range(n - horizon):
        if not np.isfinite(a[i]) or a[i] <= 0:
            continue
        up = close[i] + 2.0 * a[i]
        dn = close[i] - 1.0 * a[i]
        lab[i], res[i] = 0, horizon
        for j in range(i + 1, i + 1 + horizon):
            tou, tod = high[j] >= up, low[j] <= dn
            if tou or tod:
                lab[i] = 1 if (tou and not tod) else 0
                res[i] = j - i
                break
    return (pd.Series(lab, index=bars.index),
            pd.Series(res, index=bars.index))


# ─── Step 3 ablation runner (§12) ────────────────────────────────────
def _ablation_report(symbol: str, timeframe: str) -> None:
    from model_trainer_v2 import ablation_table, load_event_dataset

    ds = load_event_dataset(symbol, timeframe)
    print(f"[ablation P] {symbol} {timeframe}: building Family P features...")
    pfeat = build_probability_theory_features(
        ds["bars"], timeframe, ds["macro"], symbol)
    ablation_table(ds["events"], ds["fm"], ds["baseline_cols"],
                   pfeat, FEATURE_GROUPS, ds["horizon"])


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        _ablation_report(args[0], args[1])
    else:
        print("Usage: python probability_engine_v2.py SYMBOL TIMEFRAME")
