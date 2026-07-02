"""probt v2 — Family Q: quantum-inspired signal processing + real-quantum-
theory-translated features. All run on classical hardware (numpy/scipy) —
this module does NOT provide quantum speedup and does NOT use quantum
hardware. See master-prompt §0 ground rule 5 and §4.2/§4.3: these are
legitimate classical mathematics that happen to share structure with
quantum-mechanical formalism, applied to time-series inference.

SKELETON ONLY — signatures + docstrings, no computation. Per §0 ground rule
6, wait for product-owner review before implementing any function body.
Every function currently raises NotImplementedError.

Hypothesis: moving averages and RSI are time-domain, first/second-moment
statistics. Spectral (FFT), scale (wavelet), phase (Hilbert) and
information-geometric (Fisher, entropy, fidelity) transforms may surface
structure — dominant cycles, self-similarity, regime transitions — that is
invisible in the time domain, ahead of a visible price breakout. Family Q
features are candidate REGIME FILTERS first, signals second; §4.3 Q8 reports
Cohen's d dropping from 0.83 (in-sample) to 0.26 (walk-forward) in the
source paper (Hammond 2605.17117) — treat every Family Q feature with that
level of skepticism until it clears the ablation gate (§8 V6).

Observed ablation effect — signal-processing branch (2026-07-01, XAUUSD 1H
n=4677 events / 4H n=1195, purged walk-forward, L1 LogReg C=0.3 + Platt):
DOES NOT CLEAR THE BRANCH GATE (§8 V6). 1H bounce stays below climatology
at every step (best cumulative BSS -0.0072); break improves marginally
(+0.0024 fft, +0.0020 cwt, AUC 0.515->0.536) but far below +0.01. On 4H all
deltas are ~0 or negative. Single bright spot: hilbert_inst_freq alone adds
+0.009..+0.012 BSS on 1H bounce regardless of insertion order without
degrading AUC — flagged for re-test in the Step 7 combined ablation, but
the branch as a whole is NOT wired into the live pipeline. EMD was skipped
(PyEMD not installed; optional per §4.2 and moot while the branch fails).
Reproduce: `python quantum_features.py SYMBOL TF`.

Observed ablation effect — real-quantum branch Q1-Q8 (2026-07-02, XAUUSD 1H
n=4257 / 4H n=1119 events, purged walk-forward): ONE retained feature.
q6_harmonic_oscillator_energy CLEARS the gate on the BREAK model: isolated
on baseline, 1H dBSS +0.0151 (BSS -> +0.0177, AUC 0.539 -> 0.582); 4H dBSS
+0.0103 (AUC 0.484 -> 0.521). No effect on bounce. Everything else
(info-geometry Q1/Q2/Q3/Q8, forward-sim Q4/Q5, macro-MI Q7) is flat to
negative on both labels — as the Hammond paper's walk-forward degradation
predicted for regime observables. Q9 not implemented (optional; the branch
as a whole did not clear, and Q6 is retained as a single feature, not a
branch). Reproduce: `python quantum_features.py SYMBOL TF quantum`.

FEATURES:

  Signal processing (§4.2) — computed on a trailing window of log-returns:
    fft_spectral_entropy      Shannon entropy of the normalized FFT power
                              spectrum of the last 128 log-returns. High =
                              broadband noise; low = dominant cycle present.
    fft_dominant_freq         Frequency bin of maximum spectral power.
    fft_dominant_amp_ratio    Amplitude of the dominant bin / noise-floor
                              amplitude (mean of the rest of the spectrum).
    cwt_energy_ratio_short    Morlet CWT (scales 2..64) energy in the
                              short-scale band (2..8) / total energy.
    cwt_max_energy_scale      Scale index of maximum instantaneous wavelet
                              energy.
    cwt_hurst_exponent        Wavelet-based Hurst exponent (self-similarity)
                              estimated from the CWT scale/energy slope.
    hilbert_inst_freq         Instantaneous frequency (derivative of
                              instantaneous phase) of the analytic signal
                              via the Hilbert transform.
    emd_fast_energy_fraction  OPTIONAL (§4.2, ablation-gated only). Energy
                              fraction in the lowest-order IMFs (fastest
                              oscillations) from Empirical Mode
                              Decomposition of the last 200 log-returns.

  Real-quantum-theory-translated (§4.3) — Q1..Q9:
    q1_von_neumann_entropy    Entropy of the normalized eigenvalue spectrum
                              of the rolling feature-correlation matrix.
                              High = diverse/uncorrelated feature
                              information (good); low = redundancy (warn).
    q2_quantum_fisher_info    Classical Fisher information of the rolling
                              return distribution w.r.t. (mean, variance),
                              averaged across parameters. Spikes on regime
                              shift (distribution becomes locally sharper).
    q3_berry_phase_fidelity   Overlap fidelity |<psi(t-1)|psi(t)>|^2 between
                              consecutive leading eigenvectors of the
                              rolling feature-correlation matrix. A
                              sustained drop signals a topological change
                              in feature relations.
    q4_fokker_planck_edge_prob  Integral of the Fokker-Planck-evolved
                              forward-return density above +2*ATR minus the
                              integral below -1*ATR, given HMM-regime
                              (mu, sigma) and horizon N bars.
    q5_path_integral_hit_prob  Fraction of `n_paths` simulated GBM paths
                              (HMM-mixture mu/sigma) that hit +2*ATR before
                              -1*ATR within the horizon.
    q6_harmonic_oscillator_energy  ((price - zone_midpoint) / zone_half_width)^2
                              while price is inside a supply/demand zone;
                              NaN otherwise. Low = trapped/mean-reverting;
                              high = escaping/breakout-imminent. Requires
                              zone bounds from zone_events.py — computed
                              here as a pure function of (price, zone_top,
                              zone_bottom) so this module has no dependency
                              on zone_events.
    q7_mutual_info_macro      Copula-based mutual information I(gold ;
                              {DXY, VIX}) over the rolling window. High =
                              gold currently trades as a function of macro;
                              low = SMC/technicals-only regime.
    q8_hamiltonian_spectral_entropy   Spectral entropy of an effective
                              Hamiltonian built from the rolling feature
                              covariance matrix (Hammond 2605.17117-style).
    q8_berry_phase_rate       Rate of Berry-phase accumulation of the
                              Hamiltonian's ground state across bars.
    q8_reduced_state_purity   Purity Tr(rho^2) of the reduced density
                              matrix built from the Hamiltonian ground
                              state. All three q8_* are REGIME FILTERS,
                              not primary signals (§4.3 Q8).
    q9_qft_boundary_score     OPTIONAL (ablation-gated only, small feature
                              vectors n=6..8). Simulated Quantum Fourier
                              Transform of the amplitude-encoded feature
                              vector; score = whether the transformed
                              representation improves the decision boundary
                              vs. fft_* on the same window. Report failure
                              if it does not beat FFT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "fft_spectral_entropy",
    "fft_dominant_freq",
    "fft_dominant_amp_ratio",
    "cwt_energy_ratio_short",
    "cwt_max_energy_scale",
    "cwt_hurst_exponent",
    "hilbert_inst_freq",
    "emd_fast_energy_fraction",  # optional, ablation-gated
    "q1_von_neumann_entropy",
    "q2_quantum_fisher_info",
    "q3_berry_phase_fidelity",
    "q4_fokker_planck_edge_prob",
    "q5_path_integral_hit_prob",
    "q6_harmonic_oscillator_energy",
    "q7_mutual_info_macro",
    "q8_hamiltonian_spectral_entropy",
    "q8_berry_phase_rate",
    "q8_reduced_state_purity",
    "q9_qft_boundary_score",  # optional, ablation-gated
]

SIGNAL_FEATURES: list[str] = FEATURES[:7]  # fft_* + cwt_* + hilbert

# Ablation groups, §12 Step 4 order (emd is appended by the runner only if
# PyEMD is installed — it is optional per §4.2).
SIGNAL_FEATURE_GROUPS: dict[str, list[str]] = {
    "fft": ["fft_spectral_entropy", "fft_dominant_freq", "fft_dominant_amp_ratio"],
    "cwt": ["cwt_energy_ratio_short", "cwt_max_energy_scale", "cwt_hurst_exponent"],
    "hilbert": ["hilbert_inst_freq"],
}

Q_WINDOW = 128          # bars of log-returns per FFT/CWT/Hilbert window
CWT_SCALES = (2, 64)    # Morlet scale range (inclusive)
CWT_SHORT_BAND = 8      # scales 2..8 = "short-scale" energy band


# ─── 1. Signal processing (§4.2) ────────────────────────────────────
def fft_features(log_returns: np.ndarray) -> tuple[float, float, float]:
    """FFT of one demeaned window of log-returns (no taper — with a 128-bar
    rectangular window the Rayleigh frequency resolution is df = 1/128
    cycles/bar at dt = 1 bar, so dt*df = 1/128 ... but the Heisenberg-Gabor
    bound constrains the JOINT localization product sigma_t*sigma_f >=
    1/(4*pi) ~= 0.0796; a length-128 rectangular window has sigma_t ~=
    128/sqrt(12) ~= 36.9 bars and sigma_f ~= 0.0027 c/b -> product ~= 0.10,
    above the bound — the analysis is physically realizable, no bug).

    Returns (spectral_entropy in [0,1] — Shannon entropy of the normalized
    power spectrum excl. DC, / log(n_bins); dominant_freq in cycles/bar
    (0, 0.5]; dominant_amp_ratio = log1p(peak power / mean non-peak power),
    log-scaled for stationarity per §3 P3).
    """
    r = log_returns - log_returns.mean()
    p = np.abs(np.fft.rfft(r)) ** 2
    p = p[1:]  # drop DC
    tot = float(p.sum())
    if tot <= 0 or len(p) < 3:
        return float("nan"), float("nan"), float("nan")
    q = p / tot
    ent = float(-(q * np.log(q + 1e-12)).sum() / np.log(len(q)))
    k = int(np.argmax(p))
    freqs = np.fft.rfftfreq(len(r))[1:]
    noise = (tot - p[k]) / (len(p) - 1)
    ratio = float(np.log1p(p[k] / noise)) if noise > 0 else float("nan")
    return ent, float(freqs[k]), ratio


def wavelet_features(log_returns: np.ndarray) -> tuple[float, float, float]:
    """Morlet CWT (pywt, scales CWT_SCALES) of one window of log-returns.

    Returns:
      energy_ratio_short  energy at scales 2..CWT_SHORT_BAND / total, [0,1]
      max_energy_scale    log2(scale of max energy at the LAST time column,
                          i.e. "instantaneous" — edge effects from the cone
                          of influence are accepted and identical for every
                          window, so cross-window comparisons stay fair),
                          normalized by log2(max scale) -> [0,1]
      hurst_exponent      wavelet Hurst: for fBm, E|W(a)|^2 ~ a^(2H+1), so
                          H = (slope of log2 E vs log2 a - 1)/2, clipped to
                          [0,1]. ~0.5 random walk, >0.5 trending/persistent,
                          <0.5 mean-reverting.
    """
    import pywt

    scales = np.arange(CWT_SCALES[0], CWT_SCALES[1] + 1)
    coef, _ = pywt.cwt(log_returns, scales, "morl")
    power = np.abs(coef) ** 2                      # (n_scales, n_time)
    e_scale = power.mean(axis=1)
    tot = float(e_scale.sum())
    if tot <= 0:
        return float("nan"), float("nan"), float("nan")
    short = float(e_scale[: CWT_SHORT_BAND - CWT_SCALES[0] + 1].sum() / tot)
    k = int(np.argmax(power[:, -1]))
    max_scale = float(np.log2(scales[k]) / np.log2(scales[-1]))
    slope = float(np.polyfit(np.log2(scales), np.log2(e_scale + 1e-300), 1)[0])
    hurst = float(np.clip((slope - 1.0) / 2.0, 0.0, 1.0))
    return short, max_scale, hurst


def hilbert_instantaneous_frequency(log_close: np.ndarray) -> float:
    """Instantaneous frequency at the window's end: linear-detrend the
    log-price window (the analytic signal of a trending series is dominated
    by the trend, drowning the phase), take scipy.signal.hilbert, unwrap the
    phase, and return the median of the last 5 phase increments / 2*pi
    (median for edge stability). Units: cycles/bar."""
    from scipy.signal import hilbert

    x = log_close - np.polyval(
        np.polyfit(np.arange(len(log_close)), log_close, 1),
        np.arange(len(log_close)))
    phase = np.unwrap(np.angle(hilbert(x)))
    inst = np.diff(phase) / (2.0 * np.pi)
    return float(np.median(inst[-5:]))


def emd_fast_energy_fraction(log_returns: np.ndarray) -> float:
    """OPTIONAL (§4.2, ablation-gated): energy fraction of the two
    lowest-order (fastest) IMFs from PyEMD.EMD. Lazy import — returns NaN
    when PyEMD is not installed (it is not a project dependency unless this
    feature clears the gate)."""
    try:
        from PyEMD import EMD
    except ImportError:
        return float("nan")
    imfs = EMD()(log_returns)
    if imfs.shape[0] < 3:
        return float("nan")
    e = (imfs ** 2).sum(axis=1)
    tot = float(e.sum())
    return float(e[:2].sum() / tot) if tot > 0 else float("nan")


def build_signal_features(
    bars: pd.DataFrame,
    at_indices: list[int] | None = None,
    window: int = Q_WINDOW,
    include_emd: bool = False,
) -> pd.DataFrame:
    """Assemble the §4.2 signal-processing features, aligned to bars.index.

    CWT is O(n_scales * window) per bar, so computing at all 13k+ bars is
    wasteful when only event bars are needed — `at_indices` restricts
    computation to those positional indices (None = every bar; the live
    engine passes just the last index). Non-computed rows stay NaN.
    """
    import feature_engineer

    bars = feature_engineer._clean_index(bars)
    logc = np.log(bars["close"].to_numpy(dtype=float))
    r = np.diff(logc, prepend=np.nan)
    n = len(bars)
    cols = SIGNAL_FEATURES + (["emd_fast_energy_fraction"] if include_emd else [])
    vals = np.full((n, len(cols)), np.nan)

    idxs = range(window, n) if at_indices is None else at_indices
    for i in idxs:
        if i < window or i >= n:
            continue
        w = r[i - window + 1: i + 1]
        if not np.all(np.isfinite(w)):
            continue
        vals[i, 0:3] = fft_features(w)
        vals[i, 3:6] = wavelet_features(w)
        vals[i, 6] = hilbert_instantaneous_frequency(logc[i - window + 1: i + 1])
        if include_emd:
            vals[i, 7] = emd_fast_energy_fraction(w)
    return pd.DataFrame(vals, index=bars.index, columns=cols)


# ─── 2. Real-quantum-theory-translated (§4.3, Q1-Q9) ────────────────
Q2_SHORT = 32  # short window compared against the full Q_WINDOW for Fisher info

Q_STATE_GROUPS: dict[str, list[str]] = {
    "info_geom": ["q1_von_neumann_entropy", "q2_quantum_fisher_info",
                  "q3_berry_phase_fidelity", "q8_hamiltonian_spectral_entropy",
                  "q8_berry_phase_rate", "q8_reduced_state_purity"],
    "forward_sim": ["q4_fokker_planck_edge_prob", "q5_path_integral_hit_prob"],
    "macro_mi": ["q7_mutual_info_macro"],
    "zone_energy": ["q6_harmonic_oscillator_energy"],
}


def _spectral(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) + eigenvectors of a symmetric matrix with
    NaN-safety."""
    m = np.nan_to_num(mat, nan=0.0)
    return np.linalg.eigh(m)


def von_neumann_entropy(corr: np.ndarray) -> float:
    """Q1. Normalized eigenvalues of a correlation matrix as a probability
    distribution -> S = -sum p ln p, normalized by ln(k) to [0,1]. High =
    features carry diverse information; low = redundancy."""
    w, _ = _spectral(corr)
    w = np.clip(w, 0.0, None)
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    p = w / tot
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def quantum_fisher_information(returns: np.ndarray, short: int = Q2_SHORT) -> float:
    """Q2. Gaussian Fisher-information quadratic form of the (mean, variance)
    shift between the trailing `short` window and the full window:
    F = (dmu^2 / var) + (dvar^2 / (2 var^2)). Spikes when the recent return
    distribution becomes easy to distinguish from its own past = regime
    shift. log1p-compressed for stationarity (§3 P3)."""
    long_w = returns
    short_w = returns[-short:]
    v = float(long_w.var())
    if v <= 0:
        return float("nan")
    dmu = float(short_w.mean() - long_w.mean())
    dv = float(short_w.var() - v)
    return float(np.log1p(dmu * dmu / v + dv * dv / (2.0 * v * v)))


def berry_phase_fidelity(corr_prev: np.ndarray, corr_cur: np.ndarray) -> float:
    """Q3. Overlap fidelity |<psi(t-1)|psi(t)>|^2 of the LEADING eigenvector
    of consecutive rolling correlation matrices. Sustained drop = the
    dominant feature-relation direction is rotating (topological change)."""
    _, v_prev = _spectral(corr_prev)
    _, v_cur = _spectral(corr_cur)
    return float(np.dot(v_prev[:, -1], v_cur[:, -1]) ** 2)


def fokker_planck_edge_probability(
    mu: float, sigma: float, up_ret: float, dn_ret: float, horizon_bars: int,
) -> float:
    """Q4. P(terminal log-return > up_ret) - P(terminal < dn_ret) after
    `horizon_bars` of drift-diffusion. With constant per-bar (mu, sigma) the
    Fokker-Planck terminal density is Gaussian N(H*mu, H*sigma^2), so the
    integrals are analytic — same object as numerically integrating the FP
    equation, minus the discretization error.
    # ponytail: rolling-window (mu, sigma) instead of an HMM regime
    # posterior; add hmmlearn mixture if this feature ever clears the gate.
    """
    from scipy.stats import norm

    s = sigma * np.sqrt(horizon_bars)
    if s <= 0:
        return float("nan")
    m = mu * horizon_bars
    return float(norm.sf((up_ret - m) / s) - norm.cdf((dn_ret - m) / s))


def path_integral_hit_probability(
    mu: float, sigma: float, up_ret: float, dn_ret: float, horizon_bars: int,
    n_paths: int = 1000, seed: int = 0,
) -> float:
    """Q5. Fraction of simulated log-return paths (per-bar N(mu, sigma)
    increments) hitting up_ret before dn_ret within the horizon. Unlike Q4
    this respects path order (first passage), not just the terminal
    density. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    cum = rng.normal(mu, sigma, size=(n_paths, horizon_bars)).cumsum(axis=1)
    hit_up = cum >= up_ret
    hit_dn = cum <= dn_ret
    t_up = np.where(hit_up.any(axis=1), hit_up.argmax(axis=1), horizon_bars + 1)
    t_dn = np.where(hit_dn.any(axis=1), hit_dn.argmax(axis=1), horizon_bars + 1)
    return float((t_up < t_dn).mean())


def harmonic_oscillator_energy(
    price: float, zone_top: float, zone_bottom: float,
) -> float:
    """Q6. ((price - midpoint) / half_width)^2 for the zone [bottom, top].
    <1 inside the zone (trapped/mean-reverting), >1 outside (escaping).
    Clipped at 25 — beyond 5 half-widths the magnitude carries no extra
    information and would dominate the scaler. NaN on degenerate bounds."""
    if not (zone_top > zone_bottom):
        return float("nan")
    half = (zone_top - zone_bottom) / 2.0
    mid = (zone_top + zone_bottom) / 2.0
    return float(min(((price - mid) / half) ** 2, 25.0))


def mutual_info_macro(
    gold: np.ndarray, dxy: np.ndarray, vix: np.ndarray,
) -> float:
    """Q7. Gaussian-copula mutual information I(gold ; {DXY, VIX}) on one
    window of paired daily returns: rank-correlate (Spearman -> Gaussian
    rho = 2 sin(pi*rho_s/6)), then MI = -0.5 ln(det R / det R_macro).
    High = gold currently trades as a function of macro; low = technicals-
    only regime."""
    from scipy.stats import spearmanr

    X = np.column_stack([gold, dxy, vix])
    rho_s = spearmanr(X).statistic
    R = 2.0 * np.sin(np.pi * np.asarray(rho_s) / 6.0)
    np.fill_diagonal(R, 1.0)
    det_full = float(np.linalg.det(R))
    det_macro = float(np.linalg.det(R[1:, 1:]))
    if det_full <= 0 or det_macro <= 0:
        return float("nan")
    return float(-0.5 * np.log(det_full / det_macro))


def hamiltonian_regime_observables(
    cov_prev: np.ndarray, cov_cur: np.ndarray,
) -> tuple[float, float, float]:
    """Q8. Treat the rolling feature COVARIANCE matrix as the effective
    Hamiltonian (Hammond 2605.17117-style). Returns:
      spectral_entropy   entropy of its normalized eigenvalue spectrum [0,1]
      berry_phase_rate   1 - |<ground(t-1)|ground(t)>|^2 of the ground state
                         (lowest-eigenvalue eigenvector) — how fast the
                         least-excited feature direction is rotating
      purity             Tr(rho^2) = sum p_i^2 of the trace-normalized
                         spectrum (participation ratio) — 1/k = maximally
                         mixed, 1 = one dominant mode.
    # ponytail: the paper's full QCML Hamiltonian construction (learned
    # couplings) is replaced by the raw covariance; build the learned
    # version only if these observables clear the ablation gate.
    Regime FILTERS, not signals — walk-forward Cohen's d in the source
    paper drops 0.83 -> 0.26.
    """
    w, v = _spectral(cov_cur)
    w_prev, v_prev = _spectral(cov_prev)
    wc = np.clip(w, 0.0, None)
    tot = wc.sum()
    if tot <= 0:
        return float("nan"), float("nan"), float("nan")
    p = wc / tot
    s = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    rate = float(1.0 - np.dot(v_prev[:, 0], v[:, 0]) ** 2)
    purity = float((p * p).sum())
    return s, rate, purity


def build_quantum_state_features(
    bars: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    timeframe: str,
    macro: pd.DataFrame | None = None,
    at_indices: list[int] | None = None,
    window: int = Q_WINDOW,
    seed: int = 42,
) -> pd.DataFrame:
    """Q1-Q5, Q7, Q8 aligned to bars.index (Q6 is zone-conditional — the
    caller computes it per event from the zone bounds). `feature_matrix` =
    the v1 universal features (the 'system state' whose correlation/
    covariance structure Q1/Q3/Q8 observe). Q7 needs `macro`; NaN without.
    `at_indices` restricts computation to event bars, as in
    build_signal_features."""
    import indicators
    from feature_engineer import _clean_index

    bars = _clean_index(bars)
    n = len(bars)
    cols = ["q1_von_neumann_entropy", "q2_quantum_fisher_info",
            "q3_berry_phase_fidelity", "q4_fokker_planck_edge_prob",
            "q5_path_integral_hit_prob", "q7_mutual_info_macro",
            "q8_hamiltonian_spectral_entropy", "q8_berry_phase_rate",
            "q8_reduced_state_purity"]
    vals = np.full((n, len(cols)), np.nan)

    F = feature_matrix.reindex(bars.index).to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    r = np.diff(np.log(close), prepend=np.nan)
    atr_abs = indicators.atr(bars[["high", "low", "close"]], 14).to_numpy(dtype=float)
    from symbols import normalize_timeframe, tf_horizon
    horizon = tf_horizon(normalize_timeframe(timeframe))

    # Q7 daily series, mapped from the last completed day (as in Family P)
    q7_by_day = None
    if macro is not None and not macro.empty and {"dxy", "vix"} <= set(macro.columns):
        dg = np.log(bars["close"].resample("D").last().dropna()).diff()
        dg.index = dg.index.normalize()
        mac = macro.copy()
        mac.index = pd.to_datetime(mac.index).normalize()
        dd = np.log(mac["dxy"].astype(float)).diff()
        dv = np.log(mac["vix"].astype(float)).diff()
        joined = pd.concat({"g": dg, "d": dd, "v": dv}, axis=1, sort=False).dropna()
        vals_q7 = np.full(len(joined), np.nan)
        arr = joined.to_numpy(dtype=float)
        for t in range(60, len(joined)):
            s = max(0, t - 120 + 1)
            vals_q7[t] = mutual_info_macro(arr[s:t + 1, 0], arr[s:t + 1, 1],
                                           arr[s:t + 1, 2])
        q7_by_day = pd.Series(vals_q7, index=joined.index).shift(1)

    def _win_ok(i: int) -> bool:
        return i >= window and np.all(np.isfinite(F[i - window + 1: i + 1]))

    idxs = range(window + 1, n) if at_indices is None else at_indices
    for i in idxs:
        if i <= window or i >= n:
            continue
        w_r = r[i - window + 1: i + 1]
        if np.all(np.isfinite(w_r)):
            vals[i, 1] = quantum_fisher_information(w_r)
            a = atr_abs[i]
            if np.isfinite(a) and a > 0 and close[i] > 0:
                up = np.log1p(2.0 * a / close[i])
                dn = np.log1p(-1.0 * a / close[i])
                mu, sig = float(w_r.mean()), float(w_r.std())
                vals[i, 3] = fokker_planck_edge_probability(mu, sig, up, dn, horizon)
                vals[i, 4] = path_integral_hit_probability(
                    mu, sig, up, dn, horizon, seed=seed + i)
        if _win_ok(i) and _win_ok(i - 1):
            Xc = F[i - window + 1: i + 1]
            Xp = F[i - window: i]
            corr_c = np.corrcoef(Xc, rowvar=False)
            corr_p = np.corrcoef(Xp, rowvar=False)
            vals[i, 0] = von_neumann_entropy(corr_c)
            vals[i, 2] = berry_phase_fidelity(corr_p, corr_c)
            vals[i, 6:9] = hamiltonian_regime_observables(
                np.cov(Xp, rowvar=False), np.cov(Xc, rowvar=False))
        if q7_by_day is not None:
            day = bars.index[i].normalize()
            if day in q7_by_day.index:
                vals[i, 5] = q7_by_day.loc[day]
            else:  # weekend/holiday bar: last completed macro day
                prior = q7_by_day.index[q7_by_day.index <= day]
                if len(prior):
                    vals[i, 5] = q7_by_day.loc[prior[-1]]
    return pd.DataFrame(vals, index=bars.index, columns=cols)


def qft_boundary_score(
    feature_vector: np.ndarray, fft_baseline_score: float,
) -> float:
    """Q9. OPTIONAL — only implement if Family Q clears its ablation gate
    and there is budget for it (§4.3). Simulate a Quantum Fourier Transform
    on an amplitude-encoded small feature vector (n=6..8) via qiskit-aer or
    a hand-rolled unitary matrix (NEVER real quantum hardware, §0 ground
    rule 5) and compare the resulting decision-boundary score against
    `fft_baseline_score`. Report failure if it does not beat FFT.
    """
    raise NotImplementedError


def build_quantum_features(
    bars: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    timeframe: str,
    macro: pd.DataFrame | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Assemble all Family Q features into a single DataFrame aligned to
    `bars.index`, mirroring feature_engineer.build_features' calling
    convention. q6_harmonic_oscillator_energy is left NaN here (bar-level
    build has no zone context) — populated at inference/labeling time by
    the zone-conditional callers instead.

    Signal-processing columns are implemented (build_signal_features);
    Q1-Q9 land in Step 5 of the build order.
    """
    raise NotImplementedError


# ─── Step 4 ablation runner (§12) ────────────────────────────────────
def _ablation_report_signal(symbol: str, timeframe: str) -> None:
    from model_trainer_v2 import ablation_table, load_event_dataset

    ds = load_event_dataset(symbol, timeframe)
    events = ds["events"]
    idxs = sorted(set(int(i) for i in events["bar_index"]))
    try:
        import PyEMD  # noqa: F401
        include_emd = True
    except ImportError:
        include_emd = False
        print("  (emd group skipped — PyEMD not installed; optional per §4.2)")
    print(f"[ablation Q-signal] {symbol} {timeframe}: computing "
          f"FFT/CWT/Hilbert at {len(idxs)} event bars...")
    qfeat = build_signal_features(ds["bars"], at_indices=idxs,
                                  include_emd=include_emd)
    groups = dict(SIGNAL_FEATURE_GROUPS)
    if include_emd:
        groups["emd"] = ["emd_fast_energy_fraction"]
    ablation_table(events, ds["fm"], ds["baseline_cols"], qfeat, groups,
                   ds["horizon"])


def _ablation_report_quantum(symbol: str, timeframe: str) -> None:
    from model_trainer_v2 import ablation_table, load_event_dataset

    ds = load_event_dataset(symbol, timeframe)
    events = ds["events"]
    idxs = sorted(set(int(i) for i in events["bar_index"]))
    print(f"[ablation Q-quantum] {symbol} {timeframe}: computing Q1-Q8 at "
          f"{len(idxs)} event bars...")
    qf = build_quantum_state_features(
        ds["bars"], ds["fm"][ds["baseline_cols"]], timeframe,
        macro=ds["macro"], at_indices=idxs)
    # Q6 is per-event (zone bounds differ across events on the same bar)
    close = ds["bars"]["close"].to_numpy(dtype=float)
    qf = qf.reindex(events.index)
    qf["q6_harmonic_oscillator_energy"] = [
        harmonic_oscillator_energy(close[int(t)], top, bot)
        for t, top, bot in zip(events["bar_index"], events["zone_top"],
                               events["zone_bottom"])
    ]
    ablation_table(events, ds["fm"], ds["baseline_cols"], qf,
                   Q_STATE_GROUPS, ds["horizon"])


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) >= 2:
        branch = args[2] if len(args) > 2 else "quantum"
        if branch == "signal":
            _ablation_report_signal(args[0], args[1])
        else:
            _ablation_report_quantum(args[0], args[1])
    else:
        print("Usage: python quantum_features.py SYMBOL TIMEFRAME [signal|quantum]")
