# probt v2 — Zone-Conditional Directional Probability Engine

*Honest final report. English original; Spanish mirror in [README.v2.es.md](README.v2.es.md).
v1 platform docs live in [README.md](README.md).*

## What v2 is

v1 asked, on every bar: *"will price move +2·ATR before −1·ATR?"* — and honestly reported
no edge (Brier ≈ 0.266 vs. 0.25 random).

v2 asks a sharper question, only at pressure points: **when price touches a supply/demand
zone, order block, or FVG — does the zone hold (bounce) or fail (break)?** Two separate
models per (symbol, timeframe), trained on zone-touch events only, with purged
walk-forward validation, Platt calibration, split-conformal intervals, and per-event-payoff
Kelly sizing.

## Executive summary (read this before trusting any number)

1. **The break model has real, calibrated skill.** XAUUSD 1H out-of-sample: Brier 0.194,
   **BSS +0.153**, **ROC-AUC 0.740**, ECE 0.040, well-calibrated in calm/normal/stressed
   VIX regimes alike. 4H confirms directionally (BSS +0.116, AUC 0.700).
2. **Most of that skill is geometry, priced correctly by the market.** The single feature
   "distance from entry to the break target" alone scores AUC 0.74. When the model is most
   confident a zone breaks, the remaining move (the reward) is smallest. Consequence:
3. **There are ~zero EV-positive trades after realistic costs.** The cost-sensitivity
   backtest (spread 0.04 ATR + slippage 0.02 ATR, EV-gated with per-event payoff) finds
   0 break trades and 7 bounce trades on the 1H OOS set. The engine's honest output is
   *odds*, and the odds say: almost never bet on these exact barrier definitions.
4. **The bounce model — the actually tradeable fixed 2R:1R question — has no skill.**
   AUC ≈ 0.49–0.52 on both timeframes, negative BSS, and **no feature family fixed it**.
   It ships flagged `live_eligible: false` and the UI shows its base rate labeled
   "not a probability".
5. **Failed experiments, reported per §11 G4:** Family P (Bayesian/Markov/copulas/EVT) —
   no measurable edge. Family Q signal processing (FFT/wavelet) — no edge (Hilbert
   instantaneous frequency is the one flagged lead: +0.009..0.012 BSS on 1H bounce,
   insufficient to rescue the model). Family Q quantum-translated Q1–Q5, Q7, Q8 — flat to
   negative, exactly the walk-forward degradation the QCML literature warns about.
   **Q6 (zone oscillator energy) is the sole quantum-branch survivor** and a strong break
   feature. Q9 and EMD were not implemented (optional; their precondition — the branch
   clearing its gate — never happened).

## Final ablation (XAUUSD 1H, 3,436 events, one shared row set, cumulative)

| step | bounce BSS | bounce AUC | break BSS | break AUC |
|---|---|---|---|---|
| baseline (13 v1 features) | −0.009 | 0.514 | −0.004 | 0.529 |
| +Family P | −0.010 | 0.508 | +0.001 | 0.535 |
| +Q signal | −0.016 | 0.499 | +0.001 | 0.540 |
| +Q quantum | −0.010 | 0.496 | **+0.013** | 0.577 |
| +Z zone-strength | −0.008 | 0.502 | **+0.062** | 0.655 |

Retained live feature sets (fixed globally, no per-pair snooping —
`model_trainer_v2.LIVE_FEATURE_RECIPE`): bounce = baseline + `hilbert_inst_freq`;
break = baseline + Z1–Z8 + `q6_harmonic_oscillator_energy` + `break_target_dist_atr`.

## Final models

| | 1H bounce | 1H break | 4H bounce | 4H break |
|---|---|---|---|---|
| events | 4,677 | 3,882 | 1,195 | 999 |
| base rate | 24.5% | 35.6% | 23.8% | 40.5% |
| OOS Brier | 0.186 | 0.194 | 0.182 | 0.213 |
| OOS BSS | −0.004 | **+0.153** | −0.001 | **+0.116** |
| ROC-AUC | 0.493 | **0.740** | 0.517 | 0.700 |
| ECE (10 bins) | 0.010 | 0.040 | 0.013 | 0.037 |
| chosen C (P4 sweep) | 0.03 | 0.03 | 0.03 | 0.1 |
| conformal coverage check | 98.6% ✓ | 88.4% ✓ | 100% ✓ | **82.5% ✗** |
| live-eligible | **no** | yes | **no** | yes |

L1 LogReg beat all five calibrated benchmarks (XGBoost, RF, GradientBoost, AdaBoost,
GaussianNB) on break Brier under the identical protocol; none came within the 0.005
replacement rule (`benchmarks_v2.json`).

## Known limitations (each verified, none cosmetic)

- **Conformal intervals are near-vacuous.** The spec's nonconformity |p − y| against a
  binary y forces q_hat ≈ max(p, 1−p): measured 0.65–0.79 (v1's shipped model: 0.79).
  Bands are valid but ±0.7 wide, and the Kelly EV gate keyed to the lower bound is
  therefore permanently shut. **Recommendation:** probability-binned conformal or
  Venn-Abers intervals — a spec change, not made unilaterally.
- **4H break coverage failed (82.5% < 88%).** Surfaced as `interval_reliable: false` on
  the live reading; do not trust the 4H band.
- **Break skill ≠ break edge.** See executive summary point 2/3. The break probability's
  practical use is as a *veto* (don't fade a zone that is 70% likely to break), not a
  trade trigger.
- **E3 (close-inside trigger) is effectively dead** (n=1 in 13.7k bars): E1/E2 priority
  plus mitigation semantics subsume it. Kept, reported.
- **The central-bank calendar ships empty** (`engine/data/external/central_bank_calendar.json`).
  Fill in real FOMC/ECB dates or readings near decisions are NOT suppressed.
- **Live zone detection uses a 2,000-bar trailing window** — zones older than that are
  invisible to the live engine (they were visible in training). Acceptable for the 5
  freshest zones per side; documented in `live_engine_v2.py`.
- Data sources D1–D3 (FRED real yields, GLD flows, COT) were **not built**: no surviving
  model wanted macro features (the copula family failed its gate), so per YAGNI they wait
  until something needs them.

## §13 scorecard (targets vs. achieved, XAUUSD 1H/4H)

| target | achieved |
|---|---|
| OOS Brier < 0.20 both models | break yes (0.194/0.213 vs base-rate-matched climatology); bounce numerically yes but skill-free |
| BSS > 0.10 | break yes (+0.153/+0.116); bounce no |
| ROC-AUC > 0.60 | break yes (0.740/0.700); bounce no |
| 90% coverage ≥ 88% | 1H yes; 4H break no (82.5%) |
| ablation "P + Z carry the lift" | half right: Z carried it; P failed entirely |
| frontend never fakes a signal | yes — no_signal is the default state, every non-model number is tagged |

## Running v2

```bash
cd engine
# events + labels (writes zone_events.csv into the pair dir)
uv run python bounce_break_labeler.py XAUUSD 1H
# per-family ablations (reproduce every table above)
uv run python probability_engine_v2.py XAUUSD 1H
uv run python quantum_features.py XAUUSD 1H signal
uv run python quantum_features.py XAUUSD 1H quantum
uv run python zone_features.py XAUUSD 1H
# full dual training + all report files
uv run python model_trainer_v2.py XAUUSD 1H
# live event-driven reading (CLI)
uv run python live_engine_v2.py XAUUSD 1H
```

API: `GET /api/reading_v2?symbol=XAUUSD&timeframe=1H` (states `no_signal` / `blackout` /
`signal`). Dashboard: the "Zone Reading (v2)" card at the top of the main page.

Per pair, training writes: `model_bounce.pkl`, `model_break.pkl`, `conformal_*.pkl`,
`features_v2.json`, `metrics_v2.json`, `ablation_report.json`, `regime_slice_report.json`,
`cost_sensitivity.json`, `coefficient_audit.json`, `benchmarks_v2.json`,
`reliability_diagram_{bounce,break}.png`. v1 files (`model.pkl`, `features.json`) are
untouched — both engines coexist.

## Addendum (2026-07-02): trader-focused frontend + usable uncertainty bands

Applied after the Step 10 report, at the product owner's request:

- **Frontend redesigned around the decision.** The dashboard now shows only:
  live price (probability removed from the ticker — it was the v1 no-edge model's),
  the SMC chart with zones, the Zone Reading card, a Regime & News context card
  (labeled "context only"), and settings. Removed: KPI grid, backtest equity chart,
  probability histogram, AI insight cards, market overview, trades table, model zoo,
  gauges (§9.3 bans them), heatmap, feature-importance section, and the sidebar's
  hardcoded "+68.0 R Total Profit". Component files remain on disk; only the page
  composition changed.
- **Wilson calibration bands replace the vacuous conformal display.** The reading's
  band is now the 90% Wilson CI of the observed OOS hit rate in the prediction's
  calibration quintile — e.g. 1H break top bin: model says >48%, reality delivered
  65.2% [62.1%, 68.2%]. The spec's conformal q_hat is still computed and stored
  (plus a Mondrian-binned variant, measured and found only marginally tighter —
  the |p−y| score against a binary outcome is wide by construction). The Kelly EV
  gate now keys off a meaningful lower bound instead of being mechanically shut.
- **Per-reading feature contributions (§9.2 Card 3).** Each calibrated tier lists
  its top-6 |coefficient × standardized value| drivers, signed — making the
  geometry dominance of the break model visible instead of implied.
- **Tested and rejected:** event-type dummy features (E1/E2 vs E4 vs E5 as inputs)
  — ΔBSS +0.0002…+0.0025, far below the +0.01 gate on both TFs; the geometry
  features already encode it. Reported per G4, not adopted.
- **Session/time-of-day features — tested, sub-gate, not adopted.** Cyclic
  hour-of-day encoding was the strongest lead all project: 1H bounce ΔBSS +0.0060
  with AUC 0.494→0.544 (the only thing that ever moved the bounce model), 1H break
  ΔBSS +0.0073; nothing on 4H. Both fall short of the +0.01 gate that rejected
  weaker candidates, so they are not in the live models — flagged as the FIRST
  re-test when tick data expands the event count.
- **Macro positioning feeds built (`engine/external_data.py`, §6 D1+D3):** FRED
  DFII10 real yields (daily, keyless) and CFTC COT managed-money gold positioning
  (weekly, publication-lagged 4 days for causality), cached under `data/external/`
  with a manifest. As MODEL features they fail the gate on every model/timeframe
  (ΔBSS +0.0001…−0.0054) — recorded in the module docstring. They ship as REGIME
  CONTEXT only: the dashboard's Regime & News card shows real-yields 1d change and
  COT crowding percentile, labeled "context, not signal".
- **Self-grading signal journal (`engine/signal_journal.py` + `/api/journal` +
  the Live Track Record card).** Every live SIGNAL reading is journaled with its
  barrier prices frozen at signal time; once the horizon elapses it is graded
  against what price actually did, and the dashboard reports predicted-vs-observed
  ("said 45% → got 41%, n=63") overall and by probability tercile. No backfilling
  from training history — only genuinely post-training signals count — and rates
  are withheld below 20 graded signals. This is the earliest-warning system for
  calibration drift and the foundation for meta-labeling the trader's own entries.

probt v2 does what a serious probability engine is supposed to do: it found one
well-calibrated, regime-stable signal (zone breaks), proved that most of its strength is
geometry the market already prices, showed that the fixed-R bounce trade has no detectable
edge on 2.4 years of data, and wired every one of those facts — including the
unflattering ones — into the numbers the operator sees. It does not promise
profitability. It promises honest odds, and it keeps that promise by mostly saying
"no bet".
