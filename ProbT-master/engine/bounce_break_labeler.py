"""probt v2 — directional-conditional Triple-Barrier labeling (§5.2).

Extends labeler.py's Triple-Barrier (labeler.label_matrix) from "every bar"
to "every zone-touch event", and splits the single label into two distinct,
independently-trained targets: does price bounce off the zone, or does it
break through? These are NOT complements of each other (a zone can be
retested repeatedly without a clean 2R move in either direction within the
horizon — both labels can be 0), so both are computed and trained separately
per (symbol, timeframe) — see model_trainer_v2.py.

Hypothesis: v1's Triple-Barrier answers "does price move 2R in either
direction from THIS bar" with no reference to structure. v2 asks a sharper
question anchored to the event: "given we are AT a demand zone right now,
does it hold (bounce) or fail (break)?" — a different, and presumably more
tradeable, question.

Observed ablation effect: TBD — this module produces the labels the whole
ablation pipeline is scored against; its own Step 2 gate is the label
balance printed by running this file as a script.

LABELS (one row per zone_events.ZoneEvent, not per bar), with
A = ATR at the event bar (indicators.atr(bars, 14), same as
feature_engineer), c = close at the event bar, H = tf_horizon(timeframe):

  demand touch:  label_bounce = 1 iff high hits c + 2A before low hits c - 1A, within H
                 label_break  = 1 iff low hits zone_bottom - 1A before high hits c + 2A, within H
  supply touch:  label_bounce = 1 iff low hits c - 2A before high hits c + 1A, within H
                 label_break  = 1 iff high hits zone_top + 1A before low hits c - 2A, within H

  label_bounce_bars / label_break_bars = bars to resolution (H if unresolved).

Tie-break (same as labeler.label_matrix): if one bar touches both the target
and its opposing barrier, the label is 0 — intrabar path order is unknown,
so the conservative outcome wins. Events without a complete H-bar forward
window (too close to the end of `bars`) are dropped, mirroring
labeler.label_matrix's trailing-horizon drop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from zone_events import ZoneEvent, detect_zone_events, events_to_frame

LABELS: list[str] = [
    "label_bounce",
    "label_break",
    "label_bounce_bars",
    "label_break_bars",
]


def _race_up_wins(
    high: np.ndarray, low: np.ndarray, start: int, horizon: int,
    win_up: float, lose_dn: float,
) -> tuple[int, int]:
    """Walk bars (start, start+horizon]; label=1 iff `win_up` is touched
    strictly before `lose_dn`. A bar touching both -> 0 (conservative
    tie-break, intrabar order unknown). Returns (label, bars_to_resolution),
    with bars = horizon when nothing resolves."""
    n = len(high)
    for j in range(start + 1, min(start + 1 + horizon, n)):
        hit_up = high[j] >= win_up
        hit_dn = low[j] <= lose_dn
        if hit_up and not hit_dn:
            return 1, j - start
        if hit_up or hit_dn:
            return 0, j - start
    return 0, horizon


def _race_down_wins(
    high: np.ndarray, low: np.ndarray, start: int, horizon: int,
    win_dn: float, lose_up: float,
) -> tuple[int, int]:
    """Mirror of _race_up_wins: the DOWNSIDE level is the win condition."""
    n = len(high)
    for j in range(start + 1, min(start + 1 + horizon, n)):
        hit_dn = low[j] <= win_dn
        hit_up = high[j] >= lose_up
        if hit_dn and not hit_up:
            return 1, j - start
        if hit_dn or hit_up:
            return 0, j - start
    return 0, horizon


def label_demand_touch(
    high: np.ndarray, low: np.ndarray, event_bar: int,
    close_t: float, zone_bottom: float, atr_abs: float, horizon: int,
) -> dict[str, int]:
    """Directional-conditional labels for a demand-zone touch (bounce = LONG)."""
    b, bb = _race_up_wins(high, low, event_bar, horizon,
                          win_up=close_t + 2.0 * atr_abs,
                          lose_dn=close_t - 1.0 * atr_abs)
    k, kb = _race_down_wins(high, low, event_bar, horizon,
                            win_dn=zone_bottom - 1.0 * atr_abs,
                            lose_up=close_t + 2.0 * atr_abs)
    return {"label_bounce": b, "label_bounce_bars": bb,
            "label_break": k, "label_break_bars": kb}


def label_supply_touch(
    high: np.ndarray, low: np.ndarray, event_bar: int,
    close_t: float, zone_top: float, atr_abs: float, horizon: int,
) -> dict[str, int]:
    """Mirror of label_demand_touch for a supply-zone touch (bounce = SHORT)."""
    b, bb = _race_down_wins(high, low, event_bar, horizon,
                            win_dn=close_t - 2.0 * atr_abs,
                            lose_up=close_t + 1.0 * atr_abs)
    k, kb = _race_up_wins(high, low, event_bar, horizon,
                          win_up=zone_top + 1.0 * atr_abs,
                          lose_dn=close_t - 2.0 * atr_abs)
    return {"label_bounce": b, "label_bounce_bars": bb,
            "label_break": k, "label_break_bars": kb}


def label_events(
    bars: pd.DataFrame,
    events: list[ZoneEvent],
    atr_abs: pd.Series,
    timeframe: str,
) -> pd.DataFrame:
    """Label every event (dispatch by zone_kind), drop events without a
    complete forward horizon window, and return one row per surviving event:
    the events_to_frame columns plus the four LABELS columns."""
    from symbols import normalize_timeframe, tf_horizon

    horizon = tf_horizon(normalize_timeframe(timeframe))
    n = len(bars)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    atr = atr_abs.to_numpy(dtype=float)

    kept: list[ZoneEvent] = []
    rows: list[dict[str, int]] = []
    for e in events:
        t = e.bar_index
        if t >= n - horizon:
            continue  # incomplete forward window
        a = atr[t]
        if not np.isfinite(a) or a <= 0:
            continue  # ATR warmup
        if e.zone_kind == "demand":
            lab = label_demand_touch(high, low, t, close[t],
                                     e.zone_bottom, a, horizon)
        else:
            lab = label_supply_touch(high, low, t, close[t],
                                     e.zone_top, a, horizon)
        kept.append(e)
        rows.append(lab)

    out = events_to_frame(kept)
    for col in LABELS:
        out[col] = [r[col] for r in rows] if rows else []
    return out


def build(symbol: str, timeframe: str) -> pd.DataFrame:
    """Detect + label all zone-touch events for a stored pair; persist to
    data/symbols/{SYMBOL}_{TF}/zone_events.csv and return the DataFrame.

    Uses refractory_bars = tf_horizon so a zone cannot re-fire while its
    previous touch's label window is still open (limits label overlap
    between training rows; see zone_events.detect_zone_events)."""
    import feature_engineer
    import indicators
    from symbols import normalize_symbol, normalize_timeframe, pair_path, tf_horizon

    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    horizon = tf_horizon(timeframe)
    # canonical index treatment (naive UTC, deduped) so event timestamps and
    # bar_index positions line up 1:1 with the v1 feature matrix
    bars = feature_engineer._clean_index(
        pd.read_csv(pair_path(symbol, timeframe, "bars.csv"), index_col=0))
    events = detect_zone_events(bars, refractory_bars=horizon)
    atr_abs = indicators.atr(bars[["high", "low", "close"]], 14)
    df = label_events(bars, events, atr_abs, timeframe)
    df.to_csv(pair_path(symbol, timeframe, "zone_events.csv"))

    print(f"[bounce_break_labeler] {symbol} {timeframe}: "
          f"{len(events)} events (refractory={horizon}) -> {len(df)} labeled "
          f"(horizon={horizon} bars)")
    if len(df):
        for lab in ("label_bounce", "label_break"):
            bal = df[lab].mean()
            bars_col = df[f"{lab}_bars"]
            print(f"  {lab}: 1={bal:.1%} 0={1 - bal:.1%} | "
                  f"avg {bars_col.mean():.1f} bars to resolution")
        both0 = ((df["label_bounce"] == 0) & (df["label_break"] == 0)).mean()
        both1 = ((df["label_bounce"] == 1) & (df["label_break"] == 1)).mean()
        print(f"  neither (both 0): {both0:.1%} | both 1 (should be rare): {both1:.1%}")
        print("  by kind:")
        for kind, gdf in df.groupby("zone_kind"):
            print(f"    {kind:7s} n={len(gdf):5d} "
                  f"bounce=1 {gdf['label_bounce'].mean():.1%} "
                  f"break=1 {gdf['label_break'].mean():.1%}")
        print("  by trigger:")
        for trig, gdf in df.groupby("trigger"):
            print(f"    {trig} n={len(gdf):5d} "
                  f"bounce=1 {gdf['label_bounce'].mean():.1%} "
                  f"break=1 {gdf['label_break'].mean():.1%}")
        age = df["zone_age_bars"]
        print(f"  zone age at event (bars): median={age.median():.0f} "
              f"p25={age.quantile(0.25):.0f} p75={age.quantile(0.75):.0f}")
    return df


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        build(args[0], args[1])
    else:
        print("Usage: python bounce_break_labeler.py SYMBOL TIMEFRAME")
