"""probt v2 — zone-touch event detection (§5.1).

The anchor of v2 (§5): predictions become CONDITIONAL on a zone-touch event
instead of running on every bar. This module answers one question per closed
bar: "did price just interact with a supply/demand zone, order block, or FVG
in a way that matters?" It does not predict anything — it only detects and
describes the event; bounce_break_labeler.py labels it and zone_features.py
scores its strength.

Hypothesis: SMC zones (order blocks, FVGs, supply/demand) mark price levels
where large participants previously acted; re-tests of those levels are the
highest-information moments to ask a directional question, and asking on
every bar (v1's approach) dilutes the model with mostly-uninformative
positions strictly between zones.

Observed ablation effect: TBD — Step 2 gate is the event-count and
label-balance sanity check on XAUUSD 1H (run this file as a script).

EVENT TRIGGER TYPES (§5.1, predicates corrected per Step-1 review — the
master prompt's E1/E2 as written described price EXITING a zone, not
entering it):

  E1  supply approached/pierced from below:
        high_t >= zone_bottom AND close_{t-1} < zone_bottom
  E2  demand approached/pierced from above:
        low_t <= zone_top AND close_{t-1} > zone_top
  E3  close_t inside [zone_bottom, zone_top] AND close_{t-1} was not inside
  E4  close_t inside an order block (smc_pro) AND close_{t-1} was not inside
        (entry semantics — a later re-entry after leaving fires again and is
        counted by touches_count)
  E5  bar t enters a previously-untouched, unmitigated FVG range (first
        entry only — one E5 per FVG lifetime):
        bull FVG (demand): low_t <= fvg_top;  bear FVG (supply): high_t >= fvg_bottom

Priority when several triggers fire for the SAME zone on the same bar:
E1/E2 > E3 (supply/demand zones) — E4/E5 apply to distinct object kinds so
they never collide with E1-E3. Distinct zones touched on the same bar each
emit their own event (they are different trade hypotheses with different
Z1-Z8 features); the labeler keeps them all and reports how often this
happens.

Directional mapping: supply zone / bear OB / bear FVG -> zone_kind="supply"
(a bounce is a SHORT); demand zone / bull OB / bull FVG -> "demand" (bounce
is a LONG).

CAUSALITY (§0 ground rule 4): compute_supply_demand / compute_smc mark
mitigation using the full series, and their default output keys drop
mitigated objects entirely (survivorship). This module therefore consumes
the *_all output keys (supply_all/demand_all, order_blocks_all/fvgs_all),
which carry every birth with its `born` bar and `mitigated_at` bar, and
reconstructs the live-zone set as of each bar t using only information
available at t:

  - an object is eligible for events at t iff born < t and
    (mitigated_at is None or mitigated_at >= t)
  - births themselves are causal in both source modules (they use only
    bars <= born); `born` is x2 for supply/demand zones, the structure-BREAK
    bar for order blocks (their `x` points into the past and must never be
    used for gating), and x+1 for FVGs (detected one bar after their middle
    candle)
  - the BigBeluga overlap-merge and max_zones cap are re-applied per bar on
    the alive supply/demand set (same helpers as supply_demand.py), so the
    eligible set at t equals exactly what compute_supply_demand would
    return if called on bars[0..t]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

import smc_pro
import supply_demand

EventTrigger = Literal["E1", "E2", "E3", "E4", "E5"]
ZoneKind = Literal["supply", "demand"]
ZoneSource = Literal["sd", "ob_swing", "ob_internal", "fvg"]


@dataclass
class ZoneEvent:
    bar_index: int
    timestamp: pd.Timestamp
    trigger: EventTrigger
    zone_kind: ZoneKind
    zone_top: float
    zone_bottom: float
    zone_delta: float          # NaN for OB/FVG events (no volume delta)
    zone_age_bars: int         # bar_index - born
    zone_pct: float            # % share of live SD delta; NaN for OB/FVG
    touches_count: int         # prior events against this same zone
    zone_id: str               # "{source}:{kind}:{born}" — stable identity
    source: ZoneSource


def collect_zone_records(
    bars: pd.DataFrame,
    include_ob: bool = True,
    include_fvg: bool = True,
) -> list[dict[str, Any]]:
    """Unified causal zone records from all three sources:
    {source, kind, top, bottom, born, death, delta, x2, zone_id}.
    `born`/`death` are the bars the object became known / was mitigated
    (death None = still alive) — consumers gate on born < t <= death for a
    lookahead-free view at bar t. Shared by detect_zone_events and
    zone_features (Z5 confluence)."""
    sd = supply_demand.compute_supply_demand(bars)
    smc = smc_pro.compute_smc(bars) if (include_ob or include_fvg) else smc_pro._empty()

    zones: list[dict[str, Any]] = []
    for z in sd["supply_all"] + sd["demand_all"]:
        zones.append({
            "source": "sd", "kind": z["kind"],
            "top": z["top"], "bottom": z["bottom"],
            "born": z["x2"], "death": z["mitigated_at"],
            "delta": float(z["delta"]), "x2": z["x2"],
        })
    if include_ob:
        for o in smc["order_blocks_all"]:
            zones.append({
                "source": f"ob_{o['scope']}",
                "kind": "demand" if o["bias"] == "bull" else "supply",
                "top": o["top"], "bottom": o["bottom"],
                "born": o["born"], "death": o.get("mitigated_at"),
                "delta": float("nan"), "x2": o["born"],
            })
    if include_fvg:
        for f in smc["fvgs_all"]:
            zones.append({
                "source": "fvg",
                "kind": "demand" if f["bias"] == "bull" else "supply",
                "top": f["top"], "bottom": f["bottom"],
                "born": f["x"] + 1, "death": f.get("mitigated_at"),
                "delta": float("nan"), "x2": f["x"] + 1,
            })
    for z in zones:
        z["zone_id"] = f"{z['source']}:{z['kind']}:{z['born']}"
    return zones


def _eligible_sd(alive: list[dict[str, Any]], max_zones: int) -> list[dict[str, Any]]:
    """Overlap-merge + recency cap, mirroring compute_supply_demand's
    post-processing so the causal per-bar set matches its full-run output."""
    merged = supply_demand._merge_overlap(alive)
    return sorted(merged, key=lambda z: z["x2"], reverse=True)[:max_zones]


def detect_zone_events(
    bars: pd.DataFrame,
    max_zones: int = 5,
    include_ob: bool = True,
    include_fvg: bool = True,
    refractory_bars: int = 0,
) -> list[ZoneEvent]:
    """Scan `bars` (OHLCV, DatetimeIndex) and return every zone-touch event,
    in bar order. Self-contained: runs supply_demand + smc_pro internally so
    callers cannot accidentally feed a non-causal snapshot.

    `refractory_bars`: minimum bars between two events on the SAME zone.
    Without it, price hovering at a zone boundary re-fires E1/E2 every bar,
    producing near-duplicate training rows whose label windows almost fully
    overlap (observed on XAUUSD 1H: one zone fired 51 times). Training
    callers should pass the Triple-Barrier horizon so a zone cannot re-fire
    while its previous touch's outcome window is still open; 0 keeps every
    event (display/diagnostics). Suppressed re-pokes do NOT increment
    touches_count — a "touch" is an event row, not a bar.

    The last bar's events are included — live_engine_v2 calls this on a
    trailing window and keeps only events at bar n-1.
    """
    n = len(bars)
    if n < 3:
        return []

    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)

    zones = collect_zone_records(bars, include_ob, include_fvg)

    # ── event-driven active sets (add at born+1, drop after death) ──
    births: dict[int, list[dict[str, Any]]] = {}
    for z in zones:
        births.setdefault(z["born"] + 1, []).append(z)

    active_sd: list[dict[str, Any]] = []
    active_ob: list[dict[str, Any]] = []
    active_fvg: list[dict[str, Any]] = []
    sd_dirty = True
    eligible_supply: list[dict[str, Any]] = []
    eligible_demand: list[dict[str, Any]] = []
    total_delta = 0.0

    touches: dict[str, int] = {}
    last_event_bar: dict[str, int] = {}
    events: list[ZoneEvent] = []

    def _inside(price: float, z: dict[str, Any]) -> bool:
        return z["bottom"] <= price <= z["top"]

    for t in range(1, n):
        # activate zones born at t-1 (eligible from the NEXT bar: born < t)
        for z in births.get(t, ()):
            if z["source"] == "sd":
                active_sd.append(z)
                sd_dirty = True
            elif z["source"] == "fvg":
                active_fvg.append(z)
            else:
                active_ob.append(z)

        # retire dead zones (death bar t means: still eligible AT t, gone after)
        n_sd = len(active_sd)
        active_sd = [z for z in active_sd if z["death"] is None or z["death"] >= t]
        if len(active_sd) != n_sd:
            sd_dirty = True
        active_ob = [z for z in active_ob if z["death"] is None or z["death"] >= t]
        active_fvg = [z for z in active_fvg if z["death"] is None or z["death"] >= t]

        if sd_dirty:
            eligible_supply = _eligible_sd(
                [z for z in active_sd if z["kind"] == "supply"], max_zones)
            eligible_demand = _eligible_sd(
                [z for z in active_sd if z["kind"] == "demand"], max_zones)
            total_delta = sum(abs(z["delta"])
                              for z in eligible_supply + eligible_demand)
            sd_dirty = False

        c_prev = close[t - 1]

        def _emit(z: dict[str, Any], trigger: EventTrigger) -> None:
            zid = z["zone_id"]
            if refractory_bars and t - last_event_bar.get(zid, -10**9) < refractory_bars:
                return
            last_event_bar[zid] = t
            pct = float("nan")
            if z["source"] == "sd" and total_delta > 0:
                pct = round(abs(z["delta"]) / total_delta * 100, 1)
            events.append(ZoneEvent(
                bar_index=t,
                timestamp=bars.index[t],
                trigger=trigger,
                zone_kind=z["kind"],
                zone_top=float(z["top"]),
                zone_bottom=float(z["bottom"]),
                zone_delta=z["delta"],
                zone_age_bars=t - z["born"],
                zone_pct=pct,
                touches_count=touches.get(zid, 0),
                zone_id=zid,
                source=z["source"],
            ))
            touches[zid] = touches.get(zid, 0) + 1

        # E1/E2/E3 — supply/demand zones
        for z in eligible_supply:
            if high[t] >= z["bottom"] and c_prev < z["bottom"]:
                _emit(z, "E1")
            elif _inside(close[t], z) and not _inside(c_prev, z):
                _emit(z, "E3")
        for z in eligible_demand:
            if low[t] <= z["top"] and c_prev > z["top"]:
                _emit(z, "E2")
            elif _inside(close[t], z) and not _inside(c_prev, z):
                _emit(z, "E3")

        # E4 — order blocks (close-entry semantics)
        for z in active_ob:
            if _inside(close[t], z) and not _inside(c_prev, z):
                _emit(z, "E4")

        # E5 — FVG first entry (then retire the FVG from further E5s)
        still_active = []
        for z in active_fvg:
            entered = (low[t] <= z["top"] if z["kind"] == "demand"
                       else high[t] >= z["bottom"])
            if entered:
                _emit(z, "E5")
            else:
                still_active.append(z)
        active_fvg = still_active

    return events


def events_to_frame(events: list[ZoneEvent]) -> pd.DataFrame:
    """Event list -> DataFrame (one row per event, bar order), indexed by
    timestamp. Empty DataFrame with the right columns when no events."""
    cols = ["bar_index", "trigger", "zone_kind", "source", "zone_id",
            "zone_top", "zone_bottom", "zone_delta", "zone_age_bars",
            "zone_pct", "touches_count"]
    if not events:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame([{
        "timestamp": e.timestamp, "bar_index": e.bar_index,
        "trigger": e.trigger, "zone_kind": e.zone_kind, "source": e.source,
        "zone_id": e.zone_id, "zone_top": e.zone_top,
        "zone_bottom": e.zone_bottom, "zone_delta": e.zone_delta,
        "zone_age_bars": e.zone_age_bars, "zone_pct": e.zone_pct,
        "touches_count": e.touches_count,
    } for e in events])
    return df.set_index("timestamp")


# ─── Step 2 sanity check (§12): event counts on a stored pair ────────
def _sanity_report(symbol: str, timeframe: str) -> None:
    import feature_engineer
    from symbols import normalize_symbol, normalize_timeframe, pair_path

    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    bars = feature_engineer._clean_index(
        pd.read_csv(pair_path(symbol, timeframe, "bars.csv"), index_col=0))
    events = detect_zone_events(bars)
    df = events_to_frame(events)
    n = len(bars)

    print(f"[zone_events] {symbol} {timeframe}: {n} bars "
          f"({bars.index[0]} -> {bars.index[-1]})")
    print(f"  events: {len(df)} total | {len(df) / n * 1000:.1f} per 1000 bars"
          f" | bars with >=1 event: {df['bar_index'].nunique()}"
          f" ({df['bar_index'].nunique() / n:.1%} of bars)")
    print(f"  by trigger: {df['trigger'].value_counts().to_dict()}")
    print(f"  by source:  {df['source'].value_counts().to_dict()}")
    print(f"  by kind:    {df['zone_kind'].value_counts().to_dict()}")
    print(f"  unique zones touched: {df['zone_id'].nunique()} | "
          f"max touches on one zone: {df['touches_count'].max()}")

    # causal-reconstruction check: eligible SD set at the final bar must
    # equal compute_supply_demand's own (alive-only) output on the full df
    sd = supply_demand.compute_supply_demand(bars)
    for side in ("supply", "demand"):
        alive = [dict(z, delta=float(z["delta"]))
                 for z in sd[f"{side}_all"] if z["mitigated_at"] is None]
        mine = {(round(z["top"], 6), round(z["bottom"], 6))
                for z in _eligible_sd(alive, 5)}
        ref = {(round(z["top"], 6), round(z["bottom"], 6)) for z in sd[side]}
        status = "OK" if mine == ref else f"MISMATCH mine={mine} ref={ref}"
        print(f"  causal-reconstruction check ({side}): {status}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        _sanity_report(args[0], args[1])
    else:
        print("Usage: python zone_events.py SYMBOL TIMEFRAME")
