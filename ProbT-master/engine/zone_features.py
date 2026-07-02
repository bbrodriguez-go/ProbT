"""probt v2 — zone-strength features Z1-Z8 (§5.3).

Derives the model inputs that describe how "strong" the touched zone is:
how much of the market's recent order flow it represents, how old and how
often-tested it is, whether other structures agree with it, and whether it
sits in a high- or low-volume price region. Per §3 P3 everything is a
ratio, count, sign or ATR-normalized distance — raw prices never enter.

Hypothesis: not all zones are equal. A fresh, wide, high-delta zone with
confluent structures and higher-timeframe agreement should hold (bounce)
more often than a thin, old, single-touch zone with no confluence.

Observed ablation effect (2026-07-02, XAUUSD 1H n=3882 / 4H n=999 events,
purged walk-forward): CLEARS THE GATE ON BREAK, FAILS ON BOUNCE.
Break cumulative BSS 1H +0.0749 (AUC 0.523 -> 0.672), 4H +0.0389 (AUC
0.509 -> 0.617); z_geom and z_flow carry the lift. Bounce: every group flat
or negative on both TFs (AUC stuck at ~0.47-0.49).

HONESTY CAVEAT (do not oversell the break number): the break label's target
sits at the zone edge, so its distance from entry VARIES per event — and
the single raw feature "distance to break target in ATR" alone scores AUC
0.740 (1H) / 0.737 (4H), MORE than the full Z model. The Z lift is mostly
barrier geometry (closer target = more likely hit first), which is real,
calibrated conditioning but NOT market edge by itself: closer target also
means smaller reward. Consequences, implemented in Step 7/8: the break
model gets distance-to-target as an explicit feature, and EV/Kelly for
break trades must use the PER-EVENT payoff b = target_dist/stop_dist, never
the fixed b=2 (which only holds for bounce trades).

Reproduce: `python zone_features.py SYMBOL TF`.

FEATURES (one row per zone-touch event):

  zone_delta_pct           Z1. |zone_delta| / total live delta, as a ratio
                            in [0,1]. OB/FVG events carry no volume delta ->
                            0.0 (no measured share), not NaN, so those rows
                            survive the training dropna.
  zone_age_bars             Z2. log1p(bars since zone birth) — log because
                            raw age is unbounded/heavy-tailed (P3).
  zone_touches               Z3. prior touch events on this zone (raw count).
  zone_width_atr             Z4. (zone_top - zone_bottom) / ATR at the event.
  zone_confluence_count       Z5. other zone records (SD zones, order blocks,
                            FVGs) alive at t whose price range overlaps the
                            touched zone. S/R clusters are NOT re-counted
                            here — the baseline features already carry
                            zone_dist_atr/at_zone from support_resistance.
  htf_alignment               Z6. smc_bias of the next-higher timeframe's
                            last CLOSED bar (from its feature_matrix.csv),
                            multiplied by the event's bounce direction
                            (+1 demand/long, -1 supply/short): +1 = HTF
                            agrees with the bounce, -1 = opposes, 0 =
                            neutral or no higher TF trained.
  distance_from_poc_atr        Z7. (close - previous COMPLETED day's POC) /
                            ATR, clipped to ±20. Previous day per
                            volume_profile.get_daily_vp_features — the
                            current day's POC would be intraday lookahead.
  volume_delta_since_zone       Z8. net signed volume (up-close bars minus
                            down-close bars) over (born, t], divided by
                            total volume over the same span -> [-1, 1].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "zone_delta_pct",
    "zone_age_bars",
    "zone_touches",
    "zone_width_atr",
    "zone_confluence_count",
    "htf_alignment",
    "distance_from_poc_atr",
    "volume_delta_since_zone",
]

Z_GROUPS: dict[str, list[str]] = {
    "z_flow": ["zone_delta_pct", "volume_delta_since_zone"],
    "z_history": ["zone_age_bars", "zone_touches"],
    "z_geom": ["zone_width_atr", "zone_confluence_count"],
    "z_context": ["htf_alignment", "distance_from_poc_atr"],
}

_HTF_LADDER = {"1m": "5m", "5m": "15m", "15m": "1H", "1H": "4H",
               "4H": "1D", "1D": "1W", "1W": None}


def _htf_bias_at(events_ts: pd.DatetimeIndex, symbol: str, timeframe: str) -> np.ndarray:
    """smc_bias of the next-higher TF's last CLOSED bar before each event
    timestamp (searchsorted right minus 2: the bar merely STARTED before the
    event may still be open — its features would leak its own close).
    Zeros when no higher TF or no trained feature matrix on disk."""
    import os

    import feature_engineer
    from symbols import pair_path

    htf = _HTF_LADDER.get(timeframe)
    out = np.zeros(len(events_ts))
    if htf is None:
        return out
    path = pair_path(symbol, htf, "feature_matrix.csv")
    if not os.path.exists(path):
        return out
    fm = feature_engineer._clean_index(pd.read_csv(path, index_col=0))
    col = f"smc_bias_{htf.lower()}"
    if col not in fm.columns:
        return out
    bias = fm[col].to_numpy(dtype=float)
    pos = fm.index.searchsorted(events_ts, side="right") - 2
    ok = pos >= 0
    out[ok] = bias[pos[ok]]
    return out


def build_zone_features_bulk(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Z1-Z8 for every event row, aligned to events.index (which may carry
    duplicate timestamps — several zones touched on one bar). `events` is
    the zone_events.csv / events_to_frame schema; `bars` the raw OHLCV.
    The live engine calls this with a single-row events frame."""
    import indicators
    import volume_profile
    from feature_engineer import _clean_index
    from zone_events import collect_zone_records

    bars = _clean_index(bars)
    close = bars["close"].to_numpy(dtype=float)
    open_ = bars["open"].to_numpy(dtype=float)
    vol = bars["volume"].to_numpy(dtype=float)
    atr = indicators.atr(bars[["high", "low", "close"]], 14).to_numpy(dtype=float)

    # cumulative signed / total volume for Z8 (index 0 = before first bar)
    signed = np.where(close > open_, vol, np.where(close < open_, -vol, 0.0))
    cs = np.concatenate(([0.0], np.cumsum(signed)))
    cv = np.concatenate(([0.0], np.cumsum(np.abs(vol))))

    # all zone records for Z5 confluence
    recs = collect_zone_records(bars)
    r_born = np.array([z["born"] for z in recs], dtype=float)
    r_death = np.array([np.inf if z["death"] is None else z["death"]
                        for z in recs], dtype=float)
    r_top = np.array([z["top"] for z in recs], dtype=float)
    r_bot = np.array([z["bottom"] for z in recs], dtype=float)
    r_id = np.array([z["zone_id"] for z in recs])

    # previous completed day's POC for Z7
    poc = volume_profile.get_daily_vp_features(bars)["poc_price"].shift(1)
    ev_days = events.index.normalize()
    poc_at_event = poc.reindex(ev_days.unique().sort_values(),
                               method="ffill").loc[ev_days].to_numpy(dtype=float)

    htf = _htf_bias_at(events.index, symbol, timeframe)

    t_arr = events["bar_index"].to_numpy(dtype=int)
    ev_top = events["zone_top"].to_numpy(dtype=float)
    ev_bot = events["zone_bottom"].to_numpy(dtype=float)
    ev_age = events["zone_age_bars"].to_numpy(dtype=float)
    ev_dir = np.where(events["zone_kind"].to_numpy() == "demand", 1.0, -1.0)
    ev_id = events["zone_id"].to_numpy()

    n_ev = len(events)
    conf = np.zeros(n_ev)
    z8 = np.full(n_ev, np.nan)
    for k in range(n_ev):
        t = t_arr[k]
        alive = (r_born < t) & (r_death >= t) & (r_id != ev_id[k])
        conf[k] = float(np.count_nonzero(
            alive & (r_top >= ev_bot[k]) & (r_bot <= ev_top[k])))
        born = t - int(ev_age[k])
        tot = cv[t + 1] - cv[born + 1]
        if tot > 0:
            z8[k] = (cs[t + 1] - cs[born + 1]) / tot

    a = atr[t_arr]
    with np.errstate(invalid="ignore", divide="ignore"):
        width = np.where(a > 0, (ev_top - ev_bot) / a, np.nan)
        poc_dist = np.clip(np.where(a > 0, (close[t_arr] - poc_at_event) / a,
                                    np.nan), -20.0, 20.0)
        # distance from entry close to the BREAK target (zone edge -1A /
        # +1A) in ATR — the strongest single break predictor (AUC 0.74, see
        # module docstring) and the basis of the per-event payoff
        # b = dist/2 (stop = 2A) that break-EV/Kelly must use instead of
        # the fixed b=2. Negative = close already beyond the target.
        break_dist = np.where(
            ev_dir > 0,
            (close[t_arr] - (ev_bot - a)) / a,
            ((ev_top + a) - close[t_arr]) / a)
        break_dist = np.clip(break_dist, -5.0, 10.0)

    return pd.DataFrame({
        "break_target_dist_atr": break_dist,
        "break_payoff_b": np.maximum(break_dist / 2.0, 0.05),
        "zone_delta_pct": np.nan_to_num(
            events["zone_pct"].to_numpy(dtype=float) / 100.0, nan=0.0),
        "zone_age_bars": np.log1p(ev_age),
        "zone_touches": events["touches_count"].to_numpy(dtype=float),
        "zone_width_atr": width,
        "zone_confluence_count": conf,
        "htf_alignment": htf * ev_dir,
        "distance_from_poc_atr": poc_dist,
        "volume_delta_since_zone": z8,
    }, index=events.index)


# ─── Step 6 ablation runner (§12) ────────────────────────────────────
def _ablation_report(symbol: str, timeframe: str) -> None:
    from model_trainer_v2 import ablation_table, load_event_dataset

    ds = load_event_dataset(symbol, timeframe)
    print(f"[ablation Z] {symbol} {timeframe}: building Z1-Z8 for "
          f"{len(ds['events'])} events...")
    zf = build_zone_features_bulk(ds["events"], ds["bars"], symbol, timeframe)
    ablation_table(ds["events"], ds["fm"], ds["baseline_cols"], zf,
                   Z_GROUPS, ds["horizon"])


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        _ablation_report(args[0], args[1])
    else:
        print("Usage: python zone_features.py SYMBOL TIMEFRAME")
