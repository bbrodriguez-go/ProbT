"""probt v2 — live_engine_v2.py: event-driven inference, dual-model output
(§5.4, §6 D4, §9.4).

Replaces live_engine.compute_reading's "always produce a Tier A reading"
behavior with the event-driven flow of §5.4: most calls return state
"no_signal"; only a genuine zone-touch event on the latest closed bar
produces the dual-model (bounce/break) reading. §9.4's honesty rules are
enforced server-side:

  - is_probability=True only for a model whose metrics_v2.json says
    live_eligible (OOS BSS > 0 and AUC > 0.52). A non-eligible model
    reports the event-conditional CLIMATOLOGY (its OOS base rate) with
    is_probability=False — never a dressed-up guess. (XAUUSD status at
    training time: M_break eligible on 1H/4H, M_bounce eligible nowhere.)
  - break-side EV and Kelly use the PER-EVENT payoff b = target_dist/2
    (see zone_features docstring: p and b are anti-correlated by
    construction; fixed b=2 would overstate break EV badly).
  - conformal coverage_check failures (e.g. XAUUSD 4H break: 82.5% < 88%)
    surface as interval_reliable=False on the reading.
  - central-bank blackout (§6 D4) suppresses the reading entirely.

KNOWN LIMITATION (report, don't hide): §4.1's split-conformal score |p - y|
against a BINARY y makes q_hat ~ max(p, 1-p) at 90% coverage — observed
q_hat 0.65-0.79 on both v2 models (and 0.79 on the v1 model, which shipped
the same formula). The bands are honest but near-vacuous, and since the
Kelly EV gate keys off the conformal LOWER bound, sizing is ~permanently
0% at current skill levels. Consistent with the Step 7 cost-sensitivity
finding (no EV+ trades), but partly mechanical. A tighter valid interval
needs a different nonconformity (e.g. probability-binned quantiles) — a
spec change, flagged for the Step 10 writeup rather than made unilaterally.

Live zone detection runs on a trailing LIVE_WINDOW of bars, not full
history. # ponytail: zones born before the window are invisible and
zone_age/touches saturate at the window edge — acceptable for zones that
matter (max 5 per side, recent); extend the window if stale-zone readings
ever matter.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

import feature_engineer
import news_sentiment
from live_engine import _pull_bars, _pull_macro
from quantum_features import build_signal_features, harmonic_oscillator_energy
from symbols import (data_dir, normalize_symbol, normalize_timeframe,
                     pair_path, symbol_name, symbol_news_query, tf_horizon)
from zone_events import ZoneEvent, detect_zone_events, events_to_frame
from zone_features import build_zone_features_bulk

LIVE_WINDOW = 2000
KELLY_CAP = 0.02
CALENDAR_PATH = os.path.join(data_dir(), "external",
                             "central_bank_calendar.json")
# an event fires E1/E2 (edge cross) before E3 before OB/FVG touches; among
# equals the biggest volume-delta zone is the one the operator would trade
_TRIGGER_RANK = {"E1": 0, "E2": 0, "E3": 1, "E4": 2, "E5": 3}


def is_blackout_window(now: datetime, calendar_path: str = CALENDAR_PATH,
                       hours_before: float = 24.0,
                       hours_after: float = 1.0) -> dict[str, Any] | None:
    """§6 D4. Hand-maintained JSON: {"events": [{"name": str, "utc":
    ISO-8601}]}. Returns {"event", "starts_in_seconds"} when `now` falls in
    [utc - hours_before, utc + hours_after] of any entry, else None.
    Missing/empty calendar = never blackout (and the reading notes it)."""
    if not os.path.exists(calendar_path):
        return None
    try:
        cal = json.load(open(calendar_path))
    except (json.JSONDecodeError, OSError):
        return None
    for e in cal.get("events", []):
        try:
            t = datetime.fromisoformat(e["utc"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if t - timedelta(hours=hours_before) <= now <= t + timedelta(hours=hours_after):
            return {"event": e.get("name", "central bank decision"),
                    "starts_in_seconds": max(int((t - now).total_seconds()), 0)}
    return None


def detect_live_event(bars: pd.DataFrame, horizon: int) -> ZoneEvent | None:
    """§5.4 step 1: run causal event detection (with the training-time
    refractory) on the trailing window; keep only events on the LAST closed
    bar; pick one by trigger rank then |zone_delta|."""
    events = detect_zone_events(bars, refractory_bars=horizon)
    last_bar = len(bars) - 1
    at_last = [e for e in events if e.bar_index == last_bar]
    if not at_last:
        return None
    return sorted(at_last, key=lambda e: (
        _TRIGGER_RANK.get(e.trigger, 9),
        -(abs(e.zone_delta) if np.isfinite(e.zone_delta) else 0.0)))[0]


def _load_model_side(symbol: str, timeframe: str, name: str) -> dict | None:
    """Model bundle + conformal + metrics for one side, or None if this
    pair has no v2 training on disk."""
    mpath = pair_path(symbol, timeframe, f"model_{name}.pkl")
    fpath = pair_path(symbol, timeframe, "features_v2.json")
    if not (os.path.exists(mpath) and os.path.exists(fpath)):
        return None
    out = {
        "bundle": joblib.load(mpath),
        "features": json.load(open(fpath))[name],
        "conformal": None, "metrics": None, "coefficients": None,
    }
    cpath = pair_path(symbol, timeframe, f"conformal_{name}.pkl")
    if os.path.exists(cpath):
        out["conformal"] = joblib.load(cpath)
    mepath = pair_path(symbol, timeframe, "metrics_v2.json")
    if os.path.exists(mepath):
        out["metrics"] = json.load(open(mepath)).get(name)
    apath = pair_path(symbol, timeframe, "coefficient_audit.json")
    if os.path.exists(apath):
        out["coefficients"] = json.load(open(apath)).get(name, {}).get("coefficients")
    return out


def score_side(side: dict | None, feature_row: pd.Series) -> dict[str, Any]:
    """§5.4 step 4 for one of bounce/break. Applies the §9.4 honesty rules
    (eligibility -> climatology fallback, coverage flag)."""
    if side is None:
        return {"probability": None, "probability_lo": None,
                "probability_hi": None, "is_probability": False,
                "interval_reliable": False,
                "note": "no v2 model trained for this pair"}
    metrics = side["metrics"] or {}
    eligible = bool(metrics.get("live_eligible"))
    missing = [c for c in side["features"] if c not in feature_row.index
               or pd.isna(feature_row[c])]
    contributions = None
    if eligible and not missing:
        X = pd.DataFrame([feature_row[side["features"]].to_numpy(dtype=float)],
                         columns=side["features"])
        xs = side["bundle"]["scaler"].transform(X)
        p = float(side["bundle"]["model"].predict_proba(xs)[0, 1])
        is_prob = True
        note = None
        # §9.2 card 3: top-6 features by |coefficient * standardized value| —
        # what is actually driving THIS prediction, signed
        if side["coefficients"]:
            contrib = [
                {"feature": f, "value": float(feature_row[f]),
                 "contribution": float(side["coefficients"].get(f, 0.0) * xs[0][i])}
                for i, f in enumerate(side["features"])
                if side["coefficients"].get(f, 0.0) != 0.0
            ]
            contrib.sort(key=lambda c: -abs(c["contribution"]))
            contributions = [
                {**c, "value": round(c["value"], 4),
                 "contribution": round(c["contribution"], 4)}
                for c in contrib[:6]]
    else:
        # climatology: the event-conditional OOS base rate, clearly flagged
        p = metrics.get("base_rate")
        is_prob = False
        note = ("model not live_eligible (no OOS skill) — showing event "
                "base rate" if eligible is False else
                f"live features missing: {missing} — showing event base rate")
    lo = hi = p
    reliable = False
    band_kind = None
    if p is not None and side["conformal"]:
        conf = side["conformal"]
        cal = conf.get("calibration_bins")
        if cal and is_prob:
            # Wilson 90% CI of the observed OOS hit rate in this
            # prediction's calibration bin: "when the model said ~p, reality
            # delivered k/n" — the trader-usable uncertainty. (The spec's
            # conformal q_hat is still recorded in conformal_*.pkl, but a
            # 90% quantile of |p - y| against a BINARY y is near-vacuous by
            # construction and not worth displaying.)
            b = int(np.digitize([p], cal["edges"])[0])
            n_b, k_b = cal["bins"][b]["n"], cal["bins"][b]["k"]
            if n_b >= 30:
                z = 1.6449  # 90%
                z2 = z * z
                center = (k_b + z2 / 2.0) / (n_b + z2)
                hw = (z / (n_b + z2)) * np.sqrt(k_b * (n_b - k_b) / n_b + z2 / 4.0)
                lo, hi = max(center - hw, 0.0), min(center + hw, 1.0)
                reliable = True
                band_kind = "calibration_wilson_90"
        if band_kind is None:
            q = float(conf.get("q_hat", 0.08))
            lo, hi = max(p - q, 0.0), min(p + q, 1.0)
            reliable = bool(conf.get("coverage_check", {}).get("passes_88pct"))
            band_kind = "conformal_90"
    return {"probability": None if p is None else round(p, 4),
            "probability_lo": None if lo is None else round(lo, 4),
            "probability_hi": None if hi is None else round(hi, 4),
            "is_probability": is_prob, "interval_reliable": reliable,
            "band_kind": band_kind,
            "note": note, "contributions": contributions}


def kelly_sizing(p: float, p_lo: float, b: float) -> dict[str, Any]:
    """§5.4 step 6. Half-Kelly at payoff b, capped; EV gate on the
    conformal LOWER bound (§9.4: sizing greys out unless even the lower
    bound beats breakeven 1/(1+b))."""
    ev_positive = bool(p_lo * b - (1.0 - p_lo) > 0)
    full = max((p * b - (1.0 - p)) / b, 0.0)
    half = 0.5 * full
    return {"payoff_b": round(b, 3),
            "ev_positive": ev_positive,
            "full_kelly_pct": round(full * 100, 3),
            "half_kelly_pct": round(half * 100, 3),
            "final_pct": round(min(half, KELLY_CAP) * 100, 3) if ev_positive else 0.0,
            "cap_pct": KELLY_CAP * 100}


def compute_reading_v2(symbol: str, timeframe: str) -> dict[str, Any]:
    """Top-level entry point for the API gateway (§5.4). states:
    "blackout" | "no_signal" | "signal"."""
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    horizon = tf_horizon(timeframe)
    now = datetime.now(timezone.utc)

    import indicators
    from symbols import symbol_has_macro

    bars = _pull_bars(symbol, timeframe).tail(LIVE_WINDOW)
    atr_abs = float(indicators.atr(bars[["high", "low", "close"]], 14).iloc[-1])
    base = {
        "symbol": symbol, "symbol_name": symbol_name(symbol),
        "timeframe": timeframe, "asof": str(bars.index[-1]),
        "generated_at": now.isoformat(),
        "price": round(float(bars["close"].iloc[-1]), 2),
        "atr_abs": round(atr_abs, 4),
        "horizon_bars": horizon,
        "calendar_loaded": os.path.exists(CALENDAR_PATH),
    }
    if symbol_has_macro(symbol):
        # regime CONTEXT only — these failed the model gate (external_data
        # docstring) and are displayed as context, never as signals
        import external_data
        mc = external_data.macro_positioning_features(bars.index[-1:]).iloc[0]
        base["macro_context"] = {
            k: (None if pd.isna(v) else round(float(v), 4))
            for k, v in mc.items()}

    bl = is_blackout_window(now)
    if bl:
        return {**base, "state": "blackout", "blackout": bl}

    event = detect_live_event(bars, horizon)
    if event is None:
        return {**base, "state": "no_signal"}

    # §5.4 step 3: full feature vector for this event
    macro = _pull_macro(symbol)
    last_row = feature_engineer.build_last_row_features(
        bars, timeframe, macro=macro, symbol=symbol)
    if last_row is None:
        return {**base, "state": "no_signal",
                "note": "insufficient bars for features"}
    ev_frame = events_to_frame([event])
    zrow = build_zone_features_bulk(ev_frame, bars, symbol, timeframe).iloc[0]
    hilbert = build_signal_features(bars, at_indices=[len(bars) - 1]).iloc[-1]
    q6 = harmonic_oscillator_energy(base["price"], event.zone_top,
                                    event.zone_bottom)
    feature_row = pd.concat([
        last_row, zrow,
        pd.Series({"q6_harmonic_oscillator_energy": q6,
                   "hilbert_inst_freq": hilbert["hilbert_inst_freq"]}),
    ])
    feature_row = feature_row[~feature_row.index.duplicated(keep="last")]

    # §5.4 steps 4-6
    scores = {n: score_side(_load_model_side(symbol, timeframe, n), feature_row)
              for n in ("bounce", "break")}
    payoff = {"bounce": 2.0,
              "break": float(zrow["break_payoff_b"])
              if np.isfinite(zrow["break_payoff_b"]) else 2.0}
    p_b = scores["bounce"]["probability"]
    p_k = scores["break"]["probability"]
    direction_block: dict[str, Any] = {"winner": None, "direction": None,
                                       "magnitude": None}
    sizing = None
    if p_b is not None and p_k is not None:
        winner = "bounce" if p_b >= p_k else "break"
        p = scores[winner]["probability"]
        is_long = (event.zone_kind == "demand") == (winner == "bounce")
        direction_block = {
            "winner": winner,
            "direction": "LONG" if is_long else "SHORT",
            "magnitude": round(max(p_b, p_k) - 0.5, 4),
            "winner_is_probability": scores[winner]["is_probability"],
        }
        sizing = kelly_sizing(p, scores[winner]["probability_lo"],
                              payoff[winner])

    return {
        **base, "state": "signal",
        "event": {
            "trigger": event.trigger, "zone_kind": event.zone_kind,
            "source": event.source, "zone_top": round(event.zone_top, 2),
            "zone_bottom": round(event.zone_bottom, 2),
            "zone_age_bars": event.zone_age_bars,
            "touches_count": event.touches_count,
            "zone_pct": event.zone_pct if np.isfinite(event.zone_pct) else None,
        },
        "tier_a_bounce": scores["bounce"],
        "tier_a_break": {**scores["break"], "payoff_b": payoff["break"]},
        "direction": direction_block,
        "sizing": sizing,
        "news": news_sentiment.get_news_sentiment(
            query=symbol_news_query(symbol)),
    }


def _print_report(r: dict[str, Any]) -> None:
    line = "=" * 60
    print(line)
    print(f"{r['symbol']} {r['timeframe']}  |  {r['asof']}  |  ${r['price']}"
          f"  |  state: {r['state'].upper()}")
    print(line)
    if r["state"] == "blackout":
        print(f"  BLACKOUT: {r['blackout']['event']} in "
              f"{r['blackout']['starts_in_seconds']}s — reading suppressed")
        return
    if r["state"] == "no_signal":
        print("  no zone touch on the latest closed bar — no reading")
        return
    e = r["event"]
    print(f"\nEVENT  {e['trigger']} {e['zone_kind'].upper()} ({e['source']}) "
          f"zone {e['zone_bottom']}-{e['zone_top']} age {e['zone_age_bars']} "
          f"touches {e['touches_count']}")
    for name in ("bounce", "break"):
        s = r[f"tier_a_{name}"]
        tag = "calibrated ML" if s["is_probability"] else "NOT a probability"
        pr = s["probability"]
        print(f"  {name:6s}: {pr if pr is None else f'{pr:.1%}':>6} "
              f"[{s['probability_lo']}, {s['probability_hi']}] ({tag})"
              f"{'' if s.get('interval_reliable') else ' [interval unverified]'}"
              f"{' — ' + s['note'] if s.get('note') else ''}")
    d = r["direction"]
    if d["winner"]:
        print(f"  direction: {d['direction']} via {d['winner']} "
              f"(magnitude {d['magnitude']})")
    if r["sizing"]:
        z = r["sizing"]
        print(f"  sizing: {z['final_pct']:.3f}% (EV+ {z['ev_positive']}, "
              f"b={z['payoff_b']}, half-Kelly capped {z['cap_pct']}%)")
    print(line)


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        _print_report(compute_reading_v2(args[0], args[1]))
    else:
        print("Usage: python live_engine_v2.py SYMBOL TIMEFRAME")
