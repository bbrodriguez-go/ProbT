"""probt v2 — self-grading signal journal (upgrade #4).

Every SIGNAL reading the live engine produces is logged; once its
Triple-Barrier window has elapsed, the journal grades it against what price
actually did (same race rules as bounce_break_labeler) and can summarize
the LIVE track record: "over the last N graded signals, when the break
model said X%, it happened Y% of the time."

Why this matters: the training-time calibration numbers are out-of-sample
but historical. The journal is the only evidence that calibration holds on
data that did not exist when the model was trained — and the earliest
warning when it stops holding (regime drift, data-feed changes).

Honesty rules:
  - Only genuinely LIVE readings are logged. No backfilling from training
    history — those events were seen by the trainer and would dress
    in-sample fit up as a live track record.
  - Barrier prices are frozen at logging time (from the reading's close,
    ATR and zone bounds), so grading can never drift from what the signal
    actually claimed.
  - The summary reports insufficient-n plainly instead of quoting rates
    off a handful of observations.

Storage: one CSV per pair at data/symbols/{SYMBOL}_{TF}/signal_journal.csv.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from bounce_break_labeler import _race_down_wins, _race_up_wins
from symbols import normalize_symbol, normalize_timeframe, pair_path, tf_horizon

_COLS = [
    "asof", "generated_at", "price", "atr_abs", "trigger", "zone_kind",
    "source", "zone_top", "zone_bottom",
    "p_bounce", "bounce_is_prob", "p_break", "break_is_prob", "payoff_b",
    # barrier prices frozen at signal time
    "bounce_win", "bounce_lose", "break_win", "break_lose",
    # filled by grade()
    "outcome_bounce", "outcome_break", "graded_at",
]


def _path(symbol: str, timeframe: str) -> str:
    return pair_path(normalize_symbol(symbol), normalize_timeframe(timeframe),
                     "signal_journal.csv")


def _load(symbol: str, timeframe: str) -> pd.DataFrame:
    p = _path(symbol, timeframe)
    if os.path.exists(p):
        return pd.read_csv(p, dtype={"asof": str})
    return pd.DataFrame(columns=_COLS)


def log_reading(reading: dict[str, Any]) -> bool:
    """Append one journal row for a state=="signal" reading. Dedupes on
    `asof` (the API recomputes every 30s; one row per closed bar). Returns
    True when a row was written."""
    if reading.get("state") != "signal":
        return False
    symbol, timeframe = reading["symbol"], reading["timeframe"]
    df = _load(symbol, timeframe)
    if len(df) and (df["asof"] == str(reading["asof"])).any():
        return False

    e = reading["event"]
    price = float(reading["price"])
    a = float(reading.get("atr_abs") or 0.0)
    if a <= 0:
        return False
    demand = e["zone_kind"] == "demand"
    row = {
        "asof": str(reading["asof"]),
        "generated_at": reading["generated_at"],
        "price": price, "atr_abs": a,
        "trigger": e["trigger"], "zone_kind": e["zone_kind"],
        "source": e["source"], "zone_top": e["zone_top"],
        "zone_bottom": e["zone_bottom"],
        "p_bounce": reading["tier_a_bounce"]["probability"],
        "bounce_is_prob": bool(reading["tier_a_bounce"]["is_probability"]),
        "p_break": reading["tier_a_break"]["probability"],
        "break_is_prob": bool(reading["tier_a_break"]["is_probability"]),
        "payoff_b": reading["tier_a_break"].get("payoff_b"),
        # same barrier definitions as bounce_break_labeler (§5.2)
        "bounce_win": price + 2 * a if demand else price - 2 * a,
        "bounce_lose": price - a if demand else price + a,
        "break_win": (e["zone_bottom"] - a) if demand else (e["zone_top"] + a),
        "break_lose": price + 2 * a if demand else price - 2 * a,
        "outcome_bounce": np.nan, "outcome_break": np.nan, "graded_at": "",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(_path(symbol, timeframe), index=False)
    return True


def grade(symbol: str, timeframe: str, bars: pd.DataFrame) -> int:
    """Grade every ungraded row whose horizon window has fully elapsed in
    `bars` (cleaned, naive-UTC OHLC). Returns the number of rows graded."""
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    horizon = tf_horizon(timeframe)
    df = _load(symbol, timeframe)
    if not len(df):
        return 0
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    idx = pd.to_datetime(df["asof"])
    graded = 0
    pending = df["graded_at"].isna() | (df["graded_at"].astype(str).str.strip() == "")
    # an all-NaN CSV column loads as float64; pandas 3 refuses string
    # assignment into it without an explicit dtype change
    df["graded_at"] = df["graded_at"].astype("object")
    for i in df.index[pending]:
        pos = bars.index.get_indexer([idx[i]])[0]
        if pos < 0 or pos + horizon >= len(bars):
            continue  # signal bar not in window yet / horizon incomplete
        demand = df.at[i, "zone_kind"] == "demand"
        if demand:
            ob, _ = _race_up_wins(high, low, pos, horizon,
                                  df.at[i, "bounce_win"], df.at[i, "bounce_lose"])
            ok, _ = _race_down_wins(high, low, pos, horizon,
                                    df.at[i, "break_win"], df.at[i, "break_lose"])
        else:
            ob, _ = _race_down_wins(high, low, pos, horizon,
                                    df.at[i, "bounce_win"], df.at[i, "bounce_lose"])
            ok, _ = _race_up_wins(high, low, pos, horizon,
                                  df.at[i, "break_win"], df.at[i, "break_lose"])
        df.at[i, "outcome_bounce"] = ob
        df.at[i, "outcome_break"] = ok
        df.at[i, "graded_at"] = pd.Timestamp.utcnow().isoformat()
        graded += 1
    if graded:
        df.to_csv(_path(symbol, timeframe), index=False)
    return graded


def summary(symbol: str, timeframe: str, min_n: int = 20) -> dict[str, Any]:
    """Live track record: predicted vs observed per model, overall and by
    predicted-probability tercile (break only — bounce has no live model).
    Rates are withheld below `min_n` graded signals."""
    df = _load(symbol, timeframe)
    g = df[~df["outcome_break"].isna()]
    out: dict[str, Any] = {
        "n_signals": int(len(df)), "n_graded": int(len(g)),
        "min_n": min_n,
        "verdict_available": bool(len(g) >= min_n),
    }
    if not len(g):
        return out

    def _block(p: pd.Series, y: pd.Series) -> dict[str, Any]:
        return {"n": int(len(y)), "predicted_mean": round(float(p.mean()), 4),
                "observed_rate": round(float(y.mean()), 4)}

    gb = g[g["break_is_prob"].astype(bool)]
    if len(gb):
        block = {"overall": _block(gb["p_break"], gb["outcome_break"])}
        if len(gb) >= min_n:
            terciles = pd.qcut(gb["p_break"], 3, labels=["low", "mid", "high"],
                               duplicates="drop")
            for name, grp in gb.groupby(terciles, observed=True):
                block[f"tercile_{name}"] = _block(grp["p_break"],
                                                  grp["outcome_break"])
        out["break"] = block
    gB = g[~g["bounce_is_prob"].astype(bool)]  # base-rate rows still checkable
    if len(gB):
        out["bounce_base_rate_check"] = _block(gB["p_bounce"],
                                               gB["outcome_bounce"])
    out["recent"] = g.tail(10)[
        ["asof", "trigger", "zone_kind", "p_break", "outcome_break",
         "p_bounce", "outcome_bounce"]].to_dict("records")
    return out


if __name__ == "__main__":
    import sys

    from feature_engineer import _clean_index
    from live_engine import _pull_bars
    args = sys.argv[1:]
    if len(args) == 2:
        bars = _clean_index(_pull_bars(args[0], args[1]))
        n = grade(args[0], args[1], bars)
        print(f"[signal_journal] graded {n} new rows")
        import json
        print(json.dumps(summary(args[0], args[1]), indent=2, default=str))
    else:
        print("Usage: python signal_journal.py SYMBOL TIMEFRAME")
