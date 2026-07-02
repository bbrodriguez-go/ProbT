"""Smart Money Concepts — faithful port of the LuxAlgo "Smart Money Concepts"
Pine indicator to server-side Python for the probt chart endpoint.

Single-pass, stateful scan over OHLCV bars producing JSON-friendly overlays:
  - swing pivots (HH / HL / LH / LL)
  - market structure breaks (BOS / CHoCH) for swing and internal scopes
  - order blocks (bullish / bearish, swing / internal) with ATR volatility filter
  - Fair Value Gaps (bullish / bearish)
  - Equal Highs / Equal Lows (EQH / EQL)
  - Premium / Equilibrium / Discount zones
  - Strong / Weak trailing High / Low

All indices are positional (iloc) integers into the trimmed output dataframe so
the web client can place boxes/lines on a category x-axis without time math.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

BULLISH = 1
BEARISH = -1


def _atr(tr: pd.Series, period: int = 200) -> pd.Series:
    return tr.rolling(period, min_periods=1).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr


def _parsed_hl(df: pd.DataFrame, atr_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Volatility-filtered highs/lows (LuxAlga parsed high/low).

    On a high-volatility bar (range >= 2*ATR) the extreme is inverted so an
    exhaustion spike does not become the OB reference.
    """
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = 2.0 * atr_arr
    high_vol = (high - low) >= vol
    parsed_high = np.where(high_vol, low, high)
    parsed_low = np.where(high_vol, high, low)
    return parsed_high, parsed_low


def _roll_max(a: np.ndarray, win: int) -> np.ndarray:
    """Max over the trailing `win` values ending at i (inclusive)."""
    out = np.full(len(a), np.nan)
    for i in range(win - 1, len(a)):
        out[i] = np.max(a[i - win + 1 : i + 1])
    return out


def _roll_min(a: np.ndarray, win: int) -> np.ndarray:
    out = np.full(len(a), np.nan)
    for i in range(win - 1, len(a)):
        out[i] = np.min(a[i - win + 1 : i + 1])
    return out


class _Pivot:
    __slots__ = ("current", "last", "crossed", "bar")

    def __init__(self) -> None:
        self.current: float = np.nan
        self.last: float = np.nan
        self.crossed: bool = False
        self.bar: int = -1


def compute_smc(
    df: pd.DataFrame,
    swing_length: int = 10,
    internal_length: int = 4,
    equal_threshold: float = 0.1,
    max_ob: int = 12,
    max_fvg: int = 20,
) -> dict[str, Any]:
    """Compute all SMC overlays for the given OHLCV dataframe.

    Returns a dict of JSON-serializable lists with positional bar indices.
    """
    n = len(df)
    if n < max(swing_length, internal_length) + 3:
        return _empty()

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)

    tr = _true_range(df)
    atr_arr = _atr(tr, 200).to_numpy(dtype=float)

    parsed_high, parsed_low = _parsed_hl(df, atr_arr)

    # rolling max/min of recent `size` bars (inclusive of current)
    swing_hi_roll = _roll_max(high, swing_length)
    swing_lo_roll = _roll_min(low, swing_length)
    int_hi_roll = _roll_max(high, internal_length)
    int_lo_roll = _roll_min(low, internal_length)

    swing_high = _Pivot()
    swing_low = _Pivot()
    int_high = _Pivot()
    int_low = _Pivot()

    swing_bias = 0
    int_bias = 0
    leg_swing = 0
    leg_int = 0
    prev_leg_swing = 0
    prev_leg_int = 0

    swings: list[dict[str, Any]] = []
    structures: list[dict[str, Any]] = []
    order_blocks: list[dict[str, Any]] = []
    eqhl: list[dict[str, Any]] = []
    fvgs: list[dict[str, Any]] = []

    def _new_pivot(scope: str, pivot: _Pivot, price: float, bar: int, is_high: bool) -> None:
        pivot.last = pivot.current
        pivot.current = price
        pivot.crossed = False
        pivot.bar = bar
        if scope == "swing":
            label = _classify(is_high, pivot.last, price)
            swings.append(
                {"scope": scope, "x": bar, "price": float(price),
                 "label": label, "is_high": is_high}
            )
            eq = _maybe_equal(is_high, pivot.last, price, bar, atr_arr, equal_threshold)
            if eq:
                eqhl.append(eq)

    def _maybe_equal(is_high: bool, prev: float, cur: float, bar: int,
                     atr_a: np.ndarray, thr: float) -> dict[str, Any] | None:
        if not np.isfinite(prev):
            return None
        if atr_a[bar] <= 0 or np.isnan(atr_a[bar]):
            return None
        if abs(prev - cur) > thr * atr_a[bar]:
            return None
        return {
            "kind": "EQH" if is_high else "EQL",
            "x": bar,
            "price": float(cur),
            "prev_price": float(prev),
        }

    def _classify(is_high: bool, prev: float, cur: float) -> str:
        if not np.isfinite(prev):
            return "HH" if is_high else "LL"
        if is_high:
            return "HH" if cur > prev else "LH"
        return "LL" if cur < prev else "HL"

    def _store_ob(scope: str, bias: int, pivot_bar: int, end_bar: int) -> None:
        lo = parsed_low[pivot_bar : end_bar + 1]
        hi = parsed_high[pivot_bar : end_bar + 1]
        if len(lo) == 0:
            return
        if bias == BULLISH:
            local = int(np.argmin(lo))
            ob_bar = pivot_bar + local
            top = float(parsed_high[ob_bar])
            bottom = float(parsed_low[ob_bar])
        else:
            local = int(np.argmax(hi))
            ob_bar = pivot_bar + local
            top = float(parsed_high[ob_bar])
            bottom = float(parsed_low[ob_bar])
        order_blocks.append(
            {"scope": scope, "bias": "bull" if bias == BULLISH else "bear",
             "x": ob_bar, "top": top, "bottom": bottom,
             "mitigated": False,
             # Bar at which the OB became KNOWN (the structure-break bar) —
             # `x` points into the past, so causal consumers must gate on
             # `born`, not `x`, to avoid lookahead (zone_events.py).
             "born": end_bar}
        )

    # ─── main scan ──────────────────────────────────────────────────
    for i in range(n):
        # leg detection (size = swing / internal)
        if i >= swing_length and np.isfinite(swing_hi_roll[i]) and np.isfinite(swing_lo_roll[i]):
            new_hi = high[i - swing_length] > swing_hi_roll[i]
            new_lo = low[i - swing_length] < swing_lo_roll[i]
            if new_hi:
                leg_swing = 0
            elif new_lo:
                leg_swing = 1
        if i >= internal_length and np.isfinite(int_hi_roll[i]) and np.isfinite(int_lo_roll[i]):
            new_hi = high[i - internal_length] > int_hi_roll[i]
            new_lo = low[i - internal_length] < int_lo_roll[i]
            if new_hi:
                leg_int = 0
            elif new_lo:
                leg_int = 1

        # pivot confirmed for swing scope
        if leg_swing != prev_leg_swing:
            if leg_swing == 0:  # bearish leg → new swing high
                _new_pivot("swing", swing_high, high[i - swing_length],
                           i - swing_length, True)
            else:  # bullish leg → new swing low
                _new_pivot("swing", swing_low, low[i - swing_length],
                           i - swing_length, False)
        if leg_int != prev_leg_int:
            if leg_int == 0:
                _new_pivot("internal", int_high, high[i - internal_length],
                           i - internal_length, True)
            else:
                _new_pivot("internal", int_low, low[i - internal_length],
                           i - internal_length, False)
        prev_leg_swing = leg_swing
        prev_leg_int = leg_int

        # structure breaks (crossover / crossunder of stored pivot level)
        if (np.isfinite(swing_high.current) and not swing_high.crossed
                and close[i] > swing_high.current and close[i - 1] <= swing_high.current):
            tag = "CHoCH" if swing_bias == BEARISH else "BOS"
            swing_bias = BULLISH
            swing_high.crossed = True
            structures.append({"scope": "swing", "bias": "bull", "type": tag,
                               "x1": swing_high.bar, "x2": i,
                               "price": float(swing_high.current)})
            _store_ob("swing", BULLISH, swing_high.bar, i)
        if (np.isfinite(swing_low.current) and not swing_low.crossed
                and close[i] < swing_low.current and close[i - 1] >= swing_low.current):
            tag = "CHoCH" if swing_bias == BULLISH else "BOS"
            swing_bias = BEARISH
            swing_low.crossed = True
            structures.append({"scope": "swing", "bias": "bear", "type": tag,
                               "x1": swing_low.bar, "x2": i,
                               "price": float(swing_low.current)})
            _store_ob("swing", BEARISH, swing_low.bar, i)
        if (np.isfinite(int_high.current) and not int_high.crossed
                and close[i] > int_high.current and close[i - 1] <= int_high.current):
            tag = "CHoCH" if int_bias == BEARISH else "BOS"
            int_bias = BULLISH
            int_high.crossed = True
            structures.append({"scope": "internal", "bias": "bull", "type": tag,
                               "x1": int_high.bar, "x2": i,
                               "price": float(int_high.current)})
            _store_ob("internal", BULLISH, int_high.bar, i)
        if (np.isfinite(int_low.current) and not int_low.crossed
                and close[i] < int_low.current and close[i - 1] >= int_low.current):
            tag = "CHoCH" if int_bias == BULLISH else "BOS"
            int_bias = BEARISH
            int_low.crossed = True
            structures.append({"scope": "internal", "bias": "bear", "type": tag,
                               "x1": int_low.bar, "x2": i,
                               "price": float(int_low.current)})
            _store_ob("internal", BEARISH, int_low.bar, i)

        # Fair Value Gap (3-candle imbalance)
        if i >= 2:
            h_prev, h_cur = high[i - 2], high[i]
            l_prev, l_cur = low[i - 2], low[i]
            if l_cur > h_prev:
                fvgs.append({"bias": "bull", "x": i - 1, "top": float(l_cur),
                             "bottom": float(h_prev), "mitigated": False})
            elif h_cur < l_prev:
                fvgs.append({"bias": "bear", "x": i - 1, "top": float(l_prev),
                             "bottom": float(h_cur), "mitigated": False})

    # ─── mitigation + dedupe ────────────────────────────────────────
    _mitigate_ob(order_blocks, high, low, close)
    _mitigate_fvg(fvgs, high, low)

    ob_swing = [o for o in order_blocks if o["scope"] == "swing" and not o["mitigated"]][-max_ob:]
    ob_int = [o for o in order_blocks if o["scope"] == "internal" and not o["mitigated"]][-max_ob:]
    fvg_live = [f for f in fvgs if not f["mitigated"]][-max_fvg:]

    # ─── premium / discount / equilibrium (last swing high & low) ────
    top = swing_high.current if np.isfinite(swing_high.current) else float(np.max(high))
    bot = swing_low.current if np.isfinite(swing_low.current) else float(np.min(low))
    zones = {
        "top": float(top),
        "bottom": float(bot),
        "equilibrium": float((top + bot) / 2),
        "premium": [float(0.95 * top + 0.05 * bot), float(top)],
        "discount": [float(bot), float(0.95 * bot + 0.05 * top)],
    }

    bias = "bull" if swing_bias == BULLISH else "bear" if swing_bias == BEARISH else "neutral"

    return {
        "swings": swings,
        "structures": structures,
        "order_blocks_swing": ob_swing,
        "order_blocks_internal": ob_int,
        "fvgs": fvg_live,
        # Unfiltered lists (incl. mitigated, with mitigated_at when set) for
        # causal historical reconstruction in zone_events.py. The filtered
        # keys above keep their original display-oriented semantics.
        "order_blocks_all": order_blocks,
        "fvgs_all": fvgs,
        "eqhl": eqhl,
        "zones": zones,
        "bias": bias,
        "strong_weak": {
            "top": float(top),
            "bottom": float(bot),
            "top_bar": int(swing_high.bar) if swing_high.bar >= 0 else 0,
            "bottom_bar": int(swing_low.bar) if swing_low.bar >= 0 else 0,
            "bias": bias,
        },
    }


def _mitigate_ob(obs: list[dict[str, Any]], high: np.ndarray, low: np.ndarray,
                 close: np.ndarray) -> None:
    """Mark an order block mitigated when price closes beyond it."""
    for o in obs:
        bar = o["x"]
        if bar >= len(close):
            continue
        end = len(close)
        if o["bias"] == "bull":
            for j in range(bar + 1, end):
                if low[j] < o["bottom"]:
                    o["mitigated"] = True
                    o["mitigated_at"] = j
                    break
        else:
            for j in range(bar + 1, end):
                if high[j] > o["top"]:
                    o["mitigated"] = True
                    o["mitigated_at"] = j
                    break


def _mitigate_fvg(fvgs: list[dict[str, Any]], high: np.ndarray, low: np.ndarray) -> None:
    for f in fvgs:
        bar = f["x"]
        end = len(low)
        if f["bias"] == "bull":
            for j in range(bar + 1, end):
                if low[j] < f["bottom"]:
                    f["mitigated"] = True
                    f["mitigated_at"] = j
                    break
        else:
            for j in range(bar + 1, end):
                if high[j] > f["top"]:
                    f["mitigated"] = True
                    f["mitigated_at"] = j
                    break


def _empty() -> dict[str, Any]:
    return {
        "swings": [], "structures": [],
        "order_blocks_swing": [], "order_blocks_internal": [],
        "fvgs": [], "order_blocks_all": [], "fvgs_all": [],
        "eqhl": [], "zones": None, "bias": "neutral",
        "strong_weak": None,
    }
