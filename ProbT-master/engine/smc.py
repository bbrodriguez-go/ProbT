"""Smart Money Concepts — swing detection, BOS structure, order blocks, FVGs.

All detections operate on CLOSED candles only. No intraday order assumptions.
Ponytail-style: dataclasses for clarity, plain pandas loops, no external libs.

Exports: find_swings, detect_structure, detect_order_blocks, detect_fvg.
"""
from dataclasses import dataclass
from typing import List, Literal

import pandas as pd

SwingKind = Literal["high", "low"]
ZoneKind = Literal[
    "support", "resistance", "ob_bull", "ob_bear", "fvg_bull", "fvg_bear"
]


@dataclass
class Swing:
    index: int
    price: float
    kind: SwingKind


@dataclass
class Zone:
    price: float
    kind: ZoneKind
    top: float
    bottom: float
    touches: int = 1


def find_swings(df: pd.DataFrame, lookback: int = 5) -> List[Swing]:
    """Fractal swings: a candle is a swing high if its high is the max of a
    centered window of (2*lookback + 1) candles. Same (mirrored) for lows."""
    if len(df) < lookback * 2 + 1:
        return []
    win = lookback * 2 + 1
    highs = df["high"].rolling(win, center=True).max()
    lows = df["low"].rolling(win, center=True).min()
    swings: List[Swing] = []
    for i in range(len(df)):
        if pd.isna(highs.iloc[i]):
            continue
        if df["high"].iloc[i] == highs.iloc[i]:
            swings.append(Swing(i, float(df["high"].iloc[i]), "high"))
        if df["low"].iloc[i] == lows.iloc[i]:
            swings.append(Swing(i, float(df["low"].iloc[i]), "low"))
    return swings


def detect_structure(df: pd.DataFrame, lookback: int = 5) -> str:
    """Break of Structure (BOS) bias on the last closed candle.

    bullish: last close breaks above the previous swing high (higher high).
    bearish: last close breaks below the previous swing low (lower low).
    neutral: no break.
    """
    swings = find_swings(df, lookback)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    if len(highs) < 2 and len(lows) < 2:
        return "neutral"
    last = float(df["close"].iloc[-1])
    if len(highs) >= 2 and last > highs[-2].price:
        return "bullish"
    if len(lows) >= 2 and last < lows[-2].price:
        return "bearish"
    return "neutral"


def detect_order_blocks(df: pd.DataFrame, max_zones: int = 3) -> List[Zone]:
    """Order blocks: the last opposite-color candle before an impulsive move.

    A move at candle i is 'impulsive' if its body > 1.6x the previous body.
    The OB is candle i-1 (the pre-impulse candle).
    """
    zones: List[Zone] = []
    if len(df) < 3:
        return zones
    for i in range(2, len(df)):
        prev_body = abs(df["close"].iloc[i - 1] - df["open"].iloc[i - 1])
        cur_body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        if prev_body == 0 or cur_body < prev_body * 1.6:
            continue
        bullish = df["close"].iloc[i] > df["open"].iloc[i]
        zones.append(
            Zone(
                price=float(df["close"].iloc[i - 1]),
                kind="ob_bull" if bullish else "ob_bear",
                top=float(df["high"].iloc[i - 1]),
                bottom=float(df["low"].iloc[i - 1]),
            )
        )
        if len(zones) >= max_zones:
            break
    return zones


def detect_fvg(df: pd.DataFrame, max_zones: int = 10) -> List[Zone]:
    """Fair Value Gaps — 3-candle imbalance pattern.

    Bullish FVG: candle[i-1].high < candle[i+1].low (gap up).
    Bearish FVG: candle[i-1].low > candle[i+1].high (gap down).
    """
    zones: List[Zone] = []
    if len(df) < 3:
        return zones
    for i in range(1, len(df) - 1):
        h_prev, l_next = df["high"].iloc[i - 1], df["low"].iloc[i + 1]
        l_prev, h_next = df["low"].iloc[i - 1], df["high"].iloc[i + 1]
        if h_prev < l_next:
            zones.append(
                Zone(
                    price=float((h_prev + l_next) / 2),
                    kind="fvg_bull",
                    top=float(l_next),
                    bottom=float(h_prev),
                )
            )
        elif l_prev > h_next:
            zones.append(
                Zone(
                    price=float((l_prev + h_next) / 2),
                    kind="fvg_bear",
                    top=float(l_prev),
                    bottom=float(h_next),
                )
            )
        if len(zones) >= max_zones:
            break
    return zones
