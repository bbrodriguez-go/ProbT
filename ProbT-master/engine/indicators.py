"""Technical indicators — Ponytail-style minimal implementations.

Exports: ema, rsi, atr, macd_line, trend_bias, multi_timeframe_confluence.
Plain pandas/numpy, no TA-Lib dependency.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (adjust=False matches TradingView)."""
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Output range [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(100)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder). df must have high, low, close."""
    h, low, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat(
        [(h - low), (h - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def macd_line(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    return ema(close, fast) - ema(close, slow)


def macd_signal(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    return ema(macd_line(close, fast, slow), signal)


def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0):
    """Returns (middle, upper, lower, pct_b). pct_b in ~[0,1] (stationary)."""
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = mid + std * sd
    lower = mid - std * sd
    width = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / width
    return mid, upper, lower, pct_b


def trend_bias(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> str:
    """EMA cross bias on the LAST closed candle. Returns bullish/bearish/neutral."""
    c = df["close"]
    if len(c) < slow:
        return "neutral"
    ema_f, ema_s = ema(c, fast).iloc[-1], ema(c, slow).iloc[-1]
    if ema_f > ema_s:
        return "bullish"
    if ema_f < ema_s:
        return "bearish"
    return "neutral"


_BIAS_MAP = {"bullish": 1, "bearish": -1, "neutral": 0}


def encode_bias(bias: str) -> int:
    return _BIAS_MAP.get(bias, 0)


def multi_timeframe_confluence(timeframes: dict) -> dict:
    """Informational MTF vote. timeframes = {'1H': bias, '4H': bias, '1D': bias}.

    WARNING: This is a heuristic vote sum, NOT a probability. Do not feed the vote
    count to a model — features must be kept separate per timeframe so L1
    regularization can handle multicollinearity (see spec Section 2.4).
    """
    votes = [_BIAS_MAP.get(timeframes.get(k, "neutral"), 0) for k in timeframes]
    bull = sum(1 for v in votes if v == 1)
    bear = sum(1 for v in votes if v == -1)
    neutral = sum(1 for v in votes if v == 0)
    if bull > bear and bull > neutral:
        bias = "bullish"
    elif bear > bull and bear > neutral:
        bias = "bearish"
    else:
        bias = "neutral"
    return {
        "bias": bias,
        "votes_bull": bull,
        "votes_bear": bear,
        "votes_neutral": neutral,
        "confluence": bull == len(votes) or bear == len(votes),
    }
