"""Layer 2b — volume_profile.py

Builds daily Volume Profile features from REAL CME volume (GC=F hourly).
MT4/MT5 broker tick-volume is NOT used — GC=F has centralized traded volume.

Single function: get_daily_vp_features(hourly_df) -> DataFrame indexed by date
with columns: poc_price, va_low, va_high.

Algorithm:
  1. For each day, distribute each hourly bar's volume evenly across its
     [low, high] range using 100 equally-spaced price bins.
  2. poc_price  = bin with highest accumulated volume (Point of Control).
  3. va_low/va_high = narrowest range containing 70% of the day's volume,
     expanding outward from the POC bin until 70% is reached.
  4. Returns NaN for days with < 3 hourly bars.

numpy + pandas only, under 50 lines.
"""
import numpy as np
import pandas as pd

_N_BINS = 100
_VALUE_AREA_PCT = 0.70


def _day_vp(day: pd.DataFrame):
    if len(day) < 3:
        return np.nan, np.nan, np.nan
    lo = float(day["low"].min())
    hi = float(day["high"].max())
    if hi <= lo:
        return np.nan, np.nan, np.nan
    edges = np.linspace(lo, hi, _N_BINS + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vol = np.zeros(_N_BINS)
    for _, row in day.iterrows():
        r_lo, r_hi = float(row["low"]), float(row["high"])
        if r_hi <= r_lo:
            continue
        mask = (centers >= r_lo) & (centers <= r_hi)
        n = int(mask.sum())
        if n > 0 and row["volume"] > 0:
            vol[mask] += float(row["volume"]) / n
    total = vol.sum()
    if total <= 0:
        return np.nan, np.nan, np.nan
    poc_idx = int(np.argmax(vol))
    poc_price = float(centers[poc_idx])
    # Expand outward from POC until 70% of total volume is captured
    acc = vol[poc_idx]
    lo_i, hi_i = poc_idx, poc_idx
    while acc < total * _VALUE_AREA_PCT:
        down = vol[lo_i - 1] if lo_i - 1 >= 0 else -1.0
        up = vol[hi_i + 1] if hi_i + 1 < _N_BINS else -1.0
        if down < 0 and up < 0:
            break
        if up >= down:
            hi_i += 1
            acc += vol[hi_i]
        else:
            lo_i -= 1
            acc += vol[lo_i]
    return poc_price, float(centers[lo_i]), float(centers[hi_i])


def get_daily_vp_features(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame indexed by calendar date with poc_price, va_low, va_high."""
    df = hourly_df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.normalize()
    else:
        dates = pd.to_datetime(df.index).normalize()
    rows = []
    for day, group in df.groupby(dates):
        poc, va_lo, va_hi = _day_vp(group)
        rows.append({"date": day, "poc_price": poc, "va_low": va_lo, "va_high": va_hi})
    out = pd.DataFrame(rows).set_index("date")
    out.index = pd.to_datetime(out.index)
    return out
