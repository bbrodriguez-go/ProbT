"""Support/Resistance zone clustering from confirmed swings.

Exports: detect_zones, nearest_zone.
"""
from typing import List, Optional

import numpy as np
import pandas as pd

from smc import find_swings


def detect_zones(
    df: pd.DataFrame, lookback: int = 5, cluster_pct: float = 0.003
) -> List[dict]:
    """Cluster swing highs/lows into S/R zones.

    cluster_pct = max price distance (as fraction of price) to merge into one zone.
    Greedy 1D clustering on sorted prices.
    """
    swings = find_swings(df, lookback)
    if not swings:
        return []
    prices = np.array([s.price for s in swings])
    kinds = np.array([s.kind for s in swings])
    order = np.argsort(prices)
    prices, kinds = prices[order], kinds[order]

    zones: List[dict] = []
    used = set()
    for i in range(len(prices)):
        if i in used:
            continue
        cluster = [i]
        tol = prices[i] * cluster_pct
        for j in range(i + 1, len(prices)):
            if j in used:
                continue
            if prices[j] - prices[cluster[-1]] <= tol:
                cluster.append(j)
            else:
                break
        for k in cluster:
            used.add(k)
        members = prices[cluster]
        member_kinds = kinds[cluster]
        lows = int(np.sum(member_kinds == "low"))
        highs = int(np.sum(member_kinds == "high"))
        kind = "support" if lows >= highs else "resistance"
        zones.append(
            {
                "price": float(np.mean(members)),
                "top": float(np.max(members)),
                "bottom": float(np.min(members)),
                "kind": kind,
                "touches": len(cluster),
            }
        )
    return zones


def nearest_zone(
    price: float, zones: List[dict], kind: Optional[str] = None
) -> Optional[dict]:
    """Nearest zone to price (by mid). Optionally filter by kind."""
    candidates = [z for z in zones if kind is None or z["kind"] == kind]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda z: abs(z["price"] - price))
    out = dict(nearest)
    out["distance_pct"] = abs(out["price"] - price) / price if price else 0.0
    return out
