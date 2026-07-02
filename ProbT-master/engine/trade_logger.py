"""Layer 6 — trade_logger.py

Manual forward-test log. Two purposes:
  1. Verify calibration in production (model says 65% -> ~65% win rate in that band).
  2. Accumulate news_score history so it can become a Tier A feature later (~200 rows).

CLI: run with no arguments for an interactive prompt.
"""
import os
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LOG_PATH = os.path.join(DATA_DIR, "trade_log.csv")
COLUMNS = [
    "timestamp", "tier_a_prob", "tier_b_bias", "nearest_zone", "news_score",
    "action_taken", "outcome", "pnl_pct", "notes",
]


def log_reading(tier_a_prob: float, tier_b_bias: str, nearest_zone: str,
                news_score: float, action_taken: str, outcome: str,
                pnl_pct: float, notes: str = "") -> None:
    if not 0.0 <= tier_a_prob <= 1.0:
        raise ValueError(f"tier_a_prob must be in [0, 1], got {tier_a_prob}")
    os.makedirs(DATA_DIR, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tier_a_prob": tier_a_prob,
        "tier_b_bias": tier_b_bias,
        "nearest_zone": nearest_zone,
        "news_score": news_score,
        "action_taken": action_taken,
        "outcome": outcome,
        "pnl_pct": pnl_pct,
        "notes": notes,
    }])
    if os.path.exists(LOG_PATH):
        row.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_PATH, index=False)
    print(f"[trade_logger] entry logged -> {LOG_PATH}")


def _cli():
    print("=== trade_logger interactive ===")
    defaults = {
        "tier_a_prob": "0.65", "tier_b_bias": "bullish",
        "nearest_zone": "support@2341.50", "news_score": "0.5",
        "action_taken": "long", "outcome": "pending", "pnl_pct": "0",
        "notes": "",
    }
    vals = {}
    for k in COLUMNS[1:]:
        vals[k] = input(f"  {k} [{defaults.get(k, '')}]: ").strip() or defaults.get(k, "")
    log_reading(
        tier_a_prob=float(vals["tier_a_prob"]),
        tier_b_bias=vals["tier_b_bias"],
        nearest_zone=vals["nearest_zone"],
        news_score=float(vals["news_score"]),
        action_taken=vals["action_taken"],
        outcome=vals["outcome"],
        pnl_pct=float(vals["pnl_pct"]),
        notes=vals["notes"],
    )


if __name__ == "__main__":
    _cli()
