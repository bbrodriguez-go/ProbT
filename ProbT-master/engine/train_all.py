"""train_all.py — bulk pipeline runner for every (symbol, timeframe) pair.

For each pair in symbols.all_keys():
  1. data_collector.collect(symbol, tf)
  2. feature_engineer.build(symbol, tf)
  3. labeler.label(symbol, tf)
  4. model_trainer.train(symbol, tf)         (skipped if rows < tf_min_bars)

Usage:
  python train_all.py                 # every pair
  python train_all.py XAUUSD          # every timeframe for XAUUSD
  python train_all.py XAUUSD 1D       # one pair

Also writes data/symbols/_index.json summarizing what got trained.
"""
from __future__ import annotations

import json
import sys
import time

# Force UTF-8 stdout/stderr on Windows so unicode arrows/etc don't crash
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import data_collector
import feature_engineer
import labeler
import model_trainer
from symbols import all_keys, normalize_symbol, symbols_dir
import os


def run_pair(symbol: str, timeframe: str) -> dict:
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE  {symbol} {timeframe}")
    print(f"{'=' * 70}")
    t0 = time.time()
    summary = {"symbol": symbol, "timeframe": timeframe, "trained": False,
               "rows": 0, "brier": None, "error": None}
    try:
        meta = data_collector.collect(symbol, timeframe)
        summary["rows"] = meta["rows"]
        summary["ticker"] = meta["ticker"]

        feature_engineer.build(symbol, timeframe)
        labeler.label(symbol, timeframe)
        metrics = model_trainer.train(symbol, timeframe)
        if metrics:
            summary["trained"] = True
            summary["brier"] = metrics["cv_brier_score"]
            summary["features"] = metrics["n_features"]
        else:
            summary["error"] = "skipped (insufficient rows for ML)"
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        print(f"[train_all] ERROR on {symbol} {timeframe}: {e}")
    summary["seconds"] = round(time.time() - t0, 1)
    return summary


def run(symbol_filter: str | None = None, tf_filter: str | None = None) -> list[dict]:
    pairs = all_keys()
    if symbol_filter:
        symbol_filter = normalize_symbol(symbol_filter)
        pairs = [p for p in pairs if p[0] == symbol_filter]
    if tf_filter:
        pairs = [p for p in pairs if p[1] == tf_filter]

    print(f"[train_all] running {len(pairs)} pair(s): "
          f"{', '.join(f'{s}_{t}' for s, t in pairs)}")
    results = []
    for symbol, tf in pairs:
        results.append(run_pair(symbol, tf))

    # ─── Persist index ────────────────────────────────────────────
    index_path = os.path.join(symbols_dir(), "_index.json")
    with open(index_path, "w") as f:
        json.dump({
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pairs": results,
        }, f, indent=2)

    # ─── Summary ──────────────────────────────────────────────────
    trained = [r for r in results if r["trained"]]
    failed = [r for r in results if r["error"] and not r["trained"]]
    print(f"\n{'=' * 70}")
    print(f"[train_all] DONE — {len(trained)}/{len(results)} trained, "
          f"{len(failed)} skipped/failed")
    for r in results:
        flag = "OK" if r["trained"] else "SKIP"
        bri = f"Brier={r['brier']}" if r["brier"] is not None else r.get("error", "")
        print(f"  [{flag}] {r['symbol']:7s} {r['timeframe']:4s} "
              f"rows={r['rows']:>5} {bri}")
    print(f"[train_all] index -> {index_path}")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 0:
        run()
    elif len(args) == 1:
        run(args[0])
    elif len(args) == 2:
        run(args[0], args[1])
    else:
        print("Usage: python train_all.py [SYMBOL] [TIMEFRAME]")
