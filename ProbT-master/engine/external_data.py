"""probt v2 — external macro data feeds (master prompt §6, D1 + D3).

  D1  FRED DFII10 (10Y TIPS real yield), daily. Gold's most reliable macro
      anchor is inverse real yields.
  D3  CFTC COT, gold futures (code 088691), disaggregated futures-only:
      managed-money net position as a percentile of its trailing 3 years.

Both are free, keyless endpoints, cached under data/external/ with a
manifest (§6). Fetchers degrade gracefully: on network failure they serve
the cached copy (any age) or return None.

CAUSALITY:
  - DFII10 for day D is treated as available from day D+1 (shift 1 day).
  - A COT report is stamped Tuesday but PUBLISHED Friday ~15:30 ET; its
    value becomes effective report_date + 4 days. Using it from the report
    date would be a 3-day lookahead.

Observed ablation effect (2026-07-02, XAUUSD 1H n=4677/3882, 4H n=1195/999,
purged walk-forward, gate = +0.01 BSS): FAILS THE GATE ON EVERY MODEL/TF
(dBSS +0.0001 bounce-1H, -0.0054 break-1H, -0.0005 bounce-4H, -0.0039
break-4H). Macro positioning does not time zone touches — same conclusion
as Family P's copula features. NOT added to any live model. These feeds are
retained for the REGIME CONTEXT display only (§9.2 card 4: real-yield 1d
change; COT crowding percentile), which is explicitly labeled "context, not
signal" in the UI. Reproduce: `python external_data.py SYMBOL TF`.
"""
from __future__ import annotations

import io
import json
import os
import time

import pandas as pd
import requests

from symbols import data_dir

_EXT_DIR = os.path.join(data_dir(), "external")
_MANIFEST = os.path.join(_EXT_DIR, "manifest.json")
_TTL_S = 24 * 3600

FRED_DFII10_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10"
CFTC_URL = ("https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
            "?cftc_contract_market_code=088691"
            "&$select=report_date_as_yyyy_mm_dd,m_money_positions_long_all,"
            "m_money_positions_short_all,open_interest_all"
            "&$order=report_date_as_yyyy_mm_dd&$limit=10000")

COT_PCTILE_WEEKS = 156  # 3 years
COT_PUBLICATION_LAG_DAYS = 4  # Tuesday report -> Friday release

FEATURES: list[str] = [
    "dfii10_change_1d",
    "dfii10_change_5d",
    "cot_mm_net_pctile",
]


def _update_manifest(name: str, rows: int, source: str) -> None:
    manifest = {}
    if os.path.exists(_MANIFEST):
        try:
            manifest = json.load(open(_MANIFEST))
        except (json.JSONDecodeError, OSError):
            manifest = {}
    manifest[name] = {"fetched_at": pd.Timestamp.utcnow().isoformat(),
                      "rows": rows, "source": source}
    os.makedirs(_EXT_DIR, exist_ok=True)
    with open(_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def _cached_or_fetch(name: str, source: str, fetch_fn) -> pd.DataFrame | None:
    """Serve the cache when younger than _TTL_S; otherwise refetch (falling
    back to a stale cache on network failure)."""
    path = os.path.join(_EXT_DIR, f"{name}.csv")
    fresh = os.path.exists(path) and (time.time() - os.path.getmtime(path)) < _TTL_S
    if fresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    try:
        df = fetch_fn()
        os.makedirs(_EXT_DIR, exist_ok=True)
        df.to_csv(path)
        _update_manifest(name, len(df), source)
        return df
    except Exception as e:  # network / schema failure -> stale cache or None
        print(f"[external_data] {name} fetch failed ({e}); "
              f"{'serving stale cache' if os.path.exists(path) else 'no cache'}")
        if os.path.exists(path):
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return None


def fetch_dfii10() -> pd.DataFrame | None:
    """Daily 10Y TIPS real yield, indexed by observation date. Column: dfii10."""
    def _fetch() -> pd.DataFrame:
        r = requests.get(FRED_DFII10_URL, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = ["date", "dfii10"]
        df["dfii10"] = pd.to_numeric(df["dfii10"], errors="coerce")
        df = df.dropna()
        df.index = pd.to_datetime(df["date"])
        return df[["dfii10"]]

    return _cached_or_fetch("dfii10", FRED_DFII10_URL, _fetch)


def fetch_cot_gold() -> pd.DataFrame | None:
    """Weekly COT gold, indexed by report date (Tuesday). Columns:
    mm_long, mm_short, mm_net, open_interest."""
    def _fetch() -> pd.DataFrame:
        r = requests.get(CFTC_URL, timeout=60)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df = df.rename(columns={
            "report_date_as_yyyy_mm_dd": "date",
            "m_money_positions_long_all": "mm_long",
            "m_money_positions_short_all": "mm_short",
            "open_interest_all": "open_interest"})
        for c in ("mm_long", "mm_short", "open_interest"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["mm_net"] = df["mm_long"] - df["mm_short"]
        df = df.dropna(subset=["mm_net"])
        out = df.set_index(pd.to_datetime(df["date"]))[
            ["mm_long", "mm_short", "mm_net", "open_interest"]]
        return out.sort_index()

    return _cached_or_fetch("cot_gold", CFTC_URL, _fetch)


def macro_positioning_features(bar_days: pd.DatetimeIndex) -> pd.DataFrame:
    """FEATURES aligned to (naive-UTC, normalized) bar days, causally lagged
    per the module docstring. NaN columns when a source is unavailable."""
    days = pd.DatetimeIndex(bar_days).normalize()
    out = pd.DataFrame(index=days)

    dfii = fetch_dfii10()
    if dfii is not None and len(dfii):
        s = dfii["dfii10"]
        daily = pd.DataFrame({
            "dfii10_change_1d": s.diff(1),
            "dfii10_change_5d": s.diff(5),
        }).shift(1)  # available next day
        mapped = daily.reindex(days.unique().sort_values(), method="ffill")
        out["dfii10_change_1d"] = mapped["dfii10_change_1d"].loc[days].to_numpy()
        out["dfii10_change_5d"] = mapped["dfii10_change_5d"].loc[days].to_numpy()
    else:
        out["dfii10_change_1d"] = float("nan")
        out["dfii10_change_5d"] = float("nan")

    cot = fetch_cot_gold()
    if cot is not None and len(cot) > 10:
        net = cot["mm_net"]
        pct = net.rolling(COT_PCTILE_WEEKS, min_periods=52).rank(pct=True)
        eff = pct.copy()
        eff.index = eff.index + pd.Timedelta(days=COT_PUBLICATION_LAG_DAYS)
        mapped = eff.reindex(days.unique().sort_values(), method="ffill")
        out["cot_mm_net_pctile"] = mapped.loc[days].to_numpy()
    else:
        out["cot_mm_net_pctile"] = float("nan")

    out.index = bar_days
    return out


# ─── ablation experiment (upgrade #5) ────────────────────────────────
def _ablation_report(symbol: str, timeframe: str) -> None:
    from model_trainer_v2 import (LIVE_FEATURE_RECIPE, evaluate_feature_set,
                                  load_event_dataset)
    from quantum_features import (build_quantum_state_features,
                                  build_signal_features,
                                  harmonic_oscillator_energy)
    from zone_features import build_zone_features_bulk

    ds = load_event_dataset(symbol, timeframe)
    events, bars, fm = ds["events"], ds["bars"], ds["fm"]
    idxs = sorted(set(int(i) for i in events["bar_index"]))
    zf = build_zone_features_bulk(events, bars, symbol, timeframe)
    qsig = build_signal_features(bars, at_indices=idxs).reindex(events.index)
    qst = build_quantum_state_features(bars, fm[ds["baseline_cols"]], timeframe,
                                       macro=ds["macro"], at_indices=idxs
                                       ).reindex(events.index)
    close = bars["close"].to_numpy(float)
    qst["q6_harmonic_oscillator_energy"] = [
        harmonic_oscillator_energy(close[int(t)], top, bot)
        for t, top, bot in zip(events["bar_index"], events["zone_top"],
                               events["zone_bottom"])]
    macro_feats = macro_positioning_features(events.index)

    everything = pd.concat(
        [events[["bar_index", "label_bounce", "label_break"]],
         fm[ds["baseline_cols"]].reindex(events.index), qsig, qst, zf,
         macro_feats], axis=1, sort=False)

    print(f"[ablation macro-positioning] {symbol} {timeframe}")
    for name in ("bounce", "break"):
        cols = ds["baseline_cols"] + LIVE_FEATURE_RECIPE[name]
        need = list(dict.fromkeys(cols + FEATURES + ["bar_index", f"label_{name}"]))
        rows = everything[need].dropna()
        bar_idx = rows["bar_index"].to_numpy()
        y = rows[f"label_{name}"]
        base = evaluate_feature_set(rows[cols], y, bar_idx, embargo=ds["horizon"])
        plus = evaluate_feature_set(rows[cols + FEATURES], y, bar_idx,
                                    embargo=ds["horizon"])
        print(f"  {name:6s} n={len(rows)} | recipe BSS {base['bss']:+.4f} "
              f"AUC {base['roc_auc']:.3f} | +macro BSS {plus['bss']:+.4f} "
              f"AUC {plus['roc_auc']:.3f} | dBSS {plus['bss']-base['bss']:+.4f}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if len(args) == 2:
        _ablation_report(args[0], args[1])
    else:
        print("Usage: python external_data.py SYMBOL TIMEFRAME")
