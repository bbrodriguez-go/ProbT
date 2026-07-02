"""symbols.py — Multi-asset / multi-timeframe catalog.

Single source of truth for tradable symbols and timeframes across the engine,
the API gateway and the web client. Keep it dependency-free.

Symbol -> yfinance ticker mapping + macro-feature availability.
Timeframe -> yfinance interval + recommended period + min bars for training.
"""
from __future__ import annotations

import os

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_SYMBOLS_DIR = os.path.join(_DATA_DIR, "symbols")


def data_dir() -> str:
    return _DATA_DIR


def symbols_dir() -> str:
    return _SYMBOLS_DIR


def pair_dir(symbol: str, timeframe: str) -> str:
    """Directory for a (symbol, timeframe) model bundle. Created if missing."""
    symbol = normalize_symbol(symbol)
    timeframe = normalize_timeframe(timeframe)
    path = os.path.join(_SYMBOLS_DIR, f"{symbol}_{timeframe}")
    os.makedirs(path, exist_ok=True)
    return path


def pair_path(symbol: str, timeframe: str, filename: str) -> str:
    return os.path.join(pair_dir(symbol, timeframe), filename)


def pair_exists(symbol: str, timeframe: str) -> bool:
    return os.path.exists(
        pair_path(symbol, timeframe, "model.pkl")
    ) and os.path.exists(
        pair_path(symbol, timeframe, "features.json")
    )


def available_pairs() -> list[tuple[str, str]]:
    """List (symbol, timeframe) bundles that have a trained model on disk."""
    out = []
    if not os.path.isdir(_SYMBOLS_DIR):
        return out
    for d in sorted(os.listdir(_SYMBOLS_DIR)):
        full = os.path.join(_SYMBOLS_DIR, d)
        if not os.path.isdir(full):
            continue
        if "_" not in d:
            continue
        symbol, tf = d.split("_", 1)
        if symbol in SYMBOLS and tf in TIMEFRAMES:
            if os.path.exists(os.path.join(full, "model.pkl")):
                out.append((symbol, tf))
    return out


# ─── Symbols ────────────────────────────────────────────────────────
# `tickers` are tried in order by data_collector._download_first.
# `has_macro` controls whether DXY/VIX/TNX features are attached (only gold).
# `news_query` is the NewsAPI `q` parameter (kept here so it stays 1:1 with
# the symbol instead of being hardcoded in news_sentiment.py).

SYMBOLS: dict[str, dict] = {
    "XAUUSD": {
        "name": "Gold",
        "tickers": ["GC=F", "XAUUSD=X"],
        "has_macro": True,
        "macro": {"dxy": ["DX-Y.NYB", "UUP"], "vix": "^VIX", "tnx": "^TNX"},
        "news_query": 'gold OR XAUUSD OR "gold price"',
    },
    "BTCUSD": {
        "name": "Bitcoin",
        "tickers": ["BTC-USD"],
        "has_macro": False,
        "macro": None,
        "news_query": "bitcoin OR BTC",
    },
    "ETHUSD": {
        "name": "Ethereum",
        "tickers": ["ETH-USD"],
        "has_macro": False,
        "macro": None,
        "news_query": "ethereum OR ETH",
    },
    "NAS100": {
        "name": "Nasdaq 100",
        "tickers": ["^NDX", "^IXIC"],
        "has_macro": False,
        "macro": None,
        "news_query": "nasdaq",
    },
    "SPX500": {
        "name": "S&P 500",
        "tickers": ["^GSPC"],
        "has_macro": False,
        "macro": None,
        "news_query": '"S&P 500" OR SPX',
    },
}

DEFAULT_SYMBOL = "XAUUSD"


def is_valid_symbol(symbol: str) -> bool:
    return symbol in SYMBOLS


def normalize_symbol(symbol: str | None) -> str:
    if symbol and symbol in SYMBOLS:
        return symbol
    return DEFAULT_SYMBOL


def symbol_tickers(symbol: str) -> list[str]:
    return SYMBOLS[normalize_symbol(symbol)]["tickers"]


def symbol_name(symbol: str) -> str:
    return SYMBOLS[normalize_symbol(symbol)]["name"]


def symbol_has_macro(symbol: str) -> bool:
    return SYMBOLS[normalize_symbol(symbol)]["has_macro"]


def symbol_news_query(symbol: str) -> str:
    return SYMBOLS[normalize_symbol(symbol)]["news_query"]


# ─── Timeframes ────────────────────────────────────────────────────
# yfinance does not accept "1H" or "4H" — it uses "60m". For 4H we download
# 60m and resample. `min_bars` is the floor for training; below it the
# training step is skipped and the engine runs in Tier-B-only mode.

TIMEFRAMES: dict[str, dict] = {
    "1m":  {"interval": "1m",  "period": "7d",   "resample": None,  "min_bars": 300,  "horizon": 10},
    "5m":  {"interval": "5m",  "period": "60d",  "resample": None,  "min_bars": 1000, "horizon": 10},
    "15m": {"interval": "15m", "period": "60d",  "resample": None,  "min_bars": 1000, "horizon": 10},
    "1H":  {"interval": "60m", "period": "730d", "resample": None,  "min_bars": 1500, "horizon": 10},
    "4H":  {"interval": "60m", "period": "730d", "resample": "4h",  "min_bars": 800,  "horizon": 10},
    "1D":  {"interval": "1d",  "period": "max",  "resample": None,  "min_bars": 400,  "horizon": 10},
    "1W":  {"interval": "1wk", "period": "max",  "resample": None,  "min_bars": 200,  "horizon": 5},
}

DEFAULT_TIMEFRAME = "1D"


def is_valid_timeframe(tf: str) -> bool:
    return tf in TIMEFRAMES


def normalize_timeframe(tf: str | None) -> str:
    if tf and tf in TIMEFRAMES:
        return tf
    return DEFAULT_TIMEFRAME


def tf_interval(tf: str) -> str:
    return TIMEFRAMES[normalize_timeframe(tf)]["interval"]


def tf_period(tf: str) -> str:
    return TIMEFRAMES[normalize_timeframe(tf)]["period"]


def tf_resample(tf: str):
    return TIMEFRAMES[normalize_timeframe(tf)]["resample"]


def tf_min_bars(tf: str) -> int:
    return TIMEFRAMES[normalize_timeframe(tf)]["min_bars"]


def tf_horizon(tf: str) -> int:
    return TIMEFRAMES[normalize_timeframe(tf)]["horizon"]


# ─── Convenience iteration ─────────────────────────────────────────
def all_keys() -> list[tuple[str, str]]:
    """All (symbol, timeframe) combinations supported by the catalog."""
    return [(s, t) for s in SYMBOLS for t in TIMEFRAMES]
