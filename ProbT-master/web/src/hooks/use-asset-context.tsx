"use client";

import { createContext, useContext, useState, useCallback, useEffect, ReactNode } from "react";

export const AVAILABLE_SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "NAS100", "SPX500"] as const;
export const AVAILABLE_TIMEFRAMES = ["1m", "5m", "15m", "1H", "4H", "1D", "1W"] as const;

export type Symbol = (typeof AVAILABLE_SYMBOLS)[number];
export type Timeframe = (typeof AVAILABLE_TIMEFRAMES)[number];

interface AssetState {
  symbol: Symbol;
  timeframe: Timeframe;
  setSymbol: (s: Symbol) => void;
  setTimeframe: (t: Timeframe) => void;
  pairKey: string;
  /** True while pair-dependent data is reloading after a user change. */
  transitioning: boolean;
  /** Mark the transition complete (called by the coordinator once data settles). */
  clearTransition: () => void;
}

const DEFAULT_SYMBOL: Symbol = "XAUUSD";
const DEFAULT_TIMEFRAME: Timeframe = "1D";

const AssetContext = createContext<AssetState | null>(null);

const STORAGE_KEY = "probt.asset";

function loadInitial(): { symbol: Symbol; timeframe: Timeframe } {
  if (typeof window === "undefined") return { symbol: DEFAULT_SYMBOL, timeframe: DEFAULT_TIMEFRAME };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { symbol: DEFAULT_SYMBOL, timeframe: DEFAULT_TIMEFRAME };
    const parsed = JSON.parse(raw);
    const symbol = (AVAILABLE_SYMBOLS as readonly string[]).includes(parsed.symbol)
      ? (parsed.symbol as Symbol)
      : DEFAULT_SYMBOL;
    const timeframe = (AVAILABLE_TIMEFRAMES as readonly string[]).includes(parsed.timeframe)
      ? (parsed.timeframe as Timeframe)
      : DEFAULT_TIMEFRAME;
    return { symbol, timeframe };
  } catch {
    return { symbol: DEFAULT_SYMBOL, timeframe: DEFAULT_TIMEFRAME };
  }
}

export function AssetProvider({ children }: { children: ReactNode }) {
  const [initial] = useState(loadInitial);
  const [symbol, setSymbolState] = useState<Symbol>(initial.symbol);
  const [timeframe, setTimeframeState] = useState<Timeframe>(initial.timeframe);
  const [transitioning, setTransitioning] = useState(false);

  // Persist on change
  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ symbol, timeframe }));
    } catch {
      /* ignore */
    }
  }, [symbol, timeframe]);

  const setSymbol = useCallback((s: Symbol) => {
    setSymbolState((prev) => {
      if (prev !== s) setTransitioning(true);
      return s;
    });
  }, []);

  const setTimeframe = useCallback((t: Timeframe) => {
    setTimeframeState((prev) => {
      if (prev !== t) setTransitioning(true);
      return t;
    });
  }, []);

  const clearTransition = useCallback(() => setTransitioning(false), []);

  const pairKey = `${symbol}_${timeframe}`;

  return (
    <AssetContext.Provider
      value={{ symbol, timeframe, setSymbol, setTimeframe, pairKey, transitioning, clearTransition }}
    >
      {children}
    </AssetContext.Provider>
  );
}

export function useAsset() {
  const ctx = useContext(AssetContext);
  if (!ctx) throw new Error("useAsset must be used within <AssetProvider>");
  return ctx;
}
