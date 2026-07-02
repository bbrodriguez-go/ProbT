"use client";

import { QueryClient, useQuery, keepPreviousData } from "@tanstack/react-query";
import type {
  Reading,
  ReadingV2,
  JournalSummary,
  BacktestResponse,
  Kpi,
  Trade,
  ModelCard,
  MarketAsset,
  HeatmapData,
  Gauge,
  Insight,
  FeatureImportance,
  HealthStatus,
  ChartResponse,
} from "@/types";
import { API_URL } from "@/lib/constants";
import { useAsset } from "@/hooks/use-asset-context";

// ─── Query client (singleton) ─────────────────────────────────────
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      gcTime: 5 * 60_000,
      refetchOnWindowFocus: false,
      retry: 1,
      placeholderData: keepPreviousData,
    },
  },
});

// ─── Base fetcher ─────────────────────────────────────────────────
async function fetcher<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json();
}

function pairQuery(symbol: string, timeframe: string) {
  return `symbol=${encodeURIComponent(symbol)}&timeframe=${encodeURIComponent(timeframe)}`;
}

// ─── Query keys ───────────────────────────────────────────────────
export const qk = {
  health: ["health"] as const,
  reading: (s: string, t: string) => ["reading", s, t] as const,
  readingV2: (s: string, t: string) => ["reading-v2", s, t] as const,
  journal: (s: string, t: string) => ["journal", s, t] as const,
  backtest: (s: string, t: string) => ["backtest", s, t] as const,
  kpis: (s: string, t: string) => ["kpis", s, t] as const,
  probDist: (s: string, t: string) => ["probability-dist", s, t] as const,
  trades: (s: string, t: string, n: number) => ["trades", s, t, n] as const,
  models: (s: string, t: string) => ["models", s, t] as const,
  market: ["market"] as const,
  heatmap: (s: string, t: string) => ["heatmap", s, t] as const,
  confidence: (s: string, t: string) => ["confidence", s, t] as const,
  insights: (s: string, t: string) => ["insights", s, t] as const,
  features: (s: string, t: string) => ["features", s, t] as const,
  chart: (s: string, t: string) => ["chart", s, t] as const,
};

// ─── Hooks ────────────────────────────────────────────────────────
export const useHealth = () =>
  useQuery({ queryKey: qk.health, queryFn: () => fetcher<HealthStatus>("/api/health"), refetchInterval: 30_000 });

export const useReading = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.reading(symbol, timeframe),
    queryFn: () => fetcher<Reading>(`/api/reading?${pairQuery(symbol, timeframe)}`),
    refetchInterval: 30_000,
  });
};

export const useReadingV2 = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.readingV2(symbol, timeframe),
    queryFn: () => fetcher<ReadingV2>(`/api/reading_v2?${pairQuery(symbol, timeframe)}`),
    refetchInterval: 30_000,
  });
};

export const useJournal = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.journal(symbol, timeframe),
    queryFn: () => fetcher<JournalSummary>(`/api/journal?${pairQuery(symbol, timeframe)}`),
    refetchInterval: 300_000,
  });
};

export const useBacktest = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.backtest(symbol, timeframe),
    queryFn: () => fetcher<BacktestResponse>(`/api/backtest?${pairQuery(symbol, timeframe)}`),
  });
};

export const useKpis = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.kpis(symbol, timeframe),
    queryFn: () => fetcher<Kpi[]>(`/api/kpis?${pairQuery(symbol, timeframe)}`),
  });
};

export const useProbabilityDist = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.probDist(symbol, timeframe),
    queryFn: () => fetcher<{ bins: number[]; counts: number[]; percentiles: Record<string, number>; current: number; mean: number; std: number }>(
      `/api/probability-dist?${pairQuery(symbol, timeframe)}`
    ),
  });
};

export const useTrades = (limit = 50) => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.trades(symbol, timeframe, limit),
    queryFn: () => fetcher<Trade[]>(`/api/trades?limit=${limit}&${pairQuery(symbol, timeframe)}`),
  });
};

export const useModels = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.models(symbol, timeframe),
    queryFn: () => fetcher<ModelCard[]>(`/api/models?${pairQuery(symbol, timeframe)}`),
  });
};

export const useMarket = () =>
  useQuery({ queryKey: qk.market, queryFn: () => fetcher<MarketAsset[]>("/api/market"), refetchInterval: 30_000 });

export const useHeatmap = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.heatmap(symbol, timeframe),
    queryFn: () => fetcher<HeatmapData>(`/api/heatmap?${pairQuery(symbol, timeframe)}`),
  });
};

export const useConfidence = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.confidence(symbol, timeframe),
    queryFn: () => fetcher<{ gauges: Gauge[] }>(`/api/confidence?${pairQuery(symbol, timeframe)}`),
  });
};

export const useInsights = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.insights(symbol, timeframe),
    queryFn: () => fetcher<Insight[]>(`/api/insights?${pairQuery(symbol, timeframe)}`),
  });
};

export const useFeatures = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.features(symbol, timeframe),
    queryFn: () => fetcher<FeatureImportance[]>(`/api/features?${pairQuery(symbol, timeframe)}`),
  });
};

export const useChart = () => {
  const { symbol, timeframe } = useAsset();
  return useQuery({
    queryKey: qk.chart(symbol, timeframe),
    queryFn: () => fetcher<ChartResponse>(`/api/chart?bars=400&${pairQuery(symbol, timeframe)}`),
    staleTime: 60_000,
  });
};

export interface SystemMetrics {
  cpu_percent: number;
  memory_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  threads: number;
  processes: number;
  python_threads: number;
  pairs_available: number;
}

export const useSystem = () =>
  useQuery({
    queryKey: ["system"] as const,
    queryFn: () => fetcher<SystemMetrics>("/api/system"),
    refetchInterval: 5_000,
  });
