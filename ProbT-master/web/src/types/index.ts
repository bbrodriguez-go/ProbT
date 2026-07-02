// ─── Domain types for the probt dashboard ─────────────────────────

export interface TierA {
  probability: number;
  probability_lo?: number;
  probability_hi?: number;
  ev_positive?: boolean;
  horizon_bars: number;
  tp_sl_ratio: string;
  is_probability: boolean;
  direction: "LONG" | "SHORT";
}

export interface NearestZone {
  kind: string | null;
  price: number | null;
  distance_pct: number | null;
  at_zone: boolean;
}

export interface TierB {
  bias: "bullish" | "bearish" | "neutral";
  nearest_zone: NearestZone;
  is_probability: boolean;
}

export interface NewsOverlay {
  score: number;
  count: number;
  summary: string;
}

export interface Sizing {
  full_kelly_pct: number;
  half_kelly_pct: number;
  final_pct: number;
  cap_pct: number;
}

export interface Reading {
  symbol: string;
  symbol_name: string;
  timeframe: string;
  asof: string;
  generated_at: string;
  price: number;
  horizon_bars: number;
  tier_a: TierA;
  tier_b: TierB;
  news: NewsOverlay;
  sizing: Sizing;
  features: Record<string, number | null>;
}

export interface EquityPoint {
  date: string;
  value: number;
  proba: number;
}

export interface BacktestMetrics {
  total_trades: number;
  win_rate: number;
  total_profit_R: number;
  total_profit_pct: number;
  sharpe: number;
  sortino: number;
  calmar: number;
  max_drawdown_R: number;
  brier_score: number;
  brier_in_sample: number;
  threshold: number;
}

export interface BacktestResponse {
  equity_curve: EquityPoint[];
  metrics: BacktestMetrics;
}

export type Trend = "up" | "down" | "neutral";
export type Accent = "green" | "blue" | "red" | "amber" | "gray";

export interface Kpi {
  id: string;
  label: string;
  value: number | string;
  unit: string;
  change: number;
  trend: Trend;
  accent: Accent;
  sparkline: number[];
}

export interface Trade {
  time: string;
  asset: string;
  type: string;
  entry: number;
  exit: number;
  pnl: number;
  status: string;
  duration: string;
  confidence: number;
}

export interface ModelCard {
  name: string;
  type: string;
  accuracy: number;
  roc_auc?: number;
  brier_skill_score?: number;
  calibrated?: boolean;
  confidence: number;
  status: "active" | "benchmark" | "queued";
  updated: string;
  train_time: string;
  brier: number | null;
  performance: number[];
}

export interface MarketAsset {
  symbol: string;
  price: number;
  change: number;
  sparkline: number[];
  volume: number;
  prediction: string;
}

export interface HeatmapData {
  labels: string[];
  matrix: number[][];
}

export interface Gauge {
  label: string;
  value: number;
  max: number;
  accent: Accent;
}

export interface Insight {
  title: string;
  icon: string;
  text: string;
  accent: Accent;
  weight: number;
}

export interface FeatureImportance {
  feature: string;
  coefficient: number;
  abs: number;
}

export interface HealthStatus {
  status: string;
  uptime_s: number;
  reading_ready: boolean;
}

// ─── Smart Money Concepts chart types ──────────────────────────────

export interface Candle {
  t: string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

export interface SwingPoint {
  scope: "swing" | "internal";
  x: number;
  price: number;
  label: string; // HH | HL | LH | LL
  is_high: boolean;
}

export interface StructureBreak {
  scope: "swing" | "internal";
  bias: "bull" | "bear";
  type: "BOS" | "CHoCH";
  x1: number;
  x2: number;
  price: number;
}

export interface OrderBlock {
  scope: "swing" | "internal";
  bias: "bull" | "bear";
  x: number;
  top: number;
  bottom: number;
  mitigated: boolean;
}

export interface FairValueGap {
  bias: "bull" | "bear";
  x: number;
  top: number;
  bottom: number;
}

export interface EqualHL {
  kind: "EQH" | "EQL";
  x: number;
  price: number;
  prev_price: number;
}

export interface SmcZones {
  top: number;
  bottom: number;
  equilibrium: number;
  premium: [number, number];
  discount: [number, number];
}

export interface SmcData {
  swings: SwingPoint[];
  structures: StructureBreak[];
  order_blocks_swing: OrderBlock[];
  order_blocks_internal: OrderBlock[];
  fvgs: FairValueGap[];
  eqhl: EqualHL[];
  zones: SmcZones | null;
  bias: "bull" | "bear" | "neutral";
}

export interface SupplyDemandZone {
  kind: "supply" | "demand";
  x1: number;
  x2: number;
  top: number;
  bottom: number;
  delta: number;
  pct: number;
}

export interface SupplyDemandData {
  supply: SupplyDemandZone[];
  demand: SupplyDemandZone[];
  total_supply: number;
  total_demand: number;
}

export interface ChartResponse {
  symbol: string;
  symbol_name: string;
  timeframe: string;
  n: number;
  swing_length: number;
  internal_length: number;
  candles: Candle[];
  smc: SmcData | null;
  supply_demand: SupplyDemandData | null;
  error?: string;
}

// ─── v2 zone-conditional reading (/api/reading_v2) ────────────────

export interface TierV2 {
  probability: number | null;
  probability_lo: number | null;
  probability_hi: number | null;
  /** false = climatology / fallback, never a model output (§9.4 honesty) */
  is_probability: boolean;
  /** false = coverage check failed for this pair — band unverified */
  interval_reliable: boolean;
  /** "calibration_wilson_90" (observed OOS hit rate in this prediction's
   *  calibration bin) or "conformal_90" (spec |p−y| quantile, near-vacuous) */
  band_kind?: "calibration_wilson_90" | "conformal_90" | null;
  /** top features by |coefficient × standardized value| for THIS prediction */
  contributions?: { feature: string; value: number; contribution: number }[] | null;
  note: string | null;
  payoff_b?: number;
}

export interface ZoneEventV2 {
  trigger: "E1" | "E2" | "E3" | "E4" | "E5";
  zone_kind: "supply" | "demand";
  source: string;
  zone_top: number;
  zone_bottom: number;
  zone_age_bars: number;
  touches_count: number;
  zone_pct: number | null;
}

export interface SizingV2 {
  payoff_b: number;
  ev_positive: boolean;
  full_kelly_pct: number;
  half_kelly_pct: number;
  final_pct: number;
  cap_pct: number;
}

export interface ReadingV2 {
  symbol: string;
  symbol_name: string;
  timeframe: string;
  asof: string;
  generated_at: string;
  price: number;
  horizon_bars: number;
  calendar_loaded: boolean;
  state: "no_signal" | "blackout" | "signal";
  blackout?: { event: string; starts_in_seconds: number };
  note?: string;
  event?: ZoneEventV2;
  tier_a_bounce?: TierV2;
  tier_a_break?: TierV2;
  direction?: {
    winner: "bounce" | "break" | null;
    direction: "LONG" | "SHORT" | null;
    magnitude: number | null;
    winner_is_probability?: boolean;
  };
  sizing?: SizingV2 | null;
  news?: NewsOverlay;
  error?: string;
}
