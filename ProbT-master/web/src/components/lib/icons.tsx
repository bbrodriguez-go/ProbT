"use client";

import {
  LayoutDashboard,
  BarChart3,
  Brain,
  TrendingUp,
  ArrowRightLeft,
  Globe,
  Grid3x3,
  Cpu,
  List,
  Gauge,
  Settings,
  ScrollText,
  Activity,
  DollarSign,
  Layers,
  Newspaper,
  CandlestickChart,
  type LucideIcon,
} from "lucide-react";

const map: Record<string, LucideIcon> = {
  "layout-dashboard": LayoutDashboard,
  "bar-chart-3": BarChart3,
  brain: Brain,
  "trending-up": TrendingUp,
  "arrow-right-left": ArrowRightLeft,
  globe: Globe,
  "grid-3x3": Grid3x3,
  cpu: Cpu,
  list: List,
  gauge: Gauge,
  settings: Settings,
  "scroll-text": ScrollText,
  activity: Activity,
  "dollar-sign": DollarSign,
  layers: Layers,
  newspaper: Newspaper,
  "candlestick-chart": CandlestickChart,
};

export function DynamicIcon({
  name,
  className,
  size = 18,
}: {
  name: string;
  className?: string;
  size?: number;
}) {
  const Icon = map[name] ?? Activity;
  return <Icon className={className} size={size} />;
}
