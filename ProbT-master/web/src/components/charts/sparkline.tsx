"use client";
import type { EChartsOption } from "echarts";

import { useMemo } from "react";
import { EChartsBase } from "./echarts-base";
import type { Accent } from "@/types";
import { accentColor } from "@/lib/chart-colors";

export function Sparkline({
  data,
  accent = "green",
  height = 36,
  fill = true,
}: {
  data: number[];
  accent?: Accent;
  height?: number;
  fill?: boolean;
}) {
  const option = useMemo<EChartsOption>(() => {
    if (!data || data.length === 0) return {};
    const color = accentColor(accent);
    return {
      grid: { top: 2, bottom: 2, left: 2, right: 2 },
      xAxis: { type: "category", show: false, data: data.map((_, i) => i) },
      yAxis: { type: "value", show: false, min: "dataMin", max: "dataMax" },
      series: [
        {
          type: "line",
          data,
          smooth: true,
          symbol: "none",
          lineStyle: { width: 1.5, color },
          areaStyle: fill
            ? {
                color: {
                  type: "linear",
                  x: 0, y: 0, x2: 0, y2: 1,
                  colorStops: [
                    { offset: 0, color: `${color}40` },
                    { offset: 1, color: `${color}00` },
                  ],
                },
              }
            : undefined,
        },
      ],
      tooltip: { show: false },
      animation: true,
      animationDuration: 800,
    }
  }, [data, accent, fill]);

  if (!data || data.length === 0) {
    return <div style={{ height }} className="rounded bg-muted/40" />;
  }

  return <EChartsBase option={option} height={height} />;
}
