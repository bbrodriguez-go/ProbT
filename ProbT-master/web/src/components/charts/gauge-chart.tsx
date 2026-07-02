"use client";
import type { EChartsOption } from "echarts";

import { useMemo } from "react";
import { EChartsBase } from "./echarts-base";
import type { Gauge } from "@/types";
import { accentColor } from "@/lib/chart-colors";

export function GaugeChart({ gauge, height = 160 }: { gauge: Gauge; height?: number }) {
  const option = useMemo<EChartsOption>(() => {
    const pct = Math.min(gauge.value / gauge.max, 1);
    const color = accentColor(gauge.accent);
    return {
      series: [
        {
          type: "gauge",
          startAngle: 200,
          endAngle: -20,
          min: 0,
          max: gauge.max,
          radius: "92%",
          center: ["50%", "62%"],
          progress: {
            show: true,
            width: 10,
            roundCap: true,
            itemStyle: { color },
          },
          axisLine: {
            lineStyle: { width: 10, color: [[1, "rgba(128,128,128,0.12)"]] },
            roundCap: true,
          },
          pointer: { show: false },
          axisTick: { show: false },
          axisLabel: { show: false },
          splitLine: { show: false },
          axisPointers: { show: false },
          anchor: { show: false },
          title: { show: false },
          detail: {
            valueAnimation: true,
            offsetCenter: [0, "-2%"],
            fontSize: 22,
            fontWeight: 700,
            color,
            formatter: (v: number) => gauge.max === 100 ? `${Math.round(v)}%` : v.toFixed(1),
          },
          data: [{ value: gauge.value }],
        },
      ],
    }
  }, [gauge]);

  return (
    <div className="flex flex-col items-center">
      <EChartsBase option={option} height={height} />
      <span className="text-xs font-medium text-muted-foreground -mt-4">{gauge.label}</span>
    </div>
  );
}
