"use client";
import type { EChartsOption } from "echarts";

import { useMemo } from "react";
import { EChartsBase } from "./echarts-base";
import type { HeatmapData } from "@/types";

export function HeatmapChart({ data, height = 380 }: { data: HeatmapData; height?: number }) {
  const option = useMemo<EChartsOption>(() => {
    const labels = data.labels;
    const n = labels.length;
    const seriesData: [number, number, number][] = [];
    let min = 1, max = -1;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const v = data.matrix[i][j];
        seriesData.push([j, i, v]);
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return {
      tooltip: {
        position: "top",
        formatter: (p: unknown) => {
          const d = (p as { data: [number, number, number] }).data;
          return `<b>${labels[d[1]]}</b><br/>↔ ${labels[d[0]]}<br/><b>${d[2].toFixed(3)}</b>`;
        },
      },
      grid: { top: 10, bottom: 80, left: 100, right: 20 },
      xAxis: {
        type: "category",
        data: labels,
        splitArea: { show: true },
        axisLabel: { rotate: 45, fontSize: 10, color: "#6b7280", interval: 0 },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      yAxis: {
        type: "category",
        data: labels,
        splitArea: { show: true },
        axisLabel: { fontSize: 10, color: "#6b7280" },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: false,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        itemWidth: 12,
        itemHeight: 80,
        textStyle: { color: "#6b7280", fontSize: 10 },
        inRange: {
          color: ["#dc2626", "#fbbf24", "#f4f4f5", "#22c55e", "#16a34a"],
        },
      },
      series: [
        {
          type: "heatmap",
          data: seriesData,
          emphasis: {
            itemStyle: { shadowBlur: 8, shadowColor: "rgba(0,0,0,0.2)" },
          },
          itemStyle: { borderRadius: 3, borderColor: "rgba(255,255,255,0.5)", borderWidth: 1 },
        },
      ],
    }
  }, [data]);

  return <EChartsBase option={option} height={height} />;
}
