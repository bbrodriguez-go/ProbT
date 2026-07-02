"use client";
import type { EChartsOption } from "echarts";

import { EChartsBase } from "./echarts-base";
import { accentColor } from "@/lib/chart-colors";

interface ProbDistData {
  bins: number[];
  counts: number[];
  percentiles: Record<string, number>;
  current: number;
  mean: number;
  std: number;
}

export function ProbabilityDist({ data, height = 280 }: { data: ProbDistData; height?: number }) {
  const green = accentColor("green");
  const blue = accentColor("blue");
  const amber = accentColor("amber");

  // Cumulative % computed via a pure scan (no external mutation).
  const total = data.counts.reduce((a, b) => a + b, 0);
  const cumPct = data.counts.reduce<number[]>(
    (acc, c, i) => {
      const prev = i === 0 ? 0 : acc[i - 1] + data.counts[i - 1];
      acc.push(total > 0 ? (((prev + c) / total) * 100) : 0);
      return acc;
    },
    [],
  );

  const binLabels = data.bins.slice(0, -1).map((b, i) =>
    `${(b * 100).toFixed(0)}–${(data.bins[i + 1] * 100).toFixed(0)}%`,
  );

  const percentileMarks = Object.entries(data.percentiles).map(([k, v]) => ({
    name: k.toUpperCase(),
    xAxis: `${(v * 100).toFixed(0)}%`,
    label: { show: true, formatter: k.toUpperCase(), fontSize: 9, color: amber },
    lineStyle: { color: amber, type: "dashed" as const, width: 1 },
  }));

  const option: EChartsOption = {
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "rgba(255,255,255,0.96)",
        borderColor: "#e5e7eb",
        borderWidth: 1,
        textStyle: { color: "#111827", fontSize: 12 },
      },
      legend: {
        data: ["Frequency", "Cumulative %"],
        bottom: 0,
        textStyle: { fontSize: 11, color: "#6b7280" },
        itemWidth: 12,
        itemHeight: 8,
      },
      grid: { top: 20, bottom: 40, left: 40, right: 40 },
      xAxis: {
        type: "category",
        data: binLabels,
        axisLabel: { fontSize: 9, color: "#6b7280", interval: 1 },
        axisLine: { lineStyle: { color: "#e5e7eb" } },
      },
      yAxis: [
        {
          type: "value",
          name: "Count",
          nameTextStyle: { fontSize: 10, color: "#6b7280" },
          axisLabel: { fontSize: 10, color: "#6b7280" },
          splitLine: { lineStyle: { color: "#f4f4f5" } },
        },
        {
          type: "value",
          name: "Cum %",
          nameTextStyle: { fontSize: 10, color: "#6b7280" },
          axisLabel: { fontSize: 10, color: "#6b7280", formatter: "{value}%" },
          splitLine: { show: false },
          max: 100,
        },
      ],
      series: [
        {
          name: "Frequency",
          type: "bar",
          data: data.counts,
          barWidth: "70%",
          itemStyle: {
            borderRadius: [4, 4, 0, 0],
            color: (p: { dataIndex: number }) => {
              const binStart = data.bins[p.dataIndex];
              return Math.abs(binStart - data.current) < 0.06 ? green : `${green}50`;
            },
          },
          markLine: {
            symbol: "none",
            silent: true,
            data: [
              {
                name: "Current",
                xAxis: binLabels[Math.min(
                  data.bins.slice(0, -1).findIndex((b) => b >= data.current),
                  binLabels.length - 1,
                )],
                label: { show: true, formatter: "Now", fontSize: 9, color: blue },
                lineStyle: { color: blue, width: 2 },
              },
              ...percentileMarks,
            ],
          },
        },
        {
          name: "Cumulative %",
          type: "line",
          yAxisIndex: 1,
          data: cumPct,
          smooth: true,
          symbol: "none",
          lineStyle: { width: 2, color: blue },
          areaStyle: { color: `${blue}10` },
        },
      ],
    };

  return <EChartsBase option={option} height={height} />;
}
