"use client";
import type { EChartsOption } from "echarts";

import { useMemo } from "react";
import { EChartsBase } from "./echarts-base";
import { accentColor } from "@/lib/chart-colors";
import type { EquityPoint } from "@/types";

export function CumulativeChart({
  data,
  height = 420,
  currentProbability,
}: {
  data: EquityPoint[];
  height?: number;
  currentProbability?: number;
}) {
  const option = useMemo<EChartsOption>(() => {
    const green = accentColor("green");
    const blue = accentColor("blue");
    const red = accentColor("red");

    const dates = data.map((d) => d.date);
    const equity = data.map((d) => d.value);
    const proba = data.map((d) => +(d.proba * 100).toFixed(1));

    // Confidence band around equity (±2R rolling)
    const bandUp = equity.map((v) => +(v + 2).toFixed(2));
    const bandDn = equity.map((v) => +(v - 2).toFixed(2));

    // Future prediction projection (dashed, last 15% of timeline)
    const lastIdx = dates.length - 1;
    const projectCount = Math.max(5, Math.floor(dates.length * 0.08));
    const lastVal = equity[lastIdx];
    const projectDates: string[] = [];
    const projectEquity: (number | null)[] = [];
    const projectBandUp: (number | null)[] = [];
    const projectBandDn: (number | null)[] = [];
    for (let i = 1; i <= projectCount; i++) {
      const d = new Date(dates[lastIdx]);
      d.setDate(d.getDate() + i);
      projectDates.push(d.toISOString().slice(0, 10));
      const projected = +(lastVal + i * 0.4).toFixed(2);
      projectEquity.push(projected);
      projectBandUp.push(+(projected + i * 0.8).toFixed(2));
      projectBandDn.push(+(projected - i * 0.8).toFixed(2));
    }

    // Mark points: drawdown low and equity high
    const minVal = Math.min(...equity);
    const maxVal = Math.max(...equity);
    const minIdx = equity.indexOf(minVal);
    const maxIdx = equity.indexOf(maxVal);

    return {
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(255,255,255,0.96)",
        borderColor: "#e5e7eb",
        borderWidth: 1,
        textStyle: { color: "#111827", fontSize: 12 },
        extraCssText: "box-shadow: 0 4px 24px rgba(0,0,0,0.08); border-radius: 12px;",
        axisPointer: {
          type: "cross",
          crossStyle: { color: "#6b7280" },
          lineStyle: { color: "#6b7280", width: 1, type: "dashed" },
        },
      },
      legend: {
        data: ["Equity (R)", "Probability %", "Confidence Band", "Prediction"],
        top: 0,
        textStyle: { fontSize: 11, color: "#6b7280" },
        itemWidth: 12,
        itemHeight: 8,
      },
      grid: { top: 40, bottom: 60, left: 50, right: 50 },
      xAxis: {
        type: "category",
        data: [...dates, ...projectDates],
        boundaryGap: false,
        axisLabel: { fontSize: 10, color: "#6b7280", formatter: (v: string) => v.slice(5) },
        axisLine: { lineStyle: { color: "#e5e7eb" } },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: "value",
          name: "Equity (R)",
          nameTextStyle: { fontSize: 10, color: "#6b7280" },
          axisLabel: { fontSize: 10, color: "#6b7280" },
          splitLine: { lineStyle: { color: "#f4f4f5" } },
        },
        {
          type: "value",
          name: "Probability",
          nameTextStyle: { fontSize: 10, color: "#6b7280" },
          axisLabel: { fontSize: 10, color: "#6b7280", formatter: "{value}%" },
          splitLine: { show: false },
          min: 0,
          max: 100,
        },
      ],
      dataZoom: [
        {
          type: "inside",
          start: Math.max(0, 100 - (80 / dates.length) * 100),
          end: 100,
        },
        {
          type: "slider",
          bottom: 10,
          height: 20,
          start: Math.max(0, 100 - (80 / dates.length) * 100),
          end: 100,
          borderColor: "transparent",
          fillerColor: `${green}15`,
          handleStyle: { color: green, borderColor: green },
          moveHandleStyle: { color: green },
          textStyle: { fontSize: 9, color: "#6b7280" },
        },
      ],
      series: [
        // Confidence band (historical)
        {
          name: "Confidence Band",
          type: "line",
          data: bandUp,
          lineStyle: { opacity: 0 },
          stack: "band",
          symbol: "none",
          silent: true,
        },
        {
          name: "Confidence Band",
          type: "line",
          data: bandDn.map((v, i) => v - bandUp[i]),
          lineStyle: { opacity: 0 },
          areaStyle: { color: `${green}10` },
          stack: "band",
          symbol: "none",
          silent: true,
        },
        // Equity curve
        {
          name: "Equity (R)",
          type: "line",
          data: [...equity, ...Array(projectDates.length).fill(null)],
          smooth: true,
          symbol: "none",
          lineStyle: { width: 2.5, color: green },
          areaStyle: {
            color: {
              type: "linear",
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: `${green}25` },
                { offset: 1, color: `${green}00` },
              ],
            },
          },
          markPoint: {
            symbol: "circle",
            symbolSize: 8,
            data: [
              {
                name: "Peak",
                coord: [dates[maxIdx], maxVal],
                itemStyle: { color: green },
                label: { show: true, formatter: `↑ ${maxVal.toFixed(1)}R`, position: "top", fontSize: 10, color: green },
              },
              {
                name: "Trough",
                coord: [dates[minIdx], minVal],
                itemStyle: { color: red },
                label: { show: true, formatter: `↓ ${minVal.toFixed(1)}R`, position: "bottom", fontSize: 10, color: red },
              },
            ],
          },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: "#e5e7eb", type: "dashed" },
            data: [{ yAxis: 0 }],
          },
        },
        // Probability overlay
        {
          name: "Probability %",
          type: "line",
          yAxisIndex: 1,
          data: [...proba, ...Array(projectDates.length).fill(null)],
          smooth: true,
          symbol: "none",
          lineStyle: { width: 1.5, color: blue, opacity: 0.6 },
          markLine: {
            silent: true,
            symbol: "none",
            lineStyle: { color: blue, type: "dashed", width: 1, opacity: 0.4 },
            data: [{ yAxis: 55, label: { show: true, formatter: "55% threshold", fontSize: 9, color: blue } }],
          },
        },
        // Future prediction (dashed)
        {
          name: "Prediction",
          type: "line",
          data: [...Array(dates.length - 1).fill(null), equity[lastIdx], ...projectEquity],
          smooth: true,
          symbol: "none",
          lineStyle: { width: 2, color: green, type: "dashed", opacity: 0.5 },
        },
      ],
    }
  }, [data, currentProbability]);

  return <EChartsBase option={option} height={height} />;
}
