"use client";

import { useEffect, useRef, useMemo, useState, useCallback } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";

export interface EChartsBaseProps {
  option: EChartsOption;
  height?: number | string;
  className?: string;
  loading?: boolean;
  /** Receives the live ECharts instance (or null on unmount) for imperative use. */
  onChartRef?: (chart: echarts.ECharts | null) => void;
  /** Event bindings applied after init: [[eventName, handler], ...]. */
  onEvents?: { event: string; handler: (params: unknown) => void }[];
}

/** Detects dark mode by observing <html> class changes. */
export function useDarkMode(): boolean {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const check = () => setDark(document.documentElement.classList.contains("dark"));
    check();
    const ob = new MutationObserver(check);
    ob.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => ob.disconnect();
  }, []);
  return dark;
}

/**
 * Reusable ECharts wrapper — init, resize, theme, disposal.
 * Lazy-safe: renders nothing until mounted client-side.
 */
export function EChartsBase({ option, height = 300, className, loading = false, onChartRef, onEvents }: EChartsBaseProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const isDark = useDarkMode();

  const styled = useMemo<EChartsOption>(
    () => ({
      backgroundColor: "transparent",
      textStyle: { color: isDark ? "#9ca3af" : "#6b7280" },
      ...option,
    }),
    [option, isDark],
  );

  // (Re-)create chart on mount + theme change
  const ensureChart = useCallback(() => {
    if (!containerRef.current) return null;
    if (chartRef.current) {
      chartRef.current.dispose();
    }
    const chart = echarts.init(containerRef.current, undefined, { renderer: "canvas" });
    chartRef.current = chart;
    return chart;
  }, []);

  useEffect(() => {
    const chart = ensureChart();
    if (!chart) return;
    onChartRef?.(chart);
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(containerRef.current!);
    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
      onChartRef?.(null);
    };
  }, [ensureChart, isDark, onChartRef]);

  // Bind event handlers
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !onEvents) return;
    const cleanups: (() => void)[] = [];
    for (const { event, handler } of onEvents) {
      chart.on(event, handler);
      cleanups.push(() => chart.off(event, handler));
    }
    return () => cleanups.forEach((c) => c());
  }, [onEvents]);

  // Apply option + loading state
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.setOption(styled, { notMerge: false, lazyUpdate: true });
    if (loading) {
      chart.showLoading("default", {
        text: "",
        color: isDark ? "#22c55e" : "#16a34a",
        textColor: "transparent",
        maskColor: "transparent",
      });
    } else {
      chart.hideLoading();
    }
  }, [styled, loading, isDark]);

  return (
    <div
      ref={containerRef}
      className={className}
      style={{
        width: "100%",
        height: typeof height === "number" ? `${height}px` : height,
      }}
    />
  );
}

export const EChartsLazy = EChartsBase;
