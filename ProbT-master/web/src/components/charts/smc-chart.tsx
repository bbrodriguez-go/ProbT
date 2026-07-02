"use client";

import { useMemo, useCallback } from "react";
import type {
  EChartsOption,
  CustomSeriesRenderItemParams,
  CustomSeriesRenderItemAPI,
  CustomSeriesRenderItemReturn,
  TooltipComponentFormatterCallbackParams,
  GridComponentOption,
  XAXisComponentOption,
  YAXisComponentOption,
} from "echarts";
import type { ECharts } from "echarts";
import { EChartsBase, useDarkMode } from "./echarts-base";
import type {
  ChartResponse,
  OrderBlock,
  FairValueGap,
  StructureBreak,
  SupplyDemandZone,
  SwingPoint,
  EqualHL,
} from "@/types";

// ─── Layer toggles ────────────────────────────────────────────────
export type SmcLayerKey =
  | "volume"
  | "premiumDiscount"
  | "orderBlocks"
  | "fvgs"
  | "supplyDemand"
  | "structure"
  | "swings";

export type SmcLayers = Record<SmcLayerKey, boolean>;

export const DEFAULT_SMC_LAYERS: SmcLayers = {
  volume: true,
  premiumDiscount: true,
  orderBlocks: true,
  fvgs: true,
  supplyDemand: true,
  structure: true,
  swings: true,
};

export const SMC_LAYER_META: {
  key: SmcLayerKey;
  label: string;
  dot: string;
}[] = [
  { key: "volume", label: "Volume", dot: "#64748b" },
  { key: "premiumDiscount", label: "Premium / Discount", dot: "#a855f7" },
  { key: "orderBlocks", label: "Order Blocks", dot: "#22c55e" },
  { key: "fvgs", label: "Fair Value Gaps", dot: "#06b6d4" },
  { key: "supplyDemand", label: "Supply / Demand", dot: "#f59e0b" },
  { key: "structure", label: "Structure", dot: "#38bdf8" },
  { key: "swings", label: "Swings", dot: "#facc15" },
];

// ─── Institutional palette (high-contrast on dark) ─────────────────
const PAL = {
  bull: "#22c55e",
  bear: "#ef4444",
  bullFill: "rgba(34,197,94,0.16)",
  bearFill: "rgba(239,68,68,0.16)",
  bullBorder: "rgba(34,197,94,0.85)",
  bearBorder: "rgba(239,68,68,0.85)",
  fvgBullFill: "rgba(6,182,212,0.14)",
  fvgBullBorder: "rgba(6,182,212,0.55)",
  fvgBearFill: "rgba(236,72,153,0.14)",
  fvgBearBorder: "rgba(236,72,153,0.55)",
  supplyFill: "rgba(245,158,11,0.16)",
  supplyBorder: "#f59e0b",
  demandFill: "rgba(59,130,246,0.16)",
  demandBorder: "#3b82f6",
  premiumFill: "rgba(239,68,68,0.06)",
  discountFill: "rgba(34,197,94,0.06)",
  eq: "#64748b",
  bosBull: "#22c55e",
  bosBear: "#ef4444",
  choch: "#e879f9",
  swingHigh: "#f87171",
  swingLow: "#4ade80",
  eqhl: "#fbbf24",
};

// ─── Overlay primitives (custom series) ────────────────────────────
type RectMode = "span" | "right" | "full";
interface RectItem {
  kind: "rect";
  x1: number;
  x2: number;
  top: number;
  bottom: number;
  fill: string;
  stroke?: string;
  lw?: number;
  mode: RectMode;
  dashBorder?: boolean;
}
interface LineItem {
  kind: "line";
  x1: number;
  x2: number;
  y: number;
  color: string;
  width: number;
  dashed?: boolean;
  full?: boolean;
}
interface TextItem {
  kind: "text";
  x: number;
  y: number;
  text: string;
  color: string;
  align?: "left" | "center" | "right";
  dy?: number;
  dx?: number;
  fs?: number;
  weight?: number;
  bg?: string;
}
type Item = RectItem | LineItem | TextItem;

interface Group {
  name: string;
  items: Item[];
  z: number;
}

function itemX(it: Item): number {
  if (it.kind === "rect") return it.x1;
  if (it.kind === "line") return it.x1;
  return it.x;
}

// ─── Visible-range helper for dynamic Y axis ──────────────────────
function visibleRange(
  candles: ChartResponse["candles"],
  startPct: number,
  endPct: number,
) {
  const n = candles.length;
  const iStart = Math.max(0, Math.floor((startPct / 100) * n));
  const iEnd = Math.min(n, Math.ceil((endPct / 100) * n));
  let lo = Infinity;
  let hi = -Infinity;
  let vMax = 0;
  for (let i = iStart; i < iEnd; i++) {
    const c = candles[i];
    if (c.l < lo) lo = c.l;
    if (c.h > hi) hi = c.h;
    if (c.v > vMax) vMax = c.v;
  }
  if (!isFinite(lo) || !isFinite(hi)) {
    return { lo: 0, hi: 1, vMax: 1, iStart: 0, iEnd: n };
  }
  return { lo, hi, vMax, iStart, iEnd };
}

const DEFAULT_WINDOW = 260;

function initialWindow(n: number) {
  return { start: Math.max(0, 100 - (DEFAULT_WINDOW / n) * 100), end: 100 };
}

// ─── Option builder ───────────────────────────────────────────────
function buildOption(
  data: ChartResponse,
  isDark: boolean,
  layers: SmcLayers,
  height: number,
): EChartsOption {
  const candles = data.candles;
  const n = candles.length;
  const smc = data.smc;
  const sd = data.supply_demand;

  const axisLine = isDark ? "#1f2937" : "#e5e7eb";
  const splitLine = isDark ? "rgba(255,255,255,0.04)" : "rgba(0,0,0,0.05)";
  const labelColor = isDark ? "#94a3b8" : "#475569";
  const crosshairColor = isDark ? "rgba(148,163,184,0.55)" : "rgba(15,23,42,0.45)";

  const times = candles.map((c) => c.t);
  const cats = candles.map((c) => [c.o, c.c, c.l, c.h]);
  const volumes = candles.map((c) => c.v);

  const win = initialWindow(n);
  const { lo, hi, vMax } = visibleRange(candles, win.start, win.end);
  const pad = (hi - lo) * 0.08 || hi * 0.002;
  const yMin = lo - pad;
  const yMax = hi + pad;
  const volMax = vMax * 1.15 || 1;

  // ─── Overlay groups ────────────────────────────────────────────
  const groups: Group[] = [];

  // Premium / Discount / Equilibrium (back layer)
  if (layers.premiumDiscount && smc?.zones) {
    const z = smc.zones;
    const pd: Item[] = [
      {
        kind: "rect", x1: 0, x2: n - 1, top: z.premium[1], bottom: z.premium[0],
        fill: PAL.premiumFill, mode: "full",
      },
      {
        kind: "rect", x1: 0, x2: n - 1, top: z.discount[1], bottom: z.discount[0],
        fill: PAL.discountFill, mode: "full",
      },
      {
        kind: "line", x1: 0, x2: n - 1, y: z.equilibrium, color: PAL.eq,
        width: 1, dashed: true, full: true,
      },
      {
        kind: "text", x: 0, y: z.premium[1], text: "PREMIUM",
        color: "rgba(239,68,68,0.85)", align: "left", dx: 6, dy: -6, fs: 9, weight: 700,
      },
      {
        kind: "text", x: 0, y: z.discount[0], text: "DISCOUNT",
        color: "rgba(34,197,94,0.85)", align: "left", dx: 6, dy: 10, fs: 9, weight: 700,
      },
      {
        kind: "text", x: n - 1, y: z.equilibrium, text: "EQ",
        color: PAL.eq, align: "right", dx: -6, dy: -6, fs: 9, weight: 700,
      },
    ];
    groups.push({ name: "Premium / Discount", items: pd, z: 1 });
  }

  // Order blocks (swing + internal)
  if (layers.orderBlocks && smc) {
    const obItems: Item[] = [];
    const pushOB = (o: OrderBlock, strong: boolean) => {
      const fill = o.bias === "bull" ? PAL.bullFill : PAL.bearFill;
      const stroke = o.bias === "bull" ? PAL.bullBorder : PAL.bearBorder;
      obItems.push({
        kind: "rect", x1: o.x, x2: n - 1, top: o.top, bottom: o.bottom,
        fill, stroke, lw: strong ? 1.4 : 1,
        mode: "right", dashBorder: !strong,
      });
      const labelY = (o.top + o.bottom) / 2;
      obItems.push({
        kind: "text", x: o.x, y: labelY,
        text: `${o.bias === "bull" ? "B" : "S"}OB${strong ? "" : "i"}`,
        color: stroke, align: "left", dx: 4, dy: 0, fs: 9, weight: 700,
      });
    };
    smc.order_blocks_swing.forEach((o) => pushOB(o, true));
    smc.order_blocks_internal.forEach((o) => pushOB(o, false));
    groups.push({ name: "Order Blocks", items: obItems, z: 4 });
  }

  // Fair Value Gaps (distinct palette)
  if (layers.fvgs && smc) {
    const fvgItems: Item[] = [];
    smc.fvgs.forEach((f: FairValueGap) => {
      const fill = f.bias === "bull" ? PAL.fvgBullFill : PAL.fvgBearFill;
      const stroke = f.bias === "bull" ? PAL.fvgBullBorder : PAL.fvgBearBorder;
      fvgItems.push({
        kind: "rect", x1: f.x, x2: n - 1, top: f.top, bottom: f.bottom,
        fill, stroke, lw: 0.8, mode: "right",
      });
    });
    groups.push({ name: "Fair Value Gaps", items: fvgItems, z: 3 });
  }

  // Supply / Demand zones (compact, no inline text — handled by widget)
  if (layers.supplyDemand && sd && (sd.supply.length || sd.demand.length)) {
    const sdItems: Item[] = [];
    const pushSD = (z2: SupplyDemandZone) => {
      const isSupply = z2.kind === "supply";
      sdItems.push({
        kind: "rect", x1: z2.x1, x2: z2.x2, top: z2.top, bottom: z2.bottom,
        fill: isSupply ? PAL.supplyFill : PAL.demandFill,
        stroke: isSupply ? PAL.supplyBorder : PAL.demandBorder,
        lw: 1.2, mode: "span",
      });
    };
    sd.supply.forEach(pushSD);
    sd.demand.forEach(pushSD);
    groups.push({ name: "Supply / Demand", items: sdItems, z: 5 });
  }

  // Structure breaks (BOS / CHoCH)
  if (layers.structure && smc) {
    const stItems: Item[] = [];
    smc.structures.forEach((s: StructureBreak) => {
      const internal = s.scope === "internal";
      const isChoch = s.type === "CHoCH";
      const col = isChoch ? PAL.choch : s.bias === "bull" ? PAL.bosBull : PAL.bosBear;
      stItems.push({
        kind: "line", x1: s.x1, x2: s.x2, y: s.price, color: col,
        width: internal ? 1 : 1.7, dashed: internal || isChoch,
      });
      const mid = Math.round((s.x1 + s.x2) / 2);
      stItems.push({
        kind: "text", x: mid, y: s.price,
        text: `${s.type}${internal ? "i" : ""}`, color: col,
        align: "center", dy: internal ? 8 : -10, fs: internal ? 9 : 10, weight: 800,
      });
    });
    groups.push({ name: "Structure", items: stItems, z: 6 });
  }

  // Swing labels + EQH/EQL
  if (layers.swings && smc) {
    const swItems: Item[] = [];
    smc.swings.forEach((s: SwingPoint) => {
      const col = s.is_high ? PAL.swingHigh : PAL.swingLow;
      swItems.push({
        kind: "text", x: s.x, y: s.price, text: s.label, color: col,
        align: "center", dy: s.is_high ? -12 : 12, fs: 9, weight: 800,
      });
    });
    smc.eqhl.forEach((e: EqualHL) => {
      swItems.push({
        kind: "text", x: e.x, y: e.price, text: e.kind, color: PAL.eqhl,
        align: "center", dy: e.kind === "EQH" ? -14 : 14, fs: 9, weight: 800,
      });
    });
    groups.push({ name: "Swings", items: swItems, z: 7 });
  }

  // ─── renderItem ────────────────────────────────────────────────
  const TEXT_MIN_BAND = 7;
  function renderGroup(rawItems: Item[]) {
    const items = rawItems.slice().sort((a, b) => itemX(a) - itemX(b));
    let lastLabelPx = -Infinity;
    return (
      params: CustomSeriesRenderItemParams,
      api: CustomSeriesRenderItemAPI,
    ): CustomSeriesRenderItemReturn => {
      const idx = params.dataIndex;
      if (idx === 0) lastLabelPx = -Infinity;
      const it = items[idx];
      const cs = params.coordSys as unknown as {
        x: number; y: number; width: number; height: number;
      };
      const bandW = n > 1 ? api.coord([1, 0])[0] - api.coord([0, 0])[0] : 8;

      if (it.kind === "rect") {
        const topP = api.coord([it.x1, it.top]);
        const botY = api.coord([it.x1, it.bottom])[1];
        let xL: number;
        let xR: number;
        if (it.mode === "full") {
          xL = cs.x; xR = cs.x + cs.width;
        } else if (it.mode === "right") {
          xL = topP[0] - bandW / 2; xR = cs.x + cs.width;
        } else {
          xL = api.coord([it.x1, it.top])[0] - bandW / 2;
          xR = api.coord([it.x2, it.top])[0] + bandW / 2;
        }
        const style: Record<string, unknown> = { fill: it.fill };
        if (it.stroke) {
          style.stroke = it.stroke;
          style.lineWidth = it.lw ?? 1;
          if (it.dashBorder) style.lineDash = [4, 3];
        }
        return {
          type: "rect",
          shape: {
            x: xL, y: topP[1],
            width: Math.max(xR - xL, 1.5),
            height: Math.max(botY - topP[1], 1),
          },
          style, silent: true, z2: 0,
        };
      }
      if (it.kind === "line") {
        const y = api.coord([it.x1, it.y])[1];
        let x1p: number; let x2p: number;
        if (it.full) { x1p = cs.x; x2p = cs.x + cs.width; }
        else { x1p = api.coord([it.x1, it.y])[0]; x2p = api.coord([it.x2, it.y])[0]; }
        const style: Record<string, unknown> = { stroke: it.color, lineWidth: it.width };
        if (it.dashed) style.lineDash = [5, 4];
        return {
          type: "line",
          shape: { x1: x1p, y1: y, x2: x2p, y2: y },
          style, silent: true, z2: 0,
        };
      }
      // text — cull overlapping labels
      const fs = it.fs ?? 10;
      if (bandW < TEXT_MIN_BAND) return { type: "group", children: [] };
      const p = api.coord([it.x, it.y]);
      const px = p[0];
      const minGap = fs * 3.4;
      if (Math.abs(px - lastLabelPx) < minGap) return { type: "group", children: [] };
      lastLabelPx = px;
      return {
        type: "text",
        style: {
          x: px + (it.dx ?? 0), y: p[1] + (it.dy ?? 0),
          text: it.text, fill: it.color,
          fontSize: fs, fontWeight: it.weight ?? 700,
          align: it.align ?? "center", verticalAlign: "middle",
        },
        silent: true, z2: 0,
      };
    };
  }

  const customSeries = groups.map((g) => ({
    type: "custom" as const,
    name: g.name,
    xAxisIndex: 0,
    yAxisIndex: 0,
    z: g.z,
    clip: true,
    animation: false,
    data: g.items.map(() => 0),
    renderItem: renderGroup(g.items),
  }));

  // ─── Layout (price grid + independent volume panel) ────────────
  const TOP = 12;
  const RIGHT = 72;
  const LEFT = 10;
  const VOL_H = layers.volume ? Math.max(54, Math.round(height * 0.16)) : 0;
  const GAP = layers.volume ? 22 : 0;
  const XAXIS_H = layers.volume ? 20 : 22;
  const SLIDER_H = 18;
  const SLIDER_GAP = 6;
  const PRICE_H = Math.max(120, height - TOP - GAP - VOL_H - XAXIS_H - SLIDER_H - SLIDER_GAP);

  const grids: GridComponentOption[] = [
    {
      top: TOP, height: PRICE_H, left: LEFT, right: RIGHT,
      containLabel: true, borderWidth: 0,
    },
  ];
  const xAxes: XAXisComponentOption[] = [
    {
      type: "category",
      data: times,
      boundaryGap: true,
      gridIndex: 0,
      axisLine: { lineStyle: { color: axisLine } },
      axisTick: { show: false },
      axisLabel: { show: false },
      splitLine: { show: false },
    },
  ];
  const yAxes: YAXisComponentOption[] = [
    {
      type: "value",
      min: yMin,
      max: yMax,
      position: "right",
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        fontSize: 11, color: labelColor, formatter: (v: number) => fmt(v),
      },
      splitLine: { lineStyle: { color: splitLine } },
    },
  ];

  if (layers.volume) {
    grids.push({
      top: TOP + PRICE_H + GAP, height: VOL_H,
      left: LEFT, right: RIGHT, containLabel: true, borderWidth: 0,
    });
    xAxes.push({
      type: "category",
      data: times,
      boundaryGap: true,
      gridIndex: 1,
      axisLine: { lineStyle: { color: axisLine } },
      axisTick: { show: false },
      axisLabel: {
        fontSize: 10, color: labelColor, hideOverlap: true,
        formatter: (v: string) => fmtTime(v),
      },
      splitLine: { show: false },
    });
    yAxes.push({
      type: "value",
      max: volMax,
      gridIndex: 1,
      position: "right",
      axisLine: { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
      axisLabel: {
        fontSize: 9, color: labelColor,
        formatter: (v: number) => shortVol(v),
      },
    });
  } else {
    // No volume panel → show time labels under the price grid
    (xAxes[0] as Record<string, unknown>).axisLabel = {
      fontSize: 10, color: labelColor, hideOverlap: true,
      formatter: (v: string) => fmtTime(v),
    };
  }

  const series: EChartsOption["series"] = [
    {
      name: "Price",
      type: "candlestick",
      data: cats,
      xAxisIndex: 0,
      yAxisIndex: 0,
      z: 2,
      barWidth: "68%",
      itemStyle: {
        color: PAL.bull, color0: PAL.bear,
        borderColor: PAL.bull, borderColor0: PAL.bear,
        borderWidth: 1,
      },
      emphasis: {
        itemStyle: {
          color: PAL.bull, color0: PAL.bear,
          borderColor: "#ffffff", borderColor0: "#ffffff",
        },
      },
    },
  ];

  if (layers.volume) {
    series.push({
      name: "Volume",
      type: "bar",
      data: volumes.map((v, i) => ({
        value: v,
        itemStyle: {
          color: candles[i].c >= candles[i].o
            ? "rgba(34,197,94,0.45)"
            : "rgba(239,68,68,0.45)",
        },
      })),
      xAxisIndex: 1,
      yAxisIndex: 1,
      z: 1,
      barWidth: "72%",
      animation: true,
    });
  }

  // ─── Contextual lookup maps for the tooltip ───────────────────
  const obMap = new Map<number, OrderBlock[]>();
  const fvgMap = new Map<number, FairValueGap[]>();
  if (smc) {
    for (const o of [...smc.order_blocks_swing, ...smc.order_blocks_internal]) {
      const arr = obMap.get(o.x) ?? [];
      arr.push(o); obMap.set(o.x, arr);
    }
    for (const f of smc.fvgs) {
      const arr = fvgMap.get(f.x) ?? [];
      arr.push(f); fvgMap.set(f.x, arr);
    }
  }

  const option: EChartsOption = {
    animation: true,
    animationDuration: 500,
    animationEasing: "cubicOut",
    backgroundColor: "transparent",
    tooltip: {
      trigger: "axis",
      triggerOn: "mousemove",
      axisPointer: {
        type: "cross",
        crossStyle: { color: crosshairColor, width: 1, type: "dashed" },
        lineStyle: { color: crosshairColor, width: 1, type: "dashed" },
        label: {
          backgroundColor: isDark ? "#0f172a" : "#0f172a",
          color: "#e2e8f0",
          fontSize: 10,
          fontWeight: 600,
        },
      },
      backgroundColor: isDark ? "rgba(9,12,20,0.95)" : "rgba(255,255,255,0.97)",
      borderColor: isDark ? "#1e293b" : "#e2e8f0",
      borderWidth: 1,
      padding: [10, 12],
      textStyle: { color: isDark ? "#e2e8f0" : "#0f172a", fontSize: 12 },
      extraCssText:
        "box-shadow: 0 8px 32px rgba(0,0,0,0.35); border-radius: 10px; backdrop-filter: blur(10px); max-width: 320px;",
      formatter: (raw: TooltipComponentFormatterCallbackParams) => {
        const p = Array.isArray(raw) ? raw : [raw];
        const i = (p[0] as { dataIndex?: number }).dataIndex;
        if (i == null || !candles[i]) return "";
        const c = candles[i];
        const up = c.c >= c.o;
        const col = up ? PAL.bull : PAL.bear;
        const chg = c.o ? ((c.c - c.o) / c.o) * 100 : 0;

        const obs = obMap.get(i) ?? [];
        const fvgs = fvgMap.get(i) ?? [];
        let pdLabel = "";
        if (smc?.zones) {
          const z = smc.zones;
          if (c.c >= z.premium[0]) pdLabel = "Premium";
          else if (c.c <= z.discount[1]) pdLabel = "Discount";
          else pdLabel = "Equilibrium";
        }

        const rows: string[] = [
          `<div style="font-weight:700;font-size:11px;color:#94a3b8;letter-spacing:.04em;margin-bottom:4px">${c.t}</div>`,
          `<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">`,
          `<span style="color:${col};font-weight:700">${up ? "▲" : "▼"} ${fmt(c.c)}</span>`,
          `<span style="color:${up ? PAL.bull : PAL.bear};font-size:10px;font-weight:600">${up ? "+" : ""}${chg.toFixed(2)}%</span>`,
          `</div>`,
          `<div style="display:grid;grid-template-columns:auto auto;gap:3px 14px;font-size:11px;color:#cbd5e1">`,
          `<span style="color:#64748b">Open</span><span style="font-weight:600;text-align:right">${fmt(c.o)}</span>`,
          `<span style="color:#64748b">High</span><span style="font-weight:600;text-align:right;color:${PAL.bull}">${fmt(c.h)}</span>`,
          `<span style="color:#64748b">Low</span><span style="font-weight:600;text-align:right;color:${PAL.bear}">${fmt(c.l)}</span>`,
          `<span style="color:#64748b">Close</span><span style="font-weight:600;text-align:right">${fmt(c.c)}</span>`,
          `<span style="color:#64748b">Volume</span><span style="font-weight:600;text-align:right">${shortVol(c.v)}</span>`,
          `</div>`,
        ];

        const chips: string[] = [];
        if (pdLabel) {
          const cc = pdLabel === "Premium" ? PAL.bear : pdLabel === "Discount" ? PAL.bull : PAL.eq;
          chips.push(`<span style="color:${cc}">◆ ${pdLabel}</span>`);
        }
        if (obs.length) {
          const o = obs[0];
          const cc = o.bias === "bull" ? PAL.bull : PAL.bear;
          chips.push(`<span style="color:${cc}">▮ ${o.scope === "swing" ? "Swing" : "Internal"} OB · ${fmt(o.bottom)}–${fmt(o.top)}</span>`);
        }
        if (fvgs.length) {
          const f = fvgs[0];
          const cc = f.bias === "bull" ? "#06b6d4" : "#ec4899";
          chips.push(`<span style="color:${cc}">▢ FVG ${f.bias === "bull" ? "bull" : "bear"}</span>`);
        }
        if (chips.length) {
          rows.push(
            `<div style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(148,163,184,0.2);display:flex;flex-wrap:wrap;gap:6px 10px;font-size:10px;font-weight:600">${chips.join("")}</div>`,
          );
        }
        return `<div style="line-height:1.45">${rows.join("")}</div>`;
      },
    },
    axisPointer: {
      link: [{ xAxisIndex: layers.volume ? [0, 1] : [0] }],
      label: { color: "#e2e8f0" },
    },
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    dataZoom: [
      {
        type: "inside",
        xAxisIndex: layers.volume ? [0, 1] : [0],
        start: win.start,
        end: win.end,
        filterMode: "none",
      },
      {
        type: "slider",
        xAxisIndex: layers.volume ? [0, 1] : [0],
        start: win.start,
        end: win.end,
        bottom: 4,
        height: SLIDER_H,
        borderColor: "transparent",
        backgroundColor: isDark ? "rgba(255,255,255,0.03)" : "rgba(0,0,0,0.03)",
        fillerColor: isDark ? "rgba(34,197,94,0.12)" : "rgba(34,197,94,0.18)",
        handleStyle: { color: PAL.bull, borderColor: PAL.bull },
        moveHandleStyle: { color: PAL.bull },
        emphasis: { handleStyle: { color: "#16a34a", borderColor: "#16a34a" } },
        textStyle: { fontSize: 9, color: labelColor },
        labelFormatter: (_: number, v: string) => fmtTime(v),
        dataBackground: {
          lineStyle: { color: isDark ? "#334155" : "#cbd5e1" },
          areaStyle: { color: isDark ? "rgba(148,163,184,0.15)" : "rgba(148,163,184,0.25)" },
        },
        brushSelect: false,
      },
    ],
    series,
  };

  // attach the custom overlays (must be appended after the candlestick)
  (option.series as unknown[]).push(...customSeries);
  return option;
}

// ─── Formatters ───────────────────────────────────────────────────
function fmt(v: number) {
  return v.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
function shortVol(v: number) {
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return `${v}`;
}
function fmtTime(v: string) {
  // "2024-01-05 14:00" → "01-05 14:00"; tolerant to date-only too.
  if (!v) return "";
  if (v.length <= 10) return v.slice(5);
  return v.slice(5, 16);
}

// ─── Component ────────────────────────────────────────────────────
export function SmcChart({
  data,
  height = 560,
  layers = DEFAULT_SMC_LAYERS,
}: {
  data: ChartResponse;
  height?: number;
  layers?: SmcLayers;
}) {
  const isDark = useDarkMode();
  const option = useMemo(
    () => buildOption(data, isDark, layers, height),
    [data, isDark, layers, height],
  );

  // Dynamic Y-axis: rescale on every zoom/pan to the visible candle range
  // (keeps the candles filling the panel with a small margin, Bloomberg-style).
  const onChartRef = useCallback(
    (chart: ECharts | null) => {
      if (!chart) return;
      const handler = (p: unknown) => {
        const ev = p as {
          start?: number; end?: number;
          batch?: { start?: number; end?: number }[];
        };
        let start = ev.start;
        let end = ev.end;
        if (start == null && ev.batch && ev.batch[0]) {
          start = ev.batch[0].start;
          end = ev.batch[0].end;
        }
        if (start == null || end == null) return;
        const candles = data.candles;
        if (!candles.length) return;
        const { lo, hi, vMax } = visibleRange(candles, start, end);
        const padP = (hi - lo) * 0.08 || hi * 0.002;
        const yAxisPatch: Record<string, unknown>[] = [
          { min: lo - padP, max: hi + padP },
        ];
        if (layers.volume) yAxisPatch.push({ max: vMax * 1.15 });
        chart.setOption({ yAxis: yAxisPatch }, { lazyUpdate: true });
      };
      chart.off("datazoom");
      chart.on("datazoom", handler);
    },
    [data.candles, layers.volume],
  );

  return <EChartsBase option={option} height={height} onChartRef={onChartRef} />;
}

// ─── Layer toggle toolbar ────────────────────────────────────────
export function SmcLayerToolbar({
  layers,
  onChange,
}: {
  layers: SmcLayers;
  onChange: (next: SmcLayers) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1">
      {SMC_LAYER_META.map((m) => {
        const active = layers[m.key];
        return (
          <button
            key={m.key}
            onClick={() => onChange({ ...layers, [m.key]: !active })}
            className={`group inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[10px] font-semibold transition-all ${
              active
                ? "border-border bg-muted/60 text-foreground"
                : "border-transparent text-muted-foreground/60 hover:text-muted-foreground"
            }`}
            title={active ? `Hide ${m.label}` : `Show ${m.label}`}
          >
            <span
              className="h-2 w-2 rounded-sm transition-opacity"
              style={{ background: m.dot, opacity: active ? 1 : 0.3 }}
            />
            {m.label}
          </button>
        );
      })}
    </div>
  );
}

// ─── Supply / Demand sentiment widget ────────────────────────────
export function SupplyDemandWidget({
  sd,
  bias,
}: {
  sd: ChartResponse["supply_demand"];
  bias?: "bull" | "bear" | "neutral";
}) {
  if (!sd) return null;
  const total = sd.total_supply + sd.total_demand || 1;
  const demandPct = (sd.total_demand / total) * 100;
  const supplyPct = 100 - demandPct;
  const net = sd.total_demand - sd.total_supply;
  const ratio = total ? Math.max(-1, Math.min(1, net / total)) : 0;
  const sentiment =
    ratio > 0.15 ? "Bullish" : ratio < -0.15 ? "Bearish" : "Balanced";
  const sentColor =
    sentiment === "Bullish"
      ? "text-success"
      : sentiment === "Bearish"
        ? "text-destructive"
        : "text-muted-foreground";
  const sentDot =
    sentiment === "Bullish" ? PAL.bull : sentiment === "Bearish" ? PAL.bear : PAL.eq;

  const barBias =
    bias === "bull" ? "Bullish structure" : bias === "bear" ? "Bearish structure" : "Range";

  return (
    <div className="w-full min-w-[210px] rounded-xl border border-border bg-muted/30 p-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-bold uppercase tracking-wider text-muted-foreground">
          Order Flow
        </span>
        <span className={`flex items-center gap-1 text-[10px] font-bold ${sentColor}`}>
          <span
            className="inline-block h-1.5 w-1.5 rounded-full"
            style={{ background: sentDot }}
          />
          {sentiment}
        </span>
      </div>

      {/* Demand vs Supply split bar */}
      <div className="mt-2 flex h-2 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full transition-all"
          style={{ width: `${demandPct}%`, background: PAL.demandBorder }}
        />
        <div
          className="h-full transition-all"
          style={{ width: `${supplyPct}%`, background: PAL.supplyBorder }}
        />
      </div>

      <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]">
        <div>
          <div className="flex items-center gap-1 text-[9px] uppercase tracking-wide text-muted-foreground">
            <span
              className="inline-block h-1.5 w-1.5 rounded-sm"
              style={{ background: PAL.demandBorder }}
            />
            Demand
          </div>
          <div className="font-bold tabular-nums text-foreground">
            {demandPct.toFixed(0)}%
            <span className="ml-1 text-[9px] font-medium text-muted-foreground">
              {shortVol(sd.total_demand)}
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="flex items-center justify-end gap-1 text-[9px] uppercase tracking-wide text-muted-foreground">
            Supply
            <span
              className="inline-block h-1.5 w-1.5 rounded-sm"
              style={{ background: PAL.supplyBorder }}
            />
          </div>
          <div className="font-bold tabular-nums text-foreground">
            {supplyPct.toFixed(0)}%
            <span className="ml-1 text-[9px] font-medium text-muted-foreground">
              {shortVol(sd.total_supply)}
            </span>
          </div>
        </div>
      </div>

      <div className="mt-1.5 flex items-center justify-between border-t border-border/60 pt-1.5 text-[9px] text-muted-foreground">
        <span>Net flow</span>
        <span
          className={`font-bold tabular-nums ${net >= 0 ? "text-success" : "text-destructive"}`}
        >
          {net >= 0 ? "+" : ""}
          {shortVol(net)} ·{" "}
          <span
            className={
              bias === "bull"
                ? "text-success"
                : bias === "bear"
                  ? "text-destructive"
                  : "text-foreground"
            }
          >
            {barBias}
          </span>
        </span>
      </div>
    </div>
  );
}

