"use client";

import { useState } from "react";
import { CandlestickChart } from "lucide-react";
import { DashboardShell } from "@/components/layout/dashboard-shell";
import { SectionCard } from "@/components/layout/section-card";
import { ZoneReadingCard } from "@/components/cards/zone-reading-card";
import { ContextCard } from "@/components/cards/context-card";
import { TrackRecordCard } from "@/components/cards/track-record-card";
import { SmcChart, SmcLayerToolbar, SupplyDemandWidget, DEFAULT_SMC_LAYERS } from "@/components/charts/smc-chart";
import type { SmcLayers } from "@/components/charts/smc-chart";
import { LiveTicker } from "@/components/widgets/live-ticker";
import { SettingsPanel } from "@/components/widgets/settings-panel";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { useChart } from "@/hooks/use-api";
import { accentText, accentBg } from "@/lib/constants";

/**
 * Deliberately minimal (probt v2 §9): the trader needs the price, the
 * zones, the event reading, and the regime/news context — nothing else.
 * The old diagnostic sections (backtest equity, KPI grid, model zoo,
 * gauges, heatmap, AI insight cards) decorated the page around a v1 model
 * with no proven edge; they now live only in the per-pair report files.
 */
export default function DashboardPage() {
  return (
    <DashboardShell>
      {/* ─── Live price ──────────────────────────────────────── */}
      <section id="section-live">
        <LiveTicker />
      </section>

      {/* ─── The decision panel: chart + zones (left), reading +
             context (right) ─────────────────────────────────── */}
      <section className="grid gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2" id="section-smc">
          <SmcChartSection />
        </div>
        <div className="space-y-4 xl:col-span-1">
          <div id="section-zone-reading">
            <ZoneReadingCard />
          </div>
          <div id="section-context">
            <ContextCard />
          </div>
          <div id="section-track-record">
            <TrackRecordCard />
          </div>
        </div>
      </section>

      {/* ─── Settings (timezone, clock) ──────────────────────── */}
      <section id="section-settings">
        <SettingsPanel />
      </section>
    </DashboardShell>
  );
}

// ─── Smart Money Concepts chart ──────────────────────────────────
function SmcChartSection() {
  const { data, isLoading } = useChart();
  const [layers, setLayers] = useState<SmcLayers>(DEFAULT_SMC_LAYERS);
  const smc = data?.smc;
  const sd = data?.supply_demand;
  const bias = smc?.bias ?? "neutral";
  const biasAccent = bias === "bull" ? "green" : bias === "bear" ? "red" : "gray";
  const biasLabel = bias === "bull" ? "Bullish" : bias === "bear" ? "Bearish" : "Neutral";

  return (
    <SectionCard
      title="Chart & Zones"
      description="Order Blocks · BOS / CHoCH · Fair Value Gaps · Supply & Demand"
      icon={<CandlestickChart className="h-4 w-4 text-info" />}
      action={
        <div className="flex flex-wrap items-center gap-1.5 text-[10px]">
          {smc && (
            <Badge variant="secondary" className={`${accentBg[biasAccent]} ${accentText[biasAccent]}`}>
              {biasLabel} bias
            </Badge>
          )}
        </div>
      }
      bodyClassName="p-2 sm:p-3 space-y-2"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <SmcLayerToolbar layers={layers} onChange={setLayers} />
        <SupplyDemandWidget sd={sd ?? null} bias={bias} />
      </div>

      {isLoading || !data || !data.candles.length ? (
        <Skeleton className="h-[560px] rounded-xl" />
      ) : (
        <SmcChart data={data} height={560} layers={layers} />
      )}
    </SectionCard>
  );
}
