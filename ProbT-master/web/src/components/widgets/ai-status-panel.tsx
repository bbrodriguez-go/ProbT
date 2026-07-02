"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Progress } from "@/components/ui/progress";
import { Brain, Signal, Target, Shield, Percent, Cpu, Clock, Server, Zap } from "lucide-react";
import { useReading, useHealth } from "@/hooks/use-api";
import { fmtPrice, fmtPct } from "@/lib/format";

export function AiStatusPanel() {
  const { data: reading } = useReading();
  const { data: health } = useHealth();

  const prob = reading ? reading.tier_a.probability : 0;
  const bias = reading?.tier_b.bias ?? "—";
  const price = reading?.price ?? 0;
  const sizing = reading?.sizing.final_pct ?? 0;
  const symbol = reading?.symbol ?? "XAUUSD";
  const timeframe = reading?.timeframe ?? "1D";
  const horizon = reading?.horizon_bars ?? 10;
  const direction = reading?.tier_a.direction ?? (prob >= 0.5 ? "LONG" : "SHORT");
  const isML = reading?.tier_a.is_probability ?? false;

  const rows = [
    { icon: Signal, label: "Current Signal", value: direction, accent: direction === "LONG" ? "text-success" : "text-destructive" },
    { icon: Brain, label: "AI Confidence", value: `${(prob * 100).toFixed(1)}%`, accent: prob > 0.6 ? "text-success" : "text-warning" },
    { icon: Shield, label: "Risk Level", value: prob > 0.6 ? "Moderate" : "Elevated", accent: "text-warning" },
    { icon: Percent, label: "Expected ROI", value: `${((prob * 2 - (1 - prob)) * 100).toFixed(1)}%`, accent: "text-info" },
    { icon: Zap, label: "Sizing (Half-Kelly, 2% cap)", value: `${sizing.toFixed(2)}%`, accent: "text-success" },
    { icon: Cpu, label: "Active Model", value: isML ? "LogReg-L1 (Logistic Regression)" : "Heuristic fallback", accent: isML ? "text-success" : "text-muted-foreground" },
    { icon: Clock, label: "Horizon (bars ahead)", value: `${horizon} bars`, accent: "text-muted-foreground" },
    { icon: Server, label: "Engine", value: health?.reading_ready ? "Ready" : "…", accent: "text-success" },
  ];

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-success/10">
            <Brain className="h-4 w-4 text-success" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">AI Status</h3>
            <p className="text-[10px] text-muted-foreground">Real-time engine</p>
          </div>
        </div>
        <Badge variant="secondary" className="bg-success/15 text-success">
          <span className="relative mr-1 flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-60 pulse-dot" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-success" />
          </span>
          LIVE
        </Badge>
      </div>

      {/* Price + Probability big */}
      <div className="mt-4 grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-muted/50 p-3">
          <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            <Target className="h-3 w-3" /> {symbol}
          </div>
          <div className="mt-1 text-xl font-bold tabular-nums">{fmtPrice(price)}</div>
        </div>
        <div className="rounded-xl bg-muted/50 p-3">
          <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground">
            <Signal className="h-3 w-3" /> Tier A · ML Signal
          </div>
          <div className="mt-1 text-xl font-bold tabular-nums text-success">
            {(prob * 100).toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Probability bar */}
      <div className="mt-3">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
          <span>Calibrated probability ({timeframe}, {horizon} bars)</span>
          <span className="font-medium">{(prob * 100).toFixed(1)}%</span>
        </div>
        <Progress value={prob * 100} className="h-2" />
      </div>

      <Separator className="my-4" />

      {/* Detail rows */}
      <div className="space-y-2.5">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-2 text-muted-foreground">
              <r.icon className="h-3.5 w-3.5" />
              <span>{r.label}</span>
            </div>
            <span className={`font-semibold tabular-nums ${r.accent}`}>{r.value}</span>
          </div>
        ))}
      </div>

      <Separator className="my-4" />

      {/* Server health */}
      <div className="grid grid-cols-3 gap-2 text-center">
        {[
          { label: "Engine", value: health?.reading_ready ? "OK" : "…", color: "text-success" },
          { label: "Uptime", value: health ? `${Math.floor(health.uptime_s / 60)}m` : "—", color: "text-foreground" },
          { label: "Status", value: health?.status ?? "—", color: "text-success" },
        ].map((s) => (
          <div key={s.label} className="rounded-lg bg-muted/50 py-1.5">
            <div className="text-[9px] uppercase tracking-wider text-muted-foreground">{s.label}</div>
            <div className={`text-xs font-semibold ${s.color}`}>{s.value}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}
