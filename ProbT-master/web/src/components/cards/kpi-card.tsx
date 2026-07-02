"use client";

import { motion } from "framer-motion";
import { ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Sparkline } from "@/components/charts/sparkline";
import { RMultiple } from "@/components/lib/r-multiple";
import type { Kpi } from "@/types";
import { accentText, accentBgSoft } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function KpiCard({ kpi, index = 0 }: { kpi: Kpi; index?: number }) {
  const TrendIcon = kpi.trend === "up" ? ArrowUpRight : kpi.trend === "down" ? ArrowDownRight : Minus;
  const showChange = kpi.change !== 0 && typeof kpi.value === "number";

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.04, type: "spring", stiffness: 200, damping: 24 }}
    >
      <Card className="group relative overflow-hidden p-4 transition-all hover:shadow-md hover:-translate-y-0.5 hover:border-foreground/10">
        {/* Accent line */}
        <div className={cn("absolute left-0 top-0 h-0.5 w-full origin-left scale-x-0 transition-transform duration-300 group-hover:scale-x-100", `bg-current ${accentText[kpi.accent]}`)} />

        <div className="flex items-start justify-between gap-2">
          <span className="text-xs font-medium text-muted-foreground">{kpi.label}</span>
          <div className={cn("flex items-center gap-0.5 rounded-md px-1 py-0.5 text-[10px] font-semibold tabular-nums", accentBgSoft[kpi.accent], accentText[kpi.accent])}>
            <TrendIcon className="h-2.5 w-2.5" />
            {showChange && <span>{kpi.change > 0 ? "+" : ""}{kpi.change}</span>}
          </div>
        </div>

        <div className="mt-2 flex items-end justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold tracking-tight tabular-nums truncate">
                {typeof kpi.value === "number" ? kpi.value.toLocaleString("en-US", { maximumFractionDigits: 2 }) : kpi.value}
              </span>
              {kpi.unit && (
                kpi.unit === "R" ? (
                  <RMultiple />
                ) : (
                  <span className="text-xs text-muted-foreground font-medium">{kpi.unit}</span>
                )
              )}
            </div>
          </div>
          {kpi.sparkline.length > 0 && (
            <div className="h-9 w-16 shrink-0">
              <Sparkline data={kpi.sparkline} accent={kpi.accent} height={36} />
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  );
}
