"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sparkline } from "@/components/charts/sparkline";
import type { ModelCard as ModelCardType } from "@/types";
import { cn } from "@/lib/utils";

const statusConfig = {
  active: { label: "Active", className: "bg-success/15 text-success" },
  benchmark: { label: "Benchmark", className: "bg-info/15 text-info" },
  queued: { label: "Queued", className: "bg-muted text-muted-foreground" },
};

export function ModelCard({ model, index = 0 }: { model: ModelCardType; index?: number }) {
  const status = statusConfig[model.status];
  const accent = model.status === "active" ? "green" : model.status === "benchmark" ? "blue" : "gray";
  const rocAuc = model.roc_auc ?? null;
  const bss = model.brier_skill_score ?? null;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.05 }}
    >
      <Card className="p-4 transition-all hover:shadow-md hover:-translate-y-0.5">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="font-semibold truncate">{model.name}</span>
              <Badge variant="secondary" className={cn("h-5 text-[10px]", status.className)}>
                {status.label}
              </Badge>
              {model.calibrated && (
                <Badge variant="outline" className="h-5 text-[10px] text-muted-foreground">
                  calibrated
                </Badge>
              )}
            </div>
            <p className="text-[11px] text-muted-foreground mt-0.5 truncate">{model.type}</p>
          </div>
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3">
          <div
            title="ROC-AUC is threshold-free: 0.5 = random, >0.55 = useful signal, >0.60 = clear edge."
          >
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">ROC-AUC</div>
            <div className="text-lg font-bold tabular-nums">
              {rocAuc != null ? rocAuc.toFixed(3) : "—"}
            </div>
          </div>
          <div
            title="1 − Brier Score. Confidence in the probability estimate (higher is better)."
          >
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Confidence</div>
            <div className="text-lg font-bold tabular-nums">
              {model.confidence > 0 ? `${(model.confidence * 100).toFixed(0)}%` : "—"}
            </div>
          </div>
        </div>

        {model.performance.length > 0 && (
          <div className="mt-2 h-8">
            <Sparkline data={model.performance} accent={accent as "green" | "blue" | "gray"} height={32} />
          </div>
        )}

        <div
          className="mt-3 flex items-center justify-between border-t border-border pt-2 text-[10px] text-muted-foreground"
          title="Brier Skill Score vs. climatology: BSS > +0.10 = real edge, +0.28 = institutional. Below 0 = no better than always predicting the base rate."
        >
          <span>
            Brier: {model.brier != null ? model.brier.toFixed(4) : "—"}
            {bss != null && <span className="ml-2">BSS: {bss >= 0 ? "+" : ""}{bss.toFixed(3)}</span>}
          </span>
          <span>Train: {model.train_time}</span>
        </div>
      </Card>
    </motion.div>
  );
}
