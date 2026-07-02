"use client";

import { motion } from "framer-motion";
import { TrendingUp, TrendingDown } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Sparkline } from "@/components/charts/sparkline";
import type { MarketAsset } from "@/types";
import { cn } from "@/lib/utils";

export function MarketTile({ asset, index = 0 }: { asset: MarketAsset; index?: number }) {
  const isUp = asset.change >= 0;
  const accent = isUp ? "green" : "red";

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.03 }}
    >
      <Card className="p-3 transition-all hover:shadow-md hover:-translate-y-0.5">
        <div className="flex items-center justify-between gap-1">
          <span className="text-xs font-semibold">{asset.symbol}</span>
          <div className={cn("flex items-center gap-0.5 text-[10px] font-semibold tabular-nums", isUp ? "text-success" : "text-destructive")}>
            {isUp ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
            {isUp ? "+" : ""}{asset.change.toFixed(2)}%
          </div>
        </div>
        <div className="mt-1 text-base font-bold tabular-nums">
          {asset.price.toLocaleString("en-US", { maximumFractionDigits: asset.price < 10 ? 4 : 2 })}
        </div>
        <div className="mt-1 h-7">
          <Sparkline data={asset.sparkline} accent={accent as "green" | "red"} height={28} />
        </div>
        <div className="mt-1 flex items-center justify-between">
          <span className="text-[9px] uppercase tracking-wider text-muted-foreground">RSI Signal</span>
          <span className={cn(
            "text-[10px] font-medium",
            asset.prediction === "bullish" && "text-success",
            asset.prediction === "bearish" && "text-destructive",
            asset.prediction === "neutral" && "text-muted-foreground",
          )}>
            {asset.prediction === "bullish" ? "↑ Bullish" : asset.prediction === "bearish" ? "↓ Bearish" : "→ Neutral"}
          </span>
        </div>
      </Card>
    </motion.div>
  );
}
