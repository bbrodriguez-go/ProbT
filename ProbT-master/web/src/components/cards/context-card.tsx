"use client";

import { Globe } from "lucide-react";
import { SectionCard } from "@/components/layout/section-card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useReading } from "@/hooks/use-api";

/**
 * Regime + news context (§9.2 cards 4-5, merged). Everything here is
 * CONTEXT, not a signal — labeled as such. Sourced from the v1 reading's
 * feature snapshot (VIX, DXY, entropy) and the news overlay.
 */
export function ContextCard() {
  const { data, isLoading } = useReading();
  const f = data?.features ?? {};
  const vix = typeof f.vix_level === "number" ? f.vix_level * 50 : null;
  const dxy1d = typeof f.dxy_return_1d === "number" ? f.dxy_return_1d * 100 : null;
  const entropy = typeof f.entropy_20 === "number" ? f.entropy_20 : null;
  const regime =
    vix == null ? null : vix < 15 ? "calm" : vix <= 25 ? "normal" : "stressed";
  const regimeCls =
    regime === "stressed"
      ? "bg-destructive/15 text-destructive"
      : regime === "calm"
        ? "bg-success/15 text-success"
        : "bg-muted text-muted-foreground";

  return (
    <SectionCard
      title="Regime & News"
      description="Context only — none of this is a trade signal"
      icon={<Globe className="h-4 w-4 text-info" />}
      bodyClassName="p-4 space-y-3"
    >
      {isLoading || !data ? (
        <Skeleton className="h-24 rounded-xl" />
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs tabular-nums">
            {regime && (
              <Badge variant="secondary" className={regimeCls}>
                {regime} regime
              </Badge>
            )}
            <span className="text-muted-foreground">
              VIX <span className="font-semibold text-foreground">{vix == null ? "—" : vix.toFixed(1)}</span>
            </span>
            <span className="text-muted-foreground">
              DXY 1d{" "}
              <span className={`font-semibold ${dxy1d != null && dxy1d > 0 ? "text-destructive" : "text-success"}`}>
                {dxy1d == null ? "—" : `${dxy1d > 0 ? "+" : ""}${dxy1d.toFixed(2)}%`}
              </span>
            </span>
            <span className="text-muted-foreground" title="< 0.6 structured · > 0.8 noise">
              entropy₂₀{" "}
              <span className="font-semibold text-foreground">{entropy == null ? "—" : entropy.toFixed(2)}</span>
            </span>
          </div>
          {data.news && (
            <div className="border-t border-border pt-2 text-xs">
              <span className="text-muted-foreground">news sentiment </span>
              <span
                className={`font-semibold tabular-nums ${
                  data.news.score > 0.1
                    ? "text-success"
                    : data.news.score < -0.1
                      ? "text-destructive"
                      : "text-muted-foreground"
                }`}
              >
                {data.news.score >= 0 ? "+" : ""}
                {data.news.score.toFixed(2)}
              </span>
              <p className="mt-1 leading-relaxed text-muted-foreground">{data.news.summary}</p>
            </div>
          )}
        </>
      )}
    </SectionCard>
  );
}
