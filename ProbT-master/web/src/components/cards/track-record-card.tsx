"use client";

import { ClipboardCheck } from "lucide-react";
import { SectionCard } from "@/components/layout/section-card";
import { Skeleton } from "@/components/ui/skeleton";
import { useJournal } from "@/hooks/use-api";
import type { JournalBlock } from "@/types";

/**
 * Live signal track record: every signal the engine emits is journaled and
 * graded once its barrier window resolves. This card is the only place the
 * app proves (or disproves) its calibration on data that did not exist at
 * training time. Below min_n graded signals it says "verdict pending"
 * instead of quoting rates off a handful of observations.
 */
export function TrackRecordCard() {
  const { data, isLoading } = useJournal();

  return (
    <SectionCard
      title="Live Track Record"
      description="Signals graded after the fact — predicted vs. what actually happened"
      icon={<ClipboardCheck className="h-4 w-4 text-info" />}
      bodyClassName="p-4 space-y-2"
    >
      {isLoading || !data ? (
        <Skeleton className="h-16 rounded-xl" />
      ) : data.error ? (
        <p className="text-xs text-muted-foreground">journal unavailable: {data.error}</p>
      ) : (
        <>
          <div className="flex gap-4 text-xs tabular-nums text-muted-foreground">
            <span>
              signals <span className="font-semibold text-foreground">{data.n_signals}</span>
            </span>
            <span>
              graded <span className="font-semibold text-foreground">{data.n_graded}</span>
            </span>
          </div>

          {!data.verdict_available ? (
            <p className="text-xs text-muted-foreground">
              verdict pending — {data.n_graded}/{data.min_n} graded live signals. Rates are withheld
              until the sample is large enough to mean anything.
            </p>
          ) : (
            data.break && (
              <div className="space-y-1 text-xs tabular-nums">
                <Row label="break · overall" block={data.break.overall} />
                {(["tercile_low", "tercile_mid", "tercile_high"] as const).map(
                  (k) => data.break?.[k] && <Row key={k} label={`break · ${k.replace("tercile_", "")} p`} block={data.break[k]} />,
                )}
                {data.bounce_base_rate_check && (
                  <Row label="bounce · base-rate check" block={data.bounce_base_rate_check} />
                )}
              </div>
            )
          )}
        </>
      )}
    </SectionCard>
  );
}

function Row({ label, block }: { label: string; block: JournalBlock }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span>
        said <span className="font-semibold">{(block.predicted_mean * 100).toFixed(0)}%</span>
        {" → got "}
        <span className="font-semibold">{(block.observed_rate * 100).toFixed(0)}%</span>
        <span className="ml-1 text-muted-foreground">(n={block.n})</span>
      </span>
    </div>
  );
}
