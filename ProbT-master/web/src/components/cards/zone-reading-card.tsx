"use client";

import { Crosshair, ShieldAlert, MoonStar } from "lucide-react";
import { SectionCard } from "@/components/layout/section-card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useReadingV2 } from "@/hooks/use-api";
import { fmtNumber } from "@/lib/format";
import type { ReadingV2, TierV2 } from "@/types";

/**
 * Zone-conditional dual-model reading (probt v2 §9): shows a probability
 * ONLY when a zone-touch event fired on the latest closed bar. The muted
 * "no zone touch" state is the normal one — the card never fakes activity,
 * never hides the conformal band, and tags every non-model number.
 */
export function ZoneReadingCard() {
  const { data, isLoading } = useReadingV2();

  return (
    <SectionCard
      title="Zone Reading (v2)"
      description="Event-driven bounce/break probabilities · conformal bands · per-event payoff Kelly"
      icon={<Crosshair className="h-4 w-4 text-info" />}
      action={
        data?.state === "signal" && data.direction?.direction ? (
          <DirectionBadge reading={data} />
        ) : undefined
      }
      bodyClassName="p-4"
    >
      {isLoading || !data ? (
        <Skeleton className="h-40 rounded-xl" />
      ) : data.error ? (
        <p className="text-xs text-muted-foreground">reading unavailable: {data.error}</p>
      ) : data.state === "blackout" ? (
        <Blackout reading={data} />
      ) : data.state === "no_signal" ? (
        <NoSignal reading={data} />
      ) : (
        <Signal reading={data} />
      )}
    </SectionCard>
  );
}

function DirectionBadge({ reading }: { reading: ReadingV2 }) {
  const d = reading.direction!;
  const long = d.direction === "LONG";
  const cls = d.winner_is_probability
    ? long
      ? "bg-success/15 text-success"
      : "bg-destructive/15 text-destructive"
    : "bg-muted text-muted-foreground";
  return (
    <Badge variant="secondary" className={cls}>
      {d.direction} via {d.winner}
      {!d.winner_is_probability && " (not a probability)"}
    </Badge>
  );
}

function Blackout({ reading }: { reading: ReadingV2 }) {
  const s = reading.blackout!.starts_in_seconds;
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return (
    <div className="flex items-center gap-3 rounded-xl border border-warning/40 bg-warning/10 p-4">
      <ShieldAlert className="h-5 w-5 shrink-0 text-warning" />
      <div className="text-sm">
        <p className="font-semibold text-warning">
          blackout: {reading.blackout!.event} in{" "}
          <span className="tabular-nums">
            {String(hh).padStart(2, "0")}:{String(mm).padStart(2, "0")}:{String(ss).padStart(2, "0")}
          </span>
        </p>
        <p className="text-xs text-muted-foreground">
          model outputs are suppressed within 24h of a central-bank decision
        </p>
      </div>
    </div>
  );
}

function NoSignal({ reading }: { reading: ReadingV2 }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-muted/20 p-4">
      <MoonStar className="h-5 w-5 shrink-0 text-muted-foreground" />
      <div className="text-sm">
        <p className="font-medium text-muted-foreground">no zone touch on the latest closed bar</p>
        <p className="text-xs text-muted-foreground/70">
          a reading is produced only when price interacts with a supply/demand zone, order block or FVG —
          this muted state is the normal one
        </p>
        {reading.note && <p className="text-xs text-warning mt-1">{reading.note}</p>}
      </div>
      {!reading.calendar_loaded && (
        <Badge variant="outline" className="ml-auto shrink-0 text-[10px] text-warning border-warning/40">
          blackout calendar missing
        </Badge>
      )}
    </div>
  );
}

function Signal({ reading }: { reading: ReadingV2 }) {
  const e = reading.event!;
  const supply = e.zone_kind === "supply";
  return (
    <div className="space-y-4">
      {/* zone context (§9.2 card 2) */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge
          variant="secondary"
          className={supply ? "bg-destructive/15 text-destructive" : "bg-success/15 text-success"}
        >
          {e.zone_kind.toUpperCase()}
        </Badge>
        <Badge variant="outline">{e.trigger}</Badge>
        <span className="font-mono text-muted-foreground">{e.source}</span>
        <span className="tabular-nums text-muted-foreground">
          ${fmtNumber(e.zone_bottom)} – ${fmtNumber(e.zone_top)}
        </span>
        <span className="text-muted-foreground">age {e.zone_age_bars} bars</span>
        <span className="text-muted-foreground">touches {e.touches_count}</span>
        {e.zone_pct != null && <span className="text-muted-foreground">Δ share {e.zone_pct}%</span>}
      </div>

      {/* dual probabilities (§9.2 card 1) */}
      <div className="grid gap-3 sm:grid-cols-2">
        <TierBlock label={supply ? "bounce (short)" : "bounce (long)"} tier={reading.tier_a_bounce!} />
        <TierBlock label={supply ? "break (long)" : "break (short)"} tier={reading.tier_a_break!} />
      </div>

      {/* sizing (§9.2 card 6) — greyed out when not EV+ */}
      {reading.sizing && <SizingBlock reading={reading} />}
    </div>
  );
}

function TierBlock({ label, tier }: { label: string; tier: TierV2 }) {
  const p = tier.probability;
  const lo = tier.probability_lo ?? p ?? 0;
  const hi = tier.probability_hi ?? p ?? 0;
  return (
    <div className="rounded-xl border border-border p-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{label}</span>
        {tier.is_probability ? (
          <Badge variant="secondary" className="ml-auto bg-info/15 text-info text-[10px]">calibrated ML</Badge>
        ) : (
          <Badge variant="outline" className="ml-auto text-[10px] text-warning border-warning/40">
            not a probability
          </Badge>
        )}
      </div>
      <div className="text-3xl font-bold tabular-nums">
        {p == null ? "—" : `${(p * 100).toFixed(1)}%`}
      </div>
      {p != null && (
        <>
          {/* probability strip: point estimate fill + conformal band ghost */}
          <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted/40">
            <div
              className="absolute top-0 h-full bg-info/25"
              style={{ left: `${lo * 100}%`, width: `${Math.max((hi - lo) * 100, 0)}%` }}
            />
            <div
              className="absolute top-0 h-full w-[2px] bg-info transition-[left] duration-[400ms] ease-out"
              style={{ left: `calc(${p * 100}% - 1px)` }}
            />
          </div>
          <p className="text-[10px] tabular-nums text-muted-foreground">
            {tier.band_kind === "calibration_wilson_90"
              ? "when the model said this, reality delivered"
              : "90% conformal band"}
            : {(lo * 100).toFixed(1)}% – {(hi * 100).toFixed(1)}%
            {!tier.interval_reliable && (
              <span className="ml-1 text-warning">· band unverified (coverage check failed or missing)</span>
            )}
            {tier.payoff_b != null && <span className="ml-1">· payoff b = {tier.payoff_b.toFixed(2)}</span>}
          </p>
        </>
      )}
      {tier.note && <p className="text-[10px] leading-snug text-warning">{tier.note}</p>}
      {tier.contributions && tier.contributions.length > 0 && (
        <Contributions rows={tier.contributions} />
      )}
    </div>
  );
}

/** §9.2 card 3: what is driving THIS prediction — top features by
 *  |coefficient × standardized value|, green pushes up, red pushes down. */
function Contributions({ rows }: { rows: NonNullable<TierV2["contributions"]> }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-9);
  return (
    <div className="space-y-1 border-t border-border pt-2">
      {rows.map((r) => {
        const pos = r.contribution >= 0;
        return (
          <div key={r.feature} className="flex items-center gap-2 text-[10px]" title={`raw value ${r.value}`}>
            <span className="w-32 shrink-0 truncate font-mono text-muted-foreground">{r.feature}</span>
            <div className="relative h-2.5 flex-1 rounded bg-muted/40">
              <div
                className={`absolute top-0 h-full rounded ${pos ? "left-1/2 bg-success/60" : "right-1/2 bg-destructive/60"}`}
                style={{ width: `${(Math.abs(r.contribution) / max) * 50}%` }}
              />
              <div className="absolute left-1/2 top-0 h-full w-px bg-border" />
            </div>
            <span className={`w-12 shrink-0 text-right tabular-nums ${pos ? "text-success" : "text-destructive"}`}>
              {pos ? "+" : ""}{r.contribution.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function SizingBlock({ reading }: { reading: ReadingV2 }) {
  const z = reading.sizing!;
  return (
    <div
      className={`flex flex-wrap items-center gap-x-4 gap-y-1 rounded-xl border border-border p-3 text-xs tabular-nums ${
        z.ev_positive ? "" : "opacity-50"
      }`}
    >
      <span className="font-semibold">sizing</span>
      <span>full Kelly {z.full_kelly_pct.toFixed(3)}%</span>
      <span>half Kelly {z.half_kelly_pct.toFixed(3)}%</span>
      <span>cap {z.cap_pct}%</span>
      <span className="font-bold">final {z.final_pct.toFixed(3)}%</span>
      <span className="text-muted-foreground">b = {z.payoff_b.toFixed(2)}</span>
      {!z.ev_positive && (
        <span className="text-muted-foreground">
          EV not positive — the conformal lower bound does not beat breakeven; no bet
        </span>
      )}
    </div>
  );
}
