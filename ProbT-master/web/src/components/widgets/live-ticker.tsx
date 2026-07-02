"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Radio, RefreshCw, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { useLiveReading } from "@/hooks/use-websocket";
import { useReading } from "@/hooks/use-api";
import { useAsset } from "@/hooks/use-asset-context";
import { useSettings } from "@/hooks/use-settings";
import { LIVE_INTERVAL_MS } from "@/lib/constants";
import { fmtPrice } from "@/lib/format";
import { fmtClockInZone } from "@/lib/format";
import { cn } from "@/lib/utils";

const SYMBOL_NAMES: Record<string, string> = {
  XAUUSD: "Gold",
  BTCUSD: "Bitcoin",
  ETHUSD: "Ethereum",
  NAS100: "Nasdaq 100",
  SPX500: "S&P 500",
};

function fmtClock(d: Date, timezone?: string, clockFormat?: "24h" | "12h"): string {
  return fmtClockInZone(d, {
    timezone: timezone ?? "UTC",
    clockFormat: clockFormat ?? "24h",
  });
}

export function LiveTicker() {
  const { symbol, timeframe } = useAsset();
  const { settings } = useSettings();
  const { reading: wsReading, connected, lastReceivedAt } = useLiveReading();
  // REST fallback keeps the ticker populated before the socket delivers its
  // first frame (or when the WS is unreachable, e.g. local dev without engine).
  const { data: restReading, isLoading } = useReading();

  const reading = wsReading ?? restReading ?? null;
  const price = reading?.price ?? 0;

  // Server-side "last update" timestamp (when the engine generated the reading).
  // Fall back to the moment we received the WS frame, then to REST fetch time.
  const generatedAt = reading?.generated_at;
  const lastUpdateMs = generatedAt
    ? (() => {
        const t = Date.parse(generatedAt);
        return Number.isNaN(t) ? lastReceivedAt : t;
      })()
    : lastReceivedAt;

  // 1s ticker so the countdown stays smooth without re-rendering the whole tree.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const nextUpdateMs = (lastReceivedAt ?? lastUpdateMs ?? now) + LIVE_INTERVAL_MS;
  const remainingMs = Math.max(0, nextUpdateMs - now);
  const remainingSec = Math.ceil(remainingMs / 1000);
  const progressPct = lastReceivedAt
    ? Math.min(100, ((LIVE_INTERVAL_MS - remainingMs) / LIVE_INTERVAL_MS) * 100)
    : 0;

  const stale = lastReceivedAt != null && now - lastReceivedAt > LIVE_INTERVAL_MS * 2;

  const name = SYMBOL_NAMES[symbol] ?? reading?.symbol_name ?? symbol;

  return (
    <motion.div
      initial={{ opacity: 0, y: -8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 200, damping: 24 }}
    >
      <Card className="relative overflow-hidden p-4 sm:p-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          {/* ─── Left: symbol + price ─── */}
          <div className="flex items-center gap-4 sm:gap-5">
            {/* Symbol badge */}
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-info/15 to-success/15 sm:h-14 sm:w-14">
              <span className="text-xs font-bold sm:text-sm">{symbol.slice(0, 3)}</span>
            </div>

            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold sm:text-base">{symbol}</span>
                <span className="hidden text-xs text-muted-foreground sm:inline">· {name}</span>
                <span className="rounded-md bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                  {timeframe}
                </span>
              </div>

              {isLoading && !reading ? (
                <Skeleton className="mt-1 h-8 w-40" />
              ) : (
                <motion.div
                  key={price}
                  initial={{ opacity: 0.55, y: 2 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.35 }}
                  className="mt-0.5 flex items-baseline gap-2"
                >
                  <span className="text-2xl font-bold tabular-nums sm:text-3xl">
                    {price ? fmtPrice(price) : "—"}
                  </span>
                </motion.div>
              )}
            </div>
          </div>

          {/* v1's bar-model probability + direction used to sit here — removed:
              that model has no proven OOS edge, and a big green number implies
              one. Directional readings come from the Zone Reading card only. */}

          {/* ─── Right: live status + countdown ─── */}
          <div className="flex items-center gap-4 sm:gap-6">
            <div className="min-w-[150px]">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider">
                  {connected ? (
                    <>
                      <span className="relative flex h-2 w-2">
                        <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-60 pulse-dot" />
                        <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
                      </span>
                      <span className="font-semibold text-success">LIVE</span>
                    </>
                  ) : (
                    <>
                      <Loader2 className="h-2.5 w-2.5 animate-spin text-muted-foreground" />
                      <span className="font-semibold text-muted-foreground">Connecting</span>
                    </>
                  )}
                </div>
                <span className="text-[10px] font-medium tabular-nums text-muted-foreground">
                  {stale ? "stale" : `${remainingSec}s`}
                </span>
              </div>

              {/* Last update — server time */}
              <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <RefreshCw className="h-3 w-3" />
                <span>Last update</span>
                <span className="font-mono font-medium tabular-nums text-foreground">
                  {lastUpdateMs ? fmtClock(new Date(lastUpdateMs), settings.timezone, settings.clockFormat) : "—"}
                </span>
              </div>

              {/* Next update — projected server time */}
              <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <Radio className="h-3 w-3" />
                <span>Next update</span>
                <span className="font-mono font-medium tabular-nums text-foreground">
                  {lastReceivedAt != null || lastUpdateMs != null
                    ? fmtClock(new Date(nextUpdateMs), settings.timezone, settings.clockFormat)
                    : "—"}
                </span>
              </div>

              {/* Countdown progress bar */}
              <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className={cn(
                    "h-full rounded-full transition-[width] duration-500 ease-linear",
                    stale ? "bg-warning" : "bg-success",
                  )}
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </Card>
    </motion.div>
  );
}
