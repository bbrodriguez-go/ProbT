"use client";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { useTrades } from "@/hooks/use-api";
import { useSettings } from "@/hooks/use-settings";
import { fmtPrice, fmtSigned, fmtDateInZone } from "@/lib/format";
import { RMultiple } from "@/components/lib/r-multiple";
import { cn } from "@/lib/utils";
import { ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";

export function LiveTradesTable() {
  const { data: trades, isLoading } = useTrades(30);
  const { settings } = useSettings();
  const zone = { timezone: settings.timezone, clockFormat: settings.clockFormat };

  return (
    <Card className="overflow-hidden">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h3 className="text-sm font-semibold">Live Trades</h3>
          <p className="text-[11px] text-muted-foreground">Backtest trade history</p>
        </div>
        <Badge variant="secondary" className="bg-success/15 text-success">
          {trades?.length ?? 0} trades
        </Badge>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border bg-muted/30 text-muted-foreground">
              <th className="px-4 py-2 text-left font-medium">Time</th>
              <th className="px-4 py-2 text-left font-medium">Asset</th>
              <th className="px-4 py-2 text-left font-medium">Type</th>
              <th className="px-4 py-2 text-right font-medium">Entry</th>
              <th className="px-4 py-2 text-right font-medium">Exit</th>
              <th className="px-4 py-2 text-right font-medium" title="Profit and loss in Risk multiples (1R = amount risked)">PnL (R)</th>
              <th className="px-4 py-2 text-center font-medium">Status</th>
              <th className="px-4 py-2 text-right font-medium">Conf.</th>
              <th className="px-4 py-2 text-right font-medium">Dur.</th>
              <th className="px-4 py-2"></th>
            </tr>
          </thead>
          <tbody>
            {isLoading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i} className="border-b border-border">
                  {Array.from({ length: 10 }).map((_, j) => (
                    <td key={j} className="px-4 py-2.5"><Skeleton className="h-3 w-12" /></td>
                  ))}
                </tr>
              ))
            ) : (
              trades?.map((t, i) => (
                <tr
                  key={i}
                  className="border-b border-border transition-colors hover:bg-muted/40"
                >
                  <td className="px-4 py-2.5 font-mono text-muted-foreground whitespace-nowrap" title={`${settings.timezone}`}>
                    {fmtDateInZone(t.time, zone)}
                  </td>
                  <td className="px-4 py-2.5 font-medium">{t.asset}</td>
                  <td className="px-4 py-2.5">
                    <span className={cn("text-[10px] font-semibold", t.type === "LONG" ? "text-success" : "text-destructive")}>
                      {t.type}
                    </span>
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums">{fmtPrice(t.entry)}</td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">{fmtPrice(t.exit)}</td>
                  <td className={cn("px-4 py-2.5 text-right font-semibold tabular-nums", t.pnl > 0 ? "text-success" : "text-destructive")}>
                    {fmtSigned(t.pnl)}<RMultiple />
                  </td>
                  <td className="px-4 py-2.5 text-center">
                    <Badge variant="secondary" className={cn("h-5 text-[10px]", t.status === "TP" ? "bg-success/15 text-success" : "bg-destructive/15 text-destructive")}>
                      {t.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-2.5 text-right tabular-nums text-muted-foreground">{(t.confidence * 100).toFixed(0)}%</td>
                  <td className="px-4 py-2.5 text-right text-muted-foreground">{t.duration}</td>
                  <td className="px-4 py-2.5">
                    <Button variant="ghost" size="icon" className="h-6 w-6">
                      <ExternalLink className="h-3 w-3" />
                    </Button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
