"use client";

import { cn } from "@/lib/utils";

/**
 * Renders the "R" trading unit (R-multiple = multiples of risk per trade).
 *
 * Always shown in small, muted text with a tooltip that explains the meaning,
 * so the abbreviation is never ambiguous to the reader.
 *
 * @example
 *   <RMultiple />                      // → "R"  (small, titled)
 *   <RMultiple full />                 // → "R · Risk multiples"
 *   <RMultiple className="..." />
 */
export function RMultiple({
  full = false,
  className,
}: {
  full?: boolean;
  className?: string;
}) {
  return (
    <span
      title="R = Risk multiples. Profit/loss measured in units of risk per trade (1R = amount risked on the trade)."
      className={cn(
        "text-[0.65em] font-medium text-muted-foreground tabular-nums",
        className,
      )}
    >
      R{full ? <span className="ml-1 opacity-80">· Risk multiples</span> : null}
    </span>
  );
}
