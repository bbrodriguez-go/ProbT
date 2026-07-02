// ─── Number / date / currency formatters ──────────────────────────

export function fmtNumber(v: number | null | undefined, decimals = 2): string {
  if (v == null || isNaN(v)) return "—";
  return v.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function fmtCompact(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(v);
}

export function fmtPct(v: number | null | undefined, decimals = 1): string {
  if (v == null || isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(decimals)}%`;
}

export function fmtSigned(v: number | null | undefined, decimals = 2): string {
  if (v == null || isNaN(v)) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(decimals)}`;
}

export function fmtPrice(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `$${fmtNumber(v, v < 10 ? 4 : 2)}`;
}

// ─── Timezone helpers ─────────────────────────────────────────────

export interface ZoneFormat {
  /** IANA timezone id, e.g. "America/Lima". */
  timezone: string;
  /** 24-hour ("24h") or 12-hour ("12h") rendering. */
  clockFormat: "24h" | "12h";
}

const DEFAULT_ZONE: ZoneFormat = {
  timezone: "UTC",
  clockFormat: "24h",
};

function hour12(format: ZoneFormat["clockFormat"]): boolean {
  return format === "12h";
}

/** Format any Date / epoch ms / ISO string in the given zone. */
export function fmtClockInZone(
  value: Date | number | string | null | undefined,
  zone: ZoneFormat = DEFAULT_ZONE,
): string {
  if (value == null) return "—";
  const d = value instanceof Date ? value : new Date(value);
  if (isNaN(d.getTime())) return "—";
  try {
    return d.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: hour12(zone.clockFormat),
      timeZone: zone.timezone,
    });
  } catch {
    return "—";
  }
}

/** Live wall-clock string in the given zone. */
export function nowInZone(zone: ZoneFormat = DEFAULT_ZONE): string {
  return fmtClockInZone(new Date(), zone);
}

/** Date (Mon DD) in the given zone. */
export function fmtDateInZone(
  value: Date | number | string | null | undefined,
  zone: ZoneFormat = DEFAULT_ZONE,
): string {
  if (value == null) return "—";
  const d = value instanceof Date ? value : new Date(value);
  if (isNaN(d.getTime())) return "—";
  try {
    return d.toLocaleDateString("en-US", {
      month: "short",
      day: "2-digit",
      timeZone: zone.timezone,
    });
  } catch {
    return "—";
  }
}

/** Short UTC offset for a zone, e.g. "UTC-05". */
export function zoneOffsetLabel(zone: string, at: Date = new Date()): string {
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: zone,
      timeZoneName: "shortOffset",
    }).formatToParts(at);
    const tz = parts.find((p) => p.type === "timeZoneName");
    if (tz && tz.value) return tz.value.replace("GMT", "UTC");
  } catch {
    /* ignore */
  }
  return "UTC";
}

/** Human-friendly city name from an IANA id, e.g. "Lima" for "America/Lima". */
export function zoneNameLabel(zone: string): string {
  const parts = zone.split("/");
  const last = parts[parts.length - 1];
  return last.replace(/_/g, " ");
}

/** Get a list of common IANA timezones (browser-supported when available). */
export function listTimezones(): string[] {
  try {
    const supported = (Intl as unknown as {
      supportedValuesOf?: (k: "timeZone") => string[];
    }).supportedValuesOf?.("timeZone");
    if (supported && supported.length) return supported;
  } catch {
    /* ignore */
  }
  return [
    "UTC",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Lima",
    "America/Mexico_City",
    "America/Bogota",
    "America/Buenos_Aires",
    "America/Sao_Paulo",
    "Europe/London",
    "Europe/Madrid",
    "Europe/Paris",
    "Europe/Berlin",
    "Europe/Moscow",
    "Asia/Dubai",
    "Asia/Singapore",
    "Asia/Tokyo",
    "Asia/Hong_Kong",
    "Australia/Sydney",
  ];
}

// ─── Legacy (no zone) helpers, kept for callers that don't need one ──

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  } catch {
    return "—";
  }
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      month: "short",
      day: "2-digit",
    });
  } catch {
    return "—";
  }
}

/** @deprecated Use nowInZone() with the user's selected timezone. */
export function utcNow(): string {
  return nowInZone(DEFAULT_ZONE);
}
