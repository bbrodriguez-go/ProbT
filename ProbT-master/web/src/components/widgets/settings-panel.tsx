"use client";

import { useEffect, useMemo, useState } from "react";
import { Settings as SettingsIcon, Clock, Globe2, RotateCcw, Check } from "lucide-react";
import { SectionCard } from "@/components/layout/section-card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSettings } from "@/hooks/use-settings";
import {
  listTimezones,
  nowInZone,
  zoneNameLabel,
  zoneOffsetLabel,
} from "@/lib/format";

function useFilteredTimezones(query: string): string[] {
  return useMemo(() => {
    const all = listTimezones();
    if (!query.trim()) return all;
    const q = query.toLowerCase();
    return all.filter((tz) => tz.toLowerCase().includes(q));
  }, [query]);
}

function ClockPreview() {
  const { settings } = useSettings();
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);
  const zone = { timezone: settings.timezone, clockFormat: settings.clockFormat };
  const offset = zoneOffsetLabel(settings.timezone);
  const name = zoneNameLabel(settings.timezone);
  const label =
    settings.clockLabel === "name"
      ? name
      : settings.clockLabel === "offset"
        ? offset
        : offset;
  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-muted/40 px-3 py-2.5">
      <Clock className="h-4 w-4 text-info" />
      <div className="flex flex-col">
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <span className="font-mono text-lg font-semibold tabular-nums">
          {nowInZone(zone)}
        </span>
      </div>
      <span className="ml-auto text-[11px] text-muted-foreground truncate">
        {name}
      </span>
    </div>
  );
}

export function SettingsPanel() {
  const { settings, setTimezone, setClockFormat, setClockLabel, reset, hydrated } =
    useSettings();
  const [query, setQuery] = useState("");

  const tzs = useFilteredTimezones(query);
  const tzsSliced = useMemo(() => tzs.slice(0, 300), [tzs]);

  return (
    <SectionCard
      title="Settings"
      description="Display preferences · timezone · clock format"
      icon={<SettingsIcon className="h-4 w-4 text-muted-foreground" />}
      action={
        <Button variant="ghost" size="sm" className="h-7 gap-1.5 text-xs" onClick={reset}>
          <RotateCcw className="h-3 w-3" /> Reset
        </Button>
      }
      bodyClassName="p-4 space-y-5"
    >
      {/* Live preview */}
      <ClockPreview />

      {/* Timezone selector */}
      <div className="space-y-1.5">
        <Label className="gap-1.5">
          <Globe2 className="h-3.5 w-3.5 text-muted-foreground" />
          Local timezone
        </Label>
        <p className="text-[11px] text-muted-foreground">
          All timestamps across the dashboard (header clock, last/next update,
          trade times, etc.) will render in this timezone.
        </p>
        <Select
          value={settings.timezone}
          onValueChange={(v) => v && setTimezone(v as string)}
        >
          <SelectTrigger className="w-full">
            <SelectValue>
              {hydrated
                ? `${settings.timezone}  ·  ${zoneOffsetLabel(settings.timezone)}  ·  ${zoneNameLabel(settings.timezone)}`
                : "Loading…"}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <div className="px-1.5 pb-1">
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search timezone…"
                className="h-8 w-full rounded-md border border-input bg-transparent px-2 text-sm outline-none focus-visible:border-ring"
              />
            </div>
            <SelectSeparator />
            <SelectGroup>
              <SelectLabel>Timezones ({tzs.length})</SelectLabel>
              {tzsSliced.map((tz) => (
                <SelectItem key={tz} value={tz}>
                  <span className="font-mono text-xs">{tz}</span>
                  <span className="text-muted-foreground">
                    · {zoneOffsetLabel(tz)} · {zoneNameLabel(tz)}
                  </span>
                </SelectItem>
              ))}
              {tzs.length > tzsSliced.length && (
                <div className="px-1.5 py-1 text-[10px] text-muted-foreground">
                  Showing first 300 — refine your search.
                </div>
              )}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      {/* Clock format */}
      <div className="space-y-1.5">
        <Label className="gap-1.5">
          <Clock className="h-3.5 w-3.5 text-muted-foreground" />
          24-hour clock
        </Label>
        <p className="text-[11px] text-muted-foreground">
          Toggle off to display times with AM/PM (12-hour).
        </p>
        <div className="flex items-center gap-2 pt-0.5">
          <Switch
            id="fmt-24h"
            checked={settings.clockFormat === "24h"}
            onCheckedChange={(c) => setClockFormat(c ? "24h" : "12h")}
          />
          <span className="text-xs text-muted-foreground">
            {settings.clockFormat === "24h" ? "24-hour" : "12-hour (AM/PM)"}
          </span>
        </div>
      </div>

      {/* Clock label */}
      <div className="space-y-1.5">
        <Label className="gap-1.5">
          <Check className="h-3.5 w-3.5 text-muted-foreground" />
          Clock label
        </Label>
        <p className="text-[11px] text-muted-foreground">
          What to show beside the header clock.
        </p>
        <div className="flex flex-wrap gap-1.5 pt-0.5">
          {(
            [
              { id: "abbr", label: "UTC offset", example: "UTC-05" },
              { id: "name", label: "City name", example: "Lima" },
              { id: "offset", label: "Full offset", example: "UTC-05:00" },
            ] as const
          ).map((opt) => (
            <Button
              key={opt.id}
              variant={settings.clockLabel === opt.id ? "default" : "outline"}
              size="sm"
              className="h-8 gap-1.5 text-xs"
              onClick={() => setClockLabel(opt.id)}
            >
              {opt.label}
              <span className="text-[10px] opacity-70">({opt.example})</span>
            </Button>
          ))}
        </div>
      </div>
    </SectionCard>
  );
}
