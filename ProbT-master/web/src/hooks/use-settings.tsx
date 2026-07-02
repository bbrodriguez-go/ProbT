"use client";

import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  ReactNode,
} from "react";

const STORAGE_KEY = "probt.settings";

export type ClockFormat = "24h" | "12h";

export interface Settings {
  /** IANA timezone name, e.g. "America/Lima". */
  timezone: string;
  /** Whether to render clocks with 24-hour ("24h") or 12-hour ("12h") format. */
  clockFormat: ClockFormat;
  /** Label shown beside the clock: "abbr" (UTC-05), "name" (Lima), "offset" (UTC-05:00). */
  clockLabel: "abbr" | "name" | "offset";
}

export const DEFAULT_SETTINGS: Settings = {
  timezone:
    typeof Intl !== "undefined"
      ? Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
      : "UTC",
  clockFormat: "24h",
  clockLabel: "abbr",
};

function detectBrowserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

function loadInitial(): Settings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_SETTINGS, timezone: detectBrowserTimezone() };
    const parsed = JSON.parse(raw);
    return {
      timezone:
        typeof parsed.timezone === "string" && parsed.timezone
          ? parsed.timezone
          : DEFAULT_SETTINGS.timezone,
      clockFormat:
        parsed.clockFormat === "12h" || parsed.clockFormat === "24h"
          ? parsed.clockFormat
          : DEFAULT_SETTINGS.clockFormat,
      clockLabel:
        parsed.clockLabel === "name" ||
        parsed.clockLabel === "offset" ||
        parsed.clockLabel === "abbr"
          ? parsed.clockLabel
          : DEFAULT_SETTINGS.clockLabel,
    };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

interface SettingsContextValue {
  settings: Settings;
  setTimezone: (tz: string) => void;
  setClockFormat: (f: ClockFormat) => void;
  setClockLabel: (l: Settings["clockLabel"]) => void;
  reset: () => void;
  /** True after the provider has hydrated on the client. */
  hydrated: boolean;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    // Defer to avoid setState synchronously inside the effect body
    // (the React Compiler flags that pattern as a cascading render).
    const id = setTimeout(() => {
      setSettings(loadInitial());
      setHydrated(true);
    }, 0);
    return () => clearTimeout(id);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* ignore */
    }
  }, [settings, hydrated]);

  const setTimezone = useCallback((tz: string) => {
    setSettings((s) => ({ ...s, timezone: tz }));
  }, []);

  const setClockFormat = useCallback((f: ClockFormat) => {
    setSettings((s) => ({ ...s, clockFormat: f }));
  }, []);

  const setClockLabel = useCallback((l: Settings["clockLabel"]) => {
    setSettings((s) => ({ ...s, clockLabel: l }));
  }, []);

  const reset = useCallback(() => {
    setSettings({ ...DEFAULT_SETTINGS, timezone: detectBrowserTimezone() });
  }, []);

  return (
    <SettingsContext.Provider
      value={{
        settings,
        setTimezone,
        setClockFormat,
        setClockLabel,
        reset,
        hydrated,
      }}
    >
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx)
    throw new Error("useSettings must be used within <SettingsProvider>");
  return ctx;
}
