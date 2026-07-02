"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Search,
  Bell,
  Settings,
  Moon,
  Sun,
  ChevronDown,
  Wifi,
  Command,
  Loader2,
} from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import { utcNow, nowInZone, zoneNameLabel, zoneOffsetLabel } from "@/lib/format";
import { useAsset, AVAILABLE_SYMBOLS, AVAILABLE_TIMEFRAMES } from "@/hooks/use-asset-context";
import { useSettings } from "@/hooks/use-settings";

export function Header({ onCommand }: { onCommand: () => void }) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  const [time, setTime] = useState("");
  const { symbol, timeframe, setSymbol, setTimeframe, transitioning } = useAsset();
  const { settings, hydrated } = useSettings();

  useEffect(() => {
    // Defer the mount flag so we don't setState synchronously inside the effect
    // (the React Compiler flags that pattern as a cascading render).
    const id = setTimeout(() => setMounted(true), 0);
    const tick = () =>
      setTime(nowInZone({ timezone: settings.timezone, clockFormat: settings.clockFormat }));
    tick();
    const tickId = setInterval(tick, 1000);
    return () => { clearTimeout(id); clearInterval(tickId); };
  }, [settings.timezone, settings.clockFormat]);

  // Avoid hydration mismatch: render the legacy UTC string on the server and
  // first paint, then swap to the user's timezone once mounted.
  const displayTime = mounted && hydrated ? time : utcNow();
  const tzLabel =
    hydrated && mounted
      ? settings.clockLabel === "name"
        ? zoneNameLabel(settings.timezone)
        : zoneOffsetLabel(settings.timezone)
      : "UTC";

  return (
    <header className="sticky top-0 z-40 h-14 border-b border-border bg-card/80 backdrop-blur-xl">
      <div className="flex h-full items-center gap-2 px-3 sm:gap-3 sm:px-4">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: "spring", stiffness: 200, damping: 20 }}
            className="flex h-8 w-8 items-center justify-center rounded-xl bg-card shadow-sm ring-1 ring-border overflow-hidden"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/probt_logo.png"
              alt="probt"
              className="h-full w-full object-cover"
              draggable={false}
            />
          </motion.div>
          <div className="hidden sm:flex items-center gap-1.5">
            <span className="text-sm font-semibold tracking-tight">probt</span>
            <span className="text-[10px] font-medium text-muted-foreground">AI</span>
          </div>
        </div>

        <Separator orientation="vertical" className="hidden sm:block h-6" />

        {/* LIVE status */}
        <div className="hidden md:flex items-center gap-2">
          <div className="flex items-center gap-1.5 rounded-lg bg-success/10 px-2 py-1">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full rounded-full bg-success opacity-60 pulse-dot" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
            </span>
            <span className="text-xs font-semibold text-success">LIVE</span>
          </div>
          <div className="flex items-center gap-1 text-xs text-muted-foreground">
            <Wifi className="h-3.5 w-3.5 text-success" />
            <span className="hidden lg:inline">Market connected</span>
          </div>
        </div>

        <Separator orientation="vertical" className="hidden lg:block h-6" />

        {/* Local time */}
        <div className="hidden lg:flex items-center gap-1.5 font-mono text-xs text-muted-foreground tabular-nums" title={settings.timezone}>
          <span className="text-[10px] uppercase tracking-wider">{tzLabel}</span>
          <span className="font-medium text-foreground">{displayTime}</span>
        </div>

        {/* Center: Search */}
        <div className="flex-1 flex justify-center px-2 sm:px-4">
          <Button
            variant="outline"
            onClick={onCommand}
            className="group h-9 w-full max-w-md justify-start gap-2 text-sm text-muted-foreground hover:text-foreground"
          >
            <Search className="h-4 w-4" />
            <span className="hidden sm:inline">Search or jump to…</span>
            <span className="sm:hidden">Search</span>
            <kbd className="ml-auto hidden sm:flex items-center gap-0.5 rounded border border-border bg-muted px-1.5 py-0.5 text-[10px] font-mono">
              <Command className="h-2.5 w-2.5" />K
            </kbd>
          </Button>
        </div>

        {/* Market selector */}
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="ghost" size="sm" className="hidden md:flex gap-1.5 font-medium" />}>
            <span className={transitioning ? "text-muted-foreground" : ""}>{symbol}</span>
            {transitioning ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {AVAILABLE_SYMBOLS.map((m) => (
              <DropdownMenuItem key={m} onClick={() => setSymbol(m)}>
                {m}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        {/* Timeframe selector */}
        <DropdownMenu>
          <DropdownMenuTrigger render={<Button variant="ghost" size="sm" className="hidden md:flex gap-1.5 font-medium" />}>
            <span className={transitioning ? "text-muted-foreground" : ""}>{timeframe}</span>
            {transitioning ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {AVAILABLE_TIMEFRAMES.map((t) => (
              <DropdownMenuItem key={t} onClick={() => setTimeframe(t)}>
                {t}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <Separator orientation="vertical" className="hidden sm:block h-6" />

        {/* Actions */}
        <div className="flex items-center gap-0.5">
          <Button variant="ghost" size="icon" className="h-9 w-9 relative">
            <Bell className="h-4 w-4" />
            <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-destructive" />
          </Button>

          {mounted && (
            <Button
              variant="ghost"
              size="icon"
              className="h-9 w-9"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
          )}

          <Button variant="ghost" size="icon" className="h-9 w-9" title="Settings"
            onClick={() => {
              const el = document.getElementById("section-settings");
              if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          >
            <Settings className="h-4 w-4" />
          </Button>

          {/* Profile */}
          <div className="ml-1 flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-info to-success text-xs font-bold text-white">
            P
          </div>
        </div>
      </div>
    </header>
  );
}
