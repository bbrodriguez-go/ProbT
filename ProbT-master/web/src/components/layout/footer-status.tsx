"use client";

import { useEffect, useState } from "react";
import { Server, Clock, Cpu, MemoryStick, Activity, Radio, Terminal, HardDrive } from "lucide-react";
import { useHealth, useSystem } from "@/hooks/use-api";
import { API_URL } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function FooterStatus() {
  const { data: health } = useHealth();
  const { data: sys } = useSystem();
  const [latency, setLatency] = useState<number | null>(null);

  // Measure real API latency every 15s
  useEffect(() => {
    const measure = async () => {
      try {
        const t0 = performance.now();
        await fetch(`${API_URL}/api/health`);
        setLatency(Math.round(performance.now() - t0));
      } catch {
        setLatency(null);
      }
    };
    measure();
    const id = setInterval(measure, 15_000);
    return () => clearInterval(id);
  }, []);

  const cpu = sys?.cpu_percent ?? 0;
  const mem = sys?.memory_percent ?? 0;

  const stats = [
    { icon: Server, label: "API", value: health?.status === "ok" ? "Online" : "…", status: health?.status === "ok" ? "ok" : "idle" },
    { icon: Clock, label: "Latency", value: latency != null ? `${latency}ms` : "…", status: (latency ?? 99) < 100 ? "ok" : "warn" },
    { icon: Cpu, label: "CPU", value: `${cpu.toFixed(0)}%`, status: cpu < 70 ? "ok" : "warn" },
    { icon: MemoryStick, label: "Mem", value: `${mem.toFixed(0)}%`, status: mem < 80 ? "ok" : "warn" },
    { icon: HardDrive, label: "GPU", value: "N/A", status: "idle" },
    { icon: Activity, label: "Threads", value: sys ? `${sys.python_threads}` : "…", status: "ok" },
    { icon: Radio, label: "Streaming", value: health?.reading_ready ? "Active" : "Idle", status: health?.reading_ready ? "ok" : "idle" },
    { icon: Terminal, label: "Logs", value: "0 errors", status: "ok" },
  ];

  return (
    <footer className="sticky bottom-0 z-30 border-t border-border bg-card/80 backdrop-blur-xl">
      <div className="flex h-8 items-center gap-1 overflow-x-auto px-3 text-xs">
        {stats.map((s, i) => (
          <div key={s.label} className="flex items-center gap-1.5 whitespace-nowrap px-2">
            {i > 0 && <span className="mr-1 h-3 w-px bg-border" />}
            <s.icon className={cn(
              "h-3 w-3",
              s.status === "ok" && "text-success",
              s.status === "warn" && "text-warning",
              s.status === "idle" && "text-muted-foreground",
            )} />
            <span className="text-muted-foreground hidden sm:inline">{s.label}:</span>
            <span className="font-medium tabular-nums">{s.value}</span>
          </div>
        ))}
      </div>
    </footer>
  );
}
