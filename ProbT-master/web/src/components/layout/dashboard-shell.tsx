"use client";

import { useState } from "react";
import { Header } from "./header";
import { Sidebar } from "./sidebar";
import { FooterStatus } from "./footer-status";
import { CommandPalette } from "./command-palette";
import { TransitionOverlay } from "./transition-overlay";
import { useAssetTransitioning } from "@/hooks/use-asset-transition";

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const [active, setActive] = useState("dashboard");
  const [cmdOpen, setCmdOpen] = useState(false);
  const transitioning = useAssetTransitioning();

  const handleSelect = (id: string) => {
    setActive(id);
    const el = document.getElementById(`section-${id}`);
    if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Header onCommand={() => setCmdOpen(true)} />
      <div className="flex flex-1">
        <Sidebar
          collapsed={collapsed}
          onToggle={() => setCollapsed(!collapsed)}
          active={active}
          onSelect={handleSelect}
        />
        <main className="relative flex-1 overflow-x-hidden">
          <div className="mx-auto max-w-[1600px] space-y-4 p-3 sm:space-y-5 sm:p-4 lg:p-6">
            {children}
          </div>
          <TransitionOverlay active={transitioning} />
        </main>
      </div>
      <FooterStatus />
      <CommandPalette open={cmdOpen} onOpenChange={setCmdOpen} onSelect={handleSelect} />
    </div>
  );
}
