"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { NAV_SECTIONS } from "@/lib/constants";
import { DynamicIcon } from "@/components/lib/icons";
import { cn } from "@/lib/utils";

export function Sidebar({
  collapsed,
  onToggle,
  active,
  onSelect,
}: {
  collapsed: boolean;
  onToggle: () => void;
  active: string;
  onSelect: (id: string) => void;
}) {
  return (
    <motion.aside
      animate={{ width: collapsed ? 64 : 240 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
      className="sticky top-14 hidden md:flex h-[calc(100vh-3.5rem)] flex-col border-r border-border bg-card/50"
    >
      {/* Collapse toggle */}
      <div className="flex h-12 items-center justify-end px-3">
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onToggle}>
          {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 space-y-6 overflow-y-auto px-2 pb-4">
        {NAV_SECTIONS.map((section) => (
          <div key={section.title} className="space-y-1">
            <AnimatePresence>
              {!collapsed && (
                <motion.p
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
                >
                  {section.title}
                </motion.p>
              )}
            </AnimatePresence>
            {section.items.map((item) => {
              const isActive = active === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => onSelect(item.id)}
                  title={collapsed ? item.label : undefined}
                  className={cn(
                    "group relative flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm font-medium transition-all",
                    isActive
                      ? "bg-success/10 text-success"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground",
                    collapsed && "justify-center",
                  )}
                >
                  {isActive && (
                    <motion.div
                      layoutId="sidebar-active"
                      className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-success"
                    />
                  )}
                  <DynamicIcon name={item.icon} size={18} className="shrink-0" />
                  {!collapsed && <span className="flex-1 text-left">{item.label}</span>}
                  {!collapsed && item.badge && (
                    <Badge variant="secondary" className="h-5 text-[10px] bg-success/15 text-success">
                      {item.badge}
                    </Badge>
                  )}
                </button>
              );
            })}
          </div>
        ))}
      </nav>

      {/* Footer: logo only — this used to show a hardcoded "+68.0 R Total
          Profit", a fake number the §9.4 honesty rules exist to forbid */}
      <div className="border-t border-border p-3">
        <div className={cn("flex items-center gap-2", collapsed && "justify-center")}>
          <div className="flex h-8 w-8 shrink-0 items-center justify-center overflow-hidden rounded-lg ring-1 ring-border bg-card">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/probt_logo.png" alt="probt" className="h-full w-full object-cover" />
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold">probt v2</div>
              <div className="text-[10px] text-muted-foreground">zone-conditional engine</div>
            </div>
          )}
        </div>
      </div>
    </motion.aside>
  );
}
