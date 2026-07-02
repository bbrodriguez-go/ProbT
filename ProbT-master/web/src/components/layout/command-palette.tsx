"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "@/components/ui/command";
import { DynamicIcon } from "@/components/lib/icons";
import { NAV_SECTIONS } from "@/lib/constants";
import { Moon, Sun, RefreshCw, Download, ExternalLink } from "lucide-react";
import { useTheme } from "next-themes";

export function CommandPalette({
  open,
  onOpenChange,
  onSelect,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  onSelect: (id: string) => void;
}) {
  const router = useRouter();
  const { setTheme, theme } = useTheme();

  // Global Ctrl+K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        onOpenChange(!open);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onOpenChange]);

  const handleSelect = (id: string) => {
    onSelect(id);
    onOpenChange(false);
  };

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput placeholder="Search or jump to…" />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        {NAV_SECTIONS.map((section) => (
          <CommandGroup key={section.title} heading={section.title}>
            {section.items.map((item) => (
              <CommandItem
                key={item.id}
                value={`${item.label} ${item.icon}`}
                onSelect={() => handleSelect(item.id)}
              >
                <DynamicIcon name={item.icon} size={16} className="text-muted-foreground" />
                <span>{item.label}</span>
                {item.badge && (
                  <span className="ml-auto text-xs text-success">{item.badge}</span>
                )}
              </CommandItem>
            ))}
          </CommandGroup>
        ))}
        <CommandSeparator />
        <CommandGroup heading="Actions">
          <CommandItem onSelect={() => { setTheme(theme === "dark" ? "light" : "dark"); onOpenChange(false); }}>
            {theme === "dark" ? <Sun size={16} className="text-muted-foreground" /> : <Moon size={16} className="text-muted-foreground" />}
            <span>Toggle theme</span>
          </CommandItem>
          <CommandItem onSelect={() => { window.location.reload(); }}>
            <RefreshCw size={16} className="text-muted-foreground" />
            <span>Refresh data</span>
          </CommandItem>
          <CommandItem onSelect={() => window.open("http://127.0.0.1:8000/docs", "_blank")}>
            <ExternalLink size={16} className="text-muted-foreground" />
            <span>API documentation</span>
          </CommandItem>
        </CommandGroup>
      </CommandList>
    </CommandDialog>
  );
}
