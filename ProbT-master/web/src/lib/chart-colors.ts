// ─── Shared chart color helpers ───────────────────────────────────
import type { Accent } from "@/types";

export function accentColor(accent: Accent): string {
  const isDark =
    typeof document !== "undefined" && document.documentElement.classList.contains("dark");
  const map: Record<Accent, [string, string]> = {
    green: ["#16a34a", "#22c55e"],
    blue: ["#2563eb", "#3b82f6"],
    red: ["#dc2626", "#ef4444"],
    amber: ["#d97706", "#f59e0b"],
    gray: ["#6b7280", "#9ca3af"],
  };
  const [light, dark] = map[accent];
  return isDark ? dark : light;
}

export const chartColors = {
  primary: accentColor("green"),
  secondary: accentColor("blue"),
  warning: accentColor("amber"),
  danger: accentColor("red"),
  purple: "#7c3aed",
  muted: "#6b7280",
};

export const chartGrid = {
  top: 20,
  bottom: 40,
  left: 50,
  right: 20,
  containLabel: false,
};

export const chartTooltip = {
  trigger: "axis",
  backgroundColor: "rgba(255,255,255,0.96)",
  borderColor: "#e5e7eb",
  borderWidth: 1,
  textStyle: { color: "#111827", fontSize: 12 },
  extraCssText: "box-shadow: 0 4px 24px rgba(0,0,0,0.08); border-radius: 12px; backdrop-filter: blur(8px);",
};

export const darkChartTooltip = {
  ...chartTooltip,
  backgroundColor: "rgba(17,17,17,0.96)",
  borderColor: "#1f1f1f",
  textStyle: { color: "#fafafa", fontSize: 12 },
};
