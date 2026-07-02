"use client";

import { motion } from "framer-motion";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export function SectionCard({
  title,
  description,
  icon,
  action,
  children,
  className,
  bodyClassName,
  delay = 0,
}: {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  delay?: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: "spring", stiffness: 200, damping: 24 }}
      className={cn("w-full", className)}
    >
      <Card className={cn("overflow-hidden", className)}>
        {(title || action) && (
          <div className="flex items-center justify-between gap-2 border-b border-border px-4 py-3">
            <div className="flex items-center gap-2 min-w-0">
              {icon}
              {title && (
                <div className="min-w-0">
                  <h3 className="text-sm font-semibold truncate">{title}</h3>
                  {description && (
                    <p className="text-[11px] text-muted-foreground truncate">{description}</p>
                  )}
                </div>
              )}
            </div>
            {action}
          </div>
        )}
        <div className={cn("p-4", bodyClassName)}>{children}</div>
      </Card>
    </motion.div>
  );
}
