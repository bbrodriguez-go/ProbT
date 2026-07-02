"use client";

import { AnimatePresence, motion } from "framer-motion";

/**
 * Elegant shimmer overlay shown while the active symbol/timeframe reloads.
 *
 * Layers (top → bottom):
 *  1. Thin animated progress bar pinned to the top edge.
 *  2. Soft veil that dims + blurs the live content beneath it.
 *  3. Diagonal shimmer sweep that gives the premium "loading" feel.
 *
 * Pointer events are disabled so the UI beneath stays interactive.
 */
export function TransitionOverlay({ active }: { active: boolean }) {
  return (
    <AnimatePresence>
      {active && (
        <motion.div
          key="transition-overlay"
          className="pointer-events-none absolute inset-0 z-30 overflow-hidden rounded-inherit"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18, ease: "easeOut" }}
          aria-hidden
        >
          {/* Top progress bar */}
          <div className="absolute left-0 right-0 top-0 h-0.5 transition-bar" />

          {/* Branded loading mark */}
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
            <motion.img
              src="/probt_logo.png"
              alt=""
              aria-hidden
              className="h-16 w-16 rounded-2xl ring-1 ring-border bg-card object-cover"
              initial={{ scale: 0.85, opacity: 0, y: 6 }}
              animate={{ scale: 1, opacity: 0.9, y: 0 }}
              exit={{ scale: 0.85, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeOut" }}
            />
          </div>

          {/* Dim + blur veil over live content */}
          <div className="absolute inset-0 bg-card/25 backdrop-blur-[3px] transition-sweep" />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
