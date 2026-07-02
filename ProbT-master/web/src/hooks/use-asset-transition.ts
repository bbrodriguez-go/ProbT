"use client";

import { useEffect, useRef } from "react";
import { useIsFetching } from "@tanstack/react-query";
import { useAsset } from "@/hooks/use-asset-context";

/** Grace window for queries to kick off after the user changes the pair. */
const KICKOFF_GRACE_MS = 180;
/** Minimum visibility once real fetching has been observed (avoids flicker). */
const MIN_DISPLAY_MS = 350;
/** Hard ceiling so a hung request can't trap the UI. */
const MAX_DISPLAY_MS = 8000;

/**
 * Returns `true` while pair-dependent data is reloading after the user
 * changes the symbol/timeframe. Drives the elegant transition overlay.
 *
 * Flow:
 *  1. Asset context flips `transitioning` on user change.
 *  2. We watch in-flight React Query calls scoped to the active pair.
 *  3. Once they settle (and a minimum elegance window passed) we clear it.
 *  4. If nothing ever enters flight (cached/instant) we clear after a
 *     short kickoff grace so the UX stays snappy.
 */
export function useAssetTransitioning(): boolean {
  const { symbol, timeframe, transitioning, clearTransition } = useAsset();
  const seenFetchingRef = useRef(false);
  const startedAtRef = useRef(0);

  // Count in-flight queries for the active pair. Pair-dependent keys have
  // the shape [type, symbol, timeframe, ...].
  const fetching = useIsFetching({
    predicate: (query) => {
      const key = query.queryKey;
      if (!Array.isArray(key) || key.length < 3) return false;
      return key[1] === symbol && key[2] === timeframe;
    },
  });

  useEffect(() => {
    if (!transitioning) {
      seenFetchingRef.current = false;
      startedAtRef.current = 0;
      return;
    }

    if (startedAtRef.current === 0) startedAtRef.current = Date.now();
    const elapsed = Date.now() - startedAtRef.current;

    // Real loading in progress — note it and keep the overlay up.
    if (fetching > 0) {
      seenFetchingRef.current = true;
      return;
    }

    // No in-flight work. Decide whether to keep waiting or clear.
    const minDisplay = seenFetchingRef.current ? MIN_DISPLAY_MS : KICKOFF_GRACE_MS;
    if (elapsed < minDisplay) return; // still within the grace / min window

    // Defer clear so we don't setState synchronously inside the effect.
    const id = window.setTimeout(clearTransition, 0);
    return () => window.clearTimeout(id);
  }, [transitioning, fetching, clearTransition, symbol, timeframe]);

  // Safety: never let a transition hang longer than MAX_DISPLAY_MS.
  useEffect(() => {
    if (!transitioning) return;
    const id = window.setTimeout(clearTransition, MAX_DISPLAY_MS);
    return () => window.clearTimeout(id);
  }, [transitioning, clearTransition]);

  return transitioning;
}
