"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { WS_URL } from "@/lib/constants";
import type { Reading } from "@/types";
import { useAsset } from "@/hooks/use-asset-context";
import { qk } from "@/hooks/use-api";

/**
 * WebSocket hook for real-time reading streaming.
 *
 * Re-opens a fresh socket whenever the selected (symbol, timeframe) changes
 * or the previous socket drops. Backoff is implemented via a `retry` counter
 * that bumps through a re-render, avoiding setTimeout recursion entirely
 * (the React Compiler flags that pattern).
 *
 * Each received reading is also written into the React Query cache under the
 * `reading` key so any consumer of `useReading()` benefits from the live push
 * — not just this hook.
 */
export function useLiveReading() {
  const { symbol, timeframe } = useAsset();
  const queryClient = useQueryClient();
  const [reading, setReading] = useState<Reading | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastReceivedAt, setLastReceivedAt] = useState<number | null>(null);
  const [retry, setRetry] = useState(0);

  // Latest pair captured in refs so the onmessage closure stays fresh without
  // forcing a socket re-open on every render.
  const symbolRef = useRef(symbol);
  const timeframeRef = useRef(timeframe);
  useEffect(() => {
    symbolRef.current = symbol;
    timeframeRef.current = timeframe;
  }, [symbol, timeframe]);

  useEffect(() => {
    let closedByUnmount = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let ws: WebSocket | null = null;

    try {
      ws = new WebSocket(WS_URL);
    } catch {
      const delay = Math.min(1000 * 2 ** retry, 30_000);
      reconnectTimer = setTimeout(() => setRetry((r) => r + 1), delay);
      return () => { if (reconnectTimer) clearTimeout(reconnectTimer); };
    }

    ws.onopen = () => {
      setConnected(true);
      setRetry(0);
      try {
        ws?.send(JSON.stringify({ type: "subscribe", symbol, timeframe }));
      } catch {
        /* ignore */
      }
    };

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "reading" && msg.data) {
          const data = msg.data as Reading;
          setReading(data);
          const now = Date.now();
          setLastReceivedAt(now);
          // Mirror the push into the React Query cache so any component that
          // uses useReading() also updates in real time.
          queryClient.setQueryData(qk.reading(symbolRef.current, timeframeRef.current), data);
        }
      } catch {
        /* ignore malformed messages */
      }
    };

    ws.onclose = () => {
      setConnected(false);
      if (closedByUnmount) return;
      const delay = Math.min(1000 * 2 ** retry, 30_000);
      reconnectTimer = setTimeout(() => setRetry((r) => r + 1), delay);
    };

    ws.onerror = () => {
      try { ws?.close(); } catch { /* ignore */ }
    };

    return () => {
      closedByUnmount = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      try { ws?.close(); } catch { /* ignore */ }
    };
  }, [symbol, timeframe, retry, queryClient]);

  return { reading, connected, lastReceivedAt };
}
