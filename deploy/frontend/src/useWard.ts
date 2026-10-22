import { useEffect, useRef, useState } from "react";
import { api, type BedSnap, type StreamMsg, type WaveChunk } from "./api";

/** Last BUFFER_S seconds of 2 s vitals per bed, for the tile sparklines and the live part of the detail chart. */
export const BUFFER_S = 1800;

export interface Buffer { loop: number; t: number[]; map: (number | null)[]; hr: (number | null)[]; spo2: (number | null)[] }

export interface WardState {
  v: number | null;                 // virtual time of the last message
  beds: Map<number, BedSnap>;
  buffers: Map<number, Buffer>;
  connected: boolean;
  tick: number;                     // increments on every message (render trigger)
}

function append(buf: Buffer | undefined, chunk: WaveChunk, loop: number): Buffer {
  const b: Buffer = !buf || buf.loop !== loop || (chunk.t.length && buf.t.length && chunk.t[0] < buf.t[buf.t.length - 1] - 1)
    ? { loop, t: [], map: [], hr: [], spo2: [] } : buf;
  for (let i = 0; i < chunk.t.length; i++) {
    if (b.t.length && chunk.t[i] <= b.t[b.t.length - 1]) continue;
    b.t.push(chunk.t[i]); b.map.push(chunk.map[i]); b.hr.push(chunk.hr[i]); b.spo2.push(chunk.spo2[i]);
  }
  const cut = b.t.findIndex((t) => t >= b.t[b.t.length - 1] - BUFFER_S);
  if (cut > 0) { b.t.splice(0, cut); b.map.splice(0, cut); b.hr.splice(0, cut); b.spo2.splice(0, cut); }
  return b;
}

/** One Server-Sent Events stream of the ward (backend app/feed.py). Changing speed reconnects from the current
 *  virtual time, so the beds continue where they were. */
export function useWard(speed: number, enabled: boolean): WardState {
  const [state, setState] = useState<WardState>({ v: null, beds: new Map(), buffers: new Map(), connected: false, tick: 0 });
  const vRef = useRef<number | null>(null);
  const bufRef = useRef(new Map<number, Buffer>());

  useEffect(() => {
    if (!enabled) return;
    const es = new EventSource(api.streamUrl(speed, vRef.current ?? undefined));
    es.onopen = () => setState((s) => ({ ...s, connected: true }));
    es.onerror = () => setState((s) => ({ ...s, connected: false }));
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data) as StreamMsg;
      vRef.current = msg.v;
      const beds = new Map<number, BedSnap>();
      for (const b of msg.beds) {
        beds.set(b.bed, b);
        const chunk = msg.wave[String(b.bed)];
        if (chunk) bufRef.current.set(b.bed, append(bufRef.current.get(b.bed), chunk, b.loop));
      }
      setState((s) => ({ v: msg.v, beds, buffers: bufRef.current, connected: true, tick: s.tick + 1 }));
    };
    return () => es.close();
  }, [speed, enabled]);
  return state;
}
