export const HORIZONS = [300, 600, 900, 1200, 1800] as const;
export type Horizon = (typeof HORIZONS)[number];
export type ModelName = "lgbm_v8" | "catboost_v8" | "logistic_v8" | "tabm_v8";
export const MODEL_ORDER: ModelName[] = ["lgbm_v8", "catboost_v8", "logistic_v8", "tabm_v8"];
export const MODEL_COLOR: Record<ModelName, string> = {
  lgbm_v8: "var(--series-1)", catboost_v8: "var(--series-2)", logistic_v8: "var(--series-3)", tabm_v8: "var(--series-4)",
};

export interface EventItem { onset_s: number; end_s: number; min_map: number; pre_incision: boolean; ongoing?: boolean }
export interface BedInfo {
  bed: number; caseid: number; age: number | null; sex: string; asa: number | null; department: string; optype: string;
  emergency: boolean; start_s: number; end_s: number; surgery_start_s: number; duration_s: number; events: EventItem[];
}
export interface WardInfo { source: string; horizons: number[]; speeds: number[]; v_now: number; engine: string; beds: BedInfo[] }
export type Vitals = Record<"map" | "sbp" | "dbp" | "hr" | "spo2", number | null>;
export interface BedSnap {
  bed: number; t: number; loop: number; elapsed_s: number; phase: "pre_incision" | "surgery"; eligible: boolean;
  row_time: number | null; vitals: Vitals;
  risk: Record<ModelName, number[] | null>; alarm: Record<ModelName, boolean[]>; n_alarms: Record<ModelName, number[]>;
  in_event: boolean; n_events: number;
}
export interface WaveChunk { t: number[]; map: (number | null)[]; hr: (number | null)[]; spo2: (number | null)[];
  sbp?: (number | null)[]; dbp?: (number | null)[] }
export interface StreamMsg { v: number; speed: number; beds: BedSnap[]; wave: Record<string, WaveChunk> }

export interface Alarm { time_s: number; kind: "true" | "false" | "censored" | "pending"; lead_s: number | null }
export interface History {
  bed: number; info: BedInfo; t: number; loop: number; start_s: number; end_s: number; surgery_start_s: number;
  time_s: number[]; eligible: boolean[];
  risk: Record<ModelName, Record<string, (number | null)[]>>;
  alarms: Record<ModelName, Record<string, Alarm[]>>;
  thresholds: Record<ModelName, Record<string, number>>;
  events: EventItem[]; wave: WaveChunk; vitals: Vitals; context: Record<string, number | null>;
}
export interface Explanation {
  available: boolean; reason?: string; time_s: number; eligible: boolean; model: ModelName; proxy_for: ModelName | null;
  horizon_s: number; probability: number; base_logit: number;
  groups: { id: string; label: string; contribution: number }[];
  features: { column: string; label: string; unit: string; group: string; contribution: number; value: number | null }[];
}
export interface Metrics {
  auroc: number | null; auroc_lo?: number | null; auroc_hi?: number | null; auprc: number | null;
  event_sensitivity: number | null; false_alarms_per_hour: number | null; alarm_ppv: number | null;
  lead_median_s: number | null; events_eligible: number | null; events_detected: number | null; n_cases?: number | null;
}
export interface ModelInfo {
  name: ModelName; title: string; n_inputs: number; explains: boolean;
  horizons: Record<string, { threshold: number; validation: Metrics; test: Metrics | null }>;
}

async function check(r: Response) {
  if (r.ok) return r;
  let msg = `${r.status}`;
  try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); }
  catch { msg = `${r.status} ${r.statusText}`; }
  throw new Error(msg);
}
async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  return (await check(await fetch(path, { signal }))).json() as Promise<T>;
}

export const api = {
  ward: () => get<WardInfo>("/api/ward"),
  models: () => get<ModelInfo[]>("/api/models"),
  history: (bed: number, v: number, waveS: number, signal?: AbortSignal) =>
    get<History>(`/api/beds/${bed}/history?v=${v}&wave_s=${waveS}`, signal),
  explain: (bed: number, v: number, model: ModelName, h: Horizon, signal?: AbortSignal) =>
    get<Explanation>(`/api/beds/${bed}/explain?v=${v}&model=${model}&horizon=${h}`, signal),
  streamUrl: (speed: number, v?: number) => `/api/ward/stream?speed=${speed}${v == null ? "" : `&v=${v}`}`,
};

// ---------- formatting
export const pct = (p: number | null | undefined, d = 0) =>
  p == null || !isFinite(p) ? "–" : (p * 100).toLocaleString("vi-VN", { maximumFractionDigits: d, minimumFractionDigits: d });
export const fmt = (x: number | null | undefined, d = 0) =>
  x == null || !isFinite(x) ? "–" : x.toLocaleString("vi-VN", { maximumFractionDigits: d, minimumFractionDigits: d });
export const clock = (s: number) => {
  const m = Math.floor(s / 60), h = Math.floor(m / 60);
  return `${String(h).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
};
export const horizonLabel = (h: number) => `${h / 60} phút`;
export const sexLabel = (s: string) => (s === "M" ? "Nam" : s === "F" ? "Nữ" : s);

// ---------- risk status (status colours are reserved; always shown with an icon + label)
export type Level = "na" | "low" | "watch" | "high" | "alarm";
export function level(p: number | null | undefined, thr: number, alarmOn: boolean): Level {
  if (p == null || !isFinite(p)) return "na";
  if (alarmOn) return "alarm";
  if (p >= thr) return "high";
  if (p >= thr * 0.5) return "watch";
  return "low";
}
export const LEVELS: Record<Level, { label: string; color: string; icon: string }> = {
  na: { label: "Không dự báo", color: "var(--muted)", icon: "–" },
  low: { label: "Nguy cơ thấp", color: "var(--good)", icon: "✓" },
  watch: { label: "Cần theo dõi", color: "var(--warning)", icon: "!" },
  high: { label: "Nguy cơ cao", color: "var(--serious)", icon: "▲" },
  alarm: { label: "Cảnh báo", color: "var(--critical)", icon: "⚠" },
};
export const LEVEL_RANK: Record<Level, number> = { alarm: 4, high: 3, watch: 2, low: 1, na: 0 };
