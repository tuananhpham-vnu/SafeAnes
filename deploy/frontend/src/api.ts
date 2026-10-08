export const HORIZONS = [300, 600, 900, 1200, 1800] as const;
export type Horizon = (typeof HORIZONS)[number];
export type ModelName = "lgbm_context" | "lgbm_numeric";

export interface CaseItem {
  caseid: number;
  duration_min: number;
  incision_min: number;
  n_events: number;
  n_events_pre_incision: number;
}
export interface EventItem { onset_s: number; end_s: number; min_map: number; pre_incision: boolean }
export interface Alarm { time_s: number; kind: "true" | "false" | "censored"; lead_s: number | null }
export interface Timeline {
  caseid: number;
  model: ModelName;
  time_s: number[];
  eligible: boolean[];
  surgery_start_s: number;
  events: EventItem[];
  vitals: Record<"map" | "sbp" | "dbp" | "hr" | "spo2" | "etco2", (number | null)[]>;
  risk: Record<string, (number | null)[]>;
  alarms: Record<string, Alarm[]>;
  thresholds: Record<string, number>;
}
export interface Explanation {
  caseid: number;
  time_s: number;
  horizon_s: number;
  probability: number;
  eligible: boolean;
  phase: "pre_incision" | "surgery";
  groups: { id: string; label: string; kind: "monitor" | "context"; contribution: number }[];
  features: { column: string; label: string; group: string; contribution: number; value: number | null; unit: string }[];
  context: Record<string, number | null>;
  data_quality: Record<string, number | null>;
}
export interface Metrics {
  auroc: number; auroc_lo: number; auroc_hi: number; auprc: number; event_sensitivity: number;
  event_sensitivity_lo: number; event_sensitivity_hi: number; false_alarms_per_hour: number; alarm_ppv: number;
  lead_median_s: number; events_eligible: number; events_detected: number; prevalence: number; ece: number;
}
export interface ModelInfo {
  name: ModelName; title: string; n_features: number;
  horizons: Record<string, { threshold: number; metrics: Metrics }>;
}
export type ReportRow = Record<string, string | number | null>;
export interface Quality {
  columns: Record<string, string>;
  coverage: Record<string, number>;
  samples: Record<string, number>;
  artefacts_removed: Record<string, number>;
  baseline_map: number | null;
  baseline_source: string;
  duration_min: number;
  warnings: string[];
}
export interface InferResult extends Timeline {
  session: string;
  phase_known: boolean;
  anestart_s: number | null;
  quality: Quality;
}
export interface InferIn { csv: string; anestart?: string; opstart?: string; baseline_map?: number | null; model: ModelName }
export interface Example { caseid: number; anestart_s: number; opstart_s: number; rows: number }

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
  cases: () => get<CaseItem[]>("/api/cases"),
  models: () => get<ModelInfo[]>("/api/models"),
  report: () => get<Record<string, ReportRow[]>>("/api/report"),
  timeline: (id: number, model: ModelName) => get<Timeline>(`/api/cases/${id}/timeline?model=${model}`),
  explain: (id: number, model: ModelName, t: number, h: Horizon, signal?: AbortSignal) =>
    get<Explanation>(`/api/cases/${id}/explain?model=${model}&t=${t}&horizon=${h}`, signal),
  examples: () => get<Example[]>("/api/examples"),
  exampleCsv: async (id: number) => (await check(await fetch(`/api/examples/${id}.csv`))).text(),
  infer: async (body: InferIn) => (await check(await fetch("/api/infer", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }))).json() as Promise<InferResult>,
  inferModel: (sid: string, model: ModelName) => get<InferResult>(`/api/infer/${sid}?model=${model}`),
  inferExplain: (sid: string, model: ModelName, t: number, h: Horizon, signal?: AbortSignal) =>
    get<Explanation>(`/api/infer/${sid}/explain?model=${model}&t=${t}&horizon=${h}`, signal),
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

/** Whether the outcome of an alarm is known at `cursor`: a true alarm once its event started, any other
 *  once its horizon has passed without one (or always, in retrospective view). */
export const alarmKnown = (a: Alarm, horizon: number, cursor: number, reveal: boolean) =>
  reveal || (a.kind === "true" ? a.time_s + (a.lead_s ?? 0) <= cursor : a.time_s + horizon <= cursor);

/** index of the last time point <= t */
export function indexAt(times: number[], t: number): number {
  let lo = 0, hi = times.length - 1;
  if (hi < 0 || t < times[0]) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

/** Whether an alarm is on at row `idx`: the last alarm at or before it, and every prediction since then at or
 *  above threshold (alarms re-arm only after the probability drops below threshold). */
export function alarmOnAt(tl: Timeline, h: string, idx: number): boolean {
  const t = tl.time_s[idx], thr = tl.thresholds[h];
  const last = [...tl.alarms[h]].reverse().find((a) => a.time_s <= t);
  if (!last) return false;
  for (let i = indexAt(tl.time_s, last.time_s); i <= idx; i++) {
    const v = tl.risk[h][i];
    if (v == null || v < thr) return false;
  }
  return true;
}
