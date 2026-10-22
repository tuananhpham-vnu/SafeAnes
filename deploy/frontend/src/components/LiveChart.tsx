import { useMemo, useRef, useState } from "react";
import { MODEL_COLOR, clock, fmt, pct, type Alarm, type EventItem, type Horizon, type ModelInfo, type ModelName } from "../api";
import { useWidth } from "./useWidth";

interface Props {
  start: number; now: number; surgeryStart: number;
  time: number[]; risk: Record<ModelName, (number | null)[]>; models: ModelInfo[]; model: ModelName; horizon: Horizon;
  alarms: Alarm[]; events: EventItem[];
  wave: { t: number[]; map: (number | null)[] };
}

const M = { l: 44, r: 16 };
const RISK_H = 180, MAP_H = 140, GAP = 28, TOP = 18, BOTTOM = 26;

function path(xs: number[], ys: (number | null)[], x: (v: number) => number, y: (v: number) => number, step = 1) {
  let d = "", pen = false;
  for (let i = 0; i < xs.length; i += step) {
    const v = ys[i];
    if (v == null) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

/** The case so far: risk of the four models at one horizon (selected model bold, with its threshold and alarms)
 *  and MAP every 2 s. Nothing after `now` exists here: the right part of the axis is the time still to come. */
export function LiveChart({ start, now, surgeryStart, time, risk, models, model, horizon, alarms, events, wave }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const W = Math.max(width, 320);
  const span = Math.max(now - start, 1800) * 1.08;
  const t1 = start + span;
  const x = (t: number) => M.l + ((t - start) / span) * (W - M.l - M.r);
  const h = String(horizon);
  const thr = models.find((m) => m.name === model)?.horizons[h].threshold ?? 0.5;
  const riskTop = TOP, mapTop = TOP + RISK_H + GAP;
  const yRisk = (p: number) => riskTop + RISK_H - p * RISK_H;
  const yMap = (v: number) => mapTop + MAP_H - ((Math.min(Math.max(v, 30), 140) - 30) / 110) * MAP_H;
  const H = mapTop + MAP_H + BOTTOM;
  const ticks = useMemo(() => {
    const m = span / 60, stp = m > 300 ? 60 : m > 120 ? 30 : 15, out: number[] = [];
    for (let k = 0; k <= m; k += stp) out.push(start + k * 60);
    return out;
  }, [start, span]);
  const step = Math.max(1, Math.floor(wave.t.length / (W * 1.5)));   // thin the 2 s MAP to ~1.5 points per pixel
  const onMove = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    const t = start + ((e.clientX - r.left - M.l) / (W - M.l - M.r)) * span;
    if (t < start || t > now || !time.length) return setHover(null);
    let i = time.findIndex((x) => x > t) - 1;
    if (i < -1) i = time.length - 1;
    setHover(Math.max(i, 0));
  };
  const ht = hover != null ? time[hover] : null;
  const order = models.map((m) => m.name).filter((n) => n !== model).concat(model);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg ref={svgRef} className="chart" width={W} height={H} role="img" aria-label="Nguy cơ và MAP từ đầu ca tới hiện tại"
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        {surgeryStart > start && (
          <g>
            <rect x={x(start)} y={riskTop} width={x(Math.min(surgeryStart, now)) - x(start)} height={RISK_H} fill="var(--phase-wash)" />
            <rect x={x(start)} y={mapTop} width={x(Math.min(surgeryStart, now)) - x(start)} height={MAP_H} fill="var(--phase-wash)" />
            <text x={x(start) + 6} y={riskTop + 13}>Trước rạch da</text>
            {surgeryStart <= now && <>
              <line x1={x(surgeryStart)} x2={x(surgeryStart)} y1={riskTop - 6} y2={mapTop + MAP_H} stroke="var(--axis)" />
              <text x={x(surgeryStart) + 4} y={riskTop - 6} style={{ fill: "var(--ink-2)" }}>Rạch da</text></>}
          </g>
        )}
        <rect x={x(now)} y={riskTop} width={Math.max(x(t1) - x(now), 0)} height={mapTop + MAP_H - riskTop} fill="var(--future-wash)" />
        {events.map((e) => (
          <g key={e.onset_s}>
            <rect x={x(e.onset_s)} y={riskTop} width={Math.max(x(e.end_s) - x(e.onset_s), 2)} height={RISK_H} fill="var(--critical-wash)" />
            <rect x={x(e.onset_s)} y={mapTop} width={Math.max(x(e.end_s) - x(e.onset_s), 2)} height={MAP_H} fill="var(--critical-wash)" />
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((p) => (
          <g key={p}>
            <line className={p === 0 ? "axis-line" : "grid-line"} x1={M.l} x2={W - M.r} y1={yRisk(p)} y2={yRisk(p)} />
            <text x={M.l - 8} y={yRisk(p) + 4} textAnchor="end" className="num">{p * 100}%</text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={yRisk(thr)} y2={yRisk(thr)} stroke="var(--critical)" strokeWidth={1} strokeDasharray="4 3" />
        <text x={W - M.r} y={yRisk(thr) - 5} textAnchor="end" style={{ fill: "var(--ink-2)" }}>Ngưỡng {pct(thr)}%</text>
        {order.map((n) => (
          <path key={n} d={path(time, risk[n] ?? [], x, yRisk)} fill="none" stroke={MODEL_COLOR[n]}
            strokeWidth={n === model ? 2.4 : 1.2} opacity={n === model ? 1 : 0.55} strokeLinejoin="round" />
        ))}
        {alarms.map((a) => {
          const i = time.indexOf(a.time_s), p = i >= 0 ? risk[model]?.[i] : null;
          if (p == null) return null;
          return <circle key={a.time_s} cx={x(a.time_s)} cy={yRisk(p)} r={5} strokeWidth={2} stroke="var(--critical)"
            fill={a.kind === "true" ? "var(--critical)" : "var(--surface)"} />;
        })}
        {[40, 65, 100, 140].map((v) => (
          <g key={v}>
            <line className={v === 40 ? "axis-line" : "grid-line"} x1={M.l} x2={W - M.r} y1={yMap(v)} y2={yMap(v)}
              style={v === 65 ? { stroke: "var(--critical)" } : undefined} />
            <text x={M.l - 8} y={yMap(v) + 4} textAnchor="end" className="num">{v}</text>
          </g>
        ))}
        <text x={M.l} y={mapTop - 8} style={{ fill: "var(--ink-2)" }}>MAP (mmHg, mỗi 2 giây)</text>
        <path d={path(wave.t, wave.map, x, yMap, step)} fill="none" stroke="var(--ink-2)" strokeWidth={1.2} strokeLinejoin="round" />
        {ticks.map((t) => <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="num">{Math.round((t - start) / 60)}′</text>)}
        <line x1={x(now)} x2={x(now)} y1={riskTop} y2={mapTop + MAP_H} stroke="var(--ink)" strokeWidth={1.5} />
        <text x={x(now)} y={riskTop - 6} textAnchor="middle" style={{ fill: "var(--ink)", fontWeight: 600 }}>bây giờ</text>
        {ht != null && <line x1={x(ht)} x2={x(ht)} y1={riskTop} y2={mapTop + MAP_H} stroke="var(--muted)" />}
      </svg>
      {ht != null && hover != null && (
        <div className="tooltip" style={{ left: Math.min(x(ht) + 12, W - 200), top: 24 }}>
          <div className="t num">{clock(ht - start)} · {ht <= surgeryStart ? "trước rạch da" : "trong mổ"}</div>
          {models.map((m) => (
            <div key={m.name} className="row"><span><i className="key-line" style={{ background: MODEL_COLOR[m.name] }} />{m.title}</span>
              <b>{risk[m.name]?.[hover] == null ? "–" : `${pct(risk[m.name][hover], 1)}%`}</b></div>
          ))}
          <div className="row"><span>Mốc</span><b>{horizon / 60} phút</b></div>
          <div className="row"><span>MAP</span><b>{fmt(nearest(wave, ht))} mmHg</b></div>
        </div>
      )}
    </div>
  );
}

function nearest(w: { t: number[]; map: (number | null)[] }, t: number): number | null {
  let lo = 0, hi = w.t.length - 1;
  if (hi < 0) return null;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (w.t[mid] <= t) lo = mid; else hi = mid - 1; }
  return w.map[lo];
}
