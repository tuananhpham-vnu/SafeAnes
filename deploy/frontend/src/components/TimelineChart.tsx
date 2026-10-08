import { useMemo, useRef, useState } from "react";
import { alarmKnown, clock, fmt, indexAt, pct, type Alarm, type Horizon, type Timeline } from "../api";
import { useWidth } from "./useWidth";

interface Props {
  tl: Timeline;
  compare?: Timeline | null; // baseline model, drawn as a second series
  horizon: Horizon;
  cursor: number;
  reveal: boolean; // retrospective: show the whole case and the outcome of every alarm
  onSeek: (t: number) => void;
  marker?: number | null; // selected time, drawn in retrospective view too
  phaseKnown?: boolean;   // false: incision time not given, no pre-incision shading
}

const M = { l: 44, r: 16 };
const RISK_H = 190, MAP_H = 150, GAP = 28, TOP = 18, BOTTOM = 26;

function path(xs: number[], ys: (number | null)[], x: (v: number) => number, y: (v: number) => number, until: number) {
  let d = "", pen = false;
  for (let i = 0; i < xs.length && xs[i] <= until; i++) {
    const v = ys[i];
    if (v == null) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}

export function TimelineChart({ tl, compare, horizon, cursor, reveal, onSeek, marker, phaseKnown = true }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const t0 = tl.time_s[0], t1 = tl.time_s[tl.time_s.length - 1];
  const W = Math.max(width, 320);
  const x = (t: number) => M.l + ((t - t0) / Math.max(t1 - t0, 1)) * (W - M.l - M.r);
  const until = reveal ? t1 : cursor;
  const h = String(horizon);
  const thr = tl.thresholds[h];
  const riskTop = TOP, mapTop = TOP + RISK_H + GAP;
  const yRisk = (p: number) => riskTop + RISK_H - p * RISK_H;
  const MAP_MIN = 30, MAP_MAX = 140;
  const yMap = (v: number) => mapTop + MAP_H - ((Math.min(Math.max(v, MAP_MIN), MAP_MAX) - MAP_MIN) / (MAP_MAX - MAP_MIN)) * MAP_H;
  const H = mapTop + MAP_H + BOTTOM;

  const ticks = useMemo(() => {
    const span = (t1 - t0) / 60, step = span > 300 ? 60 : span > 120 ? 30 : 15;
    const out: number[] = [];
    for (let m = 0; m <= span; m += step) out.push(t0 + m * 60);
    return out;
  }, [t0, t1]);

  const events = tl.events.filter((e) => reveal || e.onset_s <= cursor);
  const alarms: Alarm[] = (tl.alarms[h] ?? []).filter((a) => a.time_s <= until);
  const riskAt = (t: number) => tl.risk[h][tl.time_s.indexOf(t)] ?? null;
  const surg = tl.surgery_start_s;

  const onMove = (e: React.PointerEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    const t = t0 + ((e.clientX - r.left - M.l) / (W - M.l - M.r)) * (t1 - t0);
    if (t < t0 || t > t1) return setHover(null);
    setHover(Math.min(indexAt(tl.time_s, t), indexAt(tl.time_s, until)));
  };
  const hi = hover;
  const ht = hi != null ? tl.time_s[hi] : null;

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg ref={svgRef} className="chart" width={W} height={H} role="img"
        aria-label={`Nguy cơ tụt huyết áp trong ${horizon / 60} phút và MAP theo thời gian, ca ${tl.caseid}`}
        onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        onClick={() => ht != null && onSeek(ht)} style={{ cursor: "crosshair", touchAction: "none" }}>
        {/* pre-incision phase */}
        {phaseKnown && surg > t0 && (
          <g>
            <rect x={x(t0)} y={riskTop} width={x(Math.min(surg, t1)) - x(t0)} height={RISK_H} fill="var(--phase-wash)" />
            <rect x={x(t0)} y={mapTop} width={x(Math.min(surg, t1)) - x(t0)} height={MAP_H} fill="var(--phase-wash)" />
            <text x={x(t0) + 6} y={riskTop + 13}>Trước rạch da</text>
            <line x1={x(surg)} x2={x(surg)} y1={riskTop - 6} y2={mapTop + MAP_H} stroke="var(--axis)" strokeWidth={1} />
            <text x={x(surg) + 4} y={riskTop - 6} style={{ fill: "var(--ink-2)" }}>Rạch da</text>
          </g>
        )}
        {/* hypotension events: MAP < 65 for >= 60 s */}
        {events.map((e) => (
          <g key={e.onset_s}>
            <rect x={x(e.onset_s)} y={riskTop} width={Math.max(x(e.end_s) - x(e.onset_s), 2)} height={RISK_H} fill="var(--critical-wash)" />
            <rect x={x(e.onset_s)} y={mapTop} width={Math.max(x(e.end_s) - x(e.onset_s), 2)} height={MAP_H} fill="var(--critical-wash)" />
          </g>
        ))}

        {/* ---- risk panel */}
        {[0, 0.25, 0.5, 0.75, 1].map((p) => (
          <g key={p}>
            <line className={p === 0 ? "axis-line" : "grid-line"} x1={M.l} x2={W - M.r} y1={yRisk(p)} y2={yRisk(p)} />
            <text x={M.l - 8} y={yRisk(p) + 4} textAnchor="end" className="num">{p * 100}%</text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={yRisk(thr)} y2={yRisk(thr)} stroke="var(--critical)" strokeWidth={1} />
        <text x={W - M.r} y={yRisk(thr) - 5} textAnchor="end" style={{ fill: "var(--ink-2)" }}>
          Ngưỡng cảnh báo {pct(thr)}%
        </text>
        {compare && (
          <path d={path(compare.time_s, compare.risk[h], x, yRisk, until)} fill="none" stroke="var(--series-2)"
            strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" opacity={0.9} />
        )}
        <path d={path(tl.time_s, tl.risk[h], x, yRisk, until)} fill="none" stroke="var(--series-1)" strokeWidth={2}
          strokeLinejoin="round" strokeLinecap="round" />
        {alarms.map((a) => {
          const p = riskAt(a.time_s);
          if (p == null) return null;
          const known = alarmKnown(a, horizon, cursor, reveal);
          const filled = known && a.kind === "true";
          return (
            <circle key={a.time_s} cx={x(a.time_s)} cy={yRisk(p)} r={5} stroke="var(--surface)" strokeWidth={2}
              fill={filled ? "var(--critical)" : "var(--surface)"} style={{ paintOrder: "stroke" }}>
              <title>{`Cảnh báo ${clock(a.time_s - t0)}${known ? (a.kind === "true" ? ` — đúng, báo trước ${fmt((a.lead_s ?? 0) / 60, 1)} phút` : a.kind === "false" ? " — sai" : " — chưa xác định") : ""}`}</title>
            </circle>
          );
        })}
        {alarms.map((a) => {
          const p = riskAt(a.time_s);
          const known = alarmKnown(a, horizon, cursor, reveal);
          return p == null || (known && a.kind === "true") ? null : (
            <circle key={`r${a.time_s}`} cx={x(a.time_s)} cy={yRisk(p)} r={4} fill="none" stroke="var(--critical)" strokeWidth={2} />
          );
        })}

        {/* ---- MAP panel */}
        {[40, 65, 100, 140].map((v) => (
          <g key={v}>
            <line className={v === 40 ? "axis-line" : "grid-line"} x1={M.l} x2={W - M.r} y1={yMap(v)} y2={yMap(v)}
              stroke={v === 65 ? "var(--critical)" : undefined} style={v === 65 ? { stroke: "var(--critical)" } : undefined} />
            <text x={M.l - 8} y={yMap(v) + 4} textAnchor="end" className="num">{v}</text>
          </g>
        ))}
        <text x={W - M.r} y={yMap(65) - 5} textAnchor="end" style={{ fill: "var(--ink-2)" }}>MAP 65 mmHg</text>
        <path d={path(tl.time_s, tl.vitals.map, x, yMap, until)} fill="none" stroke="var(--series-1)" strokeWidth={2}
          strokeLinejoin="round" strokeLinecap="round" />
        <text x={M.l} y={mapTop - 8} style={{ fill: "var(--ink-2)" }}>MAP (mmHg)</text>

        {/* x axis */}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - 6} textAnchor="middle" className="num">{Math.round((t - t0) / 60)}′</text>
        ))}

        {/* now cursor + hover crosshair */}
        {(!reveal || marker != null) && (
          <line x1={x(marker ?? cursor)} x2={x(marker ?? cursor)} y1={riskTop} y2={mapTop + MAP_H} stroke="var(--ink)" strokeWidth={1.5} />
        )}
        {ht != null && <line x1={x(ht)} x2={x(ht)} y1={riskTop} y2={mapTop + MAP_H} stroke="var(--muted)" strokeWidth={1} />}
        {ht != null && hi != null && tl.risk[h][hi] != null && (
          <circle cx={x(ht)} cy={yRisk(tl.risk[h][hi]!)} r={4} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
        )}
      </svg>
      {ht != null && hi != null && (
        <div className="tooltip" style={{ left: Math.min(x(ht) + 12, W - 190), top: 24 }}>
          <div className="t num">{clock(ht - t0)}{phaseKnown ? ` · ${ht <= surg ? "trước rạch da" : "trong mổ"}` : ""}</div>
          <div className="row"><span><i className="key-line" style={{ background: "var(--series-1)" }} />Nguy cơ {horizon / 60}′</span>
            <b>{tl.risk[h][hi] == null ? "không dự báo" : `${pct(tl.risk[h][hi], 1)}%`}</b></div>
          {compare && (
            <div className="row"><span><i className="key-line" style={{ background: "var(--series-2)" }} />Baseline</span>
              <b>{compare.risk[h][hi] == null ? "–" : `${pct(compare.risk[h][hi], 1)}%`}</b></div>
          )}
          <div className="row"><span>MAP</span><b>{fmt(tl.vitals.map[hi])} mmHg</b></div>
          <div className="row"><span>Nhịp tim</span><b>{fmt(tl.vitals.hr[hi])}/phút</b></div>
          <div className="note" style={{ marginTop: 4 }}>Bấm để tua tới thời điểm này</div>
        </div>
      )}
    </div>
  );
}
