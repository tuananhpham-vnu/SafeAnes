import { useState } from "react";
import {
  HORIZONS, LEVELS, alarmKnown, clock, fmt, horizonLabel, pct, type Alarm, type Explanation, type Horizon, type Level, type Timeline,
} from "../api";
import { useWidth } from "./useWidth";

export function StatusPill({ lv }: { lv: Level }) {
  const s = LEVELS[lv];
  return (
    <span className="status" role="status">
      <span className="dot" style={{ background: s.color }} aria-hidden />
      <span aria-hidden>{s.icon}</span> {s.label}
    </span>
  );
}

/* ---------- hero: one number per view */
export function RiskHero({ p, thr, lv, horizon, phase, eligible }: {
  p: number | null; thr: number; lv: Level; horizon: Horizon; phase: string; eligible: boolean;
}) {
  const fill = lv === "na" ? "var(--muted)" : LEVELS[lv].color;
  return (
    <div className="hero">
      <div className="hero-top">
        <h2>Nguy cơ tụt huyết áp trong {horizonLabel(horizon)}</h2>
        <StatusPill lv={lv} />
      </div>
      <div className="hero-value" aria-live="polite">
        {p == null ? "–" : pct(p)}<small>{p == null ? "" : "%"}</small>
      </div>
      <div className="meter" aria-hidden>
        <div className="meter-fill" style={{ width: `${(p ?? 0) * 100}%`, background: fill }} />
        <div className="meter-thr" style={{ left: `calc(${thr * 100}% - 1px)` }} title={`Ngưỡng ${pct(thr)}%`} />
      </div>
      <div className="meter-scale num" style={{ position: "relative" }}><span>0%</span>
        <span style={{ position: "absolute", left: `${thr * 100}%`, transform: "translateX(-50%)" }}>ngưỡng {pct(thr)}%</span><span>100%</span></div>
      <div className="hero-caption">
        {!eligible
          ? "Không dự báo ở thời điểm này (đang tụt, ngay sau một đợt tụt, hoặc thiếu MAP)."
          : `Xác suất đã hiệu chỉnh: cứ 100 thời điểm có nguy cơ như vậy thì khoảng ${pct(p)} thời điểm có tụt huyết áp (MAP < 65 mmHg ≥ 1 phút) bắt đầu trong ${horizonLabel(horizon)} tới.`}
      </div>
      <div className="note">Giai đoạn: <b style={{ color: "var(--ink-2)" }}>{phase === "pre_incision" ? "sau khởi mê, trước rạch da" : "trong mổ"}</b></div>
    </div>
  );
}

/* ---------- horizon profile: 5 columns, value on the cap, threshold tick per column */
export function HorizonProfile({ tl, idx, horizon, onPick }: {
  tl: Timeline; idx: number; horizon: Horizon; onPick: (h: Horizon) => void;
}) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hov, setHov] = useState<number | null>(null);
  const W = Math.max(w, 240), H = 170, top = 22, bottom = 26, band = (W - 8) / HORIZONS.length, bw = Math.min(24, band * 0.5);
  const y = (p: number) => top + (H - top - bottom) * (1 - p);
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg className="chart" width={W} height={H} role="img" aria-label="Nguy cơ theo từng mốc thời gian dự báo">
        {[0, 0.5, 1].map((p) => <line key={p} className={p === 0 ? "axis-line" : "grid-line"} x1={0} x2={W} y1={y(p)} y2={y(p)} />)}
        {HORIZONS.map((h, i) => {
          const p = tl.risk[String(h)][idx], thr = tl.thresholds[String(h)];
          const cx = 4 + band * i + band / 2, sel = h === horizon;
          const yy = y(p ?? 0), r = Math.min(4, (y(0) - yy) / 2);
          return (
            <g key={h} onClick={() => onPick(h)} onPointerEnter={() => setHov(i)} onPointerLeave={() => setHov(null)} style={{ cursor: "pointer" }}>
              <rect x={cx - band / 2} y={top - 16} width={band} height={H - top - bottom + 16 + 20} fill="transparent" />
              {p != null && p > 0 && (
                <path d={`M${cx - bw / 2},${y(0)} V${yy + r} Q${cx - bw / 2},${yy} ${cx - bw / 2 + r},${yy} H${cx + bw / 2 - r} Q${cx + bw / 2},${yy} ${cx + bw / 2},${yy + r} V${y(0)} Z`}
                  fill="var(--series-1)" opacity={sel ? 1 : hov === i ? 0.85 : 0.55} />
              )}
              <line x1={cx - bw / 2 - 5} x2={cx + bw / 2 + 5} y1={y(thr)} y2={y(thr)} stroke="var(--critical)" strokeWidth={2} strokeLinecap="round" />
              <text x={cx} y={Math.min(p == null ? y(0) : yy, y(thr)) - 7} textAnchor="middle" className="num" style={{ fill: "var(--ink)", fontWeight: sel ? 700 : 500 }}>
                {p == null ? "–" : `${pct(p)}%`}{p != null && p >= thr ? " ▲" : ""}
              </text>
              <text x={cx} y={H - 8} textAnchor="middle" style={{ fill: sel ? "var(--ink)" : "var(--muted)", fontWeight: sel ? 600 : 400 }}>{h / 60}′</text>
            </g>
          );
        })}
      </svg>
      <div className="legend" style={{ marginTop: 4 }}>
        <span><i className="key-rect" style={{ background: "var(--series-1)" }} />Xác suất</span>
        <span><i className="key-line" style={{ background: "var(--critical)" }} />Ngưỡng cảnh báo của mốc</span>
      </div>
      {hov != null && (
        <div className="tooltip" style={{ left: Math.min(4 + band * hov + band / 2 + 8, W - 170), top: 8 }}>
          <div className="t">Trong {horizonLabel(HORIZONS[hov])} tới</div>
          <div className="row"><span>Xác suất</span><b>{pct(tl.risk[String(HORIZONS[hov])][idx], 1)}%</b></div>
          <div className="row"><span>Ngưỡng</span><b>{pct(tl.thresholds[String(HORIZONS[hov])], 1)}%</b></div>
        </div>
      )}
    </div>
  );
}

/* ---------- vitals at the cursor */
export function Vitals({ tl, idx }: { tl: Timeline; idx: number }) {
  const items: [keyof Timeline["vitals"], string, string, number][] = [
    ["map", "MAP", "mmHg", 1], ["hr", "Nhịp tim", "/phút", 0], ["spo2", "SpO₂", "%", 0], ["etco2", "EtCO₂", "mmHg", 0]];
  return (
    <div className="vitals">
      {items.map(([k, label, unit]) => {
        const v = tl.vitals[k][idx];
        const low = k === "map" && v != null && v < 65;
        return (
          <div key={k} className={`vital${low ? " low" : ""}`}>
            <div className="lbl"><span>{label}</span>{low && <span style={{ color: "var(--critical)" }}>⚠ dưới 65</span>}</div>
            <div className="val num">{fmt(v)}<span className="unit">{unit}</span></div>
          </div>
        );
      })}
      <div className="vital" style={{ gridColumn: "span 2" }}>
        <div className="lbl"><span>HA tâm thu / tâm trương</span></div>
        <div className="val num">{fmt(tl.vitals.sbp[idx])} / {fmt(tl.vitals.dbp[idx])}<span className="unit">mmHg</span></div>
      </div>
    </div>
  );
}

/* ---------- why: diverging bars (red raises, blue lowers), grouped by physiology */
function DBar({ v, max }: { v: number; max: number }) {
  const f = Math.min(Math.abs(v) / max, 1) * 50;
  return (
    <div className="dbar" aria-hidden>
      <div className="mid" />
      <div className="b" style={{ left: v >= 0 ? "50%" : `${50 - f}%`, width: `${f}%`, background: v >= 0 ? "var(--up)" : "var(--down)",
        borderRadius: v >= 0 ? "0 4px 4px 0" : "4px 0 0 4px" }} />
    </div>
  );
}

export function Drivers({ ex, loading }: { ex: Explanation | null; loading: boolean }) {
  const [view, setView] = useState<"groups" | "features">("groups");
  if (!ex) return <div className="empty">{loading ? "Đang tính…" : "Chọn một thời điểm có dự báo."}</div>;
  const rows = view === "groups"
    ? ex.groups.map((g) => ({ k: g.id, name: g.label, tag: g.kind === "context" ? "ngữ cảnh ca" : "", v: g.contribution, val: "" }))
    : ex.features.map((f) => ({ k: f.column, name: f.label, tag: "", v: f.contribution, val: f.value == null ? "thiếu" : `${fmt(f.value, Math.abs(f.value) < 10 ? 1 : 0)} ${f.unit}` }));
  const max = Math.max(...rows.map((r) => Math.abs(r.v)), 0.25);
  const c = ex.context;
  return (
    <div style={{ opacity: loading ? 0.6 : 1, transition: "opacity .2s" }}>
      <div className="chart-title" style={{ marginBottom: 10 }}>
        <div className="legend">
          <span><i className="key-rect" style={{ background: "var(--up)" }} />Làm tăng nguy cơ</span>
          <span><i className="key-rect" style={{ background: "var(--down)" }} />Làm giảm nguy cơ</span>
        </div>
        <div className="seg" role="group" aria-label="Mức chi tiết">
          <button aria-pressed={view === "groups"} onClick={() => setView("groups")}>Nhóm</button>
          <button aria-pressed={view === "features"} onClick={() => setView("features")}>Đặc trưng</button>
        </div>
      </div>
      <div className="drivers">
        {rows.map((r) => (
          <div key={r.k} className="driver" title={`${r.v >= 0 ? "+" : ""}${r.v.toFixed(2)} log-odds`}>
            <div className="name">{r.name}{r.tag && <em>{r.tag}</em>}</div>
            <DBar v={r.v} max={max} />
            <div className="val">{view === "features" ? r.val : `${r.v >= 0 ? "▲" : "▼"} ${Math.abs(r.v).toFixed(2)}`}</div>
          </div>
        ))}
      </div>
      <div className="ctx">
        <div><b className="num">{c.f1a_min_since_anestart == null ? "–" : `${fmt(c.f1a_min_since_anestart)}′`}</b><span>từ lúc khởi mê</span></div>
        <div><b className="num">{fmt(c.f1a_n_prev_events)}</b><span>đợt tụt trước trong ca</span></div>
        <div><b className="num">{c.f1a_map_rel_case_median == null ? "–" : `${c.f1a_map_rel_case_median > 0 ? "+" : ""}${fmt(c.f1a_map_rel_case_median)}`}</b><span>MAP so với trung vị ca (mmHg)</span></div>
      </div>
      <p className="note" style={{ marginTop: 10 }}>
        Đóng góp TreeSHAP trên thang log-odds của xác suất đã hiệu chỉnh, cho mốc {horizonLabel(ex.horizon_s)}. Đây là đóng góp
        thống kê của mô hình, không phải chẩn đoán nguyên nhân.
      </p>
    </div>
  );
}

/* ---------- alarm log */
export function AlarmLog({ alarms, horizon, t0, cursor, reveal, onSeek }: {
  alarms: Alarm[]; horizon: Horizon; t0: number; cursor: number; reveal: boolean; onSeek: (t: number) => void;
}) {
  const shown = alarms.filter((a) => reveal || a.time_s <= cursor).slice().reverse();
  if (!shown.length) return <div className="empty">Chưa có cảnh báo.</div>;
  return (
    <ul className="log">
      {shown.map((a) => {
        const known = alarmKnown(a, horizon, cursor, reveal);
        const tag = !known ? { t: "Đang chờ kết quả", c: "var(--muted)", i: "…" }
          : a.kind === "true" ? { t: "Đúng", c: "var(--critical)", i: "●" }
          : a.kind === "false" ? { t: "Cảnh báo sai", c: "var(--muted)", i: "○" } : { t: "Không xác định", c: "var(--muted)", i: "?" };
        return (
          <li key={a.time_s} onClick={() => onSeek(a.time_s)} tabIndex={0} onKeyDown={(e) => e.key === "Enter" && onSeek(a.time_s)}>
            <span className="num">{clock(a.time_s - t0).slice(0, 5)}</span>
            <span>{known && a.kind === "true" ? `Báo trước ${fmt((a.lead_s ?? 0) / 60, 1)} phút` : "Cảnh báo tụt huyết áp"}</span>
            <span className="pill"><span style={{ color: tag.c }}>{tag.i}</span>{tag.t}</span>
          </li>
        );
      })}
    </ul>
  );
}
