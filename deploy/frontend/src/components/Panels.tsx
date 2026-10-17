import { useState } from "react";
import {
  HORIZONS, LEVELS, alarmKnown, clock, fmt, horizonLabel, pct, type Alarm, type Chain, type Comp, type Explanation, type Horizon, type Level,
  type Timeline,
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

/* ---------- why: driver -> HR / SV / SVR -> MAP -> risk (component-forecast chain) */
const COMPS = ["hr", "sv", "svr"] as const;
const signed = (v: number | null | undefined, d = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${fmt(v, d)}%`);
// a falling component pushes MAP down and risk up: red; rising: blue (same palette as the risk drivers)
const tone = (v: number | null | undefined) => (v == null ? "var(--muted)" : v < 0 ? "var(--up)" : "var(--down)");

function ChainDiagram({ c, p }: { c: Chain; p: number | null }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const W = Math.max(w, 320);
  // driver nodes: the groups with the largest total effect over the three components
  const tot = new Map<string, number>();
  COMPS.forEach((k) => c.drivers[k].forEach((d) => tot.set(d.group, (tot.get(d.group) ?? 0) + Math.abs(d.pp))));
  const groups = [...tot.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6).map(([g]) => g);
  const rowH = 30, H = Math.max(groups.length, 3) * rowH + 40;
  const x0 = 4, x1 = W * 0.36, x2 = W * 0.64, x3 = W * 0.86, nodeW = Math.min(150, W * 0.3);
  const gy = (i: number) => 26 + i * rowH + (H - 40 - groups.length * rowH) / 2;
  const cy = (i: number) => 26 + i * ((H - 40) / 3) + (H - 40) / 6 - 8;
  const maxPP = Math.max(0.5, ...COMPS.flatMap((k) => c.drivers[k].map((d) => Math.abs(d.pp))));
  const maxF = Math.max(1, ...COMPS.map((k) => Math.abs(c.forecast[k] ?? 0)));
  const midY = (H - 14) / 2;
  return (
    <div ref={ref}>
      <svg className="chart" width={W} height={H} role="img"
        aria-label="Chuỗi giải thích: yếu tố dẫn, thành phần huyết động, MAP và nguy cơ">
        <text x={x0} y={12} className="axis-label">Yếu tố dẫn</text>
        <text x={x1} y={12} className="axis-label">Dự báo 15 phút tới</text>
        <text x={x3 - 20} y={12} className="axis-label">Kết quả</text>
        {COMPS.map((k, j) => c.drivers[k].filter((d) => groups.includes(d.group)).map((d) => {
          const i = groups.indexOf(d.group), sy = gy(i) + 9, ty = cy(j) + 9;
          return <path key={`${k}-${d.group}`} d={`M${x0 + nodeW},${sy} C${(x0 + nodeW + x1) / 2},${sy} ${(x0 + nodeW + x1) / 2},${ty} ${x1},${ty}`}
            fill="none" stroke={tone(d.pp)} strokeOpacity={0.55} strokeWidth={1 + 7 * Math.abs(d.pp) / maxPP}>
            <title>{`${d.group} → ${c.labels[k]}: ${d.pp > 0 ? "+" : ""}${d.pp.toFixed(1)} điểm %`}</title></path>;
        }))}
        {COMPS.map((k, j) => (
          <path key={`m-${k}`} d={`M${x1 + nodeW * 0.78},${cy(j) + 9} C${(x1 + x2) / 2 + 30},${cy(j) + 9} ${(x1 + x2) / 2 + 30},${midY} ${x2},${midY}`}
            fill="none" stroke={tone(c.forecast[k])} strokeOpacity={0.55} strokeWidth={1 + 7 * Math.abs(c.forecast[k] ?? 0) / maxF} />
        ))}
        <line x1={x2 + 78} x2={x3 - 24} y1={midY} y2={midY} stroke="var(--axis)" strokeWidth={2} markerEnd="" />
        {groups.map((g, i) => (
          <g key={g}><rect x={x0} y={gy(i)} width={nodeW} height={18} rx={5} fill="var(--surface-2)" />
            <text x={x0 + 6} y={gy(i) + 13} style={{ fill: "var(--ink-2)", fontSize: 11 }}>{g.length > 26 ? g.slice(0, 25) + "…" : g}</text></g>
        ))}
        {COMPS.map((k, j) => (
          <g key={k}><rect x={x1} y={cy(j)} width={nodeW * 0.78} height={18} rx={5} fill="var(--surface-2)"
            stroke={c.dominant === k ? tone(c.forecast[k]) : "none"} strokeWidth={1.5} />
            <text x={x1 + 6} y={cy(j) + 13} style={{ fill: "var(--ink)", fontSize: 11, fontWeight: c.dominant === k ? 700 : 500 }}>
              {k.toUpperCase()} <tspan style={{ fill: tone(c.forecast[k]) }}>{signed(c.forecast[k])}</tspan></text></g>
        ))}
        <g><rect x={x2} y={midY - 9} width={78} height={18} rx={5} fill="var(--surface-2)" />
          <text x={x2 + 6} y={midY + 4} style={{ fill: "var(--ink)", fontSize: 11, fontWeight: 700 }}>
            MAP <tspan style={{ fill: tone(c.forecast.map) }}>{signed(c.forecast.map)}</tspan></text></g>
        <g><rect x={x3 - 22} y={midY - 9} width={W - x3 + 18} height={18} rx={5} fill="var(--surface-2)" />
          <text x={x3 - 16} y={midY + 4} style={{ fill: "var(--ink)", fontSize: 11, fontWeight: 700 }}>Nguy cơ {pct(p)}%</text></g>
      </svg>
    </div>
  );
}

function Split({ title, v }: { title: string; v: Record<Comp, number | null> }) {
  const max = Math.max(5, ...(["map", ...COMPS] as Comp[]).map((k) => Math.abs(v[k] ?? 0)));
  return (
    <div>
      <div className="note" style={{ marginBottom: 6 }}>{title}</div>
      <div className="drivers">
        {(["map", ...COMPS] as Comp[]).map((k) => (
          <div key={k} className="driver">
            <div className="name" style={{ fontWeight: k === "map" ? 650 : 400 }}>{k === "map" ? "MAP = HR × SV × SVR" : k.toUpperCase()}</div>
            {/* bar direction follows the value; colour: a fall (raises risk) red */}
            <div className="dbar" aria-hidden><div className="mid" />
              <div className="b" style={{ left: (v[k] ?? 0) >= 0 ? "50%" : `${50 - Math.min(Math.abs(v[k] ?? 0) / max, 1) * 50}%`,
                width: `${Math.min(Math.abs(v[k] ?? 0) / max, 1) * 50}%`, background: tone(v[k]) }} /></div>
            <div className="val">{signed(v[k])}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function WhyChain({ ex, loading }: { ex: Explanation | null; loading: boolean }) {
  if (!ex) return <div className="empty">{loading ? "Đang tính…" : "Chọn một thời điểm có dự báo."}</div>;
  const c = ex.chain;
  if (!c) return <div className="empty">Bản dữ liệu demo chưa có mô hình dự báo thành phần (chạy export_bundle.py --chain-models).</div>;
  const dom = c.dominant ? c.labels[c.dominant] : null;
  return (
    <div style={{ opacity: loading ? 0.6 : 1, transition: "opacity .2s" }}>
      <p style={{ margin: "0 0 10px", fontSize: 14 }}>
        {dom
          ? <>15 phút tới MAP dự kiến <b style={{ color: tone(c.forecast.map) }}>{signed(c.forecast.map)}</b>, chủ yếu do <b>{dom}</b> {signed(c.forecast[c.dominant!])}
            {c.drivers[c.dominant!][0] && <>; yếu tố đẩy mạnh nhất: <b>{c.drivers[c.dominant!][0].group}</b></>}.</>
          : <>15 phút tới MAP dự kiến <b>{signed(c.forecast.map)}</b>; không thành phần nào dự kiến giảm.</>}
      </p>
      <ChainDiagram c={c} p={ex.probability} />
      <div className="grid2">
        <Split title={`10 phút qua · ${c.past_pattern}`} v={c.past} />
        <Split title="15 phút tới (dự báo)" v={c.forecast} />
      </div>
      {c.no_change && (
        <div className="banner soft" style={{ marginTop: 12 }}>
          <span aria-hidden>↺</span>
          <div>Nếu <b>không có</b> thay đổi thuốc mê / thở máy trong 10 phút qua, mô hình dự báo MAP {signed(c.no_change.map)} thay vì {signed(c.forecast.map)}
            (SVR {signed(c.no_change.svr)} thay vì {signed(c.forecast.svr)}). Đây là ước lượng theo mô hình, không phải hiệu quả nhân quả.</div>
        </div>
      )}
      <div className="chips">
        {c.findings.map((f) => <span key={f} className="chip">{f}</span>)}
        {c.profile.map((f) => <span key={f} className="chip prof">{f}</span>)}
        {c.personal.hr_per_map != null && (
          <span className="chip" title="Độ dốc HR theo MAP trong 30 phút: âm = nhịp tim tăng bù khi MAP giảm; gần 0 = bù trừ yếu">
            HR theo MAP: {fmt(c.personal.hr_per_map, 2)} nhịp/mmHg{c.personal.hr_per_map > -0.1 ? " (bù trừ yếu)" : ""}</span>
        )}
        {c.personal.lowest_map != null && <span className="chip">MAP thấp nhất từ đầu ca: {fmt(c.personal.lowest_map)} mmHg</span>}
      </div>
      <p className="note" style={{ marginTop: 10 }}>
        HR, SV, SVR: SV và SVR là proxy tính từ huyết áp (Liljestrand–Zander), chỉ tin được về hướng thay đổi.
        Độ dày mũi tên theo mức đóng góp (TreeSHAP, điểm %). Đây là phân tích kết hợp, không phải chẩn đoán nguyên nhân.
        {c.missing_groups.length > 0 && <> Thiếu dữ liệu: {c.missing_groups.join(", ")}.</>}
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
