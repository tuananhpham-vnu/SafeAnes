import { useState } from "react";
import {
  HORIZONS, LEVELS, MODEL_COLOR, clock, fmt, horizonLabel, pct, type Alarm, type Explanation, type Horizon, type Level,
  type ModelInfo, type ModelName, type Vitals,
} from "../api";

export function StatusPill({ lv, compact = false }: { lv: Level; compact?: boolean }) {
  const s = LEVELS[lv];
  return (
    <span className={`status${compact ? " compact" : ""}`} role="status">
      <span className="dot" style={{ background: s.color }} aria-hidden />
      <span aria-hidden>{s.icon}</span> {s.label}
    </span>
  );
}

/* ---------- hero: one number per view */
export function RiskHero({ p, thr, lv, horizon, phase, eligible, model }: {
  p: number | null; thr: number; lv: Level; horizon: Horizon; phase: string; eligible: boolean; model: string;
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
          : `${model}: cứ 100 thời điểm có nguy cơ như vậy thì khoảng ${pct(p)} thời điểm có tụt huyết áp (MAP < 65 mmHg ≥ 1 phút) bắt đầu trong ${horizonLabel(horizon)} tới.`}
      </div>
      <div className="note">Giai đoạn: <b style={{ color: "var(--ink-2)" }}>{phase === "pre_incision" ? "sau khởi mê, trước rạch da" : "trong mổ"}</b></div>
    </div>
  );
}

/* ---------- the 4 models x 5 horizons at the current row; threshold marked, selected cell outlined */
export function ModelMatrix({ risk, alarm, models, model, horizon, onPick }: {
  risk: Record<ModelName, number[] | null>; alarm: Record<ModelName, boolean[]>; models: ModelInfo[];
  model: ModelName; horizon: Horizon; onPick: (m: ModelName, h: Horizon) => void;
}) {
  return (
    <table className="matrix">
      <thead><tr><th>Mô hình</th>{HORIZONS.map((h) => <th key={h}>{h / 60}′</th>)}</tr></thead>
      <tbody>
        {models.map((m) => (
          <tr key={m.name} className={m.name === model ? "hl" : ""}>
            <td><i className="key-line" style={{ background: MODEL_COLOR[m.name] }} /> {m.title}</td>
            {HORIZONS.map((h, k) => {
              const p = risk[m.name]?.[k] ?? null, thr = m.horizons[String(h)].threshold;
              const on = alarm[m.name]?.[k];
              return (
                <td key={h} onClick={() => onPick(m.name, h)} className={`cell${m.name === model && h === horizon ? " sel" : ""}`}
                  title={`Ngưỡng ${pct(thr, 1)}%`}>
                  <span className="num" style={{ color: on ? "var(--critical)" : p != null && p >= thr ? "var(--serious)" : undefined,
                    fontWeight: on || (p != null && p >= thr) ? 700 : 500 }}>
                    {p == null ? "–" : `${pct(p)}%`}{on ? " ⚠" : p != null && p >= thr ? " ▲" : ""}
                  </span>
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ---------- vitals now */
export function VitalsPanel({ v }: { v: Vitals }) {
  const items: [keyof Vitals, string, string][] = [["map", "MAP", "mmHg"], ["hr", "Nhịp tim", "/phút"], ["spo2", "SpO₂", "%"]];
  return (
    <div className="vitals">
      {items.map(([k, label, unit]) => {
        const x = v[k];
        const low = k === "map" && x != null && x < 65;
        return (
          <div key={k} className={`vital${low ? " low" : ""}`}>
            <div className="lbl"><span>{label}</span>{low && <span style={{ color: "var(--critical)" }}>⚠ dưới 65</span>}</div>
            <div className="val num">{fmt(x)}<span className="unit">{unit}</span></div>
          </div>
        );
      })}
      <div className="vital">
        <div className="lbl"><span>HA tâm thu / tâm trương</span></div>
        <div className="val num">{fmt(v.sbp)} / {fmt(v.dbp)}<span className="unit">mmHg</span></div>
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

const METHOD: Record<ModelName, string> = {
  lgbm_v8: "TreeSHAP", catboost_v8: "SHAP của CatBoost", logistic_v8: "hệ số × giá trị chuẩn hoá", tabm_v8: "",
};

export function Drivers({ ex, loading, titles }: { ex: Explanation | null; loading: boolean; titles: Record<string, string> }) {
  const [view, setView] = useState<"groups" | "features">("features");
  if (!ex) return <div className="empty">{loading ? "Đang tính…" : "Chưa có dự báo ở thời điểm này."}</div>;
  if (!ex.available) return <div className="empty">{ex.reason ?? "Không có giải thích."}</div>;
  const rows = view === "groups"
    ? ex.groups.map((g) => ({ k: g.id, name: g.label, v: g.contribution, val: "" }))
    : ex.features.map((f) => ({ k: f.column, name: f.label, v: f.contribution,
      val: f.value == null ? "thiếu" : `${fmt(f.value, Math.abs(f.value) < 10 ? 1 : 0)} ${f.unit}` }));
  const max = Math.max(...rows.map((r) => Math.abs(r.v)), 0.25);
  return (
    <div style={{ opacity: loading ? 0.6 : 1, transition: "opacity .2s" }}>
      {ex.proxy_for && (
        <div className="banner soft" style={{ marginBottom: 10 }}>
          <span aria-hidden>ⓘ</span>
          <div>{titles[ex.proxy_for]} không có cách phân rã theo từng đặc trưng; đây là giải thích của <b>{titles[ex.model]}</b> trên cùng
            dữ liệu, dùng để tham khảo.</div>
        </div>
      )}
      <div className="chart-title" style={{ marginBottom: 10 }}>
        <div className="legend">
          <span><i className="key-rect" style={{ background: "var(--up)" }} />Làm tăng nguy cơ</span>
          <span><i className="key-rect" style={{ background: "var(--down)" }} />Làm giảm nguy cơ</span>
        </div>
        <div className="seg" role="group" aria-label="Mức chi tiết">
          <button aria-pressed={view === "features"} onClick={() => setView("features")}>Đặc trưng</button>
          <button aria-pressed={view === "groups"} onClick={() => setView("groups")}>Nhóm</button>
        </div>
      </div>
      <div className="drivers">
        {rows.map((r) => (
          <div key={r.k} className="driver" title={`${r.v >= 0 ? "+" : ""}${r.v.toFixed(2)} log-odds`}>
            <div className="name">{r.name}</div>
            <DBar v={r.v} max={max} />
            <div className="val">{view === "features" ? r.val : `${r.v >= 0 ? "▲" : "▼"} ${Math.abs(r.v).toFixed(2)}`}</div>
          </div>
        ))}
      </div>
      <p className="note" style={{ marginTop: 10 }}>
        Đóng góp ({METHOD[ex.model]}) trên thang log-odds của xác suất đã hiệu chỉnh, mốc {horizonLabel(ex.horizon_s)}. Đây là đóng
        góp thống kê của mô hình, không phải chẩn đoán nguyên nhân.
      </p>
    </div>
  );
}

/* ---------- alarm log (outcome as known now: the server never looks ahead) */
export function AlarmLog({ alarms, t0 }: { alarms: Alarm[]; t0: number }) {
  const shown = alarms.slice().reverse();
  if (!shown.length) return <div className="empty">Chưa có cảnh báo.</div>;
  return (
    <ul className="log">
      {shown.map((a) => {
        const tag = a.kind === "pending" ? { t: "Đang chờ kết quả", c: "var(--muted)", i: "…" }
          : a.kind === "true" ? { t: "Đúng", c: "var(--critical)", i: "●" }
          : a.kind === "false" ? { t: "Cảnh báo sai", c: "var(--muted)", i: "○" } : { t: "Không xác định", c: "var(--muted)", i: "?" };
        return (
          <li key={a.time_s}>
            <span className="num">{clock(a.time_s - t0).slice(0, 5)}</span>
            <span>{a.kind === "true" ? `Báo trước ${fmt((a.lead_s ?? 0) / 60, 1)} phút` : "Cảnh báo tụt huyết áp"}</span>
            <span className="pill"><span style={{ color: tag.c }}>{tag.i}</span>{tag.t}</span>
          </li>
        );
      })}
    </ul>
  );
}
