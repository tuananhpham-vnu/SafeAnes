import { useEffect, useMemo, useState } from "react";
import {
  HORIZONS, MODEL_COLOR, api, clock, fmt, horizonLabel, level, pct, sexLabel, type BedSnap, type Explanation, type History,
  type Horizon, type ModelInfo, type ModelName,
} from "../api";
import type { Buffer } from "../useWard";
import { LiveChart } from "./LiveChart";
import { AlarmLog, Drivers, ModelMatrix, RiskHero, VitalsPanel } from "./Panels";

const CONTEXT: [string, string, (v: number) => string][] = [
  ["f1a_min_since_anestart", "từ lúc khởi mê", (v) => `${fmt(v)}′`],
  ["f1a_min_since_opstart", "từ lúc rạch da", (v) => `${fmt(v)}′`],
  ["f1a_min_since_last_event", "từ đợt tụt gần nhất", (v) => `${fmt(v)}′`],
  ["f1a_map_min_m15", "MAP thấp nhất 15 phút (mmHg)", (v) => fmt(v)],
  ["map_drop_pct", "MAP giảm so với nền", (v) => `${fmt(v)}%`],
  ["f1b_ppf", "Ce propofol (µg/mL)", (v) => fmt(v, 1)],
  ["f1b_rftn_d5", "Δ Ce remifentanil 5 phút (ng/mL)", (v) => `${v > 0 ? "+" : ""}${fmt(v, 1)}`],
  ["f2_exp_sevo", "Sevoflurane thở ra (%)", (v) => fmt(v, 1)],
];

export function BedDetail({ bed, v, snap, buf, models, model, horizon, onModel, onHorizon, onBack }: {
  bed: number; v: number | null; snap: BedSnap | undefined; buf: Buffer | undefined; models: ModelInfo[];
  model: ModelName; horizon: Horizon; onModel: (m: ModelName) => void; onHorizon: (h: Horizon) => void; onBack: () => void;
}) {
  const [hist, setHist] = useState<History | null>(null);
  const [baseWave, setBaseWave] = useState<{ t: number[]; map: (number | null)[]; loop: number } | null>(null);
  const [ex, setEx] = useState<Explanation | null>(null);
  const [exLoading, setExLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const rowTime = snap?.row_time ?? null, loop = snap?.loop ?? 0;
  const titles = useMemo(() => Object.fromEntries(models.map((m) => [m.name, m.title])), [models]);

  // whole case so far once (2 s MAP since the start), then the risk history each time a new 30 s row arrives
  useEffect(() => {
    if (v == null) return;
    const ctrl = new AbortController();
    const first = !baseWave || baseWave.loop !== loop;
    api.history(bed, v, first ? 12 * 3600 : 0, ctrl.signal).then((h) => {
      setHist(h);
      if (first) setBaseWave({ t: h.wave.t, map: h.wave.map, loop });
      setError(null);
    }).catch((e) => { if (e.name !== "AbortError") setError(String(e)); });
    return () => ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bed, rowTime, loop]);

  useEffect(() => {
    if (v == null || rowTime == null) { setEx(null); return; }
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      setExLoading(true);
      api.explain(bed, v, model, horizon, ctrl.signal)
        .then((e) => { setEx(e); setExLoading(false); })
        .catch((e) => { if (e.name !== "AbortError") { setExLoading(false); setError(String(e)); } });
    }, 250);
    return () => { clearTimeout(id); ctrl.abort(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bed, rowTime, model, horizon]);

  // chart MAP: the history loaded on open, continued with the live 2 s samples of the stream
  const wave = useMemo(() => {
    if (!baseWave) return { t: [], map: [] as (number | null)[] };
    const last = baseWave.t.length ? baseWave.t[baseWave.t.length - 1] : -Infinity;
    const t = [...baseWave.t], map = [...baseWave.map];
    if (buf && buf.loop === loop) buf.t.forEach((x, i) => { if (x > last) { t.push(x); map.push(buf.map[i]); } });
    return { t, map };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseWave, buf, buf?.t.length, loop]);

  if (!hist || !snap) return <div className="card empty">{error ?? "Đang tải giường…"}</div>;
  const info = hist.info, k = HORIZONS.indexOf(horizon), h = String(horizon);
  const mi = models.find((m) => m.name === model)!;
  const thr = mi.horizons[h].threshold;
  const p = snap.risk[model]?.[k] ?? null;
  const lv = level(p, thr, snap.alarm[model][k]);
  const riskH = Object.fromEntries(models.map((m) => [m.name, hist.risk[m.name]?.[h] ?? []])) as Record<ModelName, (number | null)[]>;
  const val = mi.horizons[h].validation, test = mi.horizons[h].test;
  return (
    <>
      <section className="controls">
        <div className="controls-row">
          <button className="btn" onClick={onBack}>← Về trạm theo dõi</button>
          <div className="bed-title">
            <b>Giường {String(bed).padStart(2, "0")}</b>
            <span>{sexLabel(info.sex)}, {fmt(info.age)} tuổi · ASA {info.asa ?? "–"} · {info.department} · {info.optype}
              {info.emergency ? " · cấp cứu" : ""} · ca VitalDB #{info.caseid}</span>
          </div>
          <span className="clock">{clock(snap.elapsed_s)} · {snap.phase === "pre_incision" ? "trước rạch da" : "trong mổ"}</span>
        </div>
      </section>
      <div className="grid">
        <div className="card span-4">
          <RiskHero p={p} thr={thr} lv={lv} horizon={horizon} phase={snap.phase} eligible={snap.eligible} model={mi.title} />
          {lv === "alarm" && (
            <div className="banner" style={{ marginTop: 12 }} role="alert">
              <span aria-hidden>⚠</span>
              <div><strong>Cảnh báo nguy cơ tụt huyết áp trong {horizonLabel(horizon)}</strong>
                Xác suất vượt ngưỡng ở 2 lần dự báo liên tiếp. Xem phần “Vì sao” bên dưới.</div>
            </div>
          )}
        </div>
        <div className="card span-4">
          <h2>4 mô hình × 5 mốc</h2>
          <p className="sub">Xác suất ở lần dự báo gần nhất ({rowTime == null ? "chưa có" : clock(rowTime - hist.start_s)}). Bấm ô để chọn.</p>
          <ModelMatrix risk={snap.risk} alarm={snap.alarm} models={models} model={model} horizon={horizon}
            onPick={(m, hh) => { onModel(m); onHorizon(hh); }} />
        </div>
        <div className="card span-4">
          <h2>Sinh hiệu</h2>
          <p className="sub">Giá trị mới nhất, cập nhật mỗi 2 giây.</p>
          <VitalsPanel v={snap.vitals} />
        </div>
        <div className="card span-12">
          <div className="chart-title">
            <div><h2>Diễn tiến từ đầu ca tới hiện tại · mốc {horizonLabel(horizon)}</h2>
              <p className="sub">Mỗi 30 giây cả 4 mô hình chạy trên dữ liệu tới thời điểm đó. Phần mờ bên phải là thời gian chưa tới.</p></div>
            <div className="legend">
              {models.map((m) => <span key={m.name}><i className="key-line" style={{ background: MODEL_COLOR[m.name], height: m.name === model ? 3 : 2 }} />{m.title}</span>)}
              <span><i className="key-rect" style={{ background: "var(--critical-wash)", outline: "1px solid var(--critical)" }} />Đợt tụt</span>
              <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--critical)" /></svg>Cảnh báo đúng</span>
              <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="var(--critical)" strokeWidth="2" /></svg>Cảnh báo sai / chờ</span>
            </div>
          </div>
          <LiveChart start={hist.start_s} now={snap.t} surgeryStart={hist.surgery_start_s} time={hist.time_s} risk={riskH}
            models={models} model={model} horizon={horizon} alarms={hist.alarms[model]?.[h] ?? []} events={hist.events} wave={wave} />
        </div>
        <div className="card span-7">
          <h2>Vì sao nguy cơ ở mức này? · {mi.title}</h2>
          <p className="sub">Yếu tố đẩy nguy cơ lên/xuống ở lần dự báo gần nhất, mốc {horizonLabel(horizon)}.</p>
          <Drivers ex={ex} loading={exLoading} titles={titles} />
        </div>
        <div className="card span-5">
          <h2>Nhật ký cảnh báo · {mi.title} · {horizonLabel(horizon)}</h2>
          <p className="sub">Kết quả chỉ biết khi thời gian đã trôi qua: đúng khi đợt tụt bắt đầu trong {horizonLabel(horizon)} sau cảnh báo.</p>
          <AlarmLog alarms={hist.alarms[model]?.[h] ?? []} t0={hist.start_s} />
          <h2 style={{ marginTop: 16 }}>Ngữ cảnh ca</h2>
          <div className="ctx">
            {CONTEXT.map(([c, lbl, f]) => (
              <div key={c}><b className="num">{hist.context[c] == null ? "–" : f(hist.context[c]!)}</b><span>{lbl}</span></div>
            ))}
          </div>
          <p className="note" style={{ marginTop: 12 }}>
            {mi.title}, mốc {horizonLabel(horizon)}: validation AUROC {fmt(val.auroc, 3)}, độ nhạy {pct(val.event_sensitivity)}%,{" "}
            {fmt(val.false_alarms_per_hour, 2)} cảnh báo sai/giờ
            {test && <> · test (mô tả) AUROC {fmt(test.auroc, 3)}, độ nhạy {pct(test.event_sensitivity)}%</>}.
          </p>
        </div>
      </div>
    </>
  );
}
