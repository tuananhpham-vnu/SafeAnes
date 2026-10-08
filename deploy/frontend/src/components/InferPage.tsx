import { useEffect, useMemo, useRef, useState } from "react";
import {
  HORIZONS, alarmOnAt, api, clock, fmt, horizonLabel, indexAt, level, pct,
  type Example, type Explanation, type Horizon, type InferResult, type ModelInfo, type ModelName,
} from "../api";
import { AlarmLog, Drivers, HorizonProfile, RiskHero, Vitals } from "./Panels";
import { TimelineChart } from "./TimelineChart";

const FORMAT = `time,map,sbp,dbp,hr,spo2,etco2,rr
08:15:02,82,121,63,74,99,35,12
08:15:04,81,,,74,,,
08:15:06,80,119,62,75,99,34,12
...`;

/** Last row with a prediction (the "now" of an uploaded record). */
const lastPredicted = (r: InferResult, h: string) => {
  for (let i = r.time_s.length - 1; i >= 0; i--) if (r.risk[h][i] != null) return i;
  return r.time_s.length - 1;
};

export function InferPage({ models }: { models: ModelInfo[] }) {
  const [examples, setExamples] = useState<Example[]>([]);
  const [csv, setCsv] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [anestart, setAnestart] = useState("");
  const [opstart, setOpstart] = useState("");
  const [baseline, setBaseline] = useState("");
  const [model, setModel] = useState<ModelName>("lgbm_context");
  const [horizon, setHorizon] = useState<Horizon>(300);
  const [res, setRes] = useState<InferResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [sel, setSel] = useState<number | null>(null);
  const [ex, setEx] = useState<Explanation | null>(null);
  const [exLoading, setExLoading] = useState(false);
  const [over, setOver] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => { api.examples().then(setExamples).catch(() => setExamples([])); }, []);

  // ?example=<caseid>: load that example and run at once (a link to try the page without a file)
  useEffect(() => {
    const id = Number(new URLSearchParams(location.search).get("example"));
    const e = examples.find((x) => x.caseid === id);
    if (!e || csv) return;
    (async () => {
      const text = await api.exampleCsv(id);
      setCsv(text); setFileName(`ca mẫu #${id} (VitalDB, Solar8000)`);
      setAnestart(String(e.anestart_s)); setOpstart(String(e.opstart_s));
      setBusy(true);
      try {
        const r = await api.infer({ csv: text, anestart: String(e.anestart_s), opstart: String(e.opstart_s), model });
        setRes(r); setSel(r.time_s[lastPredicted(r, String(horizon))]);
      } catch (err) { setErr((err as Error).message); } finally { setBusy(false); }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [examples]);

  const readFile = (f: File) => {
    if (f.size > 30e6) { setErr("File quá lớn (tối đa 30 MB)."); return; }
    setErr(null); setFileName(f.name);
    f.text().then(setCsv);
  };
  const loadExample = async (id: number) => {
    const e = examples.find((x) => x.caseid === id);
    setErr(null);
    setCsv(await api.exampleCsv(id));
    setFileName(`ca mẫu #${id} (VitalDB, Solar8000)`);
    if (e) { setAnestart(String(e.anestart_s)); setOpstart(String(e.opstart_s)); }
    setBaseline("");
  };
  const run = async () => {
    if (!csv) return;
    setBusy(true); setErr(null);
    try {
      const r = await api.infer({ csv, anestart, opstart, baseline_map: baseline === "" ? null : Number(baseline), model });
      setRes(r);
      setSel(r.time_s[lastPredicted(r, String(horizon))]);
    } catch (e) { setErr((e as Error).message); setRes(null); }
    finally { setBusy(false); }
  };
  // switching model on an existing upload: recompute risks from the stored features (no re-upload)
  useEffect(() => {
    if (!res || res.model === model) return;
    api.inferModel(res.session, model).then(setRes).catch((e) => setErr((e as Error).message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const h = String(horizon);
  const idx = res && sel != null ? indexAt(res.time_s, sel) : 0;
  const tAt = res ? res.time_s[idx] : 0;
  useEffect(() => {
    if (!res) return;
    const ctrl = new AbortController();
    setExLoading(true);
    api.inferExplain(res.session, model, tAt, horizon, ctrl.signal)
      .then((e) => { setEx(e); setExLoading(false); })
      .catch((e) => { if (e.name !== "AbortError") { setExLoading(false); setErr((e as Error).message); } });
    return () => ctrl.abort();
  }, [res, model, tAt, horizon]);

  const p = res ? res.risk[h][idx] : null;
  const thr = res ? res.thresholds[h] : 0.5;
  const alarmOn = useMemo(() => (res ? alarmOnAt(res, h, idx) : false), [res, h, idx]);
  const lv = level(p, thr, alarmOn);
  const t0 = res?.time_s[0] ?? 0;
  const phase = res && res.phase_known && tAt > res.surgery_start_s ? "surgery" : "pre_incision";

  const download = () => {
    if (!res) return;
    const head = ["time_s", "elapsed", ...HORIZONS.map((x) => `risk_${x / 60}min`), "map", "alarm_5min"];
    const alarm5 = new Set(res.alarms["300"].map((a) => a.time_s));
    const rows = res.time_s.map((t, i) => [t, clock(t - t0), ...HORIZONS.map((x) => res.risk[String(x)][i] ?? ""),
      res.vitals.map[i] ?? "", alarm5.has(t) ? 1 : 0].join(","));
    const blob = new Blob([[head.join(","), ...rows].join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `safeanes_du_bao_${(fileName || "ket_qua").replace(/[^\w.-]+/g, "_")}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="grid">
      <div className="card span-12">
        <h2>Dự báo trên dữ liệu của bạn</h2>
        <p className="sub">Tải lên sinh hiệu monitor của một ca (CSV). Hệ thống tính đặc trưng bằng đúng pipeline lúc huấn luyện,
          rồi trả về nguy cơ mỗi 30 giây cho 5 mốc, các cảnh báo và lý do. Dữ liệu chỉ nằm trong bộ nhớ của máy chủ local, không lưu xuống đĩa.</p>
        <div className="grid" style={{ gap: 16 }}>
          <div className="span-7">
            <div className={`drop${over ? " over" : ""}`} role="button" tabIndex={0}
              onClick={() => fileRef.current?.click()} onKeyDown={(e) => e.key === "Enter" && fileRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) readFile(f); }}>
              <b>Kéo thả file CSV vào đây hoặc bấm để chọn</b>
              <span className="note">CSV, TSV hoặc phân cách bằng dấu chấm phẩy · tối đa 30 MB · tối thiểu 5 phút dữ liệu</span>
              <input ref={fileRef} type="file" accept=".csv,.tsv,.txt" hidden onChange={(e) => e.target.files?.[0] && readFile(e.target.files[0])} />
            </div>
            <div className="actions">
              {csv && <span className="filechip">📄 {fileName} · {fmt(csv.split("\n").length - 1)} dòng</span>}
              {examples.length > 0 && (
                <label className="field">hoặc thử ca mẫu
                  <select value="" onChange={(e) => e.target.value && loadExample(Number(e.target.value))}>
                    <option value="">— chọn —</option>
                    {examples.map((x) => <option key={x.caseid} value={x.caseid}>#{x.caseid}</option>)}
                  </select>
                </label>
              )}
              {examples[0] && <a className="note" href={`/api/examples/${examples[0].caseid}.csv`} download>Tải file mẫu</a>}
            </div>
            <div className="form-grid">
              <label>Thời điểm khởi mê<input value={anestart} onChange={(e) => setAnestart(e.target.value)} placeholder="vd 08:05 hoặc 0" />
                <small>Cùng đồng hồ với cột thời gian</small></label>
              <label>Thời điểm rạch da<input value={opstart} onChange={(e) => setOpstart(e.target.value)} placeholder="vd 08:40" />
                <small>Bỏ trống nếu chưa rạch da</small></label>
              <label>MAP nền (mmHg)<input value={baseline} inputMode="decimal" onChange={(e) => setBaseline(e.target.value.replace(",", "."))} placeholder="tự tính nếu bỏ trống" />
                <small>MAP trước khởi mê, nếu có</small></label>
              <label>Mô hình<select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
                {models.map((m) => <option key={m.name} value={m.name}>{m.title}</option>)}</select></label>
            </div>
            <div className="actions">
              <button className="btn primary" disabled={!csv || busy} onClick={run}>{busy ? "Đang tính…" : "Dự báo"}</button>
              {res && <button className="btn" onClick={download}>Tải kết quả (CSV)</button>}
              {err && <span role="alert" style={{ color: "var(--critical)", fontSize: 13 }}>⚠ {err}</span>}
            </div>
          </div>
          <div className="span-5">
            <h2>Định dạng file</h2>
            <p className="sub">Một cột thời gian và các cột sinh hiệu; ô trống được phép (mỗi máy ghi một nhịp khác nhau).</p>
            <div className="fmt">{FORMAT}</div>
            <ul className="note" style={{ paddingLeft: 18, marginBottom: 0 }}>
              <li><b>Bắt buộc:</b> thời gian (giây, HH:MM:SS hoặc ngày giờ) và <b>MAP động mạch xâm lấn</b>.</li>
              <li><b>Nên có:</b> SBP, DBP, HR, SpO₂, EtCO₂, nhịp thở; NIBP để tính MAP nền.</li>
              <li>Nhận tên cột kiểu VitalDB (<code>Solar8000/ART_MBP</code>, <code>ART_SBP</code>…) hoặc <code>map</code>, <code>sbp</code>…</li>
              <li>Nhiễu (ngoài ngưỡng, flush, giá trị đứng yên, nhảy vọt) được loại như lúc huấn luyện.</li>
            </ul>
          </div>
        </div>
      </div>

      {res && (
        <>
          <div className="card span-12" style={{ padding: "12px 16px" }}>
            <div className="chart-title">
              <div className="legend" style={{ gap: 18 }}>
                <span><b className="num">{fmt(res.quality.duration_min)}</b>&nbsp;phút dữ liệu</span>
                <span><b className="num">{fmt(res.time_s.length)}</b>&nbsp;thời điểm dự báo</span>
                <span><b className="num">{res.events.length}</b>&nbsp;đợt tụt đã xảy ra</span>
                <span><b className="num">{res.alarms[h].length}</b>&nbsp;cảnh báo ở mốc {horizonLabel(horizon)}</span>
                <span>MAP nền: <b className="num">{res.quality.baseline_map == null ? "không có" : `${fmt(res.quality.baseline_map)} mmHg`}</b>&nbsp;({res.quality.baseline_source})</span>
              </div>
              <div className="seg" role="group" aria-label="Mốc dự báo">
                {HORIZONS.map((x) => <button key={x} aria-pressed={x === horizon} onClick={() => setHorizon(x)}>{x / 60}′</button>)}
              </div>
            </div>
            {res.quality.warnings.map((w) => <div key={w} className="warn"><span className="i">⚠</span>{w}</div>)}
          </div>

          <div className="card span-4">
            <RiskHero p={p} thr={thr} lv={lv} horizon={horizon} phase={phase} eligible={res.eligible[idx]} />
            <p className="note" style={{ marginTop: 8 }}>Tại {clock(tAt - t0)} từ đầu dữ liệu{idx === lastPredicted(res, h) ? " (thời điểm mới nhất có dự báo)" : ""}. Bấm vào biểu đồ để xem thời điểm khác.</p>
            {lv === "alarm" && (
              <div className="banner" style={{ marginTop: 8 }} role="alert"><span aria-hidden>⚠</span>
                <div><strong>Đang cảnh báo nguy cơ tụt huyết áp trong {horizonLabel(horizon)}</strong>Xác suất vượt ngưỡng {pct(thr)}% ở 2 lần dự báo liên tiếp.</div></div>
            )}
          </div>
          <div className="card span-4">
            <h2>Nguy cơ theo mốc thời gian</h2>
            <p className="sub">Bấm cột để đổi mốc.</p>
            <HorizonProfile tl={res} idx={idx} horizon={horizon} onPick={setHorizon} />
          </div>
          <div className="card span-4">
            <h2>Sinh hiệu tại thời điểm chọn</h2>
            <p className="sub">Sau khi làm sạch nhiễu (giá trị gần nhất ≤ 30 giây).</p>
            <Vitals tl={res} idx={idx} />
          </div>

          <div className="card span-12">
            <div className="chart-title">
              <div><h2>Diễn tiến · {fileName}</h2>
                <p className="sub">Đợt tụt và kết quả từng cảnh báo tính từ chính dữ liệu đã tải (MAP &lt; 65 mmHg ≥ 1 phút).</p></div>
              <div className="legend">
                <span><i className="key-line" style={{ background: "var(--series-1)" }} />Nguy cơ / MAP</span>
                <span><i className="key-rect" style={{ background: "var(--critical-wash)", outline: "1px solid var(--critical)" }} />Đợt tụt</span>
                <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--critical)" /></svg>Cảnh báo đúng</span>
                <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="var(--critical)" strokeWidth="2" /></svg>Cảnh báo sai / chưa rõ</span>
                {res.phase_known && <span><i className="key-rect" style={{ background: "var(--phase-wash)", outline: "1px solid var(--axis)" }} />Trước rạch da</span>}
              </div>
            </div>
            <TimelineChart tl={res} horizon={horizon} cursor={tAt} reveal marker={tAt} phaseKnown={res.phase_known} onSeek={setSel} />
          </div>

          <div className="card span-7">
            <h2>Vì sao nguy cơ ở mức này?</h2>
            <p className="sub">Tại {clock(tAt - t0)}, mốc {horizonLabel(horizon)}.</p>
            <Drivers ex={ex} loading={exLoading} />
          </div>
          <div className="card span-5">
            <h2>Cảnh báo · mốc {horizonLabel(horizon)}</h2>
            <p className="sub">2 lần dự báo liên tiếp vượt ngưỡng; nghỉ 5 phút; “chưa rõ” khi dữ liệu kết thúc trước khi hết mốc.</p>
            <AlarmLog alarms={res.alarms[h]} horizon={horizon} t0={t0} cursor={tAt} reveal onSeek={setSel} />
            <h2 style={{ marginTop: 18 }}>Chất lượng dữ liệu</h2>
            <p className="sub">Tỷ lệ thời điểm dự báo có giá trị gần nhất của từng sinh hiệu.</p>
            {Object.entries(res.quality.coverage).map(([k, v]) => (
              <div key={k} className="cov"><span>{({ map: "MAP", sbp: "HA tâm thu", dbp: "HA tâm trương", hr: "Nhịp tim", spo2: "SpO₂", etco2: "EtCO₂", rr: "Nhịp thở" } as Record<string, string>)[k] ?? k}</span>
                <span className="bar"><i style={{ width: `${v * 100}%` }} /></span><span className="num" style={{ textAlign: "right" }}>{pct(v)}%</span></div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
