import { useEffect, useMemo, useRef, useState } from "react";
import {
  HORIZONS, alarmOnAt, api, clock, horizonLabel, indexAt, level,
  type CaseItem, type Explanation, type Horizon, type ModelInfo, type ModelName, type ReportRow, type Timeline,
} from "./api";
import { InferPage } from "./components/InferPage";
import { ModelReport } from "./components/ModelReport";
import { AlarmLog, Drivers, HorizonProfile, RiskHero, Vitals } from "./components/Panels";
import { TimelineChart } from "./components/TimelineChart";

const SPEEDS = [{ k: 30, l: "30×" }, { k: 120, l: "2′/s" }, { k: 600, l: "10′/s" }];

export default function App() {
  const [tab, setTab] = useState<"infer" | "monitor" | "model">(() =>
    location.hash === "#model" ? "model" : location.hash === "#replay" ? "monitor" : "infer");
  useEffect(() => {
    history.replaceState(null, "", tab === "model" ? "#model" : tab === "monitor" ? "#replay" : location.pathname + location.search);
  }, [tab]);
  const [theme, setTheme] = useState<"light" | "dark" | null>(() => {
    try { return (localStorage.getItem("theme") as "light" | "dark" | null) ?? null; } catch { return null; }
  });
  const [cases, setCases] = useState<CaseItem[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [report, setReport] = useState<Record<string, ReportRow[]>>({});
  const [caseId, setCaseId] = useState<number | null>(null);
  const [model, setModel] = useState<ModelName>("lgbm_context");
  const [compareOn, setCompareOn] = useState(false);
  const [horizon, setHorizon] = useState<Horizon>(300);
  const [tl, setTl] = useState<Timeline | null>(null);
  const [base, setBase] = useState<Timeline | null>(null);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(120);
  const [reveal, setReveal] = useState(false);
  const [ex, setEx] = useState<Explanation | null>(null);
  const [exLoading, setExLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
    try { theme ? localStorage.setItem("theme", theme) : localStorage.removeItem("theme"); } catch { /* private mode */ }
  }, [theme]);

  useEffect(() => {
    Promise.all([api.cases(), api.models(), api.report()])
      .then(([c, m, r]) => {
        setCases(c); setModels(m); setReport(r);
        const first = c.find((x) => x.n_events_pre_incision > 0 && x.n_events > x.n_events_pre_incision) ?? c[0];
        if (first) setCaseId(first.caseid);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (caseId == null) return;
    let live = true;
    setPlaying(false);
    api.timeline(caseId, model).then((t) => {
      if (!live) return;
      setTl(t);
      setCursor((c) => (tl?.caseid === caseId ? c : Math.min(t.surgery_start_s + 3600, t.time_s[t.time_s.length - 1])));
    }).catch((e) => setError(String(e)));
    const other: ModelName = model === "lgbm_context" ? "lgbm_numeric" : "lgbm_context";
    api.timeline(caseId, other).then((t) => live && setBase(t)).catch(() => setBase(null));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId, model]);

  // playback: one 30 s step every 30/speed seconds of wall time
  const raf = useRef<number | null>(null);
  useEffect(() => {
    if (!playing || !tl) return;
    let last = performance.now();
    const end = tl.time_s[tl.time_s.length - 1];
    const step = (now: number) => {
      const dt = (now - last) / 1000;
      last = now;
      setCursor((c) => {
        const n = c + dt * speed;
        if (n >= end) { setPlaying(false); return end; }
        return n;
      });
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [playing, speed, tl]);

  const idx = tl ? indexAt(tl.time_s, cursor) : 0;
  const tAt = tl ? tl.time_s[idx] : 0;

  // explanation at the cursor (throttled while playing; stale answers are dropped)
  useEffect(() => {
    if (!tl || caseId == null) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      setExLoading(true);
      api.explain(caseId, model, tAt, horizon, ctrl.signal)
        .then((e) => { setEx(e); setExLoading(false); })
        .catch((e) => { if (e.name !== "AbortError") { setExLoading(false); setError(String(e)); } });
    }, playing ? 600 : 120);
    return () => { clearTimeout(id); ctrl.abort(); };
  }, [tl, caseId, model, tAt, horizon, playing]);

  const h = String(horizon);
  const p = tl ? tl.risk[h][idx] : null;
  const thr = tl ? tl.thresholds[h] : 0.5;
  const alarmOn = useMemo(() => (tl ? alarmOnAt(tl, h, idx) : false), [tl, h, idx]);
  const lv = level(p, thr, alarmOn);
  const phase = tl && tAt <= tl.surgery_start_s ? "pre_incision" : "surgery";
  const t0 = tl?.time_s[0] ?? 0, tEnd = tl ? tl.time_s[tl.time_s.length - 1] : 0;
  const nextEvent = tl?.events.find((e) => e.onset_s > tAt);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden>
            <svg width="20" height="20" viewBox="0 0 32 32"><path d="M3 18h7l3-8 4 13 3-7h9" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
          <div><h1>SafeAnes Monitor</h1><p>Dự báo tụt huyết áp trong mổ · bản demo nghiên cứu, không dùng cho lâm sàng</p></div>
        </div>
        <nav className="tabs" role="tablist">
          <button className="tab" role="tab" aria-selected={tab === "infer"} onClick={() => setTab("infer")}>Dự báo</button>
          <button className="tab" role="tab" aria-selected={tab === "monitor"} onClick={() => setTab("monitor")}>Phát lại ca mẫu</button>
          <button className="tab" role="tab" aria-selected={tab === "model"} onClick={() => setTab("model")}>Mô hình & kết quả</button>
        </nav>
        <button className="icon-btn" aria-label="Đổi giao diện sáng/tối" title="Sáng / tối"
          onClick={() => setTheme(document.documentElement.matches("[data-theme=dark]") ||
            (!theme && matchMedia("(prefers-color-scheme: dark)").matches) ? "light" : "dark")}>◐</button>
      </header>

      {error && <div className="banner" role="alert"><div><strong>Không kết nối được backend</strong>{error}</div></div>}

      {tab === "infer" ? <InferPage models={models} /> : tab === "model" ? <ModelReport models={models} report={report} /> : (
        <>
          <section className="controls" aria-label="Điều khiển">
            <div className="controls-row">
            <label className="field">Ca
              <select value={caseId ?? ""} onChange={(e) => setCaseId(Number(e.target.value))}>
                {cases.map((c) => (
                  <option key={c.caseid} value={c.caseid}>
                    #{c.caseid} · {c.duration_min}′ · {c.n_events} đợt tụt{c.n_events_pre_incision ? ` (${c.n_events_pre_incision} trước rạch da)` : ""}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">Mô hình
              <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
                {models.map((m) => <option key={m.name} value={m.name}>{m.title}</option>)}
              </select>
            </label>
            <div className="field">Mốc
              <div className="seg" role="group" aria-label="Mốc dự báo">
                {HORIZONS.map((x) => <button key={x} aria-pressed={x === horizon} onClick={() => setHorizon(x)}>{x / 60}′</button>)}
              </div>
            </div>
            <label className="toggle spacer"><input type="checkbox" checked={reveal} onChange={(e) => setReveal(e.target.checked)} />Xem hồi cứu cả ca</label>
            <label className="toggle"><input type="checkbox" checked={compareOn} onChange={(e) => setCompareOn(e.target.checked)} />So với baseline</label>
            </div>
            <div className="controls-row">
            <div className="player">
              <button className="play" aria-label={playing ? "Tạm dừng" : "Phát"} onClick={() => {
                if (!tl) return;
                if (cursor >= tEnd) setCursor(t0);
                setPlaying((x) => !x);
              }}>{playing ? "❚❚" : "▶"}</button>
              <input type="range" min={t0} max={tEnd} step={30} value={cursor} aria-label="Thời điểm"
                onChange={(e) => { setPlaying(false); setCursor(Number(e.target.value)); }} />
              <span className="clock">{clock(tAt - t0)}</span>
              <div className="seg" role="group" aria-label="Tốc độ phát">
                {SPEEDS.map((s) => <button key={s.k} aria-pressed={speed === s.k} onClick={() => setSpeed(s.k)}>{s.l}</button>)}
              </div>
            </div>
            </div>
          </section>

          {!tl ? <div className="card empty">Đang tải ca…</div> : (
            <div className="grid">
              <div className="card span-4">
                <RiskHero p={p} thr={thr} lv={lv} horizon={horizon} phase={phase} eligible={tl.eligible[idx]} />
                {lv === "alarm" && (
                  <div className="banner" style={{ marginTop: 12 }} role="alert">
                    <span aria-hidden>⚠</span>
                    <div><strong>Cảnh báo nguy cơ tụt huyết áp trong {horizonLabel(horizon)}</strong>
                      Xác suất vượt ngưỡng ở 2 lần dự báo liên tiếp. Xem phần “Vì sao” để biết yếu tố đang đẩy nguy cơ lên.</div>
                  </div>
                )}
              </div>
              <div className="card span-4">
                <h2>Nguy cơ theo mốc thời gian</h2>
                <p className="sub">Xác suất có đợt tụt bắt đầu trong 5–30 phút tới. Bấm cột để đổi mốc.</p>
                <HorizonProfile tl={tl} idx={idx} horizon={horizon} onPick={setHorizon} />
              </div>
              <div className="card span-4">
                <h2>Sinh hiệu</h2>
                <p className="sub">Giá trị gần nhất (Solar 8000, lưới 2 giây).</p>
                <Vitals tl={tl} idx={idx} />
                {reveal && nextEvent && (
                  <p className="note" style={{ marginTop: 10 }}>Đợt tụt kế tiếp sau {Math.round((nextEvent.onset_s - tAt) / 60)} phút (MAP thấp nhất {Math.round(nextEvent.min_map)} mmHg).</p>
                )}
              </div>

              <div className="card span-12">
                <div className="chart-title">
                  <div><h2>Diễn tiến ca #{tl.caseid}</h2>
                    <p className="sub">{reveal ? "Hồi cứu: hiện toàn bộ ca và kết quả của từng cảnh báo." : "Phát lại như đang theo dõi: chỉ thấy dữ liệu tới thời điểm hiện tại."}</p></div>
                  <div className="legend">
                    <span><i className="key-line" style={{ background: "var(--series-1)" }} />{models.find((m) => m.name === model)?.title ?? model}</span>
                    {compareOn && base && <span><i className="key-line" style={{ background: "var(--series-2)" }} />{models.find((m) => m.name === base.model)?.title}</span>}
                    <span><i className="key-rect" style={{ background: "var(--critical-wash)", outline: "1px solid var(--critical)" }} />Đợt tụt (MAP &lt; 65 ≥ 1′)</span>
                    <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="var(--critical)" /></svg>Cảnh báo đúng</span>
                    <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="var(--critical)" strokeWidth="2" /></svg>Cảnh báo sai / chưa rõ</span>
                    <span><i className="key-rect" style={{ background: "var(--phase-wash)", outline: "1px solid var(--axis)" }} />Trước rạch da</span>
                  </div>
                </div>
                <TimelineChart tl={tl} compare={compareOn ? base : null} horizon={horizon} cursor={tAt} reveal={reveal}
                  onSeek={(t) => { setPlaying(false); setCursor(t); }} />
              </div>

              <div className="card span-7">
                <h2>Vì sao nguy cơ ở mức này?</h2>
                <p className="sub">Yếu tố đẩy nguy cơ lên/xuống tại {clock(tAt - t0)}, mốc {horizonLabel(horizon)}.</p>
                <Drivers ex={ex} loading={exLoading} />
              </div>
              <div className="card span-5">
                <h2>Nhật ký cảnh báo · mốc {horizonLabel(horizon)}</h2>
                <p className="sub">Phát khi 2 lần dự báo liên tiếp vượt ngưỡng; không lặp lại trong 5 phút và tới khi nguy cơ xuống dưới ngưỡng.</p>
                <AlarmLog alarms={tl.alarms[h]} horizon={horizon} t0={t0} cursor={tAt} reveal={reveal} onSeek={(t) => { setPlaying(false); setCursor(t); }} />
              </div>
            </div>
          )}
        </>
      )}
      <footer className="foot">UC04 · VitalDB (tập validation) · mô hình LightGBM W = 120 giây, samples v3 · không thay thế đánh giá lâm sàng</footer>
    </div>
  );
}
