import { useEffect, useState } from "react";
import { HORIZONS, MODEL_COLOR, MODEL_ORDER, api, type Horizon, type ModelInfo, type ModelName, type WardInfo } from "./api";
import { BedDetail } from "./components/BedDetail";
import { Home } from "./components/Home";
import { WardGrid } from "./components/WardGrid";
import { useWard } from "./useWard";

type Route = { page: "home" } | { page: "ward" } | { page: "bed"; bed: number };

function parse(hash: string): Route {
  const m = hash.match(/^#\/bed\/(\d+)/);
  if (m) return { page: "bed", bed: Number(m[1]) };
  return hash.startsWith("#/ward") ? { page: "ward" } : { page: "home" };
}

const SPEEDS = [{ k: 1, l: "1×" }, { k: 5, l: "5×" }, { k: 20, l: "20×" }];

export default function App() {
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  useEffect(() => {
    const on = () => { setRoute(parse(location.hash)); window.scrollTo(0, 0); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const go = (hash: string) => { location.hash = hash; };

  const [theme, setTheme] = useState<"light" | "dark" | null>(() => {
    try { return (localStorage.getItem("theme") as "light" | "dark" | null) ?? null; } catch { return null; }
  });
  useEffect(() => {
    if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
    try { theme ? localStorage.setItem("theme", theme) : localStorage.removeItem("theme"); } catch { /* private mode */ }
  }, [theme]);

  const [ward, setWard] = useState<WardInfo | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    Promise.all([api.ward(), api.models()])
      .then(([w, m]) => {
        setWard(w);
        setModels(MODEL_ORDER.map((n) => m.find((x) => x.name === n)).filter((x): x is ModelInfo => !!x));
      })
      .catch((e) => setError(String(e)));
  }, []);

  const [model, setModel] = useState<ModelName>("lgbm_v8");
  const [horizon, setHorizon] = useState<Horizon>(300);
  const [speed, setSpeed] = useState(1);
  const live = route.page !== "home";
  const st = useWard(speed, live && !!ward);

  return (
    <div className="app">
      <header className="topbar">
        <button className="brand" onClick={() => go("#/")} aria-label="Về trang giới thiệu">
          <div className="brand-mark" aria-hidden>
            <svg width="20" height="20" viewBox="0 0 32 32"><path d="M3 18h7l3-8 4 13 3-7h9" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /></svg>
          </div>
          <div><h1>SafeAnes Monitor</h1><p>Dự báo tụt huyết áp trong mổ · bản nghiên cứu, không dùng cho lâm sàng</p></div>
        </button>
        {live && (
          <span className={`live${st.connected ? " on" : ""}`} title={st.connected ? "Đang nhận dữ liệu" : "Mất kết nối, đang thử lại"}>
            <i />{st.connected ? "LIVE" : "mất kết nối"}
          </span>
        )}
        <button className="icon-btn" aria-label="Đổi giao diện sáng/tối" title="Sáng / tối"
          onClick={() => setTheme(document.documentElement.matches("[data-theme=dark]") ||
            (!theme && matchMedia("(prefers-color-scheme: dark)").matches) ? "light" : "dark")}>◐</button>
      </header>

      {error && <div className="banner" role="alert"><div><strong>Không kết nối được máy chủ</strong>{error}</div></div>}

      {live && models.length > 0 && (
        <section className="controls" aria-label="Điều khiển">
          <div className="controls-row">
            <div className="field">Mô hình
              <div className="seg" role="group" aria-label="Mô hình">
                {models.map((m) => (
                  <button key={m.name} aria-pressed={m.name === model} onClick={() => setModel(m.name)}>
                    <i className="key-line" style={{ background: m.name === model ? "#fff" : MODEL_COLOR[m.name], marginRight: 6 }} />{m.title}
                  </button>
                ))}
              </div>
            </div>
            <div className="field">Mốc
              <div className="seg" role="group" aria-label="Mốc dự báo">
                {HORIZONS.map((x) => <button key={x} aria-pressed={x === horizon} onClick={() => setHorizon(x)}>{x / 60}′</button>)}
              </div>
            </div>
            <div className="field spacer">Tốc độ
              <div className="seg" role="group" aria-label="Tốc độ phát">
                {SPEEDS.map((s) => <button key={s.k} aria-pressed={speed === s.k} onClick={() => setSpeed(s.k)}>{s.l}</button>)}
              </div>
            </div>
          </div>
        </section>
      )}

      {route.page === "home" && <Home ward={ward} models={models} onStart={() => go("#/ward")} />}
      {route.page === "ward" && ward && (
        <WardGrid beds={ward.beds} snaps={st.beds} buffers={st.buffers} models={models} model={model} horizon={horizon}
          onOpen={(b) => go(`#/bed/${b}`)} />
      )}
      {route.page === "bed" && ward && models.length > 0 && (
        <BedDetail key={route.bed} bed={route.bed} v={st.v} snap={st.beds.get(route.bed)} buf={st.buffers.get(route.bed)}
          models={models} model={model} horizon={horizon} onModel={setModel} onHorizon={setHorizon} onBack={() => go("#/ward")} />
      )}
      <footer className="foot">UC04 · VitalDB, tập test (mô tả) · 4 mô hình trên 71 đặc trưng features_v8, W = 120 giây · không thay thế đánh giá lâm sàng</footer>
    </div>
  );
}
