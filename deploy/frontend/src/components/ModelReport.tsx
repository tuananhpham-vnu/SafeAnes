import { useState } from "react";
import { HORIZONS, fmt, horizonLabel, pct, type ModelInfo, type ReportRow } from "../api";

const NAMES: Record<string, string> = {
  map_threshold: "Ngưỡng MAP", map_logistic: "Hồi quy logistic MAP", lgbm_numeric: "LightGBM chỉ số",
  lgbm_wave: "LightGBM + sóng", lgbm_context: "LightGBM + ngữ cảnh ca", catboost_numeric: "CatBoost chỉ số",
  dl_conv_tf: "Conv1D + Transformer", tabm_numeric: "TabM", tabm_context: "TabM + ngữ cảnh",
};
const ci = (v: number, lo?: number, hi?: number, p = true) =>
  `${p ? `${pct(v, 1)}%` : fmt(v, 3)}${lo != null && hi != null && isFinite(lo) ? ` [${p ? pct(lo, 1) : fmt(lo, 3)}–${p ? pct(hi, 1) : fmt(hi, 3)}]` : ""}`;

export function ModelReport({ models, report }: { models: ModelInfo[]; report: Record<string, ReportRow[]> }) {
  const [h, setH] = useState<number>(300);
  const cmp = (report.compare_interim ?? []).filter((r) => Number(r.horizon) === h && (r.window == null || Number(r.window) === 120));
  const thr = (report.phase_thresholds_W120 ?? []).filter((r) => Number(r.horizon) === h && r.phase === "all");
  const order = ["map_threshold", "map_logistic", "lgbm_numeric", "lgbm_wave", "catboost_numeric", "dl_conv_tf", "lgbm_context"];
  const byModel = (v: string, m: string, ph: string) => cmp.find((r) => r.version === v && r.model === m && r.phase === ph);
  return (
    <div className="grid">
      <div className="card span-12">
        <div className="chart-title">
          <div><h2>Chỉ số trên tập validation (248 ca, 245 bệnh nhân) — W = 120 giây</h2>
            <p className="sub">Ngưỡng chọn để ≤ 1 cảnh báo sai/giờ; độ nhạy tính theo đợt tụt; khoảng tin cậy 95% bootstrap theo bệnh nhân.</p></div>
          <div className="seg" role="group" aria-label="Mốc dự báo">
            {HORIZONS.map((x) => <button key={x} aria-pressed={x === h} onClick={() => setH(x)}>{x / 60}′</button>)}
          </div>
        </div>
        <table>
          <thead><tr><th>Mô hình</th><th>AUROC</th><th>AUPRC</th><th>Độ nhạy theo đợt</th><th>Cảnh báo sai/giờ</th><th>PPV cảnh báo</th><th>Báo trước (trung vị)</th><th>Ngưỡng</th></tr></thead>
          <tbody>
            {models.map((m) => {
              const x = m.horizons[String(h)], mt = x.metrics;
              return (
                <tr key={m.name} className={m.name === "lgbm_context" ? "hl" : ""}>
                  <td>{m.title}</td><td>{ci(mt.auroc, mt.auroc_lo, mt.auroc_hi, false)}</td><td>{fmt(mt.auprc, 3)}</td>
                  <td>{ci(mt.event_sensitivity, mt.event_sensitivity_lo, mt.event_sensitivity_hi)}</td>
                  <td>{fmt(mt.false_alarms_per_hour, 2)}</td><td>{pct(mt.alarm_ppv, 1)}%</td>
                  <td>{fmt(mt.lead_median_s / 60, 1)} phút</td><td>{pct(x.threshold, 1)}%</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card span-7">
        <h2>Theo giai đoạn của ca, so với dữ liệu v2 — mốc {horizonLabel(h)}</h2>
        <p className="sub">v2 chỉ có phần trong mổ. v3 thêm giai đoạn sau khởi mê, trước rạch da (≈ 13% thời gian, ≈ 1/3 số đợt tụt).</p>
        {cmp.length ? (
          <table>
            <thead><tr><th>Mô hình</th><th>v2 · trong mổ</th><th>v3 · trong mổ</th><th>v3 · trước rạch da</th></tr></thead>
            <tbody>
              {order.map((m) => {
                const a = byModel("v2", m, "surgery"), b = byModel("v3", m, "surgery"), c = byModel("v3", m, "pre_incision");
                if (!a && !b) return null;
                const cell = (r?: ReportRow) => r ? `${fmt(Number(r.auroc), 3)} · ${pct(Number(r.event_sensitivity), 1)}% · ${fmt(Number(r.false_alarms_per_hour), 2)}` : "–";
                return <tr key={m} className={m === "lgbm_context" ? "hl" : ""}><td>{NAMES[m] ?? m}</td><td>{cell(a)}</td><td>{cell(b)}</td><td>{cell(c)}</td></tr>;
              })}
            </tbody>
          </table>
        ) : <div className="empty">Chưa có bảng so sánh.</div>}
        <p className="note">Mỗi ô: AUROC · độ nhạy theo đợt · cảnh báo sai/giờ (ngưỡng chung của cả ca).</p>
      </div>

      <div className="card span-5">
        <h2>Ngưỡng có giữ được trên dữ liệu khác?</h2>
        <p className="sub">Chọn ngưỡng trên tập calibration (như khi triển khai), báo cáo trên validation.</p>
        {thr.length ? (
          <table>
            <thead><tr><th>Mô hình</th><th>Ngưỡng chọn trên validation</th><th>Chọn trên calibration</th></tr></thead>
            <tbody>
              {[...new Set(thr.map((r) => String(r.model)))].map((m) => {
                const v = thr.find((r) => r.model === m && r.policy === "single_val"), c = thr.find((r) => r.model === m && r.policy === "single_cal");
                return <tr key={m} className={m === "lgbm_context" ? "hl" : ""}><td>{NAMES[m] ?? m}</td>
                  <td>{pct(Number(v?.event_sensitivity), 1)}%</td><td>{pct(Number(c?.event_sensitivity), 1)}%</td></tr>;
              })}
            </tbody>
          </table>
        ) : <div className="empty">Chỉ có cho mốc 5′ và 10′.</div>}
      </div>

      <div className="card span-12">
        <h2>So với các hệ thống cùng loại</h2>
        <p className="sub">Bối cảnh thiết kế; số liệu không so trực tiếp được vì khác dữ liệu, định nghĩa nhãn và cách đánh giá.</p>
        <div className="peer">
          <div><h3>Acumen HPI (Edwards)</h3><ul>
            <li>Một chỉ số 0–100 từ sóng động mạch, cảnh báo khi ≥ 85</li>
            <li>Màn hình phụ: tiền tải, co bóp, hậu tải</li>
            <li>RCT HYPE giảm thời gian tụt; Mulder 2024: HPI gần tương đương ngưỡng MAP</li></ul></div>
          <div><h3>Prescience (Lundberg 2018)</h3><ul>
            <li>Dự báo biến cố trong gây mê, giải thích từng thời điểm bằng SHAP</li>
            <li>Bác sĩ dự đoán tốt hơn khi có giải thích</li>
            <li>Dự báo thiếu oxy, không phải tụt huyết áp</li></ul></div>
          <div><h3>SafeAnes (bản này)</h3><ul>
            <li>Xác suất đã hiệu chỉnh cho 5 mốc 5–30 phút, không phải một chỉ số</li>
            <li>Gồm cả giai đoạn sau khởi mê; ngữ cảnh ca tăng AUROC có ý nghĩa ở mọi mốc</li>
            <li>Cảnh báo theo đúng chính sách đã đánh giá (2 lần liên tiếp, nghỉ 5 phút), đánh giá theo đợt tụt</li></ul></div>
        </div>
      </div>
    </div>
  );
}
