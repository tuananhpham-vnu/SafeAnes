import { HORIZONS, MODEL_COLOR, fmt, pct, type ModelInfo, type WardInfo } from "../api";

export function Home({ ward, models, onStart }: { ward: WardInfo | null; models: ModelInfo[]; onStart: () => void }) {
  const nEvents = ward?.beds.reduce((s, b) => s + b.events.length, 0) ?? 0;
  const best = (h: number) => Math.max(...models.map((m) => m.horizons[String(h)].validation.auroc ?? 0));
  const top = models.length ? models.reduce((a, b) => ((b.horizons["300"].validation.auroc ?? 0) > (a.horizons["300"].validation.auroc ?? 0) ? b : a)) : null;
  return (
    <div className="home">
      <section className="home-hero card">
        <div className="home-copy">
          <p className="eyebrow">UC04 · dự báo sớm tụt huyết áp trong mổ</p>
          <h2 className="home-title">Báo trước 5–30 phút nguy cơ MAP xuống dưới 65 mmHg</h2>
          <p className="home-lead">
            Tụt huyết áp trong mổ (MAP &lt; 65 mmHg kéo dài ≥ 1 phút) liên quan tới tổn thương thận, tim sau mổ.
            Hệ thống đọc sinh hiệu, thuốc mê, thở máy và thông tin trước mổ để ước lượng xác suất một đợt tụt bắt đầu trong
            5, 10, 15, 20 hoặc 30 phút tới, và phát cảnh báo khi xác suất vượt ngưỡng hai lần liên tiếp.
          </p>
          <div className="actions">
            <button className="btn primary big" onClick={onStart} disabled={!ward}>Start — vào trạm theo dõi →</button>
            <span className="note">{ward ? "Mô phỏng thời gian thực trên dữ liệu đã ghi" : "Đang kết nối máy chủ…"}</span>
          </div>
        </div>
        <div className="kpis" aria-label="Tóm tắt">
          <div className="kpi"><b className="num">{ward?.beds.length ?? "–"}</b><span>giường theo dõi</span></div>
          <div className="kpi"><b className="num">{nEvents || "–"}</b><span>đợt tụt huyết áp trong các ca</span></div>
          <div className="kpi"><b className="num">{models.length || "–"}</b><span>mô hình chạy song song</span></div>
          <div className="kpi accent"><b className="num">{top ? fmt(top.horizons["300"].validation.auroc, 3) : "–"}</b>
            <span>AUROC 5 phút · {top?.title ?? ""}</span></div>
        </div>
      </section>

      <div className="grid">
        <div className="card span-4">
          <h2>Dữ liệu</h2>
          <ul className="plain">
            <li>VitalDB (Bệnh viện ĐH Quốc gia Seoul): monitor Solar 8000, bơm TCI, máy thở, khí máu, hồ sơ trước mổ.</li>
            <li>Mô hình học trên tập train (2.393 ca), chọn ngưỡng và hiệu chỉnh trên calibration/validation.</li>
            <li>Màn hình này phát lại <b>{ward?.beds.length ?? 50} ca của tập test</b>, như đang theo dõi trực tiếp.</li>
          </ul>
        </div>
        <div className="card span-4">
          <h2>Mô phỏng thời gian thực</h2>
          <ul className="plain">
            <li><b>Mỗi 2 giây</b>: sinh hiệu mới (MAP, nhịp tim, SpO₂) từ bản ghi gốc của monitor.</li>
            <li><b>Mỗi 30 giây</b>: máy chủ chạy cả 4 mô hình trên dữ liệu tới thời điểm đó (71 đặc trưng) và cập nhật cảnh báo.</li>
            <li>Không dùng dữ liệu tương lai. Kết quả của một cảnh báo (đúng/sai) chỉ hiện khi thời gian đã trôi qua.</li>
            <li>Tốc độ 1× là thời gian thật; 5× và 20× để xem nhanh.</li>
          </ul>
        </div>
        <div className="card span-4">
          <h2>Lưu ý</h2>
          <ul className="plain">
            <li>Bản nghiên cứu, <b>không dùng cho quyết định lâm sàng</b>.</li>
            <li>Tập test đã được nhóm dùng để xác nhận các mô hình trước đây, nên số liệu trên test chỉ mang tính mô tả.</li>
            <li>Giải thích “vì sao” là đóng góp thống kê của mô hình, không phải nguyên nhân.</li>
          </ul>
        </div>

        <div className="card span-12">
          <h2>Bốn mô hình · AUROC theo mốc dự báo</h2>
          <p className="sub">Cùng 71 đặc trưng (W = 120 giây). Validation: 248 ca (in đậm: cao nhất mỗi mốc). Test: toàn bộ tập test, chỉ mô tả.</p>
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr><th>Mô hình</th>{HORIZONS.map((h) => <th key={h}>{h / 60} phút</th>)}<th>Độ nhạy / cảnh báo sai mỗi giờ (5′)</th></tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.name}>
                    <td><i className="key-line" style={{ background: MODEL_COLOR[m.name] }} /> {m.title}</td>
                    {HORIZONS.map((h) => {
                      const v = m.horizons[String(h)].validation, t = m.horizons[String(h)].test;
                      const isBest = v.auroc != null && v.auroc >= best(h) - 1e-9;
                      return <td key={h} className="num"><span className={isBest ? "best" : ""}>{fmt(v.auroc, 3)}</span>
                        <span className="muted"> · test {fmt(t?.auroc, 3)}</span></td>;
                    })}
                    <td className="num">{pct(m.horizons["300"].validation.event_sensitivity)}% / {fmt(m.horizons["300"].validation.false_alarms_per_hour, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
