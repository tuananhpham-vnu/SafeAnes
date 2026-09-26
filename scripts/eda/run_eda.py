"""EDA of the full VitalDB cohort for UC04 -> reports/EDA/{REPORT.md, figures/, *.csv}.

  python scripts/eda/run_eda.py [--workers 8] [--limit N] [--reuse]

Heavy intermediate outputs (per-event table, trajectories) go to data/eda_cache (not in git);
--reuse skips recomputing them. Narrative interpretation lives in reports/EDA/notes.md and is
included verbatim; every number in REPORT.md is computed here.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from safeanes import eda, eda_plots as plots

ROOT = Path(__file__).resolve().parents[2]
OUT, FIG, CACHE = ROOT / "reports/EDA", ROOT / "reports/EDA/figures", ROOT / "data/eda_cache"


def md(frame, index=True, floatfmt=2):
    frame = frame.copy()
    if index:
        frame = frame.reset_index()
    cols = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        cells = []
        for v in row:
            if isinstance(v, (float, np.floating)):
                cells.append("–" if np.isnan(v) else f"{v:,.{floatfmt}f}".rstrip("0").rstrip(".") if abs(v) < 1e6 else f"{v:,.0f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    for d in (OUT, FIG, CACHE):
        d.mkdir(parents=True, exist_ok=True)

    manifest = eda.load_manifest(ROOT)
    flow = eda.cohort_flow(manifest)
    demo = eda.demographics(manifest)
    cats = {c: eda.category_table(manifest, c) for c in ("department", "optype", "approach", "asa")}
    avail = eda.track_availability(ROOT, manifest)
    print("track quality...", flush=True)
    quality = eda.track_quality(ROOT, manifest)
    flow.to_csv(OUT / "cohort_flow.csv", index=False)
    demo.to_csv(OUT / "demographics.csv")
    avail.to_csv(OUT / "track_availability.csv", index=False)
    quality.to_csv(OUT / "track_quality.csv", index=False)
    plots.availability(avail, FIG / "track_availability.png")

    if args.reuse and (CACHE / "events.parquet").exists():
        cases = pd.read_parquet(CACHE / "cases.parquet")
        events = pd.read_parquet(CACHE / "events.parquet")
        traj = np.load(CACHE / "trajectories.npy")
    else:
        print("per-case analysis...", flush=True)
        cases, events, traj = eda.analyze_cohort(ROOT, manifest, workers=args.workers, limit=args.limit)
        cases.to_parquet(CACHE / "cases.parquet")
        events.to_parquet(CACHE / "events.parquet")
        np.save(CACHE / "trajectories.npy", traj)
    ev = events[events.kind.eq("event")]
    dec = eda.decompose(events)
    centroids, clustered = eda.cluster_changes(dec)
    print("proxy validation / NIBP / labs...", flush=True)
    proxy = eda.proxy_vs_ev1000(ROOT, manifest)
    nibp_pairs, nibp = eda.nibp_vs_art(ROOT, manifest)
    bleed = eda.labs_bleeding(ROOT, manifest, cases)
    subgroups = eda.subgroup_rates(manifest, cases)

    # ---- numeric summaries
    hours = cases.monitored_h.sum()
    labels = {h: {k: int(cases[f"y{h}_{k}"].sum()) for k in ("pos", "neg", "cens")} for h in (300, 600)}
    ev_summary = pd.Series({
        "số ca phân tích outcome": len(cases), "số bệnh nhân": cases.subjectid.nunique(),
        "giờ theo dõi (lưới 2 s)": round(hours, 1), "số biến cố IOH": len(ev),
        "% ca có ≥1 biến cố": round((cases.n_events > 0).mean() * 100, 1),
        "biến cố / giờ": round(len(ev) / hours, 3),
        "thời lượng biến cố, median [IQR] (s)": eda._iqr(ev.duration_s),
        "MAP thấp nhất, median [IQR]": eda._iqr(ev.min_map),
        "AUC<65, median [IQR] (mmHg·phút)": eda._iqr(ev.auc65_mmHg_min),
        "% biến cố trong 20 phút đầu sau rạch da": round((ev.min_after_opstart <= 20).mean() * 100, 1),
        "% biến cố đủ lịch sử cho y5 (eligible_300)": round(ev.eligible_300.mean() * 100, 1),
        "% biến cố đủ lịch sử cho y10 (eligible_600)": round(ev.eligible_600.mean() * 100, 1),
        "% thời gian MAP<65 (median theo ca)": round(cases.pct_time_map_lt65.median(), 2),
        **{f"y{h//60} dương / âm / censored (decision row eligible)":
           f"{labels[h]['pos']:,} / {labels[h]['neg']:,} / {labels[h]['cens']:,} "
           f"(prevalence {labels[h]['pos'] / max(labels[h]['pos'] + labels[h]['neg'], 1) * 100:.1f}%)" for h in (300, 600)},
    })
    coverage = pd.Series({f"ca có {k}": int(cases[f"has_{k}"].sum()) for k in ["beats", *eda.EXTRA]})
    pattern = dec[dec.kind.isin(["event", "control"])].groupby("kind").pattern.value_counts().unstack(0).fillna(0).astype(int)
    pattern_pct = (pattern / pattern.drop(index="thiếu dữ liệu beat", errors="ignore").sum() * 100).round(1)
    pattern_tab = pattern.join(pattern_pct, rsuffix=" (% có beat)")
    delta_cols = ["dlog_map", "dlog_bt_sv_lz", "dlog_bt_hr", "dlog_bt_svr_lz", "d_bt_ppv", "dlog_bt_dpdt",
                  "d_bis", "d_ppf_ce", "d_rftn_ce", "d_mac", "d_cvp", "dlog_ev_sv", "dlog_ev_svr" if "dlog_ev_svr" in events else "d_ev_svr", "d_ev_svv"]
    delta_cols = [c for c in delta_cols if c in events]
    delta = events.groupby("kind")[delta_cols].agg(["median", "count"]).T.unstack(1)
    delta.columns = [f"{k} {s}" for k, s in delta.columns]
    rho = bleed[["events_per_h", "hb_drop", "intraop_ebl", "intraop_rbc", "intraop_crystalloid", "duration_h"]].corr("spearman")["events_per_h"].round(3)
    proxy_sum = proxy.describe().T[["count", "50%", "25%", "75%"]].round(3) if len(proxy) else pd.DataFrame()

    for name, table in {"events.csv": ev.drop(columns=[c for c in ev if c.startswith(("base_", "late_"))]),
                        "case_summary.csv": cases, "pattern_counts.csv": pattern_tab, "pre_event_changes.csv": delta,
                        "clusters.csv": centroids, "proxy_vs_ev1000.csv": proxy, "bleeding.csv": bleed,
                        "event_summary.csv": ev_summary.to_frame("giá trị")}.items():
        table.to_csv(OUT / name, index=name not in ("events.csv", "case_summary.csv", "proxy_vs_ev1000.csv", "bleeding.csv"))
    (OUT / "nibp_vs_art.json").write_text(json.dumps(nibp, ensure_ascii=False, indent=2), encoding="utf-8")
    for col, table in subgroups.items():
        table.to_csv(OUT / f"subgroup_{col}.csv")

    # ---- figures
    plots.events_overview(events, cases, FIG / "events_overview.png")
    plots.trajectories(events, traj, FIG / "trajectories_core.png",
                       ["map", "hr", "bt_sv_lz", "bt_svr_lz", "bt_co_lz", "bt_ppv", "bt_dpdt", "bt_pp"])
    plots.trajectories(events, traj, FIG / "trajectories_context.png",
                       ["bis", "ppf_ce", "rftn_ce", "mac", "cvp", "peep", "ev_sv", "ev_svr", "ev_svv", "vg_svv",
                        "phen_rate", "nibp_mbp"])
    plots.decomposition(dec, FIG / "decomposition.png")
    if len(centroids):
        plots.clusters(centroids, FIG / "clusters.png")
    if len(proxy):
        plots.proxy_validation(proxy, FIG / "proxy_vs_ev1000.png")
    if len(nibp_pairs):
        plots.nibp_art(nibp_pairs, FIG / "nibp_vs_art.png")
    plots.bleeding(bleed, FIG / "bleeding.png")
    snippets = sorted((ROOT / "data/beats_full/snippets").glob("*.npz"))
    if snippets:
        plots.beat_snippets(snippets, FIG / "beat_snippets.png")

    # ---- report
    notes = (OUT / "notes.md").read_text(encoding="utf-8") if (OUT / "notes.md").exists() else "_(chưa có nhận xét)_"
    beats_index = json.loads((ROOT / "data/beats_full/index.json").read_text(encoding="utf-8"))
    bi = pd.DataFrame(beats_index["cases"]).T
    beat_line = (f"{len(bi):,} ca có beat features; lỗi tải/xử lý {len(beats_index['errors'])} ca; "
                 f"tỉ lệ beat hợp lệ median {bi.valid_frac.astype(float).median():.3f} "
                 f"(ca có <50% beat hợp lệ: {(bi.valid_frac.astype(float) < .5).sum()})")
    report = f"""# EDA VitalDB cho UC04 — dự báo sớm tụt huyết áp và phân tách nguyên nhân

Sinh tự động bởi `scripts/eda/run_eda.py` (module `safeanes.eda`, notebook `notebooks/05_eda_vitaldb.ipynb`).
Insight về feature, nhiễu, quan hệ và feature ảnh hưởng nhất: [INSIGHTS.md](INSIGHTS.md).
Mọi con số dưới đây được tính từ dữ liệu trong `data/`; bảng đầy đủ ở các CSV cùng thư mục.

**Quy tắc phạm vi.** Mô tả cohort và mức sẵn có track dùng toàn bộ ca. Mọi phân tích chạm tới outcome
(biến cố IOH, nhãn, quỹ đạo trước biến cố, tín hiệu cơ chế) **chỉ dùng ca ngoài global test**
(`evaluation_group != unseen_test`) để không mở tập test đã khóa. IOH = MAP < 65 mmHg liên tục ≥ 60 s
theo `Protocol` (docs/PROTOCOL.md). SV/CO/SVR từ waveform là **proxy pulse-contour chưa hiệu chỉnh**
(Liljestrand–Zander), chỉ so sánh tương đối trong cùng ca; không phải đo lường hay nhãn nguyên nhân.

## Nhận xét chính

{notes}

## 1. Cohort

{md(flow, index=False)}

{md(demo)}

Khoa phẫu thuật (eligible):

{md(cats['department'])}

Loại phẫu thuật (top):

{md(cats['optype'])}

Đường mổ / ASA:

{md(cats['approach'])}

{md(cats['asa'])}

## 2. Track liên quan UC04

![availability](figures/track_availability.png)

{md(avail, index=False)}

Waveform SNUADC/ART → beat features (`data/beats_full`): {beat_line}.

![beats](figures/beat_snippets.png)

Chất lượng / chu kỳ lấy mẫu (mẫu ngẫu nhiên tối đa 80 ca mỗi track):

{md(quality, index=False)}

NIBP vs ART MAP (cặp đo NIBP với ART trong 10 s trước): {', '.join(f'{k}: {v}' for k, v in nibp.items())}.

![nibp](figures/nibp_vs_art.png)

## 3. Biến cố IOH (ngoài global test)

{md(ev_summary.to_frame('giá trị'))}

![events](figures/events_overview.png)

Độ phủ tín hiệu bổ trợ trong phạm vi outcome (số ca có ít nhất một giá trị trên lưới):

{md(coverage.to_frame('số ca'))}

## 4. Quỹ đạo trước biến cố

Căn theo mốc khởi phát (−15 → +5 phút); đối chứng = mốc ngẫu nhiên (tối đa 2/ca) có MAP ≥ 65,
không có khởi phát trong ±15 phút. Đường = median, dải = IQR. Tín hiệu tỉ lệ được chuẩn hóa theo nền −15→−10 phút của chính ca.

![core](figures/trajectories_core.png)

![context](figures/trajectories_context.png)

Median thay đổi "trễ (−2→0 phút) so với nền (−15→−10 phút)":

{md(delta)}

## 5. Tín hiệu gợi ý cơ chế (mô tả, không phải nhãn)

Phân rã ΔlogMAP ≈ ΔlogSV + ΔlogHR + ΔlogSVR trên proxy beat; "thành phần chi phối" = thành phần giảm mạnh nhất.
Mẫu hình heuristic chỉ để xem trước phân bố, luật ở `safeanes.eda._pattern`.

![decomposition](figures/decomposition.png)

{md(pattern_tab)}

Cụm k-means (k=5) trên thay đổi trước IOH:

{md(centroids) if len(centroids) else '_không đủ dữ liệu_'}

![clusters](figures/clusters.png)

### Proxy so với EV1000 (ca có EV1000, ngoài global test)

{md(proxy_sum) if len(proxy_sum) else '_chưa có ca_'}

![proxy](figures/proxy_vs_ev1000.png)

### Mất máu

Spearman ρ với số biến cố/giờ: {', '.join(f'{k}: {v}' for k, v in rho.items() if k != 'events_per_h')}.
Ca có Hb trong mổ: {int(bleed.hb_intra_min.notna().sum())}; EBL không có mốc thời gian (chỉ tổng cuối ca).

![bleeding](figures/bleeding.png)

## 6. Phân nhóm

""" + "\n\n".join(f"**{col}**\n\n{md(t)}" for col, t in subgroups.items()) + """

## 7. Giới hạn dữ liệu cho UC04

- Bolus phenylephrine/ephedrine, dịch truyền, máu và EBL chỉ có **tổng cuối ca** (không có mốc thời gian);
  chỉ bơm tiêm điện Orchestra có timeline. Không thể dùng thời điểm can thiệp làm nhãn hoặc biến đầu vào đầy đủ.
- CO/SV/SVR đo bằng thiết bị (EV1000/Vigileo) chỉ có ở một phần nhỏ cohort và có thiên lệch chọn ca (mổ lớn).
- Proxy pulse-contour chưa hiệu chỉnh, nhạy với damping/artefact đường động mạch; cần SQI và kiểm chứng.
- Không có siêu âm tim, không có nhãn cơ chế chuyên gia.
"""
    (OUT / "REPORT.md").write_text(eda.apply_captions(report, eda.load_captions(OUT / "figure_captions.md")), encoding="utf-8")
    print("xong:", OUT / "REPORT.md", flush=True)


if __name__ == "__main__":
    sys.exit(main())
