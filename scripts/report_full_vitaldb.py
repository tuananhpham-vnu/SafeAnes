"""Full-cohort E07 metrics from every case; primary conclusion on global test only."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from safeanes.data import write_json
from safeanes.evaluation import aggregate_cases, window_metrics, quality_gates

ROOT = Path(__file__).resolve().parents[1]
OUT, REPORT = ROOT / "artifacts/E07b", ROOT / "reports/E07"
NAMES = ("tabm_ensemble", "catboost_numeric", "catboost_all", "map", "lightgbm_monotone")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(a, b):
    columns = ["events_detected", "events_eligible", "true_alarms", "false_alarms", "evaluable_seconds"]
    a = a.groupby("subjectid")[columns].sum().sort_index()
    b = b.groupby("subjectid")[columns].sum().sort_index()
    assert a.index.equals(b.index)
    np.testing.assert_allclose(a.events_eligible, b.events_eligible)
    np.testing.assert_allclose(a.evaluable_seconds, b.evaluable_seconds)
    def rates(x):
        def ratio(u, v):
            return np.divide(u, v, out=np.full(np.shape(u), np.nan, dtype=float), where=v > 0)
        return np.column_stack((ratio(x[:, 0], x[:, 1]), ratio(x[:, 2], x[:, 2] + x[:, 3]), ratio(3600*x[:, 3], x[:, 4])))
    rng = np.random.default_rng(20260919)
    weights = rng.multinomial(len(a), np.full(len(a), 1/len(a)), size=5000)
    deltas = rates(weights @ a.to_numpy()) - rates(weights @ b.to_numpy())
    point = (rates(a.sum().to_numpy()[None, :]) - rates(b.sum().to_numpy()[None, :]))[0]
    metrics = {}
    for i, key in enumerate(("event_sensitivity", "alarm_ppv", "false_alarms_per_hour")):
        valid = deltas[:, i][np.isfinite(deltas[:, i])]
        lo, hi = np.quantile(valid, [.025, .975]) if len(valid) else (None, None)
        metrics[key] = {"difference": float(point[i]) if np.isfinite(point[i]) else None,
            "low": float(lo) if lo is not None else None, "high": float(hi) if hi is not None else None,
            "valid_replicates": len(valid)}
    return {"unit": "subjectid", "subjects": len(a), "repeats": 5000, "metrics": metrics}


def main():
    meta = pd.read_csv(REPORT / "cohort_manifest.csv")
    selected = meta[meta.eligible]
    correction = json.loads((OUT / "registration.json").read_text())["identity"]
    assert sha(ROOT / "artifacts/E07/registration.json") == correction["parent_registration_sha256"]
    for path, expected in correction["sources"].items():
        assert sha(OUT / "source" / path) == expected
    identity = json.loads((ROOT / "artifacts/E07/registration.json").read_text())["identity"]
    for path, expected in identity["inputs"].items():
        assert sha(ROOT / path) == expected, f"Registered input changed: {path}"
    for path, expected in identity["sources"].items():
        assert sha(ROOT / "artifacts/E07/source" / path) == expected
    files = {int(p.stem): p for p in (OUT / "cases").glob("*.joblib")}
    assert set(files) == set(selected.caseid), f"Incomplete: {len(files)}/{len(selected)} cases"
    thresholds = json.loads((OUT / "thresholds.json").read_text())
    groups = selected.set_index("caseid").evaluation_group.to_dict()
    subject_ids = selected.set_index("caseid").subjectid.to_dict()
    frames, case_rows, audits, index = [], [], [], []
    for caseid, path in sorted(files.items()):
        item = joblib.load(path)
        assert item["group"] == groups[caseid]
        assert sha(ROOT / item["preprocessed_path"]) == item["preprocessed_sha256"]
        frame = item["frame"]
        assert frame.caseid.eq(caseid).all() and frame.subjectid.eq(subject_ids[caseid]).all()
        frame["group"] = groups[caseid]
        for name in NAMES:
            p = item["probabilities"][name]
            assert p.shape == (len(frame), 2) and np.isfinite(p[frame.eligible]).all()
            for i, h in enumerate((300, 600)):
                frame[f"p_{name}_{h}"] = p[:, i]
        assert np.all(item["probabilities"]["tabm_ensemble"][frame.eligible, 1] >= item["probabilities"]["tabm_ensemble"][frame.eligible, 0])
        frames.append(frame)
        case_rows.extend([{**row, "group": groups[caseid]} for row in item["case_stats"]])
        audits.append({**item["audit"], "group": groups[caseid]})
        index.append({"caseid": caseid, "result_sha256": sha(path), "preprocessed_sha256": item["preprocessed_sha256"]})
    windows = pd.concat(frames, ignore_index=True)
    del frames
    cases = pd.DataFrame(case_rows)
    cases[["caseid", "subjectid", "horizon"]] = cases[["caseid", "subjectid", "horizon"]].astype(int)
    assert not cases.duplicated(["caseid", "model", "horizon"]).any()
    assert len(cases) == len(selected) * 10
    cases.to_csv(REPORT / "case_metrics.csv.gz", index=False)
    pd.DataFrame(audits).to_csv(REPORT / "quality.csv", index=False)
    pd.DataFrame(index).to_csv(REPORT / "artifact_index.csv", index=False)
    rows, support = [], []
    scopes = ["unseen_test", "unseen_validation", "unseen_calibration", "unseen_train", "development_seen", "all_eligible"]
    for scope in scopes:
        w = windows if scope == "all_eligible" else windows[windows.group.eq(scope)]
        c = cases if scope == "all_eligible" else cases[cases.group.eq(scope)]
        for h in (300, 600):
            known = w.eligible & w[f"y_{h}"].ge(0)
            representative = c[c.model.eq("tabm_ensemble") & c.horizon.eq(h)]
            aggregate = aggregate_cases(representative)
            support.append({"scope": scope, "horizon": h, "patients": int(w.subjectid.nunique()),
                "cases": int(w.caseid.nunique()), "windows": len(w), "known_windows": int(known.sum()),
                "positive_windows": int(w.loc[known, f"y_{h}"].sum()), "events_all": aggregate["events_all"],
                "events_eligible": aggregate["events_eligible"], "evaluable_hours": aggregate["evaluable_hours"]})
            for name in NAMES:
                m = window_metrics(w.loc[known, f"y_{h}"], w.loc[known, f"p_{name}_{h}"])
                m.update(aggregate_cases(c[c.model.eq(name) & c.horizon.eq(h)]))
                m["threshold"] = thresholds[f"{name}_{h}"]
                gates = quality_gates(m, h)
                rows.append({"scope": scope, "model": name, "horizon": h, **m, "all_gates": gates["all_point_targets_met"]})
    table = pd.DataFrame(rows)
    table.to_csv(REPORT / "comparison.csv", index=False)
    pd.DataFrame(support).to_csv(REPORT / "data_support.csv", index=False)
    comparisons = []
    for h in (300, 600):
        a = cases[cases.group.eq("unseen_test") & cases.model.eq("tabm_ensemble") & cases.horizon.eq(h)]
        for name in ("catboost_numeric", "catboost_all", "map"):
            b = cases[cases.group.eq("unseen_test") & cases.model.eq(name) & cases.horizon.eq(h)]
            comparisons.append({"horizon": h, "model": "tabm_ensemble", "comparator": name,
                "primary": name == "catboost_numeric", **paired(a, b)})
    write_json(REPORT / "paired_final_test.json", comparisons)
    write_json(REPORT / "verification.json", {"completed_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_cases": len(meta), "eligible_cases": len(selected), "completed_cases": len(files),
        "exclusions": meta.loc[~meta.eligible, "exclusion_reason"].value_counts().to_dict(),
        "source_input_hashes": "passed", "result_preprocessed_hashes": "passed", "finite_predictions_on_eligible": "passed",
        "subject_groups": "fixed before unseen track download", "retuning": False, "tests_executed": False,
        "final_test_opened": True, "primary_group": "unseen_test", "report_source_sha256": sha(Path(__file__))})
    render(table, support, comparisons, meta)
    print(table[table.scope.eq("unseen_test")].to_string(index=False), flush=True)


def render(table, support, comparisons, meta):
    f = lambda x: "N/A" if x is None or pd.isna(x) else f"{x:.3f}"
    lines = ["# E07 — toàn bộ VitalDB đủ điều kiện", "",
        f"Đã audit {len(meta):,} ca và xử lý đủ {int(meta.eligible.sum()):,} ca đủ điều kiện protocol v1. "
        "Model/ngưỡng đã khóa trước khi tải và đánh giá cohort mới. Final test được mở theo yêu cầu đánh giá toàn bộ VitalDB; "
        "không train lại, recalibrate hoặc tune. Không chạy tests/. "
        "[Sửa thực thi E07b](../../docs/experiments/E07_CORRECTION.md) giữ đường input CSV giống E05/E06; "
        "bản nháp feature trực tiếp được giữ riêng và không dùng kết luận.", "",
        "[Plan khóa trước chạy](../../docs/experiments/E07_PLAN.md) · [Metrics](comparison.csv) · [Cỡ mẫu](data_support.csv) · "
        "[CI ghép cặp final test](paired_final_test.json) · [Kiểm chứng](verification.json)", "",
        "## Loại trừ", "", "| Lý do | Ca |", "|---|---:|"]
    for reason, count in meta.loc[~meta.eligible, "exclusion_reason"].value_counts().items():
        lines.append(f"| {reason} | {count} |")
    lines += ["", "## Cỡ mẫu từng nhóm", "", "| Nhóm | Phút | Bệnh nhân | Ca | Event đủ điều kiện | Giờ |", "|---|---:|---:|---:|---:|---:|"]
    for row in support:
        lines.append(f"| {row['scope']} | {row['horizon']//60} | {row['patients']} | {row['cases']} | {row['events_eligible']} | {row['evaluable_hours']:.2f} |")
    for scope in table.scope.unique():
        lines += ["", f"## {scope}", "", "| Model | Phút | AUROC | AP | Detect/event | Recall | PPV | FA/giờ | ECE | Đủ gate |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for row in table[table.scope.eq(scope)].to_dict("records"):
            lines.append("| " + " | ".join([row["model"], str(row["horizon"]//60), f(row["auroc"]), f(row["average_precision"]),
                f"{row['events_detected']}/{row['events_eligible']}", f(row["event_sensitivity"]), f(row["alarm_ppv"]),
                f(row["false_alarms_per_hour"]), f(row["ece"]), str(row["all_gates"])]) + " |")
    lines += ["", "## Chênh lệch final test — 5.000 bootstrap ghép cặp bệnh nhân", "",
        "Δ = TabM ensemble − đối chứng; recall/PPV dương tốt hơn, FA/giờ âm tốt hơn. Primary là CatBoost numeric; "
        "các đối chứng khác là secondary. CI95% percentile, có điều kiện trên model/ngưỡng đã fit; không hiệu chỉnh nhiều so sánh.", "",
        "| Phút | Đối chứng | Δ recall [95% CI] | Δ PPV [95% CI] | Δ FA/giờ [95% CI] |", "|---|---|---|---|---|"]
    for c in comparisons:
        cells = [str(c["horizon"]//60), c["comparator"]]
        for key in ("event_sensitivity", "alarm_ppv", "false_alarms_per_hour"):
            m = c["metrics"][key]
            cells.append(f"{f(m['difference'])} [{f(m['low'])}, {f(m['high'])}]")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Giới hạn diễn giải", "",
        "- `unseen_test` là nhóm xác nhận chính, giữ theo subjectid trước E01. `development_seen` gồm mọi ca "
        "của bệnh nhân đã dùng ở development300, kể cả lần mổ mới của cùng người. `all_eligible` chứa dữ liệu phát triển, "
        "nên là thống kê mô tả toàn cohort, không phải test độc lập.",
        "- Các nhóm unseen train/calibration/validation chưa dùng bởi các model đã khóa, nhưng không gộp thay kết luận primary final test.",
        "- Những ca không đủ điều kiện không có dự báo hợp lệ với pipeline hiện tại. Không thay MAP động mạch bằng NIBP "
        "hoặc báo điểm toàn bộ 6.388 ca như thể không có loại trừ.",
        "- Eligibility/censoring còn loại một phần thời gian/sự kiện; FA/giờ theo exposure 30 giây của protocol v1. "
        "Coverage gate là eligible/scheduled, khác availability của model.",
        "- Đây là hồi cứu một nguồn VitalDB; chưa kiểm định ngoài hoặc tiến cứu. Sau E07, final test đã được xem; "
        "không tune bằng kết quả này rồi tiếp tục gọi cùng nhóm là final test chưa xem."]
    (REPORT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
