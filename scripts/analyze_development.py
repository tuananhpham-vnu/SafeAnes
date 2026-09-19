"""Descriptive subgroup/lead-time analysis of predeclared E05 primary models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from safeanes.config import Protocol
from safeanes.data import write_json
from safeanes.evaluation import evaluate_predictions, bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]


def lead_interval(frame, events, alarms, horizon, repeats=200):
    subject_for_case = frame.drop_duplicates("caseid").set_index("caseid").subjectid.to_dict()
    eligible = set(zip(events.loc[events[f"eligible_{horizon}"], "caseid"], events.loc[events[f"eligible_{horizon}"], "onset"]))
    best = {}
    for a in alarms:
        if a["outcome"] == "true":
            key = (a["caseid"], a["time"] + a["lead_seconds"])
            if key in eligible:
                best[key] = max(best.get(key, 0), a["lead_seconds"])
    by_subject = {}
    for (caseid, _), lead in best.items():
        by_subject.setdefault(subject_for_case[caseid], []).append(lead)
    subjects = frame.subjectid.unique()
    rng, medians = np.random.default_rng(Protocol().seed), []
    for _ in range(repeats):
        sampled = [lead for subject in rng.choice(subjects, len(subjects), replace=True) for lead in by_subject.get(subject, [])]
        if sampled:
            medians.append(float(np.median(sampled)))
    return {"conditional_on_detected_eligible_events": True, "detected_events": len(best),
        "median_seconds": float(np.median(list(best.values()))) if best else None,
        "ci95_low": float(np.quantile(medians, .025)) if medians else None,
        "ci95_high": float(np.quantile(medians, .975)) if medians else None,
        "valid_replicates": len(medians), "repeats": repeats, "unit": "subjectid"}


def main():
    reports = ROOT / "reports/E05"
    results = json.loads((reports / "comparison.json").read_text())
    required = {f"{name}_all_20260917_{h}" for name in ("lightgbm", "catboost") for h in (300, 600)}
    assert required.issubset(results), "Wait for the four predeclared horizon models"
    events = pd.read_csv(ROOT / "data/development300/events.csv")
    records = []
    for name in ("lightgbm", "catboost"):
        for h in (300, 600):
            key = f"{name}_all_20260917_{h}"
            frame = pd.read_csv(ROOT / "artifacts/E05" / key / "test.csv.gz")
            frame = frame[~frame.historical_subject]
            threshold = results[key]["fixed"]["selection_on_validation"]["metrics"]["threshold"]
            groups = {"all_new": np.ones(len(frame), dtype=bool), "age_lt65": frame.static_age.lt(65),
                "age_ge65": frame.static_age.ge(65), "asa_lt3": frame.static_asa.lt(3), "asa_ge3": frame.static_asa.ge(3),
                "age_missing": frame.static_age.isna(), "asa_missing": frame.static_asa.isna()}
            for group_name, mask in groups.items():
                subset = frame[mask]
                if not len(subset):
                    records.append({"model": key, "group": group_name, "subjects": 0, "status": "no patients"})
                    continue
                m, _, alarms = evaluate_predictions(subset, events, h, threshold)
                lead = lead_interval(subset, events, alarms, h)
                assert lead["median_seconds"] == m["lead_seconds_median"]
                records.append({"model": key, "group": group_name, "subjects": int(subset.subjectid.nunique()),
                    "metrics": m, "ci95": bootstrap_ci(subset, events, h, threshold, repeats=200),
                    "lead_time": lead})
            print("Analyzed", key, flush=True)
    write_json(reports / "subgroups_lead_time.json", records)
    lines = ["# E05 — phân nhóm và lead time mô tả", "",
        "Chỉ hai cấu hình định trước LightGBM/CatBoost all-features seed 20260917, policy fixed, trên bệnh nhân mới. Không chọn threshold lại theo subgroup. Phân tích mô tả sau benchmark, không là kiểm định xác nhận ưu thế subgroup.", "",
        "| Model | Nhóm | Bệnh nhân | Detect/event | Recall CI95 | Median lead (s) | Lead CI95 (s) |",
        "|---|---|---:|---:|---|---:|---|"]
    fmt = lambda x: "N/A" if x is None else f"{x:.3f}"
    for r in records:
        if not r["subjects"]:
            lines.append(f"| {r['model']} | {r['group']} | 0 | N/A | N/A | N/A | N/A |")
            continue
        m, lead, ci = r["metrics"], r["lead_time"], r["ci95"]["intervals"]["event_sensitivity"]
        lines.append(f"| {r['model']} | {r['group']} | {r['subjects']} | {m['events_detected']}/{m['events_eligible']} | {fmt(ci['low'])}–{fmt(ci['high'])} | {fmt(lead['median_seconds'])} | {fmt(lead['ci95_low'])}–{fmt(lead['ci95_high'])} |")
    lines += ["", "Lead time lấy cảnh báo sớm nhất của mỗi event đủ điều kiện đã được phát hiện, bootstrap 200 theo bệnh nhân kể cả bệnh nhân không có phát hiện. Chỉ tính median ở replicate có event được phát hiện; số replicate hợp lệ nằm trong JSON. Đây là CI **có điều kiện đã phát hiện**, không đại diện các event bị bỏ sót.",
        "", "Nhóm tuổi/ASA có thể rất ít bệnh nhân/event; không suy kết luận công bằng hoặc tương đương từ CI rộng/đè nhau. Nhóm missing được báo kể cả khi không có bệnh nhân.",
        "", "[JSON đầy đủ PPV/FAH/AUROC/coverage và CI](subgroups_lead_time.json) · [Báo cáo chính](REPORT.md)"]
    (reports / "SUBGROUPS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
