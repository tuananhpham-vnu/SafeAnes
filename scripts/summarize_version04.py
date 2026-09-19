"""Summarize every seed without choosing a winner on pilot_test."""
from pathlib import Path
import json
import numpy as np
from safeanes.data import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "reports/v0_4"
    results = json.loads((directory / "comparison.json").read_text())
    assert len(results) == 32, "Incomplete version"
    lines = ["# Nhận xét E04 (v0.2)", "",
        "Đã chạy đủ 12 model CatBoost (hai cấu hình weighting × ba seed × hai horizon), cộng replay TCN/ensemble với hai policy: 32 operating point. Có cải thiện phát hiện trên pilot nhưng chưa cấu hình nào đạt toàn bộ mục tiêu.", "",
        "CatBoost không weighting ở 10 phút phát hiện 2/4 event ở cả ba seed, so với 1/4 của LightGBM cũ; PPV trung bình 0.433 và FA/giờ trung bình 0.220. Ở 5 phút với lưới cũ, recall trung bình 55.6% nhưng dao động 0–100%. Chưa đủ bằng chứng về độ ổn định hoặc khả năng tổng quát hóa.", "",
        "## Ba seed CatBoost — báo cáo cả khoảng dao động", "",
        "Mỗi cấu hình dùng seeds 20260917/18/19, cùng patient split. Đây là độ nhạy theo seed, không phải ba cohort độc lập.", "",
        "| Weight | Phút | Policy | Recall trung bình [min,max] | PPV trung bình | FA/giờ trung bình | AUROC trung bình | Validation recall trung bình |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    groups = []
    for weight in ("None", "SqrtBalanced"):
        for h in (300, 600):
            for policy in ("fixed", "adaptive"):
                rows = [results[f"catboost_{weight}_{seed}_{h}_{policy}"] for seed in (20260917, 20260918, 20260919)]
                mean = lambda field, scope="pilot_test": float(np.mean([r[scope][field] for r in rows if r[scope][field] is not None])) if any(r[scope][field] is not None for r in rows) else None
                recalls = [r["pilot_test"]["event_sensitivity"] for r in rows]
                group = {"weight": weight, "horizon": h, "policy": policy,
                    "test_recall_mean": mean("event_sensitivity"), "test_recall_min": min(recalls), "test_recall_max": max(recalls),
                    "test_ppv_mean_defined": mean("alarm_ppv"), "test_ppv_defined_seeds": sum(r["pilot_test"]["alarm_ppv"] is not None for r in rows),
                    "test_fah_mean": mean("false_alarms_per_hour"), "test_auroc_mean": mean("auroc"),
                    "validation_recall_mean": float(np.mean([r["selection_on_validation"]["metrics"]["event_sensitivity"] for r in rows]))}
                groups.append(group)
                fmt = lambda x: "N/A" if x is None else f"{x:.3f}"
                lines.append("| " + " | ".join([weight, str(h//60), policy,
                    f"{group['test_recall_mean']:.3f} [{min(recalls):.3f},{max(recalls):.3f}]",
                    f"{fmt(group['test_ppv_mean_defined'])} ({group['test_ppv_defined_seeds']}/3 seed có cảnh báo)",
                    fmt(group["test_fah_mean"]), fmt(group["test_auroc_mean"]), fmt(group["validation_recall_mean"])]) + " |")
    baseline = json.loads((ROOT / "artifacts/pilot_lightgbm_v1/report.json").read_text())["models"]
    lines += ["", "## Đối chứng LightGBM v0.1 cùng pilot", "",
        "| Horizon | AUROC | AP | Event phát hiện | PPV | FA/giờ |",
        "|---|---:|---:|---:|---:|---:|"]
    for h in (300, 600):
        m = baseline[f"lightgbm_{h}"]["pilot_test"]
        lines.append(f"| {h//60} phút | {m['auroc']:.3f} | {m['average_precision']:.3f} | {m['events_detected']}/{m['events_eligible']} | {m['alarm_ppv']:.3f} | {m['false_alarms_per_hour']:.3f} |")
    lines += ["", "LightGBM là run một seed với lưới cũ, không phải so sánh nhiều seed cân bằng. Không dùng chênh lệch này để khẳng định ưu thế thống kê.",
        "", "## Diễn giải và giới hạn", "",
        "- TCN giữ nguyên weights: đổi lưới ngưỡng làm recall test từ 0 lên 1/3 ở 5 phút và 1/4 ở 10 phút. PPV=0.5; AUROC không đổi. Đây là cải thiện policy có giới hạn, không phải cải thiện kiến trúc.",
        "- Ensemble vẫn không phát cảnh báo ở cả hai policy; thêm thành viên không bảo đảm cải thiện operating point.",
        f"- Số operating point đạt toàn bộ gate: {sum(r['gates']['all_point_targets_met'] for r in results.values())}/{len(results)}. Không hạ gate hoặc thay đổi event denominator để làm đẹp kết quả.",
        "- FA/giờ <=0.5 là điều kiện chọn trên validation, không bảo đảm test cũng <=0.5. Bảng test phải giữ nguyên cả trường hợp vượt ngân sách.",
        "- PPV trung bình chỉ tính seed có cảnh báo; số seed được ghi cạnh giá trị. Không coi N/A là 0 hoặc 1.",
        "- Chỉ 9 bệnh nhân test và 3/4 biến cố đủ điều kiện. Bootstrap 200 theo bệnh nhân nằm trong comparison.json; khoảng dao động seed không thay CI. Pilot_test đã được xem trước nên kết quả không xác nhận khả năng tổng quát hóa.",
        "- Hai classifier CatBoost theo horizon độc lập, chưa bảo đảm p10 >= p5. Số vi phạm và kiểm tra tái tạo prediction từ model lưu nằm trong [verification.json](verification.json).",
        "- Cohort 60 ca được giữ cố định để tách tác động mô hình/ngưỡng. Bước kế tiếp là mở rộng development, giữ nguyên role của bệnh nhân cũ, khóa validation và một tập đánh giá mới trước khi thử TabPFN/TabICL. Không tiếp tục chọn cấu hình theo điểm pilot_test hiện tại.", "",
        "## Tái lập", "", "```powershell", '$env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"',
        "python scripts/run_version04.py", "python scripts/summarize_version04.py", "python -m pytest -q", "```", "",
        "Runner lưu model/calibrator, predictions và toàn bộ validation curve tại artifacts/v0_4; registration chứa source/input hashes và phiên bản thư viện. Source đã chụp riêng; runner từ chối chạy tiếp nếu identity thay đổi. Ngưỡng vận hành nằm trong *_results.json, tương ứng từng horizon/policy; model joblib không tự chọn policy.",
        "", "[Bảng đầy đủ](REPORT.md) · [Nguồn nghiên cứu](../../docs/SOURCES.md#review-e04) · [Thiết kế](../../docs/versions/V0_4_PLAN.md)"]
    write_json(directory / "seed_summary.json", groups)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    labels = [f"{w} / {p}" for w in ("None", "SqrtBalanced") for p in ("fixed", "adaptive")]
    for ax, h in zip(axes, (300, 600)):
        for i, (weight, policy) in enumerate((w, p) for w in ("None", "SqrtBalanced") for p in ("fixed", "adaptive")):
            recall = [results[f"catboost_{weight}_{seed}_{h}_{policy}"]["pilot_test"]["event_sensitivity"]
                      for seed in (20260917, 20260918, 20260919)]
            ax.plot([min(recall), max(recall)], [i, i], color="steelblue", alpha=.5)
            ax.scatter(recall, np.array([-.09, 0, .09])+i, s=40, color="steelblue")
        reference = baseline[f"lightgbm_{h}"]["pilot_test"]["event_sensitivity"]
        ax.axvline(reference, color="darkorange", linestyle="--", label="Previous LightGBM (1 seed)")
        ax.set(xlim=(-.03, 1.03), yticks=range(4), yticklabels=labels,
               xlabel="Pilot test event recall", title=f"{h//60} min: {3 if h==300 else 4} eligible events")
        ax.grid(axis="x", alpha=.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", fontsize=9)
    fig.suptitle("CatBoost: each dot is one seed; ranges are not confidence intervals")
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(directory / "seed_recall.png", dpi=160)
    plt.close(fig)
    lines += ["", "![Độ nhạy recall theo seed](seed_recall.png)", "",
              "Mỗi chấm là một seed; đoạn nối là min–max, không phải khoảng tin cậy. Đường đứt là run LightGBM cũ một seed."]
    (directory / "ASSESSMENT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
