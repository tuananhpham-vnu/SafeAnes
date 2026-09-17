"""Measured research reports and replay figures from saved prediction artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import write_json
from .evaluation import evaluate_predictions
from .sequences import load_dataset


def calibration_diagnostics(frame, horizon):
    known = frame.eligible & frame[f"y_{horizon}"].ge(0) & frame.probability.notna()
    y = frame.loc[known, f"y_{horizon}"].to_numpy()
    p = frame.loc[known, "probability"].to_numpy()
    ids = np.minimum((p * 10).astype(int), 9)
    bins = [{"left": i / 10, "right": (i + 1) / 10, "count": int((ids == i).sum()),
             "mean_probability": float(p[ids == i].mean()) if np.any(ids == i) else None,
             "event_fraction": float(y[ids == i].mean()) if np.any(ids == i) else None}
            for i in range(10)]
    slope, intercept = None, None
    if len(np.unique(y)) == 2 and np.std(p) > 1e-12:
        bounded = np.clip(p, 1e-6, 1 - 1e-6)
        logit = np.log(bounded / (1 - bounded)).reshape(-1, 1)
        # Diagnostic fit only; never applied back to test predictions.
        fit = LogisticRegression(C=1e6, max_iter=2000).fit(logit, y)
        slope, intercept = float(fit.coef_[0, 0]), float(fit.intercept_[0])
    return {"bins": bins, "slope": slope, "intercept": intercept,
            "note": "Descriptive diagnostic on held-out pilot predictions; not a fitted deployment calibrator"}


def subgroup_metrics(frame, events, horizon, threshold, protocol):
    groups = {"age_lt65": frame.static_age.lt(65), "age_ge65": frame.static_age.ge(65),
              "asa_le2": frame.static_asa.le(2), "asa_ge3": frame.static_asa.ge(3)}
    result = {}
    for name, selected in groups.items():
        subset = frame[selected]
        if not len(subset):
            continue
        # Case-static groups preserve complete case timelines and alarm state.
        metrics, _, _ = evaluate_predictions(subset, events, horizon, threshold, protocol)
        result[name] = {"patients": int(subset.subjectid.nunique()), "cases": int(subset.caseid.nunique()), **metrics}
    return result


def reliability_figure(diagnostics, key, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = [b for b in diagnostics["bins"] if b["count"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot([0, 1], [0, 1], "--", color="#888888", label="Ideal calibration")
    axes[0].plot([b["mean_probability"] for b in bins], [b["event_fraction"] for b in bins],
                 "o-", color="#186a80", label="Pilot test")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed event fraction")
    axes[0].legend()
    axes[1].bar([b["left"] + .05 for b in diagnostics["bins"]],
                [b["count"] for b in diagnostics["bins"]], width=.09, color="#7a45a5")
    axes[1].set(xlim=(0, 1), xlabel="Predicted probability bin", ylabel="Known eligible windows")
    figure.suptitle(f"{key} | reliability, 10 fixed bins | exploratory pilot")
    figure.savefig(target, dpi=140)
    plt.close(figure)


def replay_figure(frame, events, alarms, horizon, threshold, caseid, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    case = frame[frame.caseid.eq(caseid)].sort_values("time")
    if case.empty:
        raise ValueError("Requested replay case not present in predictions")
    figure, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    minutes = case.time / 60
    axes[0].plot(minutes, case.map_current, color="#186a80", lw=1.5)
    axes[0].axhline(65, color="#ad3842", ls="--", lw=1)
    axes[0].set_ylabel("MAP (mmHg)")
    axes[1].step(minutes, case.probability, where="post", color="#7a45a5")
    if threshold <= 1:
        axes[1].axhline(threshold, color="#ad3842", ls="--", label=f"Threshold {threshold:.3f}")
        axes[1].legend(loc="upper right")
    else:
        axes[1].text(.02, .90, "No-alert fallback selected on validation (threshold > 1)",
                     transform=axes[1].transAxes, color="#ad3842")
    axes[1].set(ylim=(-.03, 1.03), ylabel=f"Risk within {horizon // 60} min")
    axes[2].step(minutes, case.history_coverage, where="post", label="MAP history coverage")
    axes[2].step(minutes, case.eligible.astype(float), where="post", label="Prediction eligible", alpha=.6)
    axes[2].set(ylim=(-.03, 1.03), ylabel="Availability", xlabel="Minutes from case start")
    axes[2].legend(loc="lower right")
    for event in events[events.caseid.eq(caseid)].itertuples():
        for ax in axes:
            ax.axvspan(event.onset / 60, event.end / 60, color="#c9444e", alpha=.15)
    colors = {"true": "#16855b", "false": "#d67716", "censored": "#666666"}
    for alarm in alarms:
        if alarm["caseid"] == caseid:
            axes[1].axvline(alarm["time"] / 60, color=colors[alarm["outcome"]], lw=1)
    figure.suptitle(f"SafeAnes research replay | case {caseid} | horizon {horizon // 60} min\n"
                   "Red shading: IOH; alarm lines: green=true, orange=false, grey=censored")
    for ax in axes:
        ax.grid(alpha=.15)
    figure.savefig(target, dpi=140)
    plt.close(figure)


def build_report(dataset, run, out):
    """No fitting or threshold selection: render the completed immutable run."""
    dataset, run, out = Path(dataset), Path(run), Path(out)
    meta, protocol, _, _ = load_dataset(dataset)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    if report["scope"] != "exploratory_pilot_not_final_test" or report["errors"]:
        raise ValueError("Completed exploratory pilot required")
    if environment["dataset_hash"] != meta["windows_sha256"]:
        raise ValueError("Report and dataset hashes differ")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Report output must be new/empty")
    out.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(dataset / "events.csv")
    roles = pd.read_csv(run / "pilot_roles.csv")
    expected_cases = set(roles.loc[roles.role.eq("pilot_test"), "caseid"])
    lines = ["# Kết quả UC04 — thử nghiệm trên dữ liệu thật", "",
        "Kết quả từ pilot nằm trong global train; chưa phải final test, kiểm định ngoài hoặc nghiệm thu lâm sàng.", "",
        f"- Nguồn artifacts: `{run.as_posix()}`.", f"- Dataset SHA-256: `{meta['windows_sha256']}`.",
        f"- Protocol: `{meta['protocol_hash']}`.",
        f"- Thiết bị thực thi: `{environment.get('device', 'cpu')}`; `{environment.get('device_name', 'baseline CPU')}`.", "",
        "| Mô hình | AUROC | AP | Event recall | PPV | FA/giờ | ECE | Phát hiện/đủ điều kiện | Đạt mọi mục tiêu? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"
    diagnostics = {}
    images, reliability_images, decisions = [], [], []
    for key, result in report["models"].items():
        m = result["pilot_test"]
        horizon, threshold = m["horizon_seconds"], m["threshold"]
        frame = pd.read_csv(run / f"{key}_predictions.csv.gz")
        if set(frame.caseid) != expected_cases or not frame.role.eq("pilot_test").all():
            raise ValueError("Prediction cohort differs from the saved pilot test roles")
        alarms = json.loads((run / f"{key}_alarms.json").read_text(encoding="utf-8"))
        columns = [key] + [fmt(m[k]) for k in ("auroc", "average_precision", "event_sensitivity", "alarm_ppv", "false_alarms_per_hour", "ece")]
        columns += [f"{m['events_detected']}/{m['events_eligible']}", "Có (pilot)" if result["gates"]["all_point_targets_met"] else "Chưa đạt"]
        lines.append("| " + " | ".join(columns) + " |")
        diagnostics[key] = {"calibration": calibration_diagnostics(frame, horizon),
            "subgroups": subgroup_metrics(frame, events, horizon, threshold, protocol),
            "ci95": result["ci95"], "targets": result["gates"],
            "selection": result["selection_on_validation"]["selection"]}
        reliability_name = f"{key}_reliability.png"
        reliability_figure(diagnostics[key]["calibration"], key, out / reliability_name)
        reliability_images.append(f"![Reliability {key}]({reliability_name})")
        decisions.append(f"- `{key}`: threshold={threshold:.6f}; "
            + ("validation chọn không phát cảnh báo; độ nhạy bằng 0, chưa có mô hình cảnh báo hữu ích."
               if threshold > 1 else f"validation: {result['selection_on_validation']['selection']}."))
        caseids = []
        for outcome in ("true", "false", "censored"):
            candidate = next((a["caseid"] for a in alarms if a["outcome"] == outcome), None)
            if candidate is not None and candidate not in caseids:
                caseids.append(candidate)
        # Include a case with an eligible missed event when one is present.
        stats = pd.read_csv(run / f"{key}_case_metrics.csv")
        missed = stats[stats.events_detected < stats.events_eligible]
        if len(missed) and int(missed.caseid.iloc[0]) not in caseids:
            caseids.append(int(missed.caseid.iloc[0]))
        if not caseids:
            caseids = [int(frame.caseid.iloc[0])]
        for caseid in caseids:
            filename = f"{key}_case_{caseid}.png"
            replay_figure(frame, events, alarms, horizon, threshold, caseid, out / filename)
            images.append(f"![{key}, ca {caseid}]({filename})")
    resources = report.get("resources")
    if resources:
        lines += ["", "## Tài nguyên đã đo", "",
            f"- Số tham số: {resources['parameters']:,}; số epoch: {resources['epochs_completed']}; checkpoint tốt nhất: epoch {resources['best_epoch']}.",
            f"- Tổng thời gian các epoch: {resources['training_seconds']:.1f} giây.",
            f"- Peak allocated VRAM: {resources['peak_allocated_vram_bytes'] if resources['peak_allocated_vram_bytes'] is not None else 'không áp dụng (CPU)'}.",
            "- Thời gian inference trong JSON gồm DataLoader và sao chép dữ liệu; không phải độ trễ ứng dụng lâm sàng."]
    lines += ["", "## Ngưỡng đã khóa trên validation", "", *decisions,
        "", "## Diễn giải và giới hạn", "",
        "- N/A nghĩa chưa tính được metric, thường do không có cảnh báo hoặc thiếu một lớp.",
        "- Ngưỡng được chọn trên validation. Nếu không đạt, giữ nhãn `exploratory_fallback_targets_unmet`; không nới tiêu chí.",
        "- Khoảng tin cậy 95% bootstrap theo bệnh nhân, calibration bins/slope/intercept và phân nhóm tuổi/ASA nằm trong [diagnostics.json](diagnostics.json). Các phân nhóm nhỏ chỉ có giá trị mô tả.",
        "- FA/giờ dùng lưới 30 giây theo protocol v1; coverage đo trên thời gian quyết định sau history. Cần audit exposure từng giây trước nghiệm thu.",
        "- Pilot nhỏ, chưa có kiểm định ngoài hoặc nhãn nguyên nhân. Thiết bị thực tế ghi ở đầu báo cáo; không suy ra benchmark T4 từ lần chạy CPU. Các biểu đồ được chọn để audit cảnh báo/bỏ sót; bảng metric dùng toàn bộ pilot_test.",
        "- Có thể so cùng dữ liệu/split với baseline nhưng calibrator của DL bảo toàn thứ tự hai horizon và khác sigmoid riêng từng horizon của baseline.",
        "", "## Hiệu chỉnh xác suất", "", *reliability_images,
        "", "## Phát lại ca", "", *images]
    write_json(out / "diagnostics.json", diagnostics)
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(out / "REPORT.md"), "replays": len(images)}
