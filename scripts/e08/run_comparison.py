"""E08 v2: compare every method on one pipeline; writes reports/E08/ and the README section.

`--render-only` rebuilds the report and README from the saved CSVs without retraining.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from safeanes.e08_methods import (AUGMENT_COPIES, AUGMENT_SEED, CALIBRATOR_PARAMS, CATBOOST_PARAMS,
                                  ENSEMBLE_MEMBERS, ENSEMBLE_NAME, HORIZONS, JITTER_SCALE,
                                  LIGHTGBM_PARAMS, LOGISTIC_PARAMS, MAP_FEATURES, METHODS, MODEL_SEED,
                                  MONOTONE_FEATURES, SMOTE_NEIGHBOURS, TABM_EXPECTED, TABM_FIXED,
                                  calibrate_member, fit_predict, frame_with_probability, load_context,
                                  pareto_front, select_threshold, sweep, threshold_grid)
from safeanes.evaluation import bootstrap_ci, evaluate_predictions, quality_gates

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/E08"
README = ROOT / "README.md"
README_BEGIN, README_END = "<!-- e08:begin -->", "<!-- e08:end -->"
POLICIES = ("recall_first", "fa_budget")
BOOTSTRAP = 200
FAMILY = {"baseline": "baseline", "lightgbm": "LightGBM", "catboost": "CatBoost", "tabm": "TabM",
          "ensemble": "ensemble"}


def evaluate(ctx, name, family, p_cal, p_val, horizon):
    cal_frame = frame_with_probability(ctx.cal_rows, p_cal)
    val_frame = frame_with_probability(ctx.val_rows, p_val)
    grid = threshold_grid(cal_frame.probability.to_numpy())
    cal_curve = sweep(cal_frame, ctx.events, horizon, ctx.protocol, grid)
    rows = []
    for policy in POLICIES:
        chosen, resolved = select_threshold(cal_curve, policy)
        threshold = float(chosen.threshold)
        m, _, _ = evaluate_predictions(val_frame, ctx.events, horizon, threshold, ctx.protocol)
        gates = quality_gates(m, horizon)
        ci = bootstrap_ci(val_frame, ctx.events, horizon, threshold, ctx.protocol, repeats=BOOTSTRAP)["intervals"]
        rows.append({"method": name, "family": family, "horizon": horizon, "policy": policy,
                     "resolved_policy": resolved, "threshold": threshold,
                     "cal_recall": chosen.event_sensitivity, "cal_ppv": chosen.alarm_ppv,
                     "cal_fa_per_hour": chosen.false_alarms_per_hour,
                     "recall": m["event_sensitivity"], "ppv": m["alarm_ppv"],
                     "fa_per_hour": m["false_alarms_per_hour"], "events_detected": m["events_detected"],
                     "events_eligible": m["events_eligible"], "auroc": m["auroc"],
                     "average_precision": m["average_precision"], "ece": m["ece"],
                     "gates_met": sum(gates["criteria"].values()), "gates_total": len(gates["criteria"]),
                     "failed_gates": ",".join(k for k, ok in gates["criteria"].items() if not ok),
                     **{f"{key}_ci_{side}": ci[metric][side]
                        for key, metric in (("recall", "event_sensitivity"), ("ppv", "alarm_ppv"),
                                            ("fa", "false_alarms_per_hour"))
                        for side in ("low", "high")}})
        print(f"  {name:26s} h={horizon} {policy:12s} recall={m['event_sensitivity']:.3f} "
              f"ppv={m['alarm_ppv'] or float('nan'):.3f} fa={m['false_alarms_per_hour']:.3f}", flush=True)
    return rows, (val_frame, grid)


def compute():
    started = time.perf_counter()
    ctx = load_context(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, audits, params, frames = [], [], [], {}

    for horizon in HORIZONS:
        mask = ctx.calibration_mask(horizon)
        y_cal = ctx.cal_rows.loc[mask, f"y_{horizon}"].astype(int).to_numpy()
        print(f"--- horizon {horizon}: calibration prevalence {y_cal.mean():.4f} ---", flush=True)
        calibrated = {}
        for spec in METHODS:
            raw_cal, raw_val, used = fit_predict(ctx, spec, horizon)
            p_cal, p_val = calibrate_member(raw_cal, raw_val, y_cal, mask)
            calibrated[spec.name] = (p_cal, p_val)
            params.append({"method": spec.name, "horizon": horizon, **used})
            audits.append({"method": spec.name, "horizon": horizon, "raw_mean": raw_cal[mask].mean(),
                           "calibrated_mean": p_cal[mask].mean(), "prevalence": y_cal.mean()})
            new_rows, frames[spec.name, horizon] = evaluate(ctx, spec.name, spec.family, p_cal, p_val, horizon)
            rows += new_rows
        p_cal = np.mean([calibrated[m][0] for m in ENSEMBLE_MEMBERS], axis=0)
        p_val = np.mean([calibrated[m][1] for m in ENSEMBLE_MEMBERS], axis=0)
        audits.append({"method": ENSEMBLE_NAME, "horizon": horizon, "raw_mean": p_cal[mask].mean(),
                       "calibrated_mean": p_cal[mask].mean(), "prevalence": y_cal.mean()})
        new_rows, frames[ENSEMBLE_NAME, horizon] = evaluate(ctx, ENSEMBLE_NAME, "ensemble", p_cal, p_val, horizon)
        rows += new_rows

    comparison, audit, used = pd.DataFrame(rows), pd.DataFrame(audits), pd.DataFrame(params)
    winners = pick_winners(comparison)
    frontier = []
    for horizon, name in winners.items():
        val_frame, grid = frames[name, horizon]
        front = pareto_front(sweep(val_frame, ctx.events, horizon, ctx.protocol, grid))
        frontier.append(front.assign(method=name, horizon=horizon))
    frontier = pd.concat(frontier, ignore_index=True)

    comparison.to_csv(OUT / "v2_method_comparison.csv", index=False)
    audit.to_csv(OUT / "v2_calibration_audit.csv", index=False)
    used.to_csv(OUT / "v2_parameters.csv", index=False)
    frontier.to_csv(OUT / "v2_pareto_frontier.csv", index=False)
    seconds = time.perf_counter() - started
    (OUT / "v2_run.json").write_text(json.dumps({"seconds": round(seconds)}), encoding="utf-8")
    print(f"computed in {seconds:.0f}s", flush=True)


def render():
    ctx = load_context(ROOT)
    comparison = pd.read_csv(OUT / "v2_method_comparison.csv")
    used = pd.read_csv(OUT / "v2_parameters.csv")
    winners = pick_winners(comparison)
    shared = shared_sections(ctx, comparison, used, winners)
    write_report(shared, comparison, pd.read_csv(OUT / "v2_calibration_audit.csv"),
                 pd.read_csv(OUT / "v2_pareto_frontier.csv"), winners,
                 json.loads((OUT / "v2_run.json").read_text(encoding="utf-8"))["seconds"])
    write_readme(shared)
    print("rendered README.md and V2_METHOD_COMPARISON.md", flush=True)


def main():
    if "--render-only" not in sys.argv:
        compute()
    render()


def pick_winners(comparison):
    """Choose on CALIBRATION with the recall_first ordering; VALIDATION only reports."""
    first = comparison[comparison.policy.eq("recall_first")]
    return {h: first[first.horizon.eq(h)].sort_values(["cal_recall", "cal_ppv", "cal_fa_per_hour"],
                                                      ascending=[False, False, True]).iloc[0].method
            for h in HORIZONS}


# --- rendering ------------------------------------------------------------------------------

def fmt(x):
    return "N/A" if x is None or pd.isna(x) else f"{x:.3f}"


def code(params):
    return "`" + ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in params.items()) + "`"


def split_summary(ctx):
    out = {}
    for key, rows in (("FIT", ctx.fit_rows), ("CALIBRATION", ctx.cal_rows), ("VALIDATION", ctx.val_rows)):
        cases = rows.caseid.unique()
        events = ctx.events[ctx.events.caseid.isin(cases)]
        out[key] = (len(cases), *(int(events[f"eligible_{h}"].astype(bool).sum()) for h in HORIZONS))
    return out


def pipeline_table(ctx):
    p, split = ctx.protocol, split_summary(ctx)
    return ["| Tham số | Giá trị |", "|---|---|",
        "| Dữ liệu | development300, chia theo `subjectid`: "
        + " · ".join(f"{k} {v[0]} ca" for k, v in split.items()) + " |",
        "| Biến cố eligible (5 / 10 phút) | "
        + " · ".join(f"{k} {v[1]}/{v[2]}" for k, v in split.items()) + " |",
        f"| Định nghĩa biến cố | MAP < {p.map_threshold:g} mmHg liên tục ≥ {p.event_seconds} s |",
        f"| Đặc trưng | {len(ctx.features)} numeric (bỏ static); `map_logistic` chỉ dùng "
        + ", ".join(f"`{f}`" for f in MAP_FEATURES) + " |",
        "| Imputation | median, fit trên FIT, áp dụng y hệt cho FIT / CALIBRATION / VALIDATION |",
        f"| Calibration | StandardScaler → LogisticRegression {code(CALIBRATOR_PARAMS)} trên log-odds của "
        "điểm thô; fit trên CALIBRATION, riêng từng phương pháp × horizon |",
        "| Lưới ngưỡng | 40 điểm cố định 0,01–0,99 + 201 quantile xác suất trên CALIBRATION |",
        "| Chọn ngưỡng | `recall_first` trên CALIBRATION: recall cao nhất → PPV cao hơn → FA/giờ thấp hơn |",
        "| Chọn phương pháp | trên CALIBRATION, cùng thứ tự `recall_first` |",
        "| Báo cáo | VALIDATION — không tham gia chọn ngưỡng hay chọn phương pháp |",
        f"| Chính sách cảnh báo | {p.alarm_persistence} decision liên tiếp vượt ngưỡng · cooldown "
        f"{p.alarm_cooldown_seconds} s · nhịp {p.cadence_seconds} s |",
        f"| Khoảng tin cậy | bootstrap theo `subjectid`, {BOOTSTRAP} lần |",
        f"| Seed | model {MODEL_SEED} · augmentation {AUGMENT_SEED} |"]


def params_table(ctx, used):
    def value(name, horizon, key):
        row = used[used.method.eq(name) & used.horizon.eq(horizon)]
        return None if row.empty or key not in row or pd.isna(row.iloc[0][key]) else row.iloc[0][key]

    monotone = ", ".join(f"`{f}`" for f in MONOTONE_FEATURES)
    lines = ["| Phương pháp | Nhóm | Đặc trưng | Siêu tham số | Cân bằng lớp | Augmentation | "
             "Window dương khi train (5 / 10 phút) |", "|---|---|---|---|---|---|---|"]
    for spec in METHODS:
        if spec.family == "baseline":
            hyper = f"StandardScaler → LogisticRegression {code(LOGISTIC_PARAMS)}"
        elif spec.family == "lightgbm":
            hyper = f"{code(LIGHTGBM_PARAMS)}; đơn điệu giảm theo {monotone}"
        elif spec.family == "catboost":
            hyper = code(CATBOOST_PARAMS)
        else:
            hyper = (f"{code(TABM_EXPECTED)}; `{TABM_FIXED}`; dừng ở epoch "
                     f"{value(spec.name, HORIZONS[0], 'best_epoch'):.0f} (backbone đông lạnh từ E06)")

        if spec.weighting == "scale_pos_20x":
            weights = " / ".join(f"{value(spec.name, h, 'scale_pos_weight'):.0f}" for h in HORIZONS)
            weighting = f"`scale_pos_weight` = 20 × âm/dương = {weights}"
        elif spec.weighting == "balanced":
            weighting = ("`class_weight='balanced'`" if spec.family == "lightgbm"
                         else "`auto_class_weights='Balanced'`")
        else:
            weighting = "không"

        augmentation = {
            "none": "không",
            "oversample": f"nhân bản nguyên văn, +{AUGMENT_COPIES} bản / window dương",
            "smote": f"SMOTE {SMOTE_NEIGHBOURS} láng giềng, +{AUGMENT_COPIES} mẫu / window dương",
            "jitter": f"nhiễu Gaussian σ = {JITTER_SCALE:g} × std, +{AUGMENT_COPIES} bản / window dương",
        }[spec.augmentation]

        positives = [value(spec.name, h, "n_positive") for h in HORIZONS]
        positives = ("— (train ở E06)" if positives[0] is None
                     else " / ".join(str(int(v)) for v in positives))
        features = f"{len(MAP_FEATURES)} (MAP)" if spec.family == "baseline" else f"{len(ctx.features)}"
        lines.append(f"| `{spec.name}` | {FAMILY[spec.family]} | {features} | {hyper} | {weighting} | "
                     f"{augmentation} | {positives} |")
    lines.append(f"| `{ENSEMBLE_NAME}` | ensemble | — | trung bình xác suất đã calibrate của "
                 + ", ".join(f"`{m}`" for m in ENSEMBLE_MEMBERS) + " | — | — | — |")
    return lines


def results_table(comparison, horizon, winner):
    sub = comparison[comparison.horizon.eq(horizon) & comparison.policy.eq("recall_first")]
    sub = sub.sort_values(["recall", "ppv", "fa_per_hour"], ascending=[False, False, True])
    lines = ["| Phương pháp | Recall (CI95) | Bắt được | PPV (CI95) | FA/giờ (CI95) | AUROC | AP | ECE |",
             "|---|---|---:|---|---|---:|---:|---:|"]
    for r in sub.itertuples():
        mark = " ★" if r.method == winner else ""
        lines.append(f"| `{r.method}`{mark} | {fmt(r.recall)} ({fmt(r.recall_ci_low)}–{fmt(r.recall_ci_high)}) | "
                     f"{int(r.events_detected)}/{int(r.events_eligible)} | "
                     f"{fmt(r.ppv)} ({fmt(r.ppv_ci_low)}–{fmt(r.ppv_ci_high)}) | "
                     f"{fmt(r.fa_per_hour)} ({fmt(r.fa_ci_low)}–{fmt(r.fa_ci_high)}) | "
                     f"{fmt(r.auroc)} | {fmt(r.average_precision)} | {fmt(r.ece)} |")
    return lines


def winner_lines(ctx, comparison, winners):
    first = comparison[comparison.policy.eq("recall_first")]
    cal_events = dict(zip(HORIZONS, split_summary(ctx)["CALIBRATION"][1:]))
    lines, consistent = [], []
    for horizon, name in winners.items():
        sub = first[first.horizon.eq(horizon)]
        top, runner = sub.sort_values(["cal_recall", "cal_ppv", "cal_fa_per_hour"],
                                      ascending=[False, False, True]).iloc[:2].itertuples()
        leader = sub.sort_values(["recall", "ppv", "fa_per_hour"], ascending=[False, False, True]).iloc[0]
        n = cal_events[horizon]
        caught, runner_caught = round(top.cal_recall * n), round(runner.cal_recall * n)
        lines.append(f"- **{horizon // 60} phút: `{name}`** — VALIDATION recall {fmt(top.recall)} "
                     f"({int(top.events_detected)}/{int(top.events_eligible)} biến cố), PPV {fmt(top.ppv)}, "
                     f"FA {fmt(top.fa_per_hour)}/giờ, AUROC {fmt(top.auroc)}; đạt {int(top.gates_met)}/"
                     f"{int(top.gates_total)} gate, còn thiếu: "
                     + ", ".join(f"`{g}`" for g in top.failed_gates.split(",")) + ".")
        if caught == runner_caught:
            lines.append(f"  - Trên CALIBRATION **hòa recall** với `{runner.method}` ({caught}/{n} biến cố), chỉ "
                         f"phân định bằng PPV {top.cal_ppv:.4f} so với {runner.cal_ppv:.4f} — thực chất "
                         "**chưa phân biệt được** các phương pháp dẫn đầu ở horizon này.")
        else:
            lines.append(f"  - Trên CALIBRATION bắt {caught}/{n} biến cố, hơn á quân `{runner.method}` "
                         f"{caught - runner_caught} biến cố.")
        if leader.method != name:
            lines.append(f"  - Trên VALIDATION, recall cao nhất lại là `{leader.method}` "
                         f"({int(leader.events_detected)}/{int(leader.events_eligible)}, PPV {fmt(leader.ppv)}, "
                         f"FA {fmt(leader.fa_per_hour)}/giờ) — thứ hạng giữa hai tập độc lập chưa ổn định.")
        elif top.family == "baseline":
            consistent.append(horizon)
    if consistent:
        lines.append(f"- Ở {', '.join(f'{h // 60} phút' for h in consistent)}, baseline MAP đứng đầu trên **cả** "
                     "CALIBRATION (dùng để chọn) lẫn VALIDATION (dùng để báo cáo): model phức tạp chưa cho "
                     "thấy lợi ích trên mẫu này — cùng hướng với "
                     "[Mulder 2024](https://pubmed.ncbi.nlm.nih.gov/38558038/) (HPI ≈ ngưỡng MAP).")
    return lines


def shared_sections(ctx, comparison, used, winners):
    totals = {h: int(comparison[comparison.horizon.eq(h)].events_eligible.iloc[0]) for h in HORIZONS}
    sections = {"winners": winner_lines(ctx, comparison, winners), "pipeline": pipeline_table(ctx),
                "params": params_table(ctx, used)}
    for horizon in HORIZONS:
        sections[f"results_{horizon}"] = results_table(comparison, horizon, winners[horizon])
        sections[f"total_{horizon}"] = totals[horizon]
    return sections


NOTES = ["- ★ = phương pháp chọn trên CALIBRATION; VALIDATION chỉ để báo cáo.",
         "- Mỗi biến cố trên VALIDATION = 4,2–4,3 điểm recall, CI95 rất rộng: chênh lệch dưới ~1 biến cố "
         "**không** đủ kết luận phương pháp nào hơn.",
         "- AUROC / AP / ECE không phụ thuộc ngưỡng — dùng để so khả năng phân biệt tách khỏi điểm vận hành.",
         "- Development validation, **không phải** bằng chứng xác nhận độc lập; chưa mở pilot_test/global test."]


def body(shared):
    lines = ["### Phương pháp được chọn", "", *shared["winners"], "",
             "### Tham số chung (áp dụng cho mọi phương pháp)", "", *shared["pipeline"], "",
             "### Tham số từng phương pháp", "", *shared["params"], ""]
    for horizon in HORIZONS:
        lines += [f"### Kết quả {horizon // 60} phút — `recall_first`, VALIDATION "
                  f"({shared[f'total_{horizon}']} biến cố)", "", *shared[f"results_{horizon}"], ""]
    return lines + ["**Đọc bảng:**", "", *NOTES]


def write_readme(shared):
    block = ["## E08 — so sánh phương pháp (v2, bản đang dùng)", "",
             "Mọi phương pháp đi qua đúng một quy trình, chỉ khác nhau ở bản thân phương pháp; FA/giờ đi "
             "kèm chính sách `recall_first` đã được chấp nhận. Sinh tự động bởi "
             "[`scripts/e08/run_comparison.py`](scripts/e08/run_comparison.py) từ chính các tham số đã chạy. "
             "[Báo cáo đầy đủ](reports/E08/V2_METHOD_COMPARISON.md) · "
             "[lịch sử version và 6 lỗi của v1](scripts/e08/README.md)", "", *body(shared)]
    text = README.read_text(encoding="utf-8")
    start, end = text.index(README_BEGIN) + len(README_BEGIN), text.index(README_END)
    README.write_text(text[:start] + "\n" + "\n".join(block) + "\n" + text[end:], encoding="utf-8")


def write_report(shared, comparison, audit, frontier, winners, seconds):
    lines = ["# E08 v2 — so sánh phương pháp", "",
             f"Chạy {seconds:.0f} giây. [Tóm tắt trong README](../../README.md) · "
             "[lịch sử version](../../scripts/e08/README.md) · "
             "[script](../../scripts/e08/run_comparison.py)", "", *body(shared), "",
             "## Cái giá của ràng buộc FA/giờ ≤ 0,5 (`fa_budget`, cùng model)", "",
             "| Phương pháp | Phút | Recall `recall_first` | Recall `fa_budget` | Δ recall | PPV `fa_budget` | FA/giờ `fa_budget` |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    first = comparison[comparison.policy.eq("recall_first")]
    budget = comparison[comparison.policy.eq("fa_budget")]
    merged = first.merge(budget, on=["method", "horizon"], suffixes=("_first", "_budget"))
    for r in merged.sort_values(["horizon", "recall_first"], ascending=[True, False]).itertuples():
        lines.append(f"| `{r.method}` | {r.horizon // 60} | {fmt(r.recall_first)} | {fmt(r.recall_budget)} | "
                     f"{(r.recall_budget or 0) - (r.recall_first or 0):+.3f} | {fmt(r.ppv_budget)} | "
                     f"{fmt(r.fa_per_hour_budget)} |")
    lines += ["", "## Mặt Pareto của phương pháp được chọn (VALIDATION)", "",
              "Các điểm không bị điểm nào trội hơn đồng thời về recall, PPV và FA/giờ — chọn điểm nào là "
              "quyết định lâm sàng, không phải kỹ thuật.", "",
              "| Phút | Phương pháp | Ngưỡng | Recall | Bắt được | PPV | FA/giờ |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in frontier.drop_duplicates(["horizon", "event_sensitivity"]).itertuples():
        lines.append(f"| {r.horizon // 60} | `{r.method}` | {fmt(r.threshold)} | {fmt(r.event_sensitivity)} | "
                     f"{int(r.events_detected)}/{int(r.events_eligible)} | {fmt(r.alarm_ppv)} | "
                     f"{fmt(r.false_alarms_per_hour)} |")
    lines += ["", "## Kiểm chứng calibration", "",
              "Mean xác suất sau calibrate phải bám prevalence thật, nếu không mọi so sánh ngưỡng và phép trung "
              "bình ensemble đều sai (lỗi F2 của v1).", "",
              "| Phương pháp | Phút | Mean thô | Mean sau calibrate | Prevalence |", "|---|---:|---:|---:|---:|"]
    for r in audit.itertuples():
        lines.append(f"| `{r.method}` | {r.horizon // 60} | {r.raw_mean:.4f} | {r.calibrated_mean:.4f} | "
                     f"{r.prevalence:.4f} |")
    (OUT / "V2_METHOD_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
