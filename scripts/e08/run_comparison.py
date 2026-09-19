"""E08: compare every method on one pipeline; writes reports/E08/ and the README section.

  --dataset development300|full   v2 = development300, v3 = full VitalDB except the locked global test
  --tabm frozen|retrain|skip      reuse the E06 backbone, retrain it on FIT, or leave TabM out
  --render-only                   rebuild the report and README from saved outputs, no training
A rerun resumes from artifacts/E08/cache/ as long as the method and evaluation code are unchanged.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from safeanes import e08_methods, evaluation, tabular_sota
from safeanes.e08_methods import (AUGMENT_COPIES, AUGMENT_SEED, CALIBRATOR_PARAMS, CATBOOST_PARAMS,
                                  DATASETS, ENSEMBLE_MEMBERS, ENSEMBLE_NAME, HORIZONS, JITTER_SCALE,
                                  LIGHTGBM_PARAMS, LOGISTIC_PARAMS, MAP_FEATURES, METHODS, MODEL_SEED,
                                  MONOTONE_FEATURES, SMOTE_NEIGHBOURS, TABM_EXPECTED, TABM_FIXED,
                                  TABM_MODES, calibrate_member, fit_predict, frame_with_probability,
                                  load_context, pareto_front, select_threshold, sweep, threshold_grid)
from safeanes.evaluation import bootstrap_ci, evaluate_predictions, quality_gates

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/E08"
CACHE = ROOT / "artifacts/E08/cache"
README = ROOT / "README.md"
README_BEGIN, README_END = "<!-- e08:begin -->", "<!-- e08:end -->"
POLICIES = ("recall_first", "fa_budget")
BOOTSTRAP = 200
VERSION = {"development300": "v2", "full": "v3"}
OUTPUTS = ("method_comparison.csv", "calibration_audit.csv", "parameters.csv", "pareto_frontier.csv",
           "run.json", "METHOD_COMPARISON.md")
FAMILY = {"baseline": "baseline", "lightgbm": "LightGBM", "catboost": "CatBoost", "tabm": "TabM",
          "ensemble": "ensemble"}
DATASET_TITLE = {"development300": "development300", "full": "toàn bộ VitalDB (trừ global test)"}
DATASET_TEXT = {
    "development300": "development300 (300 ca của E05), chia theo `subjectid`",
    "full": "toàn bộ VitalDB đủ điều kiện, theo nhóm bệnh nhân của E07: FIT = `development_seen` + "
            "`unseen_train`, CALIBRATION = `unseen_calibration`, VALIDATION = `unseen_validation`; "
            "global test `unseen_test` không được đọc",
}
TABM_TEXT = {"frozen": "backbone đông lạnh từ E06 (train trên development300), chỉ calibrate lại",
             "retrain": "train lại trên FIT của lần chạy này", "skip": "không chạy"}


# --- compute -------------------------------------------------------------------------------

def evaluate(ctx, name, family, p_cal, p_val, horizon):
    cal_frame = frame_with_probability(ctx.cal_rows, p_cal)
    val_frame = frame_with_probability(ctx.val_rows, p_val)
    cal_curve = sweep(cal_frame, ctx.events, horizon, ctx.protocol, threshold_grid(cal_frame.probability.to_numpy()))
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
    return rows


def fingerprint(dataset, tabm):
    """Cached results are reused only if the code that produced them is unchanged."""
    parts = [Path(module.__file__).read_bytes() for module in (e08_methods, evaluation, tabular_sota)]
    parts += [inspect.getsource(evaluate).encode(), repr((BOOTSTRAP, POLICIES, dataset, tabm)).encode()]
    return hashlib.sha256(b"\0".join(parts)).hexdigest()


def cached(path, key, produce):
    if path.exists():
        stored = joblib.load(path)
        if stored.get("fingerprint") == key:
            print(f"  resume {path.name}", flush=True)
            return stored
    result = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**result, "fingerprint": key}, path)
    return result


def run_method(ctx, spec, horizon, tabm, mask, y_cal, cache):
    raw_cal, raw_val, used = fit_predict(ctx, spec, horizon, tabm, cache / "tabm_models")
    p_cal, p_val = calibrate_member(raw_cal, raw_val, y_cal, mask)
    return {"rows": evaluate(ctx, spec.name, spec.family, p_cal, p_val, horizon),
            "audit": {"method": spec.name, "horizon": horizon, "raw_mean": raw_cal[mask].mean(),
                      "calibrated_mean": p_cal[mask].mean(), "prevalence": y_cal.mean()},
            "params": {"method": spec.name, "horizon": horizon, **used}, "p_cal": p_cal, "p_val": p_val}


def split_summary(ctx):
    out = {}
    for key, rows in (("FIT", ctx.fit_rows), ("CALIBRATION", ctx.cal_rows), ("VALIDATION", ctx.val_rows)):
        events = ctx.events[ctx.events.caseid.isin(rows.caseid.unique())]
        out[key] = [int(rows.caseid.nunique()), int(rows.subjectid.nunique()),
                    *(int(events[f"eligible_{h}"].astype(bool).sum()) for h in HORIZONS)]
    return out


def compute(dataset, tabm):
    started = time.perf_counter()
    ctx = load_context(ROOT, dataset)
    methods = [s for s in METHODS if not (s.family == "tabm" and tabm == "skip")]
    members = [m for m in ENSEMBLE_MEMBERS if m in {s.name for s in methods}]
    cache, key = CACHE / f"{dataset}_tabm-{tabm}", fingerprint(dataset, tabm)
    rows, audits, params, probabilities = [], [], [], {}

    for horizon in HORIZONS:
        mask = ctx.calibration_mask(horizon)
        y_cal = ctx.cal_rows.loc[mask, f"y_{horizon}"].astype(int).to_numpy()
        print(f"--- {dataset} horizon {horizon}: calibration prevalence {y_cal.mean():.4f} ---", flush=True)
        for spec in methods:
            result = cached(cache / f"{spec.name}_{horizon}.joblib", key,
                            lambda: run_method(ctx, spec, horizon, tabm, mask, y_cal, cache))
            rows += result["rows"]
            audits.append(result["audit"])
            params.append(result["params"])
            probabilities[spec.name, horizon] = result["p_cal"], result["p_val"]
        p_cal = np.mean([probabilities[m, horizon][0] for m in members], axis=0)
        p_val = np.mean([probabilities[m, horizon][1] for m in members], axis=0)
        probabilities[ENSEMBLE_NAME, horizon] = p_cal, p_val
        audits.append({"method": ENSEMBLE_NAME, "horizon": horizon, "raw_mean": p_cal[mask].mean(),
                       "calibrated_mean": p_cal[mask].mean(), "prevalence": y_cal.mean()})
        rows += evaluate(ctx, ENSEMBLE_NAME, "ensemble", p_cal, p_val, horizon)

    comparison = pd.DataFrame(rows)
    frontier = []
    for horizon, name in pick_winners(comparison).items():
        p_cal, p_val = probabilities[name, horizon]
        grid = threshold_grid(frame_with_probability(ctx.cal_rows, p_cal).probability.to_numpy())
        curve = sweep(frame_with_probability(ctx.val_rows, p_val), ctx.events, horizon, ctx.protocol, grid)
        frontier.append(pareto_front(curve).assign(method=name, horizon=horizon))

    p = ctx.protocol
    info = {"version": VERSION[dataset], "dataset": dataset, "tabm": tabm,
            "seconds": round(time.perf_counter() - started), "ensemble_members": members,
            "n_features": len(ctx.features), "splits": split_summary(ctx),
            "protocol": {"map_threshold": p.map_threshold, "event_seconds": p.event_seconds,
                         "alarm_persistence": p.alarm_persistence,
                         "alarm_cooldown_seconds": p.alarm_cooldown_seconds,
                         "cadence_seconds": p.cadence_seconds}}
    archive_previous(info)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT / "method_comparison.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "calibration_audit.csv", index=False)
    pd.DataFrame(params).to_csv(OUT / "parameters.csv", index=False)
    pd.concat(frontier, ignore_index=True).to_csv(OUT / "pareto_frontier.csv", index=False)
    (OUT / "run.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"computed in {info['seconds']}s", flush=True)


def archive_previous(info):
    """The newest run owns reports/E08/; a run of another version moves to reports/E08/version/."""
    current = OUT / "run.json"
    if not current.exists():
        return
    old = json.loads(current.read_text(encoding="utf-8"))
    if (old["version"], old["dataset"], old["tabm"]) == (info["version"], info["dataset"], info["tabm"]):
        return
    target = OUT / "version" / f"{old['version']}_{old['dataset']}_tabm-{old['tabm']}"
    target.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (OUT / name).exists():
            shutil.move(str(OUT / name), str(target / name))
    report = target / "METHOD_COMPARISON.md"
    if report.exists():
        text = report.read_text(encoding="utf-8").replace("](../../", "](../../../../")
        report.write_text(f"> **Bản {old['version']} — đã thay thế** bởi "
                          f"[{info['version']}](../../METHOD_COMPARISON.md). Giữ lại để truy vết.\n\n" + text,
                          encoding="utf-8")
    print(f"archived {old['version']} outputs -> {target.relative_to(ROOT).as_posix()}", flush=True)


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


def pipeline_table(info):
    p, splits = info["protocol"], info["splits"]
    return ["| Tham số | Giá trị |", "|---|---|",
        f"| Dữ liệu | {DATASET_TEXT[info['dataset']]} |",
        "| Ca / bệnh nhân | " + " · ".join(f"{k} {v[0]} ca / {v[1]} BN" for k, v in splits.items()) + " |",
        "| Biến cố eligible (5 / 10 phút) | " + " · ".join(f"{k} {v[2]}/{v[3]}" for k, v in splits.items()) + " |",
        f"| Định nghĩa biến cố | MAP < {p['map_threshold']:g} mmHg liên tục ≥ {p['event_seconds']} s |",
        f"| Đặc trưng | {info['n_features']} numeric (bỏ static); `map_logistic` chỉ dùng "
        + ", ".join(f"`{f}`" for f in MAP_FEATURES) + " |",
        "| Imputation | median từng cột, tính trên FIT, điền y hệt cho FIT / CALIBRATION / VALIDATION |",
        f"| Calibration | StandardScaler → LogisticRegression {code(CALIBRATOR_PARAMS)} trên log-odds của "
        "điểm thô; fit trên CALIBRATION, riêng từng phương pháp × horizon |",
        f"| TabM | {TABM_TEXT[info['tabm']]} |",
        "| Lưới ngưỡng | 40 điểm cố định 0,01–0,99 + 201 quantile xác suất trên CALIBRATION |",
        "| Chọn ngưỡng | `recall_first` trên CALIBRATION: recall cao nhất → PPV cao hơn → FA/giờ thấp hơn |",
        "| Chọn phương pháp | trên CALIBRATION, cùng thứ tự `recall_first` |",
        "| Báo cáo | VALIDATION — không tham gia chọn ngưỡng hay chọn phương pháp |",
        f"| Chính sách cảnh báo | {p['alarm_persistence']} decision liên tiếp vượt ngưỡng · cooldown "
        f"{p['alarm_cooldown_seconds']} s · nhịp {p['cadence_seconds']} s |",
        f"| Khoảng tin cậy | bootstrap theo `subjectid`, {BOOTSTRAP} lần |",
        f"| Seed | model {MODEL_SEED} · augmentation {AUGMENT_SEED} |"]


def params_table(info, used):
    def value(name, horizon, key):
        row = used[used.method.eq(name) & used.horizon.eq(horizon)]
        return None if row.empty or key not in row or pd.isna(row.iloc[0][key]) else row.iloc[0][key]

    monotone = ", ".join(f"`{f}`" for f in MONOTONE_FEATURES)
    lines = ["| Phương pháp | Nhóm | Đặc trưng | Siêu tham số | Cân bằng lớp | Augmentation | "
             "Window dương khi train (5 / 10 phút) |", "|---|---|---|---|---|---|---|"]
    for spec in (s for s in METHODS if s.name in set(used.method)):
        if spec.family == "baseline":
            hyper = f"StandardScaler → LogisticRegression {code(LOGISTIC_PARAMS)}"
        elif spec.family == "lightgbm":
            hyper = f"{code(LIGHTGBM_PARAMS)}; đơn điệu giảm theo {monotone}"
        elif spec.family == "catboost":
            hyper = code(CATBOOST_PARAMS)
        else:
            hyper = (f"{code(TABM_EXPECTED)}; `{TABM_FIXED}`; dừng ở epoch "
                     f"{value(spec.name, HORIZONS[0], 'best_epoch'):.0f} ({TABM_TEXT[info['tabm']]})")
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
        positives = "—" if positives[0] is None else " / ".join(str(int(v)) for v in positives)
        features = f"{len(MAP_FEATURES)} (MAP)" if spec.family == "baseline" else str(info["n_features"])
        lines.append(f"| `{spec.name}` | {FAMILY[spec.family]} | {features} | {hyper} | {weighting} | "
                     f"{augmentation} | {positives} |")
    lines.append(f"| `{ENSEMBLE_NAME}` | ensemble | — | trung bình xác suất đã calibrate của "
                 + ", ".join(f"`{m}`" for m in info["ensemble_members"]) + " | — | — | — |")
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


def winner_lines(info, comparison, winners):
    first = comparison[comparison.policy.eq("recall_first")]
    cal_events = dict(zip(HORIZONS, info["splits"]["CALIBRATION"][2:]))
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


def body(info, comparison, used, winners):
    totals = dict(zip(HORIZONS, info["splits"]["VALIDATION"][2:]))
    lines = ["### Phương pháp được chọn", "", *winner_lines(info, comparison, winners), "",
             "### Tham số chung (áp dụng cho mọi phương pháp)", "", *pipeline_table(info), "",
             "### Tham số từng phương pháp", "", *params_table(info, used), ""]
    for horizon in HORIZONS:
        lines += [f"### Kết quả {horizon // 60} phút — `recall_first`, VALIDATION ({totals[horizon]} biến cố)",
                  "", *results_table(comparison, horizon, winners[horizon]), ""]
    step = sorted(100 / v for v in totals.values())
    untouched = ("chưa mở pilot_test/global test" if info["dataset"] == "development300"
                 else "global test `unseen_test` không được đọc")
    return lines + ["**Đọc bảng:**", "",
        "- ★ = phương pháp chọn trên CALIBRATION; VALIDATION chỉ để báo cáo.",
        f"- Mỗi biến cố trên VALIDATION = {step[0]:.1f}–{step[-1]:.1f} điểm recall; chênh lệch nằm gọn trong "
        "CI95 **không** đủ kết luận phương pháp nào hơn.",
        "- AUROC / AP / ECE không phụ thuộc ngưỡng — dùng để so khả năng phân biệt tách khỏi điểm vận hành.",
        f"- Development validation, **không phải** bằng chứng xác nhận độc lập; {untouched}."]


def write_readme(info, lines):
    block = ['<a id="e08"></a>', "",
             f"## E08 {info['version']} — so sánh phương pháp trên {DATASET_TITLE[info['dataset']]}", "",
             "Mọi phương pháp đi qua đúng một quy trình, chỉ khác nhau ở bản thân phương pháp; FA/giờ đi "
             "kèm chính sách `recall_first` đã được chấp nhận. Sinh tự động bởi "
             "[`scripts/e08/run_comparison.py`](scripts/e08/run_comparison.py) từ chính các tham số đã chạy. "
             "[Báo cáo đầy đủ](reports/E08/METHOD_COMPARISON.md) · "
             "[các version và lỗi đã sửa](scripts/e08/README.md)", "", *lines]
    text = README.read_text(encoding="utf-8")
    start, end = text.index(README_BEGIN) + len(README_BEGIN), text.index(README_END)
    README.write_text(text[:start] + "\n" + "\n".join(block) + "\n" + text[end:], encoding="utf-8")


def write_report(info, lines, comparison, audit, frontier):
    out = [f"# E08 {info['version']} — so sánh phương pháp trên {DATASET_TITLE[info['dataset']]}", "",
           f"Chạy {info['seconds']} giây · TabM: {TABM_TEXT[info['tabm']]}. "
           "[Tóm tắt trong README](../../README.md#e08) · [các version](../../scripts/e08/README.md) · "
           "[script](../../scripts/e08/run_comparison.py)", "", *lines, "",
           "## Cái giá của ràng buộc FA/giờ ≤ 0,5 (`fa_budget`, cùng model)", "",
           "| Phương pháp | Phút | Recall `recall_first` | Recall `fa_budget` | Δ recall | PPV `fa_budget` | FA/giờ `fa_budget` |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    first = comparison[comparison.policy.eq("recall_first")]
    budget = comparison[comparison.policy.eq("fa_budget")]
    merged = first.merge(budget, on=["method", "horizon"], suffixes=("_first", "_budget"))
    for r in merged.sort_values(["horizon", "recall_first"], ascending=[True, False]).itertuples():
        out.append(f"| `{r.method}` | {r.horizon // 60} | {fmt(r.recall_first)} | {fmt(r.recall_budget)} | "
                   f"{(r.recall_budget or 0) - (r.recall_first or 0):+.3f} | {fmt(r.ppv_budget)} | "
                   f"{fmt(r.fa_per_hour_budget)} |")
    out += ["", "## Mặt Pareto của phương pháp được chọn (VALIDATION)", "",
            "Các điểm không bị điểm nào trội hơn đồng thời về recall, PPV và FA/giờ — chọn điểm nào là "
            "quyết định lâm sàng, không phải kỹ thuật.", "",
            "| Phút | Phương pháp | Ngưỡng | Recall | Bắt được | PPV | FA/giờ |", "|---:|---|---:|---:|---:|---:|---:|"]
    for r in frontier.drop_duplicates(["horizon", "event_sensitivity"]).itertuples():
        out.append(f"| {r.horizon // 60} | `{r.method}` | {fmt(r.threshold)} | {fmt(r.event_sensitivity)} | "
                   f"{int(r.events_detected)}/{int(r.events_eligible)} | {fmt(r.alarm_ppv)} | "
                   f"{fmt(r.false_alarms_per_hour)} |")
    out += ["", "## Kiểm chứng calibration", "",
            "Mean xác suất sau calibrate phải bám prevalence thật, nếu không mọi so sánh ngưỡng và phép trung "
            "bình ensemble đều sai (lỗi F2 của v1).", "",
            "| Phương pháp | Phút | Mean thô | Mean sau calibrate | Prevalence |", "|---|---:|---:|---:|---:|"]
    for r in audit.itertuples():
        out.append(f"| `{r.method}` | {r.horizon // 60} | {r.raw_mean:.4f} | {r.calibrated_mean:.4f} | "
                   f"{r.prevalence:.4f} |")
    (OUT / "METHOD_COMPARISON.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def render():
    info = json.loads((OUT / "run.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(OUT / "method_comparison.csv")
    used = pd.read_csv(OUT / "parameters.csv")
    lines = body(info, comparison, used, pick_winners(comparison))
    write_report(info, lines, comparison, pd.read_csv(OUT / "calibration_audit.csv"),
                 pd.read_csv(OUT / "pareto_frontier.csv"))
    write_readme(info, lines)
    print(f"rendered E08 {info['version']} ({info['dataset']}) into README.md and METHOD_COMPARISON.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=DATASETS, default="development300")
    parser.add_argument("--tabm", choices=TABM_MODES, default="frozen")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if not args.render_only:
        compute(args.dataset, args.tabm)
    render()


if __name__ == "__main__":
    main()
