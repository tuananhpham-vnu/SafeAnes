"""Fixed-comparison charts and one deterministic case replay from E05 outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports, artifacts = ROOT / "reports/E05", ROOT / "artifacts/E05"
    r = json.loads((reports / "comparison.json").read_text())
    names = [("MAP", "map", 20260917), ("Logistic", "logistic", 20260917), ("LightGBM", "lightgbm", 20260917),
             ("CB seed17", "catboost", 20260917), ("CB seed18", "catboost", 20260918), ("CB seed19", "catboost", 20260919)]
    assert {f"{name}_all_{seed}_{h}" for _, name, seed in names for h in (300, 600)}.issubset(r)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for col, h in enumerate((300, 600)):
        for row, field in enumerate(("event_sensitivity", "alarm_ppv", "false_alarms_per_hour")):
            values = [r[f"{name}_all_{seed}_{h}"]["fixed"]["scopes"]["new_patients"]["metrics"][field] for _, name, seed in names]
            ax = axes[row, col]
            ax.bar(range(6), [0 if x is None else x for x in values], color=["#999999"]*3+["#2166ac"]*3)
            for index, value in enumerate(values):
                if value is None:
                    ax.text(index, .01, "N/A", ha="center", fontsize=8)
            ax.set_xticks(range(6), [x[0] for x in names], rotation=25, ha="right")
            ax.set_ylabel(field)
            if row < 2:
                ax.set_ylim(0, 1)
            else:
                ax.axhline(.5, color="firebrick", linestyle="--", label="Validation budget")
                ax.legend(fontsize=8)
            ax.set_title(f"{h//60} min, fixed policy, new patients")
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("E05 / release v0.2: same 37 new patients; no best-seed selection")
    fig.tight_layout()
    fig.savefig(reports / "primary_comparison.png", dpi=150)
    plt.close(fig)

    # Select the lowest case ID with an eligible event, without ranking model errors.
    events = pd.read_csv(ROOT / "data/development300/events.csv")
    first = pd.read_csv(artifacts / "catboost_all_20260917_300/test.csv.gz")
    fresh_ids = first.loc[~first.historical_subject, "caseid"].unique()
    candidates = events[events.caseid.isin(fresh_ids) & events.eligible_300]
    caseid = int(candidates.caseid.min())
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    example = first[first.caseid.eq(caseid)]
    axes[0].plot(example.time/60, example.map_current, label="Causal MAP at decision")
    axes[0].axhline(65, color="firebrick", linestyle="--")
    axes[0].set_ylabel("MAP (mmHg)")
    for index, h in enumerate((300, 600), start=1):
        key = f"catboost_all_20260917_{h}"
        pred = pd.read_csv(artifacts / key / "test.csv.gz")
        pred = pred[pred.caseid.eq(caseid)]
        axes[index].plot(pred.time/60, pred.probability, label="Calibrated risk")
        threshold = r[key]["fixed"]["selection_on_validation"]["metrics"]["threshold"]
        axes[index].axhline(threshold, color="darkorange", linestyle="--", label="Validation threshold")
        for alarm in json.loads((artifacts / key / "fixed_new_patients_alarms.json").read_text()):
            if alarm["caseid"] == caseid:
                axes[index].axvline(alarm["time"]/60, color="purple", alpha=.4)
        axes[index].set_ylabel(f"{h//60} min risk")
        axes[index].legend(fontsize=8)
    for ax in axes:
        for event in events[events.caseid.eq(caseid)].itertuples():
            ax.axvspan(event.onset/60, event.end/60, color="red", alpha=.12)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Minutes since case start")
    fig.suptitle(f"E05 case {caseid}: CB seed17, red = labeled IOH; purple = emitted alarm")
    fig.tight_layout()
    fig.savefig(reports / "case_replay.png", dpi=150)
    plt.close(fig)
    (reports / "FIGURES.md").write_text("# E05 — hình từ kết quả thật\n\n![Primary comparison](primary_comparison.png)\n\nCác seed dùng cùng bệnh nhân; không chọn seed tốt nhất theo test. N/A nghĩa không có cảnh báo evaluable.\n\n![Case replay](case_replay.png)\n\nCa minh họa được chọn bằng caseid nhỏ nhất có event đủ điều kiện trong nhóm mới; model/seed cố định, không chọn ca theo mức đẹp của dự báo. Vùng đỏ là episode hậu nghiệm dùng cho đánh giá, không phải input biết trước của model. Protocol có gộp episode cách nhau dưới 120 giây khi dữ liệu giữa chúng đủ hỗ trợ; vùng đỏ vì vậy có thể chứa đoạn MAP hồi phục ngắn.\n\nCa 55 có những spike MAP trên 250 mmHg vẫn nằm trong bounds hiện tại. Đây là dấu hiệu cần xem lại QC/nguồn tín hiệu, chưa đủ để kết luận artifact. Không loại ca hoặc sửa bounds sau khi xem test; đánh giá QC nâng cao cần đợt development đăng ký riêng.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
