"""Explanation cards for the alarms of the 15-minute model -> reports/weekly/week14/mechanism/.

  $env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
  python scripts/mechanism_cards.py [--train-cases 700]

Risk model: LightGBM on the 66 monitor + 31 context inputs (as lgbm_context of week 13, W = 120, h = 15 min), fitted on
a train subsample, Platt-calibrated on the calibration split. Alarms: the week-13 policy (2 consecutive rows above the
threshold, cooldown 5 min, threshold with at most 1 false alarm per hour on validation).

Outputs
  mechanisms.csv      share of each physiological pattern for true alarms, false alarms, missed events (5 min before
                      onset) and control rows far from any event
  findings.csv        share of each clinical finding in the same groups
  alarm_cards.csv     one row per alarm: risk, decomposition, pattern, findings, top model themes
  cards.md            readable cards for example alarms
  figures/case_*.png  timelines around example events: MAP, risk, HR / SV / SVR proxies, anaesthetic depth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uc04 import feature_eda as fe
from uc04 import mechanism as mc
from uc04.calibration import Platt
from uc04.columns import CONTEXT_COLS, numeric_cols
from uc04.metrics import Evaluator, summarize_counts
from uc04.thresholds import candidates, choose_threshold

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data/samples_v3"
MANIFEST = ROOT / "data/vitaldb_full/meta/cohort_manifest.csv"
OUT = ROOT / "reports/weekly/week14/mechanism"
FIG = OUT / "figures"
SEED, W, H = 20260917, 120, 900
LAGGED = ["map_mean_w120", "sbp_mean_w120", "dbp_mean_w120", "hr_mean_w120"]


def physiology(feats: pd.DataFrame) -> pd.DataFrame:
    f = mc.add_lagged(feats, LAGGED)
    d = mc.decompose(f)
    d["mechanism"] = mc.primary_mechanism(d)
    return pd.concat([f[["caseid", "time"]], d, mc.clinical_flags(f)], axis=1)


def shares(frame: pd.DataFrame, groups: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    mech = pd.DataFrame({g: frame.loc[m, "mechanism"].value_counts(normalize=True) for g, m in groups.items()})
    mech = mech.reindex(list(mc.MECHANISMS)).fillna(0)
    mech.insert(0, "pattern", [mc.MECHANISMS[k] for k in mech.index])
    flags = [*mc.FLAG_TEXT, "anaesthetic_deepening", *mc.PROFILE_TEXT]
    fl = pd.DataFrame({g: frame.loc[m, flags].mean() for g, m in groups.items()})
    fl.insert(0, "finding", [mc.FLAG_TEXT.get(k, mc.PROFILE_TEXT.get(k, "Thuốc mê sâu thêm (bất kỳ)")) for k in flags])
    n = pd.Series({g: int(m.sum()) for g, m in groups.items()}, name="n")
    return mech, fl, n


def timeline(case: pd.DataFrame, onset: float, alarm_t: float | None, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    c = case[(case["time"] >= onset - 2400) & (case["time"] <= onset + 300)].copy()
    t = (c["time"] - onset) / 60
    fig, ax = plt.subplots(4, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 0.8, 1.1, 0.9]})
    ax[0].plot(t, c["map_current"], color="#c0392b", lw=1)
    ax[0].axhline(65, color="grey", ls="--", lw=0.8)
    ax[0].set_ylabel("MAP (mmHg)")
    ax[1].plot(t, c["risk"] * 100, color="#2a6fdb", lw=1)
    ax[1].set_ylabel("Nguy cơ 15 ph (%)")
    lhr, lsv, lsvr = mc.lz(c["sbp_mean_w120"], c["dbp_mean_w120"], c["map_mean_w120"], c["hr_mean_w120"])
    for v, name, color in ((lhr, "HR", "#16a085"), (lsv, "SV proxy", "#8e44ad"), (lsvr, "SVR proxy", "#e67e22")):
        x = v.to_numpy()
        base = np.median(x[np.isfinite(x)][:6]) if np.isfinite(x).any() else np.nan  # first 3 min with data
        ax[2].plot(t, (np.exp(x - base) - 1) * 100, color=color, lw=1, label=name)
    ax[2].axhline(0, color="grey", lw=0.6)
    ax[2].set_ylabel("% so với đầu cửa sổ")
    ax[2].legend(fontsize=7, frameon=False, ncol=3)
    drew = False
    for col, name, color in (("f1b_ppf", "Ce propofol (µg/mL)", "#34495e"), ("f1b_rftn", "Ce remifentanil (ng/mL)", "#7f8c8d"),
                             ("f1b_mac", "MAC", "#d35400")):
        if col in c and (c[col].fillna(0) > 0).any():
            ax[3].plot(t, c[col], lw=1, label=name, color=color)
            drew = True
    if drew:
        ax[3].legend(fontsize=7, frameon=False)
    ax[3].set_ylabel("Thuốc mê")
    ax[3].set_xlabel("Phút so với lúc bắt đầu tụt")
    for a in ax:
        a.axvline(0, color="#c0392b", lw=0.8)
        if alarm_t is not None:
            a.axvline((alarm_t - onset) / 60, color="#2a6fdb", lw=0.8, ls=":")
    ax[0].set_title(title, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    global SAMPLES, MANIFEST, OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cases", type=int, default=700)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--samples", type=Path, default=SAMPLES)
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="VitalDB clinical table (cases API columns)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    SAMPLES, MANIFEST, OUT = args.samples, args.manifest, args.out
    FIG = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    model_cols = numeric_cols(W) + list(CONTEXT_COLS)
    extra = [c for c in dict.fromkeys(mc.NEEDED) if c not in model_cols]
    cases = fe.sample_cases(SAMPLES / "labels/train.parquet", args.train_cases, SEED)
    lab, X = fe.load_rows(SAMPLES, "train", model_cols, MANIFEST, cases)
    m = fe.known(lab, H)
    booster = fe.fit_lgbm(X.loc[m, model_cols], lab[f"y_{H}"].to_numpy()[m], W, SEED)
    del lab, X
    clab, CX = fe.load_rows(SAMPLES, "calibration", model_cols, MANIFEST)
    mc_ = fe.known(clab, H)
    platt = Platt().fit(booster.predict(CX.loc[mc_, model_cols], raw_score=True), clab[f"y_{H}"].to_numpy()[mc_])
    del clab, CX
    print("model ready", flush=True)

    vlab, V = fe.load_rows(SAMPLES, "validation", model_cols + extra, MANIFEST, all_main=True)
    elig = vlab["eligible"].to_numpy(bool)
    raw = booster.predict(V[model_cols], raw_score=True)
    risk = np.where(elig, platt.predict(raw), np.nan)
    known = fe.known(vlab, H)
    print("validation AUROC", fe.auroc(vlab[f"y_{H}"].to_numpy()[known & elig], raw[known & elig]), flush=True)

    events = pd.read_parquet(SAMPLES / "events.parquet")
    ev = Evaluator(vlab, events, H, persistence=2, cooldown=300.0, early_lead=300.0)
    ps = ev.sort(risk)
    choice = choose_threshold(candidates(risk[elig]), lambda t: ev.quick(ps, t), budget=1.0)
    res = ev.evaluate(ps, choice["threshold"])
    summ = summarize_counts(res.case_table.sum(numeric_only=True))
    print({k: summ[k] for k in ("event_sensitivity", "false_alarms_per_hour", "alarm_ppv")}, flush=True)

    P = physiology(V)
    P["risk"] = risk
    key = pd.MultiIndex.from_frame(P[["caseid", "time"]])
    alarms = res.alarms[res.alarms["kind"] != "censored"]
    a_idx = key.get_indexer(pd.MultiIndex.from_frame(alarms[["caseid", "time"]]))
    true_rows = a_idx[(alarms["kind"] == "true").to_numpy()]
    false_rows = a_idx[(alarms["kind"] == "false").to_numpy()]

    vev = events[(events["split"] == "validation") & events[f"eligible_{H}"] & ~events["suspect_artefact"]]
    vev = vev[vev["caseid"].isin(vlab["caseid"])]
    detected_onsets = set()
    for (c, t, lead) in alarms.loc[alarms["kind"] == "true", ["caseid", "time", "lead"]].itertuples(index=False):
        detected_onsets.add((c, t + lead))
    missed = vev[[(c, o) not in detected_onsets for c, o in zip(vev["caseid"], vev["onset"])]]
    # last prediction row at or before onset - 5 min of each missed event
    grid = P[["caseid", "time"]].assign(row=np.arange(len(P))).sort_values("time")
    q = missed[["caseid", "onset"]].assign(time=(missed["onset"] - 300).astype(float)).sort_values("time")
    hit = pd.merge_asof(q, grid.astype({"time": float}), on="time", by="caseid", direction="backward",
                        tolerance=60.0)
    pre = hit["row"].dropna().astype(int).to_numpy()
    y = vlab[f"y_{H}"].to_numpy()
    far = elig & (y == 0) & (vlab["time"].to_numpy() > 0)
    rng = np.random.default_rng(SEED)
    ctrl = rng.choice(np.flatnonzero(far), min(20000, int(far.sum())), replace=False)

    def mask(idx):
        b = np.zeros(len(P), bool)
        b[idx] = True
        return b

    groups = {"Cảnh báo đúng": mask(true_rows), "Cảnh báo sai": mask(false_rows),
              "Đợt bị bỏ sót (5 ph trước)": mask(pre), "Đối chứng (y15 = 0)": mask(ctrl)}
    mech, flags, n = shares(P, groups)
    mech.to_csv(OUT / "mechanisms.csv", index=False, float_format="%.4f")
    flags.to_csv(OUT / "findings.csv", index=False, float_format="%.4f")

    # cards for every alarm, with model-layer themes from TreeSHAP
    rows = np.r_[true_rows, false_rows]
    contrib = booster.predict(V.loc[rows, model_cols], pred_contrib=True)[:, :-1]
    cards, out = [], []
    for i, r in enumerate(rows):
        card = mc.make_card(V.loc[r], P.loc[r, ["dlog_map", "dlog_hr", "dlog_sv", "dlog_svr"]], P.at[r, "mechanism"],
                            P.loc[r], risk[r], pd.Series(contrib[i], index=model_cols))
        kind = "true" if i < len(true_rows) else "false"
        cards.append((kind, card))
        out.append({"caseid": card.caseid, "time": card.time, "kind": kind, "risk": card.risk,
                    "mechanism": card.mechanism, **card.decomposition, "findings": " | ".join(card.findings),
                    "profile": ", ".join(card.profile),
                    "top_theme": card.model_themes[0][0] if card.model_themes else ""})
    alarm_tab = pd.DataFrame(out)
    alarm_tab.to_csv(OUT / "alarm_cards.csv", index=False, float_format="%.4f")
    themes = alarm_tab.groupby("kind")["top_theme"].value_counts(normalize=True).unstack(0).fillna(0)
    themes.to_csv(OUT / "model_themes.csv", float_format="%.4f")

    # examples: one true alarm per pattern (most confident first), plus one false alarm
    ex = []
    for k in ("vasodilation", "low_sv", "bradycardia", "stable_low", "mixed"):
        cand = [c for kind, c in cards if kind == "true" and c.mechanism == k]
        cand.sort(key=lambda c: (mc.FLAG_TEXT["recent_event"] in c.findings, -c.risk))
        ex += cand[:1 if k != "vasodilation" else 2]
    fa = [c for kind, c in cards if kind == "false"]
    fa.sort(key=lambda c: (mc.FLAG_TEXT["recent_event"] in c.findings, -c.risk))
    ex += fa[:1]
    lines = ["# Thẻ giải thích cảnh báo (ví dụ, validation)", "",
             f"Mô hình: LightGBM 66 chỉ số + 31 ngữ cảnh, W = 120 s, mốc 15 phút. Ngưỡng {choice['threshold']:.3f} "
             f"(≤ 1 cảnh báo sai/giờ): độ nhạy theo đợt {summ['event_sensitivity']:.1%}, "
             f"{summ['false_alarms_per_hour']:.2f} cảnh báo sai/giờ, PPV {summ['alarm_ppv']:.1%}.", ""]
    case_rows = V.assign(risk=risk)
    for j, c in enumerate(ex[:args.examples + 1]):
        kind = next(k for k, cc in cards if cc is c)
        lead = alarms[(alarms["caseid"] == c.caseid) & (alarms["time"] == c.time)]["lead"].iloc[0]
        onset = c.time + lead if np.isfinite(lead) else c.time + 600
        fig = f"case_{c.caseid}_{int(c.time)}.png"
        title = (f"Ca {c.caseid} — {'cảnh báo đúng' if kind == 'true' else 'cảnh báo sai'}, "
                 f"{mc.MECHANISMS[c.mechanism]}" + (f", báo trước {lead / 60:.1f} ph" if np.isfinite(lead) else ""))
        timeline(case_rows[case_rows["caseid"] == c.caseid], onset, c.time, FIG / fig,
                 title if kind == "true" else title + " (trục 0 = cảnh báo + 10 ph)")
        lines += [f"## Ví dụ {j + 1}: {title}", "", "```", c.text(), "```", "", f"![](figures/{fig})", ""]
    (OUT / "cards.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({
        "threshold": choice["threshold"], **{k: summ[k] for k in ("event_sensitivity", "false_alarms_per_hour",
                                                                    "alarm_ppv", "events_detected", "events_eligible")},
        "n": n.to_dict(), "train_cases": len(cases)}, indent=2, ensure_ascii=False))
    print(mech.to_string(), "\n", flags.to_string(), "\n", n.to_string())


if __name__ == "__main__":
    main()
