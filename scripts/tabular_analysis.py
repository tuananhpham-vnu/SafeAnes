"""NB02 part d (02d): combine tabular results, choose W*, SHAP, ablation, confidence.

    python scripts/tabular_analysis.py --inputs <dir of 02a> <dir of 02b> <dir of 02c> [--work .]

`--inputs` are work directories written by train_tabular.py (on Kaggle: the
outputs of notebooks 02a, 02b, 02c attached as inputs; locally: the repo root).
If HF is set up, missing combinations are pulled from runs_repo first.

Writes (only this notebook writes them):
  reports/tabular_validation.csv, reports/calibration_check_tabular.csv, reports/w_star.json,
  reports/shap_W<W>_h<h>.csv, reports/shap_groups.csv, reports/ablation.csv,
  reports/tabular_confidence/W<W>_h<h>.csv (+ cut points json)
Stops if any of the 65 combinations is missing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uc04.bootstrap import Bootstrap, ci, paired_ci
from uc04.columns import ABLATION_GROUPS, ablation_groups, window_cols
from uc04.confidence import check_by_risk_decile, seed_levels
from uc04.hub import hub_from_config
from uc04.io import write_json
from uc04.metrics import row_metrics
from uc04.provenance import make_provenance, write_provenance
from uc04.runtime import common_args, load_data, samples_inputs, setup
from uc04.tabular import (MODELS, TabularData, combo_name, evaluate_probs, fit_and_calibrate, predict_split,
                          _write_csv)

ABLATION_HORIZONS = (300, 600)
CONFIDENCE_HORIZONS = (300, 600)


def expected_names(windows, horizons) -> list[str]:
    return [combo_name(m, W, h) for m in MODELS for h in horizons for W in windows
            if not (m == "map_threshold" and W != windows[0])]


def gather(inputs: list[Path], sub: str, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    found = {}
    for d in inputs:
        for f in (d / "reports" / sub).glob("*.csv") if (d / "reports" / sub).exists() else []:
            found.setdefault(f.stem, f)
    rows = [pd.read_csv(found[n]) for n in names if n in found]
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), [n for n in names if n not in found]


def find_model(inputs: list[Path], rel: str) -> Path | None:
    for d in inputs:
        p = d / "artifacts" / rel / "model.joblib"
        if p.exists():
            return p
    return None


def group_of(col: str) -> str:
    for g, prefixes in ABLATION_GROUPS.items():
        if col.startswith(prefixes):
            return g
    return "beat_quality"  # bt_valid_frac, bt_n_beats


def main(argv=None) -> int:
    ap = common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--inputs", nargs="+", help="work dirs of 02a/02b/02c (default: --work)")
    ap.add_argument("--bootstrap", type=int)
    ap.add_argument("--skip", nargs="*", default=[], choices=["shap", "ablation", "confidence"])
    args = ap.parse_args(argv)
    cfg, samples, work, push, subset = setup(args)
    inputs = [Path(p) for p in (args.inputs or [work])]
    windows, horizons = list(cfg.features.windows_seconds), list(cfg.labels.horizons_seconds)
    boot = cfg.evaluation.bootstrap_repeats_validation if args.bootstrap is None else args.bootstrap
    hub = hub_from_config(cfg, work, push=push)
    t0 = time.time()

    # ---- 1. combine; pull missing combos from HF when available
    names = expected_names(windows, horizons)
    table, missing = gather(inputs, "tabular_validation", names)
    if missing and hub.enabled:
        for n in missing:
            try:
                hub.pull(f"reports/tabular_validation/{n}.csv", local_dir=work)
                hub.pull(f"reports/calibration_check/{n}.csv", local_dir=work)
            except Exception as exc:
                print(f"  could not pull {n}: {exc!r}")
        table, missing = gather([*inputs, work], "tabular_validation", names)
    if missing:
        print(f"missing {len(missing)} of {len(names)} combinations: {missing}", file=sys.stderr)
        return 1
    mt = table[table.model == "map_threshold"]
    table = pd.concat([table[table.model != "map_threshold"]] + [mt.assign(window=W) for W in windows],
                      ignore_index=True).sort_values(["model", "window", "horizon"])
    rep = work / "reports"
    _write_csv(rep / "tabular_validation.csv", table)
    calib, _ = gather([*inputs, work], "calibration_check", names)
    _write_csv(rep / "calibration_check_tabular.csv", calib)
    written = ["reports/tabular_validation.csv", "reports/calibration_check_tabular.csv"]

    # ---- 2. W*
    lw = table[(table.model == "lgbm_wave") & table.horizon.isin([300, 600]) & (table.status == "ok")]
    per_w = lw.groupby("window").event_sensitivity.mean()
    w_star = int(per_w.idxmax())
    write_json(rep / "w_star.json", {"w_star": w_star, "mean_event_sensitivity_5_10min": per_w.to_dict(),
                                     "rule": "lgbm_wave, mean event sensitivity at 5 and 10 min on validation"})
    written.append("reports/w_star.json")
    print(f"W* = {w_star}  ({per_w.round(4).to_dict()})")

    # ---- 3. SHAP (lgbm_wave, every W and h, 50k validation rows)
    if "shap" not in args.skip:
        import shap
        shares = []
        for W in windows:
            cols = window_cols(W)
            lab, feat, _ = load_data(samples, cols, subset, splits=("validation",))
            lab, feat = lab["validation"], feat["validation"]
            rows = np.flatnonzero(lab["eligible"].to_numpy())
            rng = np.random.default_rng(cfg.seed)
            rows = rng.choice(rows, min(cfg.lightgbm.shap_rows, len(rows)), replace=False)
            for h in horizons:
                path = find_model([*inputs, work], f"tabular/lgbm_wave/W{W}/h{h}")
                if path is None:
                    print(f"  SHAP: no model for W{W} h{h}")
                    continue
                est = joblib.load(path)["model"]
                sv = shap.TreeExplainer(est.model).shap_values(feat.iloc[rows][est.columns])
                sv = sv[1] if isinstance(sv, list) else sv
                imp = pd.DataFrame({"column": est.columns, "mean_abs_shap": np.abs(sv).mean(0)})
                imp["group"] = imp.column.map(group_of)
                _write_csv(rep / f"shap_W{W}_h{h}.csv", imp.sort_values("mean_abs_shap", ascending=False))
                written.append(f"reports/shap_W{W}_h{h}.csv")
                g = imp.groupby("group").mean_abs_shap.sum()
                shares += [{"window": W, "horizon": h, "group": k, "share": v / g.sum()} for k, v in g.items()]
        _write_csv(rep / "shap_groups.csv", pd.DataFrame(shares))
        written.append("reports/shap_groups.csv")
        print(f"SHAP done ({time.time() - t0:.0f} s)")

    need_retrain = [s for s in ("ablation", "confidence") if s not in args.skip]
    if need_retrain:
        labels, feats, events = load_data(samples, window_cols(w_star), subset)
        data = TabularData(w_star, labels, feats, events)

    # ---- 4. ablation at W*, 5 and 10 min, paired bootstrap vs the full lgbm_wave
    if "ablation" not in args.skip:
        full_cols = window_cols(w_star)
        rows_out = []
        for h in ABLATION_HORIZONS:
            est, platt, status = fit_and_calibrate(cfg, "lgbm_wave", data, h)
            if status != "ok":
                continue
            p_full = predict_split(est, platt, data.labels["validation"], data.features["validation"])
            s_full, _, res_full, ev = evaluate_probs(cfg, data, h, p_full, 0, cfg.seed)
            subj_rows = data.labels["validation"]["subjectid"].to_numpy()[ev.order][ev.eligible]
            bs = Bootstrap(subj_rows, res_full.case_table["subjectid"].to_numpy(), boot, cfg.seed)
            d_full = bs.metrics(res_full.case_table, ev.y[ev.eligible], ev.sort(p_full)[ev.eligible])
            rows_out.append({"horizon": h, "dropped": "none", **{k: s_full.get(k) for k in
                             ("event_sensitivity", "false_alarms_per_hour", "auroc", "auprc")}})
            for gname, gcols in ablation_groups(full_cols).items():
                keep = [c for c in full_cols if c not in set(gcols)]
                est_a, platt_a, st = fit_and_calibrate(cfg, "lgbm_wave", data, h, columns=keep)
                if st != "ok":
                    continue
                p_a = predict_split(est_a, platt_a, data.labels["validation"], data.features["validation"])
                s_a, _, res_a, _ = evaluate_probs(cfg, data, h, p_a, 0, cfg.seed)
                d_a = bs.metrics(res_a.case_table, ev.y[ev.eligible], ev.sort(p_a)[ev.eligible])
                diff = paired_ci(d_a, d_full)
                rows_out.append({"horizon": h, "dropped": gname, "n_dropped_columns": len(gcols),
                                 **{k: s_a.get(k) for k in ("event_sensitivity", "false_alarms_per_hour", "auroc",
                                                            "auprc")},
                                 **{f"delta_{k}": v for k, v in diff.items()}})
                print(f"  ablation h{h} -{gname:<11} sens {s_a['event_sensitivity']:.3f} "
                      f"(full {s_full['event_sensitivity']:.3f})", flush=True)
        _write_csv(rep / "ablation.csv", pd.DataFrame(rows_out))
        written.append("reports/ablation.csv")

    # ---- 5. confidence levels from 5 seeds with subsampling (W*, 5 and 10 min)
    if "confidence" not in args.skip:
        out_dir = rep / "tabular_confidence"
        for h in CONFIDENCE_HORIZONS:
            probs = []
            for seed in cfg.lightgbm.confidence_seeds:
                est, platt, st = fit_and_calibrate(cfg, "lgbm_wave", data, h, seed=seed,
                                                   extra=cfg.lightgbm.confidence_params)
                if st == "ok":
                    probs.append(predict_split(est, platt, data.labels["validation"], data.features["validation"]))
            if len(probs) < 2:
                continue
            lab = data.labels["validation"]
            el = lab["eligible"].to_numpy(bool)
            mean_p, sd, level, cuts = seed_levels(np.vstack(probs), el)  # SD on the logit scale
            y = lab[f"y_{h}"].to_numpy()
            rows_c = []
            for lv in ("high", "medium", "low"):
                m = el & (level == lv)
                rows_c.append({"window": w_star, "horizon": h, "confidence": lv, "rows": int(m.sum()),
                               "sd_logit_max": float(np.nanmax(sd[m])) if m.any() else None,
                               "mean_probability": float(np.nanmean(mean_p[m])) if m.any() else None,
                               **row_metrics(y[m], mean_p[m])})
            dec_tab, verdict = check_by_risk_decile(y[el], mean_p[el], level[el])
            _write_csv(out_dir / f"W{w_star}_h{h}.csv", pd.DataFrame(rows_c))
            _write_csv(out_dir / f"W{w_star}_h{h}_by_risk_decile.csv", dec_tab)
            write_json(out_dir / f"W{w_star}_h{h}_cuts.json", {"scale": "logit", "sd_tertiles": cuts.tolist(),
                                                               "seeds": len(probs), "check": verdict})
            written += [f"reports/tabular_confidence/W{w_star}_h{h}.csv",
                        f"reports/tabular_confidence/W{w_star}_h{h}_by_risk_decile.csv",
                        f"reports/tabular_confidence/W{w_star}_h{h}_cuts.json"]
            print(f"  confidence h{h}: check {'PASSED' if verdict['passed'] else 'FAILED'} "
                  f"(low worse in {verdict['deciles_low_worse_calibration']}/{verdict['deciles_compared']} deciles; "
                  f"pooled calibration error low {verdict['pooled_calibration_error_low']:.4f} vs high "
                  f"{verdict['pooled_calibration_error_high']:.4f}; Brier low {verdict['pooled_brier_low']:.4f} vs "
                  f"high {verdict['pooled_brier_high']:.4f})", flush=True)
        print(f"confidence done ({time.time() - t0:.0f} s)")

    write_provenance(rep / "tabular_analysis", make_provenance(cfg, "tabular_analysis",
                                                              inputs=samples_inputs(samples)))
    written.append("reports/tabular_analysis")
    hub.register(*written)
    hub.push(written, message="02d tabular analysis")
    pd.set_option("display.width", 220)
    show = table[table.horizon.isin([300, 600]) & (table.window == w_star)][
        ["model", "horizon", "event_sensitivity", "event_sensitivity_lo", "event_sensitivity_hi",
         "false_alarms_per_hour", "alarm_ppv", "auroc", "prevalence"]]
    print("\nValidation at W* (5 and 10 min):\n" + show.to_string(index=False))
    print(f"done in {time.time() - t0:.0f} s; outputs in {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
