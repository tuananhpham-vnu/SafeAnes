"""Stage 1.4 trial: events, labels, eligibility and PPV on a few TRAIN cases.

    python scripts/try_labels.py [--cases 12 345 678] [--n 3]

Refuses any case whose split is not "train" (principle 1). Prints a summary per
case and sanity checks on clocks (beat `avail`, 2 s grid) before the full run.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from uc04.columns import WINDOWS
from uc04.config import load_config
from uc04.events import detect_events
from uc04.io import load_case, read_features, read_qc
from uc04.labels import LabelRules, label_case
from uc04.ppv import beat_table, ppv30_features


def run_case(prep, row, cfg) -> dict:
    t0 = time.time()
    z = load_case(prep, row.caseid)
    start = float(z["start"])
    lm = z["label_map"]
    channels = [str(c) for c in z["channels"]]
    grid_map = z["values"][:, channels.index("map")]
    f = read_features(prep, row.caseid, ["time", "map_current", "map_missing_w120", "rr_current", "etco2_current",
                                         *[f"bt_ppv_w{W}" for W in WINDOWS]])
    L = cfg.labels
    ev = detect_events(lm, start, threshold=L.map_threshold, min_seconds=L.event_seconds,
                       merge_gap=L.merge_gap_seconds, suspect_min_map=L.suspect_artefact_min_map)
    rules = LabelRules.from_config(L)
    lab = label_case(f.time.to_numpy(), f.map_current.to_numpy(), f.map_missing_w120.to_numpy(), lm, start, ev,
                     rules=rules, grid_map=grid_map)
    beats = beat_table(z["beats"], z["beat_table_columns"])
    F = cfg.features
    ppv = ppv30_features(f.time.to_numpy(), start, beats, f.rr_current.to_numpy(), f.etco2_current.to_numpy(),
                         F.windows_seconds, sub=F.ppv_subwindow_seconds, min_beats=F.ppv_min_beats,
                         max_period_cv=F.ppv_max_period_cv, rr_range=F.ventilated_rr,
                         min_etco2=F.ventilated_min_etco2)
    end = start + len(lm)
    return {
        "caseid": row.caseid, "seconds": len(lm), "rows": len(f), "events": len(ev),
        "suspect": int(ev.suspect_artefact.sum()) if len(ev) else 0,
        "event_minutes": round(float(ev.duration.sum()) / 60, 1) if len(ev) else 0.0,
        "grid_ok": len(grid_map) * 2 >= len(lm) - 2,
        "beats": len(beats),
        "avail_in_case": bool(len(beats) == 0 or (beats.avail.min() >= start - 5 and beats.avail.max() <= end + 5)),
        "lab": lab, "ppv": ppv, "feat": f, "sec": round(time.time() - t0, 2),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/uc04_v2.json")
    ap.add_argument("--cases", type=int, nargs="*")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    prep = cfg.path("prep")
    qc = read_qc(prep).set_index("caseid", drop=False)
    if args.cases:
        rows = qc.loc[args.cases]
    else:
        pool = qc[(qc.split == "train") & qc.has_wave & (qc.map_coverage >= cfg.input.min_map_coverage)]
        rows = pool.sample(args.n, random_state=args.seed)
    if (rows.split != "train").any():
        print(f"refusing non-train cases: {rows[rows.split != 'train'].caseid.tolist()}", file=sys.stderr)
        return 1

    for row in rows.itertuples():
        r = run_case(prep, row, cfg)
        lab, ppv, f = r["lab"], r["ppv"], r["feat"]
        print(f"\n=== case {r['caseid']}: {r['seconds']} s, {r['rows']} rows, {r['events']} events "
              f"({r['event_minutes']} min, suspect {r['suspect']}), {r['beats']} beats, {r['sec']} s")
        print(f"    clocks: 2 s grid covers case {r['grid_ok']}; beat avail inside case {r['avail_in_case']}")
        for rule in ("event", "any_low"):
            el = lab[f"eligible_{rule}"]
            reasons = lab.loc[~el, f"abstain_reason_{rule}"].value_counts().to_dict()
            print(f"    eligible[{rule}] {el.mean():.1%}  abstain {reasons}")
        el = lab["eligible_event"]
        print("    horizon  lenient(+/0/-1 on eligible)      strict(+/0/-1)")
        for h in cfg.labels.horizons_seconds:
            a = lab.loc[el, f"y_{h}_lenient"].value_counts().reindex([1, 0, -1], fill_value=0).tolist()
            b = lab.loc[el, f"y_{h}_strict"].value_counts().reindex([1, 0, -1], fill_value=0).tolist()
            print(f"    {h:>6}   {str(a):<32} {b}")
        for W in WINDOWS:
            old, new = f[f"bt_ppv_w{W}"], ppv[f"bt_ppv30_w{W}"]
            both = old.notna() & new.notna()
            print(f"    PPV w{W:<3}: NaN old {old.isna().mean():.0%} new {new.isna().mean():.0%}; "
                  f"median old {old.median():.1f} new {new.median():.1f}"
                  f"{'' if not both.any() else f'; new <= old in {(new[both] <= old[both] + 1e-6).mean():.0%} rows'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
