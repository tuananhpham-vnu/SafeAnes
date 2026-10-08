"""Check app/extract.py against samples v3 and write the example uploads of the "Dự báo" page.

    python deploy/backend/check_extract.py --config configs/uc04_v3.json --work artifacts/v3

For every demo case (data/cases.json): a raw CSV is rebuilt from the VitalDB Solar8000 tracks
(data/vitaldb_full/raw) as a user would export it, run through extract.build, and compared with the
samples v3 rows at the same times: per column agreement and, end to end, the lgbm_context probabilities
against val_predictions. Writes data/examples/<caseid>.csv (+ anestart/opstart in data/examples.json) and
data/extract_check.json. Run from the repo root with PYTHONPATH="src;.local_deps;deploy/backend".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.engine import HORIZONS, Model
from app.extract import build

HERE = Path(__file__).resolve().parent
TRACKS = {"map": "Solar8000/ART_MBP", "sbp": "Solar8000/ART_SBP", "dbp": "Solar8000/ART_DBP", "hr": "Solar8000/HR",
          "spo2": "Solar8000/PLETH_SPO2", "etco2": "Solar8000/ETCO2", "rr": "Solar8000/RR_CO2",
          "nibp_mbp": "Solar8000/NIBP_MBP", "nibp_sbp": "Solar8000/NIBP_SBP"}


def raw_csv(vitaldb: Path, tracks: pd.DataFrame, caseid: int) -> pd.DataFrame:
    parts = []
    for key, tname in TRACKS.items():
        tid = tracks[(tracks.caseid == caseid) & (tracks.tname == tname)].tid
        if len(tid):
            d = pd.read_csv(vitaldb / "raw" / f"{tid.iloc[0]}.csv.gz")
            parts.append(d.rename(columns={d.columns[0]: "time", d.columns[1]: key}).groupby("time").last())
    return pd.concat(parts, axis=1).sort_index().reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="data/samples_v3")
    ap.add_argument("--work", default="artifacts/v3")
    ap.add_argument("--vitaldb", default="data/vitaldb_full")
    ap.add_argument("--data", default=str(HERE / "data"))
    args = ap.parse_args()
    data, samples, vdb = Path(args.data), Path(args.samples), Path(args.vitaldb)
    model = Model(data / "models" / "lgbm_context")
    cases = json.loads((data / "cases.json").read_text(encoding="utf-8"))
    tracks = pd.read_csv(vdb / "meta" / "uc04_tracks.csv")
    meta = pd.read_csv(vdb / "meta" / "cases.csv.gz").set_index("caseid")
    feats = pd.read_parquet(samples / "features" / "validation.parquet", columns=["caseid", "time", *[
        c for c in model.columns if not c.startswith("f1a_")]])
    f5 = pd.read_parquet(samples / "features_v5" / "validation.parquet",
                         columns=["caseid", "time", *[c for c in model.columns if c.startswith("f1a_")]])
    ref_all = feats.merge(f5, on=["caseid", "time"], how="inner")
    index = pd.read_parquet(samples / "case_index.parquet").set_index("caseid")
    (data / "examples").mkdir(exist_ok=True)
    per_col, probs, examples = {c: [] for c in model.columns}, [], {}
    for c in cases:
        cid = c["caseid"]
        df = raw_csv(vdb, tracks, cid)
        ane, op = float(meta.loc[cid, "anestart"]), float(meta.loc[cid, "opstart"])
        df.to_csv(data / "examples" / f"{cid}.csv", index=False, float_format="%.6g")
        examples[cid] = {"anestart_s": ane, "opstart_s": op, "rows": len(df)}
        ref_t = ref_all[ref_all.caseid == cid].time.to_numpy()
        out = build(df, ane, op, None, grid_start=float(index.loc[cid, "start"]), times=ref_t)
        got = out["features"].assign(time=out["times"]).set_index("time")
        ref = ref_all[ref_all.caseid == cid].set_index("time")
        common = got.index.intersection(ref.index)
        for col in model.columns:
            a, b = got.loc[common, col].to_numpy(float), ref.loc[common, col].to_numpy(float)
            both = np.isfinite(a) & np.isfinite(b)
            close = np.isclose(a[both], b[both], rtol=1e-3, atol=1e-3)
            per_col[col].append((both.sum(), close.sum(), (np.isfinite(a) != np.isfinite(b)).sum(), len(a)))
        p_got = {h: model.horizons[h].prob(model.horizons[h].raw(model.matrix(got.loc[common]))) for h in HORIZONS}
        for h in HORIZONS:
            vp = pd.read_parquet(Path(args.work) / "artifacts/tabular/lgbm_context/W120" / f"h{h}" / "val_predictions.parquet")
            vp = vp[vp.caseid == cid].set_index("time").reindex(common)
            ok = vp.eligible.to_numpy(bool) & np.isfinite(vp.probability.to_numpy())
            probs.append(pd.DataFrame({"caseid": cid, "h": h, "ref": vp.probability.to_numpy()[ok], "got": p_got[h][ok]}))
        print(f"case {cid}: {len(common)}/{len(ref)} sample times matched, raw rows {len(df)}")
    (data / "examples.json").write_text(json.dumps(examples, indent=1), encoding="utf-8")

    cols = []
    for col, parts in per_col.items():
        both, close, nan_mismatch, n = (sum(p[i] for p in parts) for i in range(4))
        cols.append({"column": col, "rows": n, "agree": close / both if both else np.nan, "nan_mismatch": nan_mismatch / n if n else np.nan})
    cols = pd.DataFrame(cols).sort_values("agree")
    P = pd.concat(probs)
    d = (P.got - P.ref).abs()
    summary = {"cases": len(cases), "rows": int(len(P) / len(HORIZONS)),
               "columns_exact_ge_99pct": int((cols.agree >= 0.99).sum()), "columns": len(cols),
               "prob_abs_diff_median": float(d.median()), "prob_abs_diff_p95": float(d.quantile(0.95)),
               "prob_corr": float(np.corrcoef(P.got, P.ref)[0, 1]),
               "worst_columns": json.loads(cols.head(12).to_json(orient="records"))}
    (data / "extract_check.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(cols.head(15).round(4).to_string(index=False))
    print({k: v for k, v in summary.items() if k != "worst_columns"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
