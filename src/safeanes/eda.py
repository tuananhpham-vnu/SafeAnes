"""Exploratory analysis of the full VitalDB cohort for UC04 (reports/EDA).

Scope rule: cohort description and track availability use every case. Anything that touches
outcomes (IOH events, labels, pre-event trajectories, mechanism signals) uses only cases
OUTSIDE the locked global test group (evaluation_group == "unseen_test"), so the EDA does
not open the final test set.

All mechanism quantities here are descriptive and use pulse-contour PROXIES
(safeanes.waveform); they are not diagnoses and not labels.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import TRACKS, UC04_TRACKS
from .data import fetch_csv, read_numeric
from .signals import causal_sample

LOCKED_GROUP = "unseen_test"
STEP = 2.0
OFFSETS = np.arange(-900, 301, 30)          # trajectory offsets (s) around onset
BASE, LATE = (-900, -600), (-120, 0)        # windows for "pre-event change"
# Extra numeric tracks put on the 2 s grid for trajectories (max age in seconds).
EXTRA = {
    "bis": ("BIS/BIS", 60), "ppf_ce": ("Orchestra/PPF20_CE", 60), "rftn_ce": ("Orchestra/RFTN20_CE", 60),
    "mac": ("Primus/MAC", 60), "cvp": ("Solar8000/CVP", 60), "nibp_mbp": ("Solar8000/NIBP_MBP", 600),
    "ev_sv": ("EV1000/SV", 60), "ev_svr": ("EV1000/SVR", 120), "ev_svv": ("EV1000/SVV", 60),
    "ev_co": ("EV1000/CO", 60), "vg_sv": ("Vigileo/SV", 60), "vg_svv": ("Vigileo/SVV", 60),
    "phen_rate": ("Orchestra/PHEN_RATE", 3600), "nepi_rate": ("Orchestra/NEPI_RATE", 3600),
    "peep": ("Primus/PEEP_MBAR", 120),
}
BEATS = ["bt_sv_lz", "bt_co_lz", "bt_svr_lz", "bt_hr", "bt_ppv", "bt_dpdt", "bt_pp"]
TRAJ = ["map", "hr", *BEATS, *EXTRA]
LOG_SIGNALS = {"map", "hr", "bt_sv_lz", "bt_co_lz", "bt_svr_lz", "bt_hr", "bt_dpdt", "bt_pp"}


class Paths:
    def __init__(self, root):
        self.root = Path(root)
        self.full = self.root / "data/vitaldb_full"
        self.raw, self.meta = self.full / "raw", self.full / "meta"
        self.cases, self.sequences = self.full / "cases", self.root / "data/sequences_full"
        self.beats = self.root / "data/beats_full"
        self.out = self.root / "reports/EDA"


# ---------------------------------------------------------------- cohort
def load_manifest(root):
    manifest = pd.read_csv(Path(root) / "reports/E07/cohort_manifest.csv")
    manifest["duration_h"] = (manifest.opend - manifest.opstart) / 3600
    manifest["outcome_scope"] = manifest.eligible & manifest.evaluation_group.ne(LOCKED_GROUP)
    return manifest


def cohort_flow(manifest):
    steps = [("Tất cả ca VitalDB", len(manifest))]
    remaining = len(manifest)
    for reason in ("missing_subjectid", "not_known_adult", "not_general_anesthesia",
                   "invalid_surgery_interval", "insufficient_duration", "missing_arterial_map"):
        n = int(manifest.exclusion_reason.eq(reason).sum())
        if n:
            remaining -= n
            steps.append((f"Loại: {reason}", -n))
    steps.append(("Đủ điều kiện (eligible)", int(manifest.eligible.sum())))
    steps.append(("  trong đó global test (khóa)", int((manifest.eligible & manifest.evaluation_group.eq(LOCKED_GROUP)).sum())))
    steps.append(("  phạm vi phân tích outcome", int(manifest.outcome_scope.sum())))
    assert remaining == manifest.eligible.sum()
    return pd.DataFrame(steps, columns=["bước", "số ca"])


def demographics(manifest):
    el = manifest[manifest.eligible]
    groups = {"eligible": el, "fit/cal/val": el[el.outcome_scope], "global test": el[~el.outcome_scope],
              "loại": manifest[~manifest.eligible]}
    rows = {}
    for name, g in groups.items():
        rows[name] = {
            "số ca": len(g), "số bệnh nhân": g.subjectid.nunique(),
            "tuổi, median [IQR]": _iqr(g.age), "nữ (%)": _pct(g.sex.eq("F")),
            "BMI, median [IQR]": _iqr(g.bmi), "ASA ≥3 (%)": _pct(g.asa.ge(3)),
            "mổ cấp cứu (%)": _pct(g.emop.eq(1)), "THA trước mổ (%)": _pct(g.preop_htn.eq(1)),
            "thời gian mổ (giờ), median [IQR]": _iqr(g.duration_h),
            "EBL (mL), median [IQR]": _iqr(g.intraop_ebl),
        }
    return pd.DataFrame(rows)


def category_table(manifest, column, top=12):
    el = manifest[manifest.eligible]
    counts = el[column].fillna("(thiếu)").value_counts()
    head = counts.head(top)
    if len(counts) > top:
        head["(khác)"] = counts.iloc[top:].sum()
    return pd.DataFrame({"số ca": head, "%": (head / len(el) * 100).round(1)})


# ---------------------------------------------------------------- tracks
def track_availability(root, manifest):
    p = Paths(root)
    trks = pd.read_csv(p.meta / "trks.csv.gz")
    el = manifest[manifest.eligible]
    rows = []
    downloaded = {f.name[:-7] for f in p.raw.glob("*.csv.gz")}
    beats = set(json.loads((p.beats / "index.json").read_text(encoding="utf-8"))["cases"]) \
        if (p.beats / "index.json").exists() else set()
    for group, names in {"core": list(TRACKS.values()), **UC04_TRACKS}.items():
        for name in dict.fromkeys(names):
            sub = trks[trks.tname.eq(name) & trks.caseid.isin(el.caseid)]
            rows.append({"nhóm": group, "track": name, "ca eligible có track": sub.caseid.nunique(),
                         "% eligible": round(sub.caseid.nunique() / len(el) * 100, 1),
                         "file đã tải": int(sub.tid.isin(downloaded).sum())})
    art = trks[trks.tname.eq("SNUADC/ART") & trks.caseid.isin(el.caseid)].caseid.nunique()
    rows.append({"nhóm": "waveform", "track": "SNUADC/ART → beat features", "ca eligible có track": art,
                 "% eligible": round(art / len(el) * 100, 1), "file đã tải": len(beats)})
    return pd.DataFrame(rows)


def track_quality(root, manifest, per_track=80, seed=0):
    """Sampling interval and value range per track on a random sample of eligible cases."""
    p = Paths(root)
    index = pd.read_csv(p.meta / "uc04_tracks.csv")
    rng = np.random.default_rng(seed)
    rows = []
    for name, sub in index.groupby("tname"):
        tids = sub.tid.to_numpy()
        tids = [t for t in rng.permutation(tids) if (p.raw / f"{t}.csv.gz").exists()][:per_track]
        intervals, values, bad = [], [], 0
        for tid in tids:
            t, v = read_numeric(fetch_csv(tid, p.raw))
            if len(t) > 1:
                intervals.append(np.median(np.diff(t)))
            bad += int((~np.isfinite(v)).sum())
            values.append(v[np.isfinite(v)])
        v = np.concatenate(values) if values else np.array([])
        rows.append({"track": name, "ca mẫu": len(tids),
                     "chu kỳ lấy mẫu median (s)": round(float(np.median(intervals)), 2) if intervals else np.nan,
                     "p1": _q(v, 1), "p50": _q(v, 50), "p99": _q(v, 99),
                     "% giá trị ≤0": round(float((v <= 0).mean() * 100), 1) if len(v) else np.nan,
                     "% không hữu hạn": round(bad / max(bad + len(v), 1) * 100, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- per-case grid
def case_grid(root, record, extra=True):
    """2 s grid for one case: 7 core tracks (sequences_full), beat features, extra tracks."""
    p = Paths(root)
    caseid = int(record["caseid"])
    array = np.load(p.sequences / f"{caseid}.npy", mmap_mode="r")
    start = float(np.ceil(record["opstart"]))
    times = start + STEP * np.arange(array.shape[0])
    frame = pd.DataFrame(np.asarray(array[:, :len(TRACKS)]), columns=list(TRACKS))
    frame.insert(0, "time", times)
    beat_path = p.beats / f"{caseid}.npz"
    if beat_path.exists():
        z = np.load(beat_path)
        cols = list(z["columns"])
        k = np.round((times - z["times"][0]) / STEP).astype(int)
        ok = (k >= 0) & (k < len(z["times"]))
        for name in BEATS + ["bt_valid_frac"]:
            col = np.full(len(times), np.nan, "float32")
            col[ok] = z["features"][k[ok], cols.index(name)]
            frame[name] = col
    else:
        for name in BEATS + ["bt_valid_frac"]:
            frame[name] = np.nan
    if extra:
        for key, (tname, max_age) in EXTRA.items():
            tid = record.get(f"x_{key}")
            frame[key] = np.nan
            if isinstance(tid, str) and (p.raw / f"{tid}.csv.gz").exists():
                t, v = read_numeric(fetch_csv(tid, p.raw))
                v = np.where(np.isfinite(v) & (v >= 0), v, np.nan)
                frame[key], _ = causal_sample(t, v, times, max_age)
    return frame


def attach_extra_tids(root, manifest):
    index = pd.read_csv(Paths(root).meta / "uc04_tracks.csv").drop_duplicates(["caseid", "tname"])
    out = manifest.copy()
    for key, (tname, _) in EXTRA.items():
        m = index[index.tname.eq(tname)].set_index("caseid").tid
        out[f"x_{key}"] = out.caseid.map(m)
    return out


def _window(frame, name, center, lo, hi):
    t = frame.time.to_numpy()
    sel = (t >= center + lo) & (t < center + hi)
    v = frame[name].to_numpy()[sel]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) >= 3 else np.nan


def _trajectory(frame, center):
    t0 = frame.time.iloc[0]
    k = np.round((center + OFFSETS - t0) / STEP).astype(int)
    ok = (k >= 0) & (k < len(frame))
    out = np.full((len(TRAJ), len(OFFSETS)), np.nan, "float32")
    for i, name in enumerate(TRAJ):
        values = frame[name].to_numpy()
        out[i, ok] = values[k[ok]]
    return out


def _deltas(frame, center):
    row = {}
    for name in TRAJ:
        base, late = _window(frame, name, center, *BASE), _window(frame, name, center, *LATE)
        row[f"base_{name}"], row[f"late_{name}"] = base, late
        if name in LOG_SIGNALS:
            row[f"dlog_{name}"] = np.log(late / base) if base > 0 and late > 0 else np.nan
        else:
            row[f"d_{name}"] = late - base
    return row


def analyze_case(args):
    """Events, labels, trajectories and pre-event changes for one outcome-scope case."""
    root, record, n_controls, seed = args
    p = Paths(root)
    import joblib
    case = joblib.load(p.cases / f"{int(record['caseid'])}.joblib")
    frame = case["frame"]
    grid = case_grid(root, record)
    t = grid.time.to_numpy()
    map_ = grid["map"].to_numpy()
    summary = {"caseid": int(record["caseid"]), "subjectid": int(record["subjectid"]),
               "monitored_h": len(grid) * STEP / 3600,
               "pct_time_map_lt65": float(np.nanmean(map_ < 65) * 100) if np.isfinite(map_).any() else np.nan,
               "n_events": len(case["events"]), "decision_rows": len(frame),
               "eligible_rows": int(frame.eligible.sum())}
    for h in (300, 600):
        y = frame.loc[frame.eligible, f"y_{h}"]
        summary[f"y{h}_pos"], summary[f"y{h}_neg"], summary[f"y{h}_cens"] = \
            int(y.eq(1).sum()), int(y.eq(0).sum()), int(y.eq(-1).sum())
    summary["has_beats"] = bool(np.isfinite(grid["bt_sv_lz"]).any())
    for key in EXTRA:
        summary[f"has_{key}"] = bool(np.isfinite(grid[key]).any())

    events, trajectories = [], []
    for i, e in enumerate(case["events"]):
        sel = (t >= e["onset"]) & (t <= e["end"])
        depth = np.clip(65 - map_[sel], 0, None)
        row = {"caseid": e["caseid"], "subjectid": e["subjectid"], "event": i, "onset": e["onset"],
               "duration_s": e["end"] - e["onset"],
               "min_after_opstart": (e["onset"] - record["opstart"]) / 60,
               "min_after_anestart": (e["onset"] - record["anestart"]) / 60,
               "min_map": float(np.nanmin(map_[sel])) if np.isfinite(map_[sel]).any() else np.nan,
               "auc65_mmHg_min": float(np.nansum(depth) * STEP / 60),
               "eligible_300": e["eligible_300"], "eligible_600": e["eligible_600"],
               "first_event": i == 0, "prev_event_gap_min":
                   (e["onset"] - case["events"][i - 1]["end"]) / 60 if i else np.nan,
               "kind": "event"}
        row.update(_deltas(grid, e["onset"]))
        events.append(row)
        trajectories.append(_trajectory(grid, e["onset"]))

    # Controls: times with MAP >= 65 and no onset within [-15, +15] min, no ongoing event.
    rng = np.random.default_rng(seed + int(record["caseid"]))
    onsets = np.array([e["onset"] for e in case["events"]])
    spans = [(e["onset"], e["end"]) for e in case["events"]]
    candidates = t[(t >= t[0] + 900) & (t <= t[-1] - 300) & (map_ >= 65)]
    ok = [c for c in candidates
          if not (len(onsets) and np.any(np.abs(onsets - c) <= 900))
          and not any(a - 900 <= c <= b for a, b in spans)]
    for j, c in enumerate(rng.choice(ok, size=min(n_controls, len(ok)), replace=False) if ok else []):
        row = {"caseid": int(record["caseid"]), "subjectid": int(record["subjectid"]), "event": j,
               "onset": float(c), "kind": "control"}
        row.update(_deltas(grid, c))
        events.append(row)
        trajectories.append(_trajectory(grid, c))
    return summary, events, trajectories


def analyze_cohort(root, manifest, workers=8, n_controls=2, seed=20260917, limit=None):
    records = attach_extra_tids(root, manifest)
    records = records[records.outcome_scope]
    records = records[[(Paths(root).sequences / f"{c}.npy").exists() and (Paths(root).cases / f"{c}.joblib").exists()
                       for c in records.caseid]]
    records = records.to_dict("records")[:limit]
    summaries, events, trajectories = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for s, e, tr in pool.map(analyze_case, [(root, r, n_controls, seed) for r in records], chunksize=8):
            summaries.append(s)
            events += e
            trajectories += tr
    events = pd.DataFrame(events)
    traj = np.stack(trajectories) if trajectories else np.zeros((0, len(TRAJ), len(OFFSETS)))
    return pd.DataFrame(summaries), events, traj


# ---------------------------------------------------------------- mechanisms (descriptive)
def decompose(events):
    """ΔlogMAP (beat) ≈ ΔlogSV + ΔlogHR + ΔlogSVR for Liljestrand-Zander proxies (exact for
    per-window values; approximately for medians). Adds dominant component + heuristic pattern."""
    out = events.copy()
    comps = out[["dlog_bt_sv_lz", "dlog_bt_hr", "dlog_bt_svr_lz"]]
    out["dlog_bt_map_sum"] = comps.sum(axis=1, min_count=3)
    complete = comps.notna().all(axis=1)
    dominant = comps[complete].idxmin(axis=1).map(
        {"dlog_bt_sv_lz": "SV", "dlog_bt_hr": "HR", "dlog_bt_svr_lz": "SVR"})
    out["dominant"] = "thiếu dữ liệu beat"
    out.loc[complete, "dominant"] = dominant
    out.loc[complete & (comps.min(axis=1) > -0.05), "dominant"] = "thay đổi nhỏ (<5%)"
    out["pattern"] = out.apply(_pattern, axis=1)
    return out


def _pattern(r):
    """Heuristic preview only (NOT a label): which physiology the pre-event change resembles."""
    if r["dominant"] in ("thiếu dữ liệu beat", "thay đổi nhỏ (<5%)"):
        return r["dominant"]
    if r["dominant"] == "SVR":
        anesthetic = (r.get("d_ppf_ce", 0) or 0) > 0.1 or (r.get("d_rftn_ce", 0) or 0) > 0.3 \
            or (r.get("d_mac", 0) or 0) > 0.1
        return "giảm SVR + tăng thuốc mê" if anesthetic else "giảm SVR"
    if r["dominant"] == "HR":
        return "giảm HR"
    ppv_up = (r.get("d_bt_ppv", np.nan) >= 3) or (r.get("late_bt_ppv", np.nan) >= 13)
    if ppv_up:
        return "giảm SV + PPV cao/tăng (gợi ý tiền tải)"
    if (r.get("dlog_bt_dpdt", 0) or 0) <= -0.1:
        return "giảm SV + giảm dP/dt (gợi ý co bóp)"
    return "giảm SV, không rõ"


def cluster_changes(events, k=5, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    cols = ["dlog_bt_sv_lz", "dlog_bt_hr", "dlog_bt_svr_lz", "d_bt_ppv", "dlog_bt_dpdt"]
    data = events[events.kind.eq("event")].dropna(subset=cols)
    if len(data) < k * 10:
        return pd.DataFrame(), data.assign(cluster=np.nan)
    X = StandardScaler().fit_transform(data[cols].clip(*np.nanpercentile(data[cols], [1, 99], axis=0)))
    labels = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    data = data.assign(cluster=labels)
    centroids = data.groupby("cluster")[cols + ["d_ppf_ce", "d_rftn_ce", "d_mac"]].median().round(3)
    centroids.insert(0, "số biến cố", data.cluster.value_counts().sort_index())
    return centroids, data


# ---------------------------------------------------------------- validation of proxies
def proxy_vs_ev1000(root, manifest, limit=None):
    """Per case with EV1000/SV: Spearman r and 5-min trend concordance of SV/SVR proxies."""
    records = attach_extra_tids(root, manifest)
    records = records[records.outcome_scope & records.x_ev_sv.notna()]
    rows = []
    for record in records.to_dict("records")[:limit]:
        if not (Paths(root).sequences / f"{record['caseid']}.npy").exists():
            continue
        g = case_grid(root, record).iloc[::15]    # 30 s
        row = {"caseid": record["caseid"]}
        for proxy, ref in (("bt_sv_lz", "ev_sv"), ("bt_svr_lz", "ev_svr"), ("bt_ppv", "ev_svv")):
            both = g[[proxy, ref]].dropna()
            row[f"n_{ref}"] = len(both)
            if len(both) < 40:
                continue
            row[f"r_{proxy}_{ref}"] = both[proxy].corr(both[ref], method="spearman")
            d = both.diff(10).dropna()                      # 5-min changes
            big = d[ref].abs() > 0.10 * both[ref].median()   # exclusion zone 10 %
            if big.sum() >= 5:
                row[f"concord_{proxy}_{ref}"] = float((np.sign(d[proxy][big]) == np.sign(d[ref][big])).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def nibp_vs_art(root, manifest, max_cases=400, seed=0):
    records = attach_extra_tids(root, manifest)
    records = records[records.eligible & records.x_nibp_mbp.notna()].sample(frac=1, random_state=seed)
    diffs = []
    for record in records.head(max_cases).to_dict("records"):
        if not (Paths(root).sequences / f"{record['caseid']}.npy").exists():
            continue
        t, v = read_numeric(fetch_csv(record["x_nibp_mbp"], Paths(root).raw))
        g = case_grid(root, record, extra=False)
        art, age = causal_sample(g.time.to_numpy(), g["map"].to_numpy(), t, 10)
        ok = np.isfinite(art) & np.isfinite(v) & (v > 0)
        diffs.append(pd.DataFrame({"caseid": record["caseid"], "nibp": v[ok], "art": art[ok]}))
    d = pd.concat(diffs) if diffs else pd.DataFrame(columns=["caseid", "nibp", "art"])
    diff = d.nibp - d.art
    summary = {"cặp đo": len(d), "số ca": d.caseid.nunique(), "bias (NIBP−ART)": round(diff.mean(), 2),
               "LoA thấp": round(diff.mean() - 1.96 * diff.std(), 2), "LoA cao": round(diff.mean() + 1.96 * diff.std(), 2),
               "% ART<65 mà NIBP≥65": round(float(((d.art < 65) & (d.nibp >= 65)).sum() / max((d.art < 65).sum(), 1) * 100), 1)}
    return d, summary


def labs_bleeding(root, manifest, cases):
    """Intra-op Hb nadir vs preop Hb, EBL; correlation with IOH burden (outcome scope only)."""
    labs = pd.read_csv(Paths(root).meta / "labs.csv.gz")
    m = manifest.set_index("caseid")
    hb = labs[labs.name.eq("hb") & labs.caseid.isin(cases.caseid)]
    hb = hb.join(m[["opstart", "opend"]], on="caseid")
    intra = hb[(hb.dt >= hb.opstart) & (hb.dt <= hb.opend)]
    per = intra.groupby("caseid").agg(hb_intra_min=("result", "min"), hb_intra_n=("result", "size"))
    out = cases.set_index("caseid").join(per).join(m[["preop_hb", "intraop_ebl", "intraop_rbc", "duration_h",
                                                     "intraop_crystalloid", "intraop_colloid"]])
    out["hb_drop"] = out.preop_hb - out.hb_intra_min
    out["events_per_h"] = out.n_events / out.monitored_h
    return out.reset_index()


def subgroup_rates(manifest, cases):
    d = cases.merge(manifest[["caseid", "age", "sex", "asa", "emop", "department", "optype", "approach",
                              "preop_htn"]], on="caseid")
    d["age_group"] = pd.cut(d.age, [17, 40, 60, 75, 120], labels=["18–40", "41–60", "61–75", ">75"])
    d["any_event"] = d.n_events > 0
    tables = {}
    for col in ("age_group", "sex", "asa", "emop", "preop_htn", "department", "optype", "approach"):
        g = d.groupby(col, observed=True).agg(ca=("caseid", "size"), co_IOH=("any_event", "mean"),
                                               biến_cố=("n_events", "sum"), giờ=("monitored_h", "sum"))
        g = g[g.ca >= 20]
        g["% ca có IOH"] = (g.co_IOH * 100).round(1)
        g["biến cố/giờ"] = (g.biến_cố / g.giờ).round(3)
        tables[col] = g[["ca", "% ca có IOH", "biến cố/giờ"]].sort_values("ca", ascending=False).head(12)
    return tables


# ---------------------------------------------------------------- helpers
def _iqr(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return "–"
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return f"{q2:.1f} [{q1:.1f}–{q3:.1f}]"


def _pct(mask):
    return round(float(mask.mean() * 100), 1) if len(mask) else np.nan


def _q(v, q):
    return round(float(np.percentile(v, q)), 2) if len(v) else np.nan


# ---------------------------------------------------------------- figure captions
_FIGURE = re.compile(r"!\[[^\]]*\]\(figures/([^)]+)\)")
_CAPTION = re.compile(r"\n\n> \*\*Hình \d+ —[^\n]*(?:\n>[^\n]*)*")


def load_captions(path):
    """reports/EDA/figure_captions.md: '## <file.png>' sections; first line = title, rest = body."""
    path = Path(path)
    if not path.exists():
        return {}
    captions, name, lines = {}, None, []
    for line in path.read_text(encoding="utf-8").splitlines() + ["## "]:
        if line.startswith("## "):
            if name:
                body = "\n".join(lines).strip()
                title, _, rest = body.partition("\n")
                captions[name] = (title.strip().rstrip("."), rest.strip())
            name, lines = line[3:].strip() or None, []
        elif name:
            lines.append(line)
    return captions


def apply_captions(markdown, captions):
    """Put a numbered caption block under every figure that has one; idempotent (old blocks replaced)."""
    markdown = _CAPTION.sub("", markdown)
    counter = iter(range(1, 1000))

    def add(match):
        name = match.group(1)
        if name not in captions:
            return match.group(0)
        title, body = captions[name]
        block = [f"**Hình {next(counter)} — {title}.**"]
        for line in body.splitlines():
            if line.startswith("**") and block[-1]:   # new paragraph for "Cách đọc" / "Nhận xét"
                block.append("")
            block.append(line)
        return match.group(0) + "\n\n" + "\n".join("> " + line if line else ">" for line in block)

    return _FIGURE.sub(add, markdown)
