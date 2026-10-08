"""SafeAnes demo API: replay validation cases with hypotension risk, alarms and explanations.

    uvicorn app.main:app --host 127.0.0.1 --port 8000        (from deploy/backend/; or deploy/run_local.ps1)
Data directory: env SAFEANES_DATA (default ./data), written by export_bundle.py. If deploy/frontend/dist
exists (npm run build), the UI is served at / too.
"""
from __future__ import annotations

import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import HORIZONS, Store, explain_row, risk_and_alarms, rounded
from .extract import InputError, build, parse_time, read_table

DATA = Path(os.environ.get("SAFEANES_DATA", Path(__file__).resolve().parents[1] / "data"))
store = Store(DATA)
app = FastAPI(title="SafeAnes API", version="0.1.0",
              description="Dự báo tụt huyết áp trong mổ (UC04, samples v3) — bản demo nghiên cứu, không dùng lâm sàng.")
app.add_middleware(CORSMiddleware,  # only the Vite dev server needs it; the built UI is same-origin
                   allow_origins=os.environ.get("SAFEANES_CORS", "http://localhost:5173").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def _model(name: str) -> str:
    if name not in store.models:
        raise HTTPException(404, f"unknown model {name}; available: {list(store.models)}")
    return name


def _case(caseid: int) -> int:
    if caseid not in store.cases:
        raise HTTPException(404, f"unknown case {caseid}")
    return caseid


@app.get("/api/health")
def health():
    return {"status": "ok", "models": list(store.models), "cases": len(store.cases)}


@app.get("/api/models")
def models():
    return [{"name": m.name, "title": m.title, "n_features": len(m.columns),
             "horizons": {h: {"threshold": hm.threshold, "metrics": hm.metrics} for h, hm in m.horizons.items()}}
            for m in store.models.values()]


@app.get("/api/report")
def report():
    return store.report


@app.get("/api/cases")
def cases():
    return [{"caseid": c["caseid"], "duration_min": round((c["end_s"] - c["start_s"]) / 60),
             "incision_min": round((c["surgery_start_s"] - c["start_s"]) / 60), "n_events": len(c["events"]),
             "n_events_pre_incision": sum(e["pre_incision"] for e in c["events"])} for c in store.cases.values()]


@app.get("/api/cases/{caseid}/timeline")
def timeline(caseid: int, model: str = "lgbm_context"):
    return store.timeline(_case(caseid), _model(model))


@app.get("/api/cases/{caseid}/explain")
def explain(caseid: int, t: float, model: str = "lgbm_context", horizon: int = Query(300)):
    if horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {HORIZONS}")
    return store.explain(_case(caseid), _model(model), t, horizon)


class PredictIn(BaseModel):
    features: dict[str, float | None]
    model: str = "lgbm_context"


@app.post("/api/predict")
def predict(body: PredictIn):
    """Risk for one time point from a feature row (the columns of GET /api/models), e.g. from a live feed."""
    m = store.models[_model(body.model)]
    missing = [c for c in m.columns if c not in body.features]
    row = pd.Series({c: body.features.get(c) for c in m.columns}, dtype=float)
    probs = m.predict(row.to_frame().T)
    return {"model": m.name, "missing_columns": missing,
            "risk": {h: {"probability": float(p[0]), "threshold": m.horizons[h].threshold,
                         "above_threshold": bool(p[0] >= m.horizons[h].threshold)} for h, p in probs.items()},
            "explanation": m.explain(row, 300)}


# ------------------------------------------------------------------ "Dự báo": the user's own data
SESSIONS: OrderedDict[str, dict] = OrderedDict()  # in memory only, never written to disk
MAX_SESSIONS = 20
MAX_BYTES = 30_000_000
EXAMPLES = DATA / "examples"


class InferIn(BaseModel):
    csv: str                          # file content (CSV / TSV / ;-separated), read in the browser
    anestart: str | None = None       # induction time, same clock as the time column (s, HH:MM[:SS] or date-time)
    opstart: str | None = None        # incision time
    baseline_map: float | None = None
    model: str = "lgbm_context"


def _opt_time(x: str | None) -> float | None:
    if x is None or str(x).strip() == "":
        return None
    v = parse_time(x)
    return None if not np.isfinite(v) else v


def _infer_result(sid: str, model: str) -> dict:
    ses, mdl = SESSIONS[sid], store.models[_model(model)]
    t, el, frame = ses["times"], ses["eligible"], ses["features"]
    onsets = np.array([e["onset_s"] for e in ses["events"]], float)
    risk, alarms = risk_and_alarms(mdl, frame, t, el, onsets, lambda h, i: t[i] + h > t[-1])  # no data after t+h
    return {"session": sid, "caseid": 0, "model": model, "time_s": t.tolist(), "eligible": el.tolist(),
            "surgery_start_s": ses["opstart"] if ses["opstart"] is not None else float(t[0]) - 1,
            "phase_known": ses["opstart"] is not None, "anestart_s": ses["anestart"], "events": ses["events"],
            "vitals": {k: rounded(v) for k, v in ses["vitals"].items()}, "risk": risk, "alarms": alarms,
            "thresholds": {str(h): m.threshold for h, m in mdl.horizons.items()}, "quality": ses["quality"]}


@app.post("/api/infer")
def infer(body: InferIn):
    """Upload vitals -> features (the training pipeline, extract.py) -> risk every 30 s, alarms, events."""
    if len(body.csv) > MAX_BYTES:
        raise HTTPException(413, "file quá lớn (tối đa 30 MB)")
    try:
        df = read_table(body.csv.encode("utf-8"))
        res = build(df, _opt_time(body.anestart), _opt_time(body.opstart), body.baseline_map)
    except InputError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:  # malformed file: say what failed, not a stack trace
        raise HTTPException(422, f"không đọc được dữ liệu: {type(exc).__name__}: {exc}")
    sid = uuid.uuid4().hex
    res["anestart"], res["opstart"] = _opt_time(body.anestart), _opt_time(body.opstart)
    SESSIONS[sid] = res
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return _infer_result(sid, body.model)


def _session(sid: str) -> dict:
    if sid not in SESSIONS:
        raise HTTPException(404, "phiên đã hết hạn, tải dữ liệu lên lại")
    return SESSIONS[sid]


@app.get("/api/infer/{sid}")
def infer_again(sid: str, model: str = "lgbm_context"):
    _session(sid)
    return _infer_result(sid, model)


@app.get("/api/infer/{sid}/explain")
def infer_explain(sid: str, t: float, model: str = "lgbm_context", horizon: int = Query(300)):
    ses, mdl = _session(sid), store.models[_model(model)]
    if horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {HORIZONS}")
    times = ses["times"]
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1))
    row = ses["features"].iloc[i]
    pre = ses["opstart"] is None or times[i] <= ses["opstart"]
    return explain_row(mdl, row, horizon, float(times[i]), bool(ses["eligible"][i]), pre)


@app.get("/api/examples")
def examples():
    p = DATA / "examples.json"
    if not p.exists():
        return []
    return [{"caseid": int(k), **v} for k, v in json.loads(p.read_text(encoding="utf-8")).items()]


@app.get("/api/examples/{caseid}.csv")
def example_csv(caseid: int):
    p = EXAMPLES / f"{caseid}.csv"
    if not p.exists():
        raise HTTPException(404, "no such example")
    return FileResponse(p, media_type="text/csv", filename=f"safeanes_example_{caseid}.csv")


# the built UI (deploy/frontend/dist) on the same origin: one local process serves UI + API
DIST = Path(os.environ.get("SAFEANES_UI", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if (DIST / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
