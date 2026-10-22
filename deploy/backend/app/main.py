"""SafeAnes ward API: 50 recorded test cases replayed as a live ward, four features_v8 models run as the clock moves.

    uvicorn app.main:app --host 0.0.0.0 --port 8000        (from deploy/backend/)
Env: SAFEANES_DATA (bundle of export_ward.py, default ./data_ward), SAFEANES_ENGINE (live | replay; replay reads
the offline test probabilities instead of running the models), SAFEANES_UI (built UI, default ../frontend/dist).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .feed import ward_stream
from .models import HORIZONS
from .ward import EPOCH, Ward

DATA = Path(os.environ.get("SAFEANES_DATA", Path(__file__).resolve().parents[1] / "data_ward"))
ward = Ward(DATA, os.environ.get("SAFEANES_ENGINE", "live"))
app = FastAPI(title="SafeAnes Ward API", version="0.2.0",
              description="Dự báo tụt huyết áp trong mổ (UC04, features_v8): trạm theo dõi mô phỏng trên 50 ca của "
                          "tập test. Bản nghiên cứu, không dùng lâm sàng.")
app.add_middleware(CORSMiddleware,  # only the Vite dev server needs it; the built UI is same-origin
                   allow_origins=os.environ.get("SAFEANES_CORS", "http://localhost:5173").split(","),
                   allow_methods=["GET"], allow_headers=["*"])
SPEEDS = (1.0, 5.0, 20.0)


def _bed(b: int) -> int:
    if b not in ward.beds:
        raise HTTPException(404, f"unknown bed {b}")
    return b


def _v(v: float | None) -> float:
    return time.time() - EPOCH if v is None else v


@app.get("/api/health")
def health():
    return {"status": "ok", "engine": ward.mode, "beds": len(ward.beds), "models": list(ward.models),
            "rows_scored": ward.rows_scored}


@app.get("/api/models")
def models():
    return [{"name": m.name, "title": m.title, "n_inputs": len(m.columns), "explains": m.explains,
             "horizons": {str(h): {k: m.meta["horizons"][str(h)][k] for k in ("threshold", "validation", "test")}
                          for h in HORIZONS}} for m in ward.models.values()]


@app.get("/api/ward")
def ward_info():
    return {"source": ward.meta["source"], "horizons": list(HORIZONS), "speeds": list(SPEEDS),
            "v_now": time.time() - EPOCH, "engine": ward.mode,
            "beds": [{**b.info, "duration_s": b.duration, "phase_s": b.phase} for b in ward.beds.values()]}


@app.get("/api/ward/stream")
async def stream(speed: float = Query(1.0), v: float | None = None):
    if speed not in SPEEDS:
        raise HTTPException(400, f"speed must be one of {SPEEDS}")
    return StreamingResponse(ward_stream(ward, speed, v), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/beds/{bed}/history")
def history(bed: int, v: float | None = None, wave_s: float = Query(1800.0, ge=0, le=43200)):
    """Everything of one bed up to now: risk of every model, alarms, known events, 2 s vitals of the last wave_s."""
    return ward.history(_bed(bed), _v(v), wave_s)


@app.get("/api/beds/{bed}/explain")
def explain(bed: int, model: str, horizon: int = 300, v: float | None = None):
    if model not in ward.models:
        raise HTTPException(404, f"unknown model {model}; available: {list(ward.models)}")
    if horizon not in HORIZONS:
        raise HTTPException(400, f"horizon must be one of {HORIZONS}")
    return ward.explain(_bed(bed), _v(v), model, horizon)


# the built UI (deploy/frontend/dist) on the same origin
DIST = Path(os.environ.get("SAFEANES_UI", Path(__file__).resolve().parents[2] / "frontend" / "dist"))
if (DIST / "index.html").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
