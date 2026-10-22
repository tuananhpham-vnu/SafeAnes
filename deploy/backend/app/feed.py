"""Transport of the ward to the browser: one Server-Sent Events stream per viewer, one message per second.

Kept apart from the ward logic so the transport (SSE here) can be swapped for WebSocket / SignalR / a message bus
without touching the models or the UI state. Each message: virtual time v, every bed's snapshot (Ward.snapshot) and
the new 2 s samples of MAP, HR, SpO2 since the previous message (the last 10 min on the first message or after a
bed restarts its case).
"""
from __future__ import annotations

import asyncio
import json
import time

import anyio

from .ward import EPOCH, Ward

TICK_S = 1.0
FIRST_WAVE_S = 600.0
TILE_VITALS = ("map", "hr", "spo2")


def _json(x) -> str:
    return json.dumps(x, separators=(",", ":"), ensure_ascii=False, default=float)


async def ward_stream(ward: Ward, speed: float, v0: float | None = None):
    wall0 = time.time()
    base = wall0 - EPOCH if v0 is None else v0
    prev: dict[int, tuple[float, int]] = {}
    yield "retry: 3000\n\n"
    while True:
        v = base + (time.time() - wall0) * speed
        snap = await anyio.to_thread.run_sync(ward.snapshot, v)
        waves = {}
        for s in snap:
            b, t, loop = s["bed"], s["t"], s["loop"]
            p = prev.get(b)
            t0 = t - FIRST_WAVE_S if p is None or p[1] != loop or t < p[0] else p[0]
            waves[b] = ward.beds[b].wave_between(t0, t, TILE_VITALS)
            prev[b] = (t, loop)
        yield f"data: {_json({'v': v, 'speed': speed, 'beds': snap, 'wave': waves})}\n\n"
        await asyncio.sleep(TICK_S)
