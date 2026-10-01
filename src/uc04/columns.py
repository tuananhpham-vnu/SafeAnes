"""Model feature columns (plan appendix A).

66 numeric + 27 waveform columns per window W, 342 unique columns over the four
windows. `ppv` selects the PPV column family: "bt_ppv30" (recomputed, section 3.3)
for models, "bt_ppv" when reading the original prep_v1 parquet.
"""
from __future__ import annotations

from typing import Iterable, Sequence

SIG = ("map", "sbp", "dbp", "hr", "spo2", "etco2", "rr", "pp", "shock_index")
STATS = ("mean", "std", "min", "max", "slope", "missing")
BEAT = ("dpdt_max", "sys_area", "ejection_time", "notch_rel", "decay_tau", "sv", "co", "svr")
BEAT_STATS = ("mean", "std", "slope")
WINDOWS = (30, 60, 90, 120)
FORBIDDEN_PREFIXES = ("nibp", "cvp", "bis", "mac", "ppf", "rftn", "peep")

PPV_MODEL = "bt_ppv30"
PPV_PREP = "bt_ppv"


def numeric_cols(W: int) -> list[str]:
    """66 columns from monitor numerics for window W."""
    return ([f"{s}_{st}_w{W}" for s in SIG for st in STATS]
            + [f"{s}_current" for s in SIG]
            + ["map_drop_pct", f"map_time_65_75_w{W}", f"map_extrap_w{W}"])


def wave_cols(W: int, ppv: str = PPV_MODEL) -> list[str]:
    """27 columns from the arterial waveform beat table for window W."""
    return ([f"bt_{b}_{st}_w{W}" for b in BEAT for st in BEAT_STATS]
            + [f"{ppv}_w{W}", f"bt_valid_frac_w{W}", f"bt_n_beats_w{W}"])


def window_cols(W: int, groups: Sequence[str] = ("numeric", "waveform"), ppv: str = PPV_MODEL) -> list[str]:
    out: list[str] = []
    for g in groups:
        if g == "numeric":
            out += numeric_cols(W)
        elif g == "waveform":
            out += wave_cols(W, ppv)
        else:
            raise ValueError(f"unknown column group {g!r}")
    return out


def all_model_cols(windows: Iterable[int] = WINDOWS, ppv: str = PPV_MODEL) -> list[str]:
    """The 342 unique columns over all windows, in a stable order."""
    seen: dict[str, None] = {}
    for W in windows:
        for c in window_cols(W, ppv=ppv):
            seen.setdefault(c, None)
    return list(seen)


def ppv_prep_to_model(col: str) -> str:
    """Map `bt_ppv_w60` -> `bt_ppv30_w60`; other columns unchanged."""
    return col.replace(f"{PPV_PREP}_w", f"{PPV_MODEL}_w", 1) if col.startswith(f"{PPV_PREP}_w") else col


def forbidden(columns: Iterable[str], prefixes: Iterable[str] = FORBIDDEN_PREFIXES) -> list[str]:
    prefixes = tuple(prefixes)
    return [c for c in columns if c.startswith(prefixes)]


def assert_allowed(columns: Iterable[str], prefixes: Iterable[str] = FORBIDDEN_PREFIXES) -> None:
    bad = forbidden(columns, prefixes)
    if bad:
        raise ValueError(f"forbidden feature columns (plan 3.2): {bad[:10]}")


ABLATION_GROUPS: dict[str, tuple[str, ...]] = {
    "map": ("map_",),
    "sbp_dbp": ("sbp_", "dbp_", "pp_"),
    "hr": ("hr_", "shock_index_"),
    "spo2": ("spo2_",),
    "etco2_rr": ("etco2_", "rr_"),
    "wave_shape": ("bt_dpdt_max_", "bt_sys_area_", "bt_ejection_time_", "bt_notch_rel_", "bt_decay_tau_"),
    "ppv": (f"{PPV_MODEL}_",),
    "sv_co_svr": ("bt_sv_", "bt_co_", "bt_svr_"),
}


def ablation_groups(columns: Sequence[str]) -> dict[str, list[str]]:
    """Columns of each ablation group (appendix A) within `columns`."""
    return {g: [c for c in columns if c.startswith(p)] for g, p in ABLATION_GROUPS.items()}
