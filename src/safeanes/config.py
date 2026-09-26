"""Protocol choices are project proposals; see docs/PROTOCOL.md and SOURCES.md."""

from dataclasses import asdict, dataclass
import hashlib
import json


TRACKS = {
    "map": "Solar8000/ART_MBP",
    "sbp": "Solar8000/ART_SBP",
    "dbp": "Solar8000/ART_DBP",
    "hr": "Solar8000/HR",
    "spo2": "Solar8000/PLETH_SPO2",
    "etco2": "Solar8000/ETCO2",
    "rr": "Solar8000/RR_CO2",
}
# Broad engineering validity checks, NOT diagnostic thresholds. Retain severe lows.
BOUNDS = {"map": (0, 300), "sbp": (0, 350), "dbp": (0, 250),
          "hr": (0, 300), "spo2": (0, 100), "etco2": (0, 150), "rr": (0, 100)}


@dataclass(frozen=True)
class Protocol:
    version: str = "uc04-numeric-v1"
    map_threshold: float = 65.0
    event_seconds: int = 60
    label_max_gap_seconds: int = 10
    feature_max_age_seconds: int = 30
    recovery_seconds: int = 120
    history_seconds: int = 600
    numeric_step_seconds: int = 2
    cadence_seconds: int = 30
    horizons_seconds: tuple[int, ...] = (300, 600)
    min_map_history_coverage: float = 0.8
    alarm_persistence: int = 2
    alarm_cooldown_seconds: int = 300
    seed: int = 20260917

    def __post_init__(self):
        for field in ("event_seconds", "label_max_gap_seconds", "feature_max_age_seconds",
                      "recovery_seconds", "history_seconds", "numeric_step_seconds",
                      "cadence_seconds", "alarm_persistence", "alarm_cooldown_seconds"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.history_seconds % self.numeric_step_seconds:
            raise ValueError("history must be divisible by numeric step")
        if not self.horizons_seconds or any(h <= 0 for h in self.horizons_seconds):
            raise ValueError("horizons must be positive")
        if tuple(sorted(set(self.horizons_seconds))) != self.horizons_seconds:
            raise ValueError("horizons must be unique and sorted")
        if not 0 <= self.min_map_history_coverage <= 1:
            raise ValueError("coverage must lie in [0, 1]")

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()



# UC04 mechanism-related numeric tracks (EDA + cause branch). Grouped by the physiology they
# inform; availability varies widely (e.g. EV1000 ~600 cases), see reports/EDA.
UC04_TRACKS = {
    "hemodynamic": [
        "Solar8000/ART_MBP", "Solar8000/ART_SBP", "Solar8000/ART_DBP", "Solar8000/HR",
        "Solar8000/PLETH_HR", "Solar8000/NIBP_MBP", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP",
        "Solar8000/CVP", "Solar8000/FEM_MBP", "Solar8000/BT", "Solar8000/PLETH_SPO2",
    ],
    "cardiac_output": [
        "EV1000/SV", "EV1000/SVI", "EV1000/SVV", "EV1000/CO", "EV1000/CI", "EV1000/SVR",
        "EV1000/SVRI", "EV1000/CVP", "EV1000/ART_MBP",
        "Vigileo/SV", "Vigileo/SVI", "Vigileo/SVV", "Vigileo/CO", "Vigileo/CI",
    ],
    "anesthetic_depth": [
        "BIS/BIS", "BIS/SQI", "BIS/SR", "Primus/MAC", "Primus/EXP_SEVO", "Primus/EXP_DES",
        "Primus/FEN2O",
        "Orchestra/PPF20_CE", "Orchestra/PPF20_CP", "Orchestra/PPF20_RATE", "Orchestra/PPF20_VOL",
        "Orchestra/RFTN20_CE", "Orchestra/RFTN20_CP", "Orchestra/RFTN20_RATE", "Orchestra/RFTN20_VOL",
        "Orchestra/RFTN50_CE", "Orchestra/RFTN50_RATE",
    ],
    "vasoactive": [
        f"Orchestra/{drug}_{kind}" for drug in
        ("PHEN", "NEPI", "EPI", "DOPA", "DOBU", "VASO", "NTG", "NPS", "PGE1", "DTZ", "ROC")
        for kind in ("RATE", "VOL")
    ],
    "ventilation": [
        "Solar8000/ETCO2", "Solar8000/RR_CO2", "Primus/ETCO2", "Primus/PEEP_MBAR",
        "Primus/PIP_MBAR", "Primus/PPLAT_MBAR", "Primus/TV", "Primus/MV", "Primus/COMPLIANCE",
    ],
}
WAVEFORM_TRACKS = {"art": "SNUADC/ART"}
WAVEFORM_HZ = 500
