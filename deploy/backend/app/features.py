"""Human-readable names and physiological groups of the model columns (Vietnamese UI)."""
from __future__ import annotations

import re

SIGNALS = {"map": "MAP", "sbp": "HA tâm thu", "dbp": "HA tâm trương", "hr": "Nhịp tim", "spo2": "SpO₂",
           "etco2": "EtCO₂", "rr": "Nhịp thở", "pp": "Hiệu áp", "shock_index": "Shock index"}
STATS = {"mean": "trung bình", "std": "độ dao động", "min": "thấp nhất", "max": "cao nhất", "slope": "độ dốc",
         "missing": "tỷ lệ thiếu"}
UNITS = {"map": "mmHg", "sbp": "mmHg", "dbp": "mmHg", "pp": "mmHg", "hr": "/phút", "spo2": "%", "etco2": "mmHg",
         "rr": "/phút", "shock_index": ""}

# group id -> (label, kind) ; kind "monitor" = short window, "context" = case context (features_v5)
GROUPS = {
    "map_now": ("MAP hiện tại & 2 phút gần nhất", "monitor"),
    "pressure": ("HA tâm thu, tâm trương, hiệu áp", "monitor"),
    "hr": ("Nhịp tim & shock index", "monitor"),
    "resp": ("SpO₂, EtCO₂, nhịp thở", "monitor"),
    "map_trend": ("Xu hướng MAP 5–30 phút", "context"),
    "map_case": ("MAP so với chính ca", "context"),
    "other_trend": ("Xu hướng nhịp tim, hiệu áp, EtCO₂", "context"),
    "phase": ("Giai đoạn ca (khởi mê, rạch da)", "context"),
    "history": ("Các đợt tụt trước trong ca", "context"),
}


def group_of(col: str) -> str:
    if col.startswith("f1a_"):
        if col in ("f1a_min_since_opstart", "f1a_min_since_anestart"):
            return "phase"
        if col in ("f1a_n_prev_events", "f1a_min_since_last_event"):
            return "history"
        if col == "f1a_map_rel_case_median":
            return "map_case"
        if col.startswith(("f1a_map_", "f1a_min_70_80")):
            return "map_trend"
        return "other_trend"
    if col.startswith("map_"):
        return "map_now"
    if col.startswith(("sbp_", "dbp_", "pp_")):
        return "pressure"
    if col.startswith(("hr_", "shock_index_")):
        return "hr"
    return "resp"


def label_of(col: str) -> str:
    """e.g. map_slope_w120 -> 'MAP · độ dốc (2 phút)'; f1a_map_min_m15 -> 'MAP · thấp nhất (15 phút)'."""
    special = {
        "map_drop_pct": "MAP giảm so với nền (%)", "map_time_65_75_w120": "Thời gian MAP 65–75 (2 phút)",
        "map_extrap_w120": "MAP dự đoán sau 5 phút", "f1a_map_rel_case_median": "MAP so với trung vị của ca",
        "f1a_map_rel_max_m15": "MAP so với đỉnh 15 phút", "f1a_min_70_80_m15": "Số phút MAP 70–80 (15 phút)",
        "f1a_min_since_opstart": "Phút từ lúc rạch da", "f1a_min_since_anestart": "Phút từ lúc khởi mê",
        "f1a_n_prev_events": "Số đợt tụt trước đó", "f1a_min_since_last_event": "Phút từ đợt tụt gần nhất",
    }
    if col in special:
        return special[col]
    m = re.fullmatch(r"(f1a_)?([a-z0-9_]+?)_(mean|std|min|max|slope|missing)_(w|m)(\d+)", col)
    if m:
        sig, st, unit, n = m.group(2), m.group(3), m.group(4), int(m.group(5))
        span = f"{n // 60} phút" if unit == "w" and n >= 60 else (f"{n} giây" if unit == "w" else f"{n} phút")
        return f"{SIGNALS.get(sig, sig)} · {STATS[st]} ({span})"
    m = re.fullmatch(r"([a-z0-9_]+)_current", col)
    if m:
        return f"{SIGNALS.get(m.group(1), m.group(1))} hiện tại"
    return col
