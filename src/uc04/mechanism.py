"""Why is the model warning? A three-layer explanation card for one prediction time t (week 14).

1. Model layer      which inputs pushed the risk up (TreeSHAP of the LightGBM risk model), summed by readable theme.
2. Physiology layer MAP = CO x SVR = SV x HR x SVR, so over the last 10 minutes
                        dlog MAP = dlog HR + dlog SV + dlog SVR
                    with SV from the Liljestrand-Zander proxy PP / (SBP + DBP) and SVR as the remainder. The unknown
                    calibration constant of the proxy cancels in the difference. The proxies come from blood pressure
                    only (INTERPRETATION_NOTES 3); on EV1000 cases their 5-min trend agrees in direction with the device
                    81-84% of the time (reports/EDA/REPORT.md, item 8), so only the direction and the share are used.
3. Clinical layer   what happened just before t that is known to lower blood pressure: anaesthetic deepening
                    (propofol / remifentanil Ce, MAC, sevoflurane, desflurane), the post-induction period, incision,
                    a recent event, high pulse pressure variation (preload), falling EtCO2 (falling cardiac output),
                    bradycardia, and the patient's pre-operative profile.

The rules below are heuristics for a hypothesis with its evidence, not a causal label: VitalDB has no timed
vasopressor boluses, fluids or blood loss, so a "cause" here can only be a finding that goes with the fall.
Every input is known at t (lagged rows come from the same case at t - 10 min).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LAG = 600  # seconds
SIG = 0.05  # |dlog| >= 0.05 (about 5 %) counts as a change
STABLE_MAP = 0.03  # |dlog MAP| below this: MAP has not moved in 10 min

NEEDED = ["map_mean_w120", "sbp_mean_w120", "dbp_mean_w120", "hr_mean_w120", "hr_current", "bt_ppv30_w120",
          "f1a_min_since_anestart", "f1a_min_since_opstart", "f1a_min_since_last_event", "f1a_n_prev_events",
          "f1b_ppf", "f1b_ppf_d10", "f1b_rftn", "f1b_rftn_d10", "f1b_mac", "f1b_mac_d10", "f1b_etco2_d5", "f1b_bis",
          "f1c_drug_exp_sevo_d5", "f1c_drug_exp_des_d5", "f1b_age", "f1b_asa", "f1b_preop_htn", "st_preop_dm",
          "f1b_position_reverse_trendelenburg", "f1b_position_trendelenburg", "f1b_position_lateral",
          "f1b_position_prone"]

THEMES = {  # model-layer themes, matched by prefix in this order
    "MAP thấp / đang giảm trong 5–30 phút": ("map_", "f1a_map_", "f1a_min_70_80"),
    "Huyết áp tâm thu / tâm trương / áp lực mạch": ("sbp_", "dbp_", "pp_", "f1a_pp_"),
    "Nhịp tim": ("hr_", "shock_index_", "f1a_hr_"),
    "Hô hấp / EtCO2 / SpO2": ("etco2_", "rr_", "spo2_", "f1a_etco2_"),
    "Hình dạng sóng ART": ("bt_",),
    "Thời điểm trong ca / tiền sử tụt": ("f1a_min_since", "f1a_n_prev"),
}

MECHANISMS = {
    "vasodilation": "Giãn mạch (SVR giảm)",
    "low_sv": "Giảm thể tích nhát bóp (SV giảm)",
    "bradycardia": "Nhịp chậm / HR giảm",
    "stable_low": "MAP thấp nhưng chưa giảm thêm",
    "mixed": "Hỗn hợp / thay đổi nhỏ",
    "no_data": "Không đủ dữ liệu",
}


def lz(sbp, dbp, map_, hr):
    """Log of HR, SV (Liljestrand-Zander, up to a constant) and SVR (MAP / (SV x HR), up to a constant)."""
    with np.errstate(all="ignore"):
        sv = np.log((sbp - dbp) / (sbp + dbp))
        lhr = np.log(hr)
        return lhr, sv, np.log(map_) - sv - lhr


def add_lagged(feats: pd.DataFrame, cols: list[str], lag: int = LAG) -> pd.DataFrame:
    """Columns `<c>_lag` = value of the same case at time - lag (NaN when that row does not exist)."""
    past = feats[["caseid", "time", *cols]].copy()
    past["time"] = past["time"] + lag
    past = past.rename(columns={c: f"{c}_lag" for c in cols})
    return feats.merge(past, on=["caseid", "time"], how="left")


def decompose(feats: pd.DataFrame) -> pd.DataFrame:
    """dlog MAP and its HR / SV / SVR parts over the last 10 min, from 2-min means. `feats` needs the `_lag`
    columns of map/sbp/dbp/hr_mean_w120 (add_lagged)."""
    now = lz(feats["sbp_mean_w120"], feats["dbp_mean_w120"], feats["map_mean_w120"], feats["hr_mean_w120"])
    lag = lz(feats["sbp_mean_w120_lag"], feats["dbp_mean_w120_lag"], feats["map_mean_w120_lag"],
             feats["hr_mean_w120_lag"])
    with np.errstate(all="ignore"):
        out = pd.DataFrame({"dlog_map": np.log(feats["map_mean_w120"]) - np.log(feats["map_mean_w120_lag"]),
                            "dlog_hr": now[0] - lag[0], "dlog_sv": now[1] - lag[1], "dlog_svr": now[2] - lag[2]},
                           index=feats.index)
    return out


def primary_mechanism(d: pd.DataFrame) -> pd.Series:
    parts = d[["dlog_hr", "dlog_sv", "dlog_svr"]].to_numpy()
    ok = np.isfinite(parts).all(axis=1) & np.isfinite(d["dlog_map"].to_numpy())
    lowest = np.argmin(np.where(np.isfinite(parts), parts, np.inf), axis=1)
    low_val = parts[np.arange(len(parts)), lowest]
    name = np.array(["bradycardia", "low_sv", "vasodilation"])[lowest]
    out = np.where(low_val <= -SIG, name, "mixed")
    out = np.where(d["dlog_map"].to_numpy() > -STABLE_MAP, np.where(low_val <= -2 * SIG, out, "stable_low"), out)
    return pd.Series(np.where(ok, out, "no_data"), index=d.index)


def clinical_flags(f: pd.DataFrame) -> pd.DataFrame:
    """Boolean findings at t known to go with hypotension (thresholds are clinical rules of thumb)."""
    g = lambda c: f[c].to_numpy(np.float64) if c in f else np.full(len(f), np.nan)  # noqa: E731
    with np.errstate(invalid="ignore"):
        flags = {
            "propofol_up": g("f1b_ppf_d10") >= 0.3,
            "remifentanil_up": g("f1b_rftn_d10") >= 0.5,
            "mac_up": g("f1b_mac_d10") >= 0.1,
            "volatile_up": (g("f1c_drug_exp_sevo_d5") >= 0.3) | (g("f1c_drug_exp_des_d5") >= 1.0),
            "post_induction": g("f1a_min_since_anestart") <= 20,
            "pre_incision": np.isnan(g("f1a_min_since_opstart")),
            "just_after_incision": g("f1a_min_since_opstart") <= 10,
            "recent_event": g("f1a_min_since_last_event") <= 15,
            "high_ppv": g("bt_ppv30_w120") >= 13,
            "etco2_falling": g("f1b_etco2_d5") <= -3,
            "hr_below_50": g("hr_current") < 50,
            "deep_bis": g("f1b_bis") < 40,
            "elderly": g("f1b_age") >= 70,
            "asa3plus": g("f1b_asa") >= 3,
            "hypertension": g("f1b_preop_htn") == 1,
            "diabetes": g("st_preop_dm") == 1,
            "head_up_or_prone": (g("f1b_position_reverse_trendelenburg") == 1) | (g("f1b_position_prone") == 1),
        }
    out = pd.DataFrame(flags, index=f.index)
    out["anaesthetic_deepening"] = out[["propofol_up", "remifentanil_up", "mac_up", "volatile_up"]].any(axis=1)
    return out


FLAG_TEXT = {
    "propofol_up": "Ce propofol tăng ≥ 0,3 µg/mL trong 10 phút",
    "remifentanil_up": "Ce remifentanil tăng ≥ 0,5 ng/mL trong 10 phút",
    "mac_up": "MAC tăng ≥ 0,1 trong 10 phút",
    "volatile_up": "Nồng độ sevo/des thở ra tăng trong 5 phút",
    "post_induction": "Trong 20 phút đầu sau khởi mê",
    "just_after_incision": "Trong 10 phút đầu sau rạch da",
    "recent_event": "Vừa có đợt tụt trong 15 phút qua (có thể đã dùng vận mạch, không có dữ liệu)",
    "high_ppv": "PPV ≥ 13% (gợi ý đáp ứng dịch / thiếu tiền tải)",
    "etco2_falling": "EtCO2 giảm ≥ 3 mmHg trong 5 phút (gợi ý cung lượng tim giảm)",
    "hr_below_50": "HR < 50 nhịp/phút",
    "deep_bis": "BIS < 40 (mê sâu)",
    "head_up_or_prone": "Tư thế đầu cao / nằm sấp (giảm hồi lưu tĩnh mạch)",
}
PROFILE_TEXT = {"elderly": "≥ 70 tuổi", "asa3plus": "ASA ≥ 3", "hypertension": "tăng huyết áp",
                "diabetes": "đái tháo đường"}


def themes(contrib: pd.Series) -> pd.Series:
    """Sum positive SHAP contributions (risk-raising) by theme."""
    out: dict[str, float] = {}
    for c, v in contrib.items():
        theme = next((t for t, pre in THEMES.items() if c.startswith(pre)), "Khác")
        out[theme] = out.get(theme, 0.0) + max(float(v), 0.0)
    return pd.Series(out).sort_values(ascending=False)


@dataclass
class Card:
    caseid: int
    time: float
    risk: float
    mechanism: str
    decomposition: dict
    findings: list[str]
    profile: list[str]
    model_themes: list[tuple[str, float]] = field(default_factory=list)

    def text(self) -> str:
        d = self.decomposition
        pct = lambda v: f"{(np.exp(v) - 1) * 100:+.0f}%"  # noqa: E731
        lines = [f"Ca {self.caseid}, t = {self.time / 60:.0f} phút: nguy cơ tụt trong 15 phút tới {self.risk:.0%}.",
                 "Mô hình dựa vào: " + "; ".join(f"{t} ({s:.0%})" for t, s in self.model_themes[:3]) + "."]
        if np.isfinite(d["dlog_map"]):
            lines.append(f"Sinh lý 10 phút qua: MAP {pct(d['dlog_map'])} = HR {pct(d['dlog_hr'])}, "
                         f"SV proxy {pct(d['dlog_sv'])}, SVR proxy {pct(d['dlog_svr'])} → {MECHANISMS[self.mechanism]}.")
        else:
            lines.append("Sinh lý: chưa đủ 10 phút dữ liệu để tách thành phần.")
        lines.append("Bối cảnh: " + ("; ".join(self.findings) if self.findings else "không thấy yếu tố đi kèm") + ".")
        if self.profile:
            lines.append("Bệnh nhân: " + ", ".join(self.profile) + ".")
        return "\n".join(lines)


def make_card(row: pd.Series, d: pd.Series, mech: str, flags: pd.Series, risk: float,
              contrib: pd.Series | None = None) -> Card:
    th = themes(contrib) if contrib is not None else pd.Series(dtype=float)
    total = th.sum() or 1.0
    return Card(int(row["caseid"]), float(row["time"]), float(risk), mech, d.to_dict(),
                [FLAG_TEXT[k] for k in FLAG_TEXT if bool(flags.get(k, False))],
                [PROFILE_TEXT[k] for k in PROFILE_TEXT if bool(flags.get(k, False))],
                [(t, v / total) for t, v in th.items() if v > 0])
