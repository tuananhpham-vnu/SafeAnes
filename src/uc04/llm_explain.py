"""LLM layer on top of the explanation card (week 14): personalised reading of an alarm for the anaesthetist.

Division of work
  learning-based  risk (LightGBM, calibrated), physiological decomposition, clinical findings  -> mechanism.Card
  LLM             turns the card + the patient's pre-operative profile into a short clinical reading, adding
                  textbook knowledge the model cannot learn from VitalDB (e.g. chronic hypertension shifts
                  cerebral/renal autoregulation to higher pressures; diabetic autonomic neuropathy blunts the heart
                  rate response; old age stiffens arteries and weakens the baroreflex).
The LLM never changes the risk or the alarm, never invents measurements, and suggests checks, not orders. It runs
outside the 30-second prediction loop (on alarm only), so its latency does not delay the warning.
"""
from __future__ import annotations

import json
import re

MODEL = "claude-opus-5-5"

SYSTEM = """Bạn hỗ trợ bác sĩ gây mê đọc một cảnh báo tụt huyết áp trong mổ do mô hình học máy phát ra.
Bạn nhận một thẻ bằng chứng (JSON) gồm: nguy cơ tụt huyết áp (MAP < 65 mmHg ≥ 1 phút) trong 15 phút tới do mô hình
tính, các nhóm tín hiệu mô hình dựa vào, phân tách thay đổi MAP 10 phút qua thành HR, SV proxy, SVR proxy
(proxy tính từ huyết áp, chỉ đáng tin về hướng thay đổi), các dấu hiệu bối cảnh, và hồ sơ trước mổ của bệnh nhân.

Quy tắc:
- Chỉ dùng số liệu có trong thẻ. Không bịa thêm số đo, thuốc, liều hay sự kiện.
- Không thay đổi mức nguy cơ. Nói rõ đây là giả thuyết đi kèm, không phải nguyên nhân đã xác định.
- Dùng kiến thức sinh lý và hướng dẫn chuyên môn để giải thích ý nghĩa với bệnh nhân này (tuổi, ASA, tăng huyết áp,
  đái tháo đường...). Nêu cơ sở kiến thức ngắn gọn.
- Đề xuất những gì nên kiểm tra hoặc cân nhắc, không ra y lệnh và không đưa liều thuốc.
- Nếu bằng chứng mâu thuẫn hoặc thiếu, nói thẳng điều đó.
- Viết tiếng Việt, ngắn, dùng thuật ngữ gây mê thông dụng."""

SCHEMA = {
    "type": "object",
    "properties": {
        "tom_tat": {"type": "string", "description": "1-2 câu: nguy cơ và giả thuyết chính"},
        "co_che_kha_di": {"type": "string", "description": "Cơ chế huyết động có khả năng nhất và mức chắc chắn"},
        "bang_chung": {"type": "array", "items": {"type": "string"}, "description": "Bằng chứng trích từ thẻ"},
        "luu_y_ca_nhan": {"type": "array", "items": {"type": "string"},
                          "description": "Ý nghĩa của hồ sơ bệnh nhân đối với cảnh báo này, kèm cơ sở kiến thức"},
        "nen_kiem_tra": {"type": "array", "items": {"type": "string"}, "description": "Việc nên kiểm tra / cân nhắc"},
        "gioi_han": {"type": "string", "description": "Điều thẻ không cho biết hoặc không chắc chắn"},
    },
    "required": ["tom_tat", "co_che_kha_di", "bang_chung", "luu_y_ca_nhan", "nen_kiem_tra", "gioi_han"],
    "additionalProperties": False,
}


def evidence(card: dict) -> dict:
    """The JSON evidence sent to the LLM, from one row of alarm_cards.csv merged with chain_h900.parquet (forecast
    components, drivers, counterfactual) when available."""
    pct = lambda v: None if v is None or v != v else round((2.718281828 ** v - 1) * 100, 1)  # noqa: E731
    split = lambda s: [x for x in str(s or "").split(" | ") if x and x != "nan"]  # noqa: E731
    ev = {
        "nguy_co_15_phut": round(float(card["risk"]), 3),
        "mo_hinh_dua_vao": card.get("top_theme") or "",
        "thay_doi_10_phut_qua_pct": {"MAP": pct(card.get("dlog_map")), "HR": pct(card.get("dlog_hr")),
                                     "SV_proxy": pct(card.get("dlog_sv")), "SVR_proxy": pct(card.get("dlog_svr"))},
        "mau_hinh_sinh_ly": card.get("mechanism_text") or card.get("mechanism"),
        "dau_hieu_boi_canh": split(card.get("findings")),
        "ho_so_benh_nhan": [x.strip() for x in str(card.get("profile") or "").split(",") if x.strip() and x != "nan"],
    }
    if card.get("f_map") is not None and card.get("f_map") == card.get("f_map"):
        ev["du_bao_15_phut_toi_pct"] = {"MAP": pct(card["f_map"]), "HR": pct(card["f_hr"]),
                                        "SV_proxy": pct(card["f_sv"]), "SVR_proxy": pct(card["f_svr"])}
        ev["thanh_phan_chi_phoi_du_kien"] = {"hr": "HR", "sv": "SV", "svr": "SVR"}.get(card.get("dominant"))
        ev["yeu_to_dan_chinh"] = {k.upper(): card.get(f"{k}_driver1") for k in ("hr", "sv", "svr")}
        ev["MAP_du_bao_neu_khong_doi_thuoc_me_tho_may_pct"] = pct(card.get("f_map_no_change"))
    if card.get("pers_baro") is not None and card.get("pers_baro") == card.get("pers_baro"):
        ev["sinh_ly_rieng_trong_ca"] = {"do_doc_HR_theo_MAP_30_phut": round(float(card["pers_baro"]), 2),
                                        "MAP_thap_nhat_tu_dau_ca": round(float(card.get("pers_case_min", float("nan"))), 0)}
    return ev


def matching_priors(card: dict, path=None) -> list[dict]:
    """Entries of configs/clinical_priors.json that apply to this card (by profile and findings text)."""
    from pathlib import Path
    path = Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "clinical_priors.json"
    if not path.exists():
        return []
    pri = {p["id"]: p for p in json.loads(path.read_text(encoding="utf-8"))["priors"]}
    text = f"{card.get('profile', '')} | {card.get('findings', '')}"
    keys = {"htn_autoregulation": "tăng huyết áp", "elderly_baroreflex": "≥ 70 tuổi", "diabetes_autonomic": "đái tháo đường",
            "asa_reserve": "ASA ≥ 3", "propofol_vasodilation": "propofol", "remifentanil_bradycardia": "remifentanil",
            "volatile_svr": "MAC", "venous_return": "Tư thế", "post_induction": "sau khởi mê"}
    return [{"id": k, "noi_dung": pri[k]["statement"], "nguon": pri[k]["references"]}
            for k, needle in keys.items() if k in pri and needle in text]


def request(card: dict) -> dict:
    """Keyword arguments of client.beta.messages.create for one card (structured JSON output, refusal fallback)."""
    return {
        "model": MODEL,
        "max_tokens": 4000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": "Thẻ bằng chứng:\n" + json.dumps(evidence(card), ensure_ascii=False,
                                                                                 indent=2)}],
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
    }


def explain(card: dict, client=None) -> dict:
    """Call Claude for one card; returns the parsed JSON, or {"error": ...} on refusal / truncation."""
    import anthropic
    client = client or anthropic.Anthropic()
    r = client.beta.messages.create(**request(card))
    if r.stop_reason in ("refusal", "max_tokens"):
        return {"error": r.stop_reason}
    return json.loads(next(b.text for b in r.content if b.type == "text"))


DOSE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|ug|ml|mL|g|đơn vị|UI|IU)(?:/(?:kg|h|giờ|phút|min))*\b", re.I)
ORDER = re.compile(r"\b(hãy tiêm|tiêm ngay|cho ngay|truyền ngay|bắt buộc phải)\b", re.I)


def check(reading: dict, card: dict) -> list[str]:
    """Problems in an LLM reading: missing fields, doses or orders, numbers that are not in the evidence."""
    problems = [f"thiếu trường {k}" for k in SCHEMA["required"] if k not in reading]
    text = json.dumps(reading, ensure_ascii=False)
    if DOSE.search(text):
        problems.append(f"có liều thuốc: {DOSE.search(text).group(0)}")
    if ORDER.search(text):
        problems.append(f"có câu mang tính y lệnh: {ORDER.search(text).group(0)}")
    allowed = {_num(x) for x in re.findall(r"-?\d+(?:[.,]\d+)?", json.dumps(evidence(card), ensure_ascii=False))}
    allowed |= {_num(x) for x in re.findall(r"-?\d+(?:[.,]\d+)?", json.dumps(matching_priors(card), ensure_ascii=False))}
    for x in re.findall(r"-?\d+(?:[.,]\d+)?", text):
        v = _num(x)
        if abs(v) >= 2 and v not in allowed and abs(v) not in allowed and not any(abs(v - a) <= 0.51 for a in allowed):
            problems.append(f"số không có trong thẻ: {x}")
    return problems


def _num(x: str) -> float:
    return round(float(x.replace(",", ".")), 1)
