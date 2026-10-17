"""Personalised LLM reading of alarm cards -> reports/weekly/week14/llm/.

  $env:PYTHONPATH = "$PWD/src;$PWD/.local_deps"
  python scripts/llm_cards.py --cards reports/weekly/week14/mechanism/alarm_cards.csv
         [--chain reports/weekly/week14/hemo/chain_h900.parquet] [--n 6] [--call] [--check readings.jsonl]

Without --call only the requests are written (requests.jsonl, prompts.md) so they can be reviewed first; --call sends
them to Claude (needs the `anthropic` package and credentials) and writes readings.md / readings.jsonl.
Examples: true alarms of patients with a pre-operative risk profile, one per physiological pattern.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from uc04 import llm_explain as le
from uc04.mechanism import MECHANISMS

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, default=ROOT / "reports/weekly/week14/mechanism/alarm_cards.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/weekly/week14/llm")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--chain", type=Path, default=ROOT / "reports/weekly/week14/hemo/chain_h900.parquet")
    ap.add_argument("--call", action="store_true")
    ap.add_argument("--check", type=Path, help="check an existing readings.jsonl against its cards and stop")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    c = pd.read_csv(args.cards)
    if args.chain.exists():  # forecast components, drivers, counterfactual (hemo_forecast)
        c = c.merge(pd.read_parquet(args.chain), on=["caseid", "time"], how="left")
    true_alarm = c["kind"].astype(str).str.lower() == "true"  # pandas may read the column as bool
    c = c[true_alarm & c["profile"].notna()].sort_values("risk", ascending=False)
    pick = pd.concat([c[c["mechanism"] == m].head(1) for m in MECHANISMS]).head(args.n)
    pick["mechanism_text"] = pick["mechanism"].map(MECHANISMS)
    cards = pick.to_dict("records")

    if args.check:
        rep = []
        for line in args.check.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            card = next((x for x in cards if x["caseid"] == r["caseid"] and x["time"] == r["time"]), None)
            probs = ["không tìm thấy thẻ"] if card is None else le.check(r["reading"], card)
            rep.append({"caseid": r["caseid"], "time": r["time"], "problems": " | ".join(probs) or "ok"})
        pd.DataFrame(rep).to_csv(args.out / "check.csv", index=False)
        print(pd.DataFrame(rep).to_string())
        return

    reqs = [le.request(x) for x in cards]
    with open(args.out / "requests.jsonl", "w", encoding="utf-8") as f:
        for x, r in zip(cards, reqs):
            f.write(json.dumps({"caseid": x["caseid"], "time": x["time"], "request": r}, ensure_ascii=False) + "\n")
    lines = ["# Prompt gửi LLM (xem trước)", "", "## System", "", "```", le.SYSTEM, "```", ""]
    for x, r in zip(cards, reqs):
        lines += [f"## Ca {x['caseid']}, t = {x['time'] / 60:.0f} phút", "", "```json", r["messages"][0]["content"],
                  "```", ""]
    (args.out / "prompts.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(reqs)} requests written to {args.out}")
    if not args.call:
        return

    out, md = [], ["# Diễn giải cá nhân hoá bằng LLM", ""]
    for x in cards:
        res = le.explain(x)
        out.append({"caseid": x["caseid"], "time": x["time"], "evidence": le.evidence(x), "reading": res,
                    "problems": [] if "error" in res else le.check(res, x)})
        md += [f"## Ca {x['caseid']}, t = {x['time'] / 60:.0f} phút ({x['mechanism_text']}; {x['profile']})", ""]
        if "error" in res:
            md += [f"Không có kết quả: {res['error']}", ""]
            continue
        md += [f"**Tóm tắt.** {res['tom_tat']}", "", f"**Cơ chế khả dĩ.** {res['co_che_kha_di']}", "",
               "**Bằng chứng:**", *[f"- {b}" for b in res["bang_chung"]], "",
               "**Lưu ý cho bệnh nhân này:**", *[f"- {b}" for b in res["luu_y_ca_nhan"]], "",
               "**Nên kiểm tra:**", *[f"- {b}" for b in res["nen_kiem_tra"]], "", f"**Giới hạn.** {res['gioi_han']}", ""]
    with open(args.out / "readings.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    (args.out / "readings.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
