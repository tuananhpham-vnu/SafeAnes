"""Register a 300-case expansion and reuse public raw cache without mutation."""
from pathlib import Path
import hashlib
import shutil
import pandas as pd
from safeanes.config import Protocol
from safeanes.data import fetch_csv, cohort_manifest, fetch_pilot, write_json
from safeanes.development import expanded_roles

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / "data/vitaldb_development300"
    raw = target / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "data/vitaldb/raw").iterdir():
        if source.is_file() and not (raw / source.name).exists():
            shutil.copy2(source, raw / source.name)
    manifest = cohort_manifest(fetch_csv("cases", raw), fetch_csv("trks", raw))
    selected = manifest[manifest.eligible & manifest.split.eq("train")].sample(frac=1, random_state=Protocol().seed).head(300).sort_values("caseid")
    historical = pd.read_csv(ROOT / "artifacts/tcn_v1/pilot_roles.csv")
    roles = expanded_roles(selected, historical)
    role_path = target / "development_roles.csv"
    if role_path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(role_path), roles.reset_index(drop=True))
    else:
        roles.to_csv(role_path, index=False)
    write_json(target / "selection.json", {"release": "0.2.0", "experiment": "E05", "cases": len(selected),
        "roles_sha256": hashlib.sha256(role_path.read_bytes()).hexdigest(),
        "plan_sha256": hashlib.sha256((ROOT / "docs/experiments/E05_PLAN.md").read_bytes()).hexdigest(),
        "role_counts": roles.groupby(["role", "historical_subject"]).size().reset_index(name="cases").to_dict("records")})
    print(roles.groupby(["role", "historical_subject"]).size().to_string(), flush=True)
    downloaded = fetch_pilot(target, limit=300, workers=4)
    assert set(downloaded.caseid) == set(roles.caseid)
    print("Downloaded", len(downloaded), "cases", flush=True)


if __name__ == "__main__":
    main()
