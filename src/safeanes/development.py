"""Stable development expansion with immutable historical patient roles."""
import hashlib
import pandas as pd


def expanded_roles(manifest, historical, seed=20260917):
    if not manifest.split.eq("train").all():
        raise ValueError("Only global train allowed")
    if historical.groupby("subjectid").role.nunique().gt(1).any():
        raise ValueError("Historical subject has conflicting roles")
    allowed = ("fit", "calibration", "validation", "pilot_test")
    if not historical.role.isin(allowed).all():
        raise ValueError("Unknown historical role")
    if not set(historical.caseid).issubset(set(manifest.caseid)):
        raise ValueError("Expansion must retain old cases")
    joined = historical.merge(manifest[["caseid", "subjectid"]], on="caseid", suffixes=("_old", "_new"), validate="one_to_one")
    if not joined.subjectid_old.eq(joined.subjectid_new).all():
        raise ValueError("Historical case changed subject")
    assignment = historical.drop_duplicates("subjectid").set_index("subjectid").role.to_dict()
    def role(subject):
        if subject in assignment:
            return assignment[subject]
        u = int.from_bytes(hashlib.sha256(f"E05:{seed}:{int(subject)}".encode()).digest()[:8], "big") / 2**64
        return "fit" if u < .60 else "calibration" if u < .75 else "validation" if u < .85 else "pilot_test"
    out = manifest[["caseid", "subjectid"]].copy()
    out["role"] = out.subjectid.map(role)
    out["historical_subject"] = out.subjectid.isin(assignment)
    return out
