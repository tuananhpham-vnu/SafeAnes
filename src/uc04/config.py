"""Config loading (plan section 4).

`load_config(path)` returns an immutable `Config`. Every section is a frozen
dataclass with `digest()` = sha256 of its canonical JSON. `Config.digest()` covers
every section except `paths`, `hub` and `kaggle`, which only decide where things
are stored and never change results.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

NON_RESULT_SECTIONS = ("paths", "hub", "kaggle")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    return value


def canonical_json(obj: Any) -> str:
    return json.dumps(_thaw(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Section:
    """Base for the frozen section dataclasses."""

    def to_dict(self) -> dict:
        return {f.name: _thaw(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def digest(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        missing = {f.name for f in dataclasses.fields(cls)
                   if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING} - set(data)
        if unknown or missing:
            raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}")
        return cls(**{k: _freeze(v) for k, v in data.items()})


@dataclass(frozen=True)
class Paths(Section):
    prep: str
    samples: str
    sealed: str  # test labels/events; outside `samples` so uploading `samples` can never include them
    artifacts: str
    reports: str


@dataclass(frozen=True)
class Input(Section):
    required: tuple
    required_for_dl: tuple
    forbidden_feature_prefixes: tuple
    max_missing_case_fraction: float
    known_failed_cases: tuple
    require_wave: bool
    min_map_coverage: float
    qc_sha256: str  # sha256 of the checked qc.csv; every later step and Kaggle compare against it


@dataclass(frozen=True)
class Features(Section):
    windows_seconds: tuple
    cadence_seconds: int
    numeric_signals: tuple
    numeric_stats: tuple
    map_extra: tuple
    beat_signals: tuple
    beat_stats: tuple
    beat_extra: tuple
    recompute_ppv: bool
    ppv_subwindow_seconds: int
    ppv_min_beats: int
    ppv_max_period_cv: float
    ventilated_rr: tuple
    ventilated_min_etco2: float


@dataclass(frozen=True)
class Labels(Section):
    map_threshold: float
    event_seconds: int
    merge_gap_seconds: int
    horizons_seconds: tuple
    post_event_exclusion_seconds: int
    exclusion_reset: str
    max_map_missing_w120: float
    label_policy: str
    negative_rule: str  # "possible_event" (default, DEVIATIONS 23) or "unknown_fraction" (plan 3.5)
    negative_max_unknown_fraction: float
    negative_max_unknown_run_seconds: int
    suspect_artefact_min_map: float

    def __post_init__(self):
        if self.exclusion_reset not in ("event", "any_low"):
            raise ValueError(f"labels.exclusion_reset must be 'event' or 'any_low', got {self.exclusion_reset!r}")
        if self.label_policy not in ("lenient", "strict"):
            raise ValueError(f"labels.label_policy must be 'lenient' or 'strict', got {self.label_policy!r}")
        if self.negative_rule not in ("possible_event", "unknown_fraction"):
            raise ValueError(f"labels.negative_rule must be 'possible_event' or 'unknown_fraction', "
                             f"got {self.negative_rule!r}")


@dataclass(frozen=True)
class Splits(Section):
    fit: str
    calibration: str
    validation: str
    test: str
    test_requires_lock: bool


@dataclass(frozen=True)
class Hub(Section):
    samples_repo: str
    samples_repo_type: str
    runs_repo: str
    runs_repo_type: str
    private: bool
    push_min_interval_seconds: int
    forbidden_path_parts: tuple
    token_env: str


@dataclass(frozen=True)
class Kaggle(Section):
    user: str
    prep_kernel_source: str
    code_dataset: str
    samples_dataset: str
    wave_dataset: str
    wave_dataset_version: int
    code_repo: str
    tabular_machine: Mapping
    dl_machine: Mapping
    samples_json_sha256: str = ""  # sha256 of samples.json on Kaggle (scripts/kaggle_run.py)
    wave_qc_sha256: str = ""  # qc.csv of wave_dataset when it differs from input.qc_sha256 (v3: prep_v1 wave)


@dataclass(frozen=True)
class Baselines(Section):
    map_threshold: Mapping
    map_logistic: Mapping


@dataclass(frozen=True)
class LightGBM(Section):
    versions: Mapping
    params: Mapping
    missing: str
    monotone_decreasing: tuple
    shap_rows: int
    confidence_seeds: tuple
    confidence_params: Mapping


@dataclass(frozen=True)
class DL(Section):
    windows_seconds: tuple
    seeds: tuple
    wave_hz: int
    tab_features: str
    tab_extra_flags: tuple
    conv: Mapping
    transformer: Mapping
    tab_mlp: tuple
    head_hidden: int
    ordered_horizons: bool
    optimizer: Mapping
    batch_size: int
    max_epochs: int
    patience: int
    train_samples_per_epoch: int
    amp: bool
    num_workers: int


@dataclass(frozen=True)
class Calibration(Section):
    tabular: str
    dl: str


@dataclass(frozen=True)
class Evaluation(Section):
    alarm_persistence: int
    alarm_cooldown_seconds: int
    alarm_rearm: str  # "drop_below" (plan 8.2) or "cooldown" (repeat every cooldown while above)
    fa_per_hour_budget: float
    threshold_candidates: int
    curve_points: int
    ece_bins: int
    early_lead_seconds: int
    bootstrap_unit: str
    bootstrap_repeats_validation: int
    bootstrap_repeats_test: int


SECTIONS = {
    "paths": Paths, "input": Input, "features": Features, "labels": Labels, "splits": Splits,
    "hub": Hub, "kaggle": Kaggle, "baselines": Baselines, "lightgbm": LightGBM, "dl": DL,
    "calibration": Calibration, "evaluation": Evaluation,
}


@dataclass(frozen=True)
class Config:
    version: str
    seed: int
    paths: Paths
    input: Input
    features: Features
    labels: Labels
    splits: Splits
    hub: Hub
    kaggle: Kaggle
    baselines: Baselines
    lightgbm: LightGBM
    dl: DL
    calibration: Calibration
    evaluation: Evaluation
    source: str = field(default="", compare=False)
    root: str = field(default=".", compare=False)

    def digest(self, *sections: str) -> str:
        """Digest of the named sections (default: every result-relevant section)."""
        names = sections or tuple(n for n in SECTIONS if n not in NON_RESULT_SECTIONS)
        parts = {"version": self.version, "seed": self.seed}
        parts.update({n: getattr(self, n).digest() for n in names})
        return sha256_text(canonical_json(parts))

    def path(self, name: str) -> Path:
        """Resolve `paths.<name>`; relative paths are relative to the repo root."""
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else Path(self.root) / p

    def with_paths(self, **overrides: str | None) -> "Config":
        """Override paths from the command line (does not change any digest)."""
        clean = {k: str(v) for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, paths=dataclasses.replace(self.paths, **clean))

    def to_dict(self) -> dict:
        out = {"version": self.version, "seed": self.seed}
        out.update({n: getattr(self, n).to_dict() for n in SECTIONS})
        return out


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    unknown = set(raw) - set(SECTIONS) - {"version", "seed"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    sections = {name: cls.from_dict(raw[name]) for name, cls in SECTIONS.items()}
    root = path.resolve().parent.parent  # configs/<file>.json -> repo root
    return Config(version=raw["version"], seed=raw["seed"], source=str(path), root=str(root), **sections)
