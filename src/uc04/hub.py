"""Hugging Face sync with guards against test data (plan 2.13, 15.3).

Rules enforced here:
- `assert_no_test` refuses any file whose path has a `sealed`/`test` component,
  whose file stem is `test`, `test_*` or `*_test`, or which is a parquet/csv with
  a `split` column containing "test". Matching is per path component, never
  substring, so `latest/`, `tests/`, `smoke` pass. The only exception is
  `reports/test_results.csv`, and only once `lock.json` exists.
- `HubSync.push` only accepts paths this notebook registered as its own, so a
  notebook never overwrites files another notebook wrote.
- A dry run (`enabled=False`) never writes to HF. `only=` restricts pushes to an
  allow-list (used by `--smoke`, which may push only its timing file).
- Network errors are retried (30, 60, 120 s) and then logged, never raised, so a
  failed push does not stop training.
"""
from __future__ import annotations

import os
import time
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Sequence

import pandas as pd
import pyarrow.parquet as pq

FORBIDDEN_PARTS = ("sealed", "test")
TEST_RESULTS = "reports/test_results.csv"
RETRY_WAITS = (30, 60, 120)


class TestDataError(RuntimeError):
    """Raised when a file about to leave the machine may contain test data."""

    __test__ = False  # not a pytest test class


class OwnershipError(RuntimeError):
    """Raised when a notebook tries to push a path it did not create."""


def _stem(name: str) -> str:
    return name.split(".", 1)[0]


def is_test_path(rel: str, forbidden_parts: Sequence[str] = FORBIDDEN_PARTS) -> bool:
    parts = PurePosixPath(rel.replace("\\", "/")).parts
    if any(p in forbidden_parts for p in parts[:-1]):
        return True
    stem = _stem(parts[-1]) if parts else ""
    return stem in forbidden_parts or stem.startswith("test_") or stem.endswith("_test")


def _split_has_test(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if "split" not in pq.read_schema(path).names:
            return False
        col = pq.read_table(path, columns=["split"]).column(0).to_pandas()
    elif suffix == ".csv":
        head = pd.read_csv(path, nrows=0)
        if "split" not in head.columns:
            return False
        col = pd.read_csv(path, usecols=["split"])["split"]
    else:
        return False
    return bool((col.astype(str) == "test").any())


def assert_no_test(files: Iterable[str], root: Path | None = None, *, lock_exists: bool = False,
                   forbidden_parts: Sequence[str] = FORBIDDEN_PARTS) -> None:
    """`files` are repo-relative posix paths; if `root` is given their content is checked too."""
    for rel in files:
        rel = rel.replace("\\", "/")
        if rel == TEST_RESULTS:
            if not lock_exists:
                raise TestDataError(f"{rel} may only be pushed after lock.json exists")
            continue
        if is_test_path(rel, forbidden_parts):
            raise TestDataError(f"refusing to push test data: {rel}")
        if root is not None:
            p = Path(root) / rel
            if p.is_file() and _split_has_test(p):
                raise TestDataError(f"refusing to push {rel}: column 'split' contains 'test'")


def _norm(rel: str | Path) -> str:
    out = PurePosixPath(str(rel).replace("\\", "/")).as_posix().strip("/")
    return "" if out == "." else out


def _under(rel: str, prefix: str) -> bool:
    return rel == prefix or rel.startswith(prefix + "/")


class HubSync:
    def __init__(self, repo_id: str, local_dir: Path, *, repo_type: str = "model",
                 min_interval_s: float = 600, enabled: bool = True, only: Sequence[str] | None = None,
                 lock_path: Path | None = None, api=None, token: str | None = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 retry_waits: Sequence[float] = RETRY_WAITS, log: Callable[[str], None] = print,
                 strip_prefix: str = ""):
        self.repo_id, self.repo_type = repo_id, repo_type
        self.strip_prefix = strip_prefix  # local "artifacts/tabular/x" -> repo "tabular/x"
        self.local_dir = Path(local_dir)
        self.min_interval_s = min_interval_s
        self.enabled = enabled
        self.only = None if only is None else {_norm(p) for p in only}
        self.lock_path = lock_path
        self._api = api
        self._token = token
        self.sleep, self.clock, self.retry_waits, self.log = sleep, clock, tuple(retry_waits), log
        self.owned: list[str] = []
        self.last_push: float | None = None

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=self._token)
        return self._api

    # ------------------------------------------------------------ ownership

    def register(self, *paths: str | Path) -> None:
        """Declare repo-relative paths (files or directories) this notebook creates."""
        for p in paths:
            rel = _norm(p)
            if rel and rel not in self.owned:
                self.owned.append(rel)

    def owns(self, rel: str) -> bool:
        rel = _norm(rel)
        return any(_under(rel, o) for o in self.owned)

    def _files(self, paths: Iterable[str]) -> list[str]:
        out: list[str] = []
        for rel in paths:
            p = self.local_dir / rel
            if p.is_dir():
                out += sorted(_norm(f.relative_to(self.local_dir)) for f in p.rglob("*")
                              if f.is_file() and not f.name.startswith(".") and ".tmp" not in f.name)
            elif p.is_file():
                out.append(rel)
        return out

    # ------------------------------------------------------------ pushing

    def push(self, paths: Sequence[str | Path] | None = None, message: str = "update") -> bool:
        """Push owned paths now. Returns True if a commit was made."""
        rels = [_norm(p) for p in (paths if paths is not None else self.owned)]
        foreign = [r for r in rels if not self.owns(r)]
        if foreign:
            raise OwnershipError(f"not created by this notebook, refusing to push: {foreign}")
        if self.only is not None:
            skipped = [r for r in rels if not any(_under(r, o) or _under(o, r) for o in self.only)]
            rels = [r for r in rels if r not in skipped]
        files = self._files(rels)
        if self.only is not None:
            files = [f for f in files if f in self.only]
        if not files:
            return False
        lock_exists = bool(self.lock_path and Path(self.lock_path).exists())
        assert_no_test(files, self.local_dir, lock_exists=lock_exists)
        if not self.enabled:
            self.log(f"[hub] dry run, not pushing {len(files)} file(s) to {self.repo_id}")
            return False
        return self._commit(files, message)

    def maybe_push(self, paths: Sequence[str | Path] | None = None, message: str = "update") -> bool:
        """Push only if `min_interval_s` has passed since the last push."""
        if self.last_push is not None and self.clock() - self.last_push < self.min_interval_s:
            return False
        return self.push(paths, message)

    def repo_path(self, rel: str) -> str:
        p = self.strip_prefix
        return rel[len(p):] if p and rel.startswith(p) else rel

    def _commit(self, files: list[str], message: str) -> bool:
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=self.repo_path(f), path_or_fileobj=str(self.local_dir / f))
               for f in files]
        for attempt, wait in enumerate((0, *self.retry_waits)):
            if wait:
                self.sleep(wait)
            try:
                self.api.create_commit(repo_id=self.repo_id, repo_type=self.repo_type,
                                       operations=ops, commit_message=message)
                self.last_push = self.clock()
                return True
            except Exception as exc:  # network or HF errors must not stop training
                self.log(f"[hub] push attempt {attempt + 1} failed: {exc!r}")
        self.log(f"[hub] WARNING: giving up on this push of {len(files)} file(s); the next push will resend them")
        return False

    # ------------------------------------------------------------ reading

    def done_set(self, prefix: str = "") -> set[str]:
        """Directories under `prefix` that already have done.json on HF (empty in a dry run)."""
        if not self.enabled:
            return set()
        prefix = _norm(prefix)
        files = self.api.list_repo_files(self.repo_id, repo_type=self.repo_type)
        return {f.rsplit("/", 1)[0] for f in files
                if f.endswith("/done.json") and (not prefix or _under(f, prefix))}

    def pull(self, path: str, revision: str | None = None, local_dir: Path | None = None) -> Path:
        """Download one file, or a directory (path ending with '/'), from HF."""
        from huggingface_hub import hf_hub_download, snapshot_download
        target = Path(local_dir or self.local_dir)
        if path.endswith("/"):
            snapshot_download(self.repo_id, repo_type=self.repo_type, revision=revision,
                              allow_patterns=[f"{_norm(path)}/*"], local_dir=target, token=self._token)
            return target / _norm(path)
        return Path(hf_hub_download(self.repo_id, _norm(path), repo_type=self.repo_type, revision=revision,
                                    local_dir=target, token=self._token))


def ensure_repo(repo_id: str, repo_type: str, private: bool = True, api=None) -> None:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)


def hf_token(env: str = "HF_TOKEN", dotenv: Path | None = None) -> str | None:
    """Token from the environment (Kaggle Secret), else from the repo's git-ignored .env (local runs).
    None means fall back to `hf auth login`. The value is never printed."""
    if os.environ.get(env):
        return os.environ[env]
    path = dotenv or Path(__file__).resolve().parents[2] / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == env and not line.lstrip().startswith("#"):
                return value.strip().strip('"').strip("'") or None
    return None


class HubRequired(SystemExit):
    """On Kaggle, results must be pushed to HF (a stopped session may lose its output)."""


def on_kaggle() -> bool:
    return os.environ.get("UC04_RUNTIME") == "kaggle" or Path("/kaggle/input").exists()


class PublicRepoError(SystemExit):
    """Results contain per-case predictions derived from patient data: never push to a public repo."""


def assert_private_repo(repo: str, repo_type: str, token: str | None, api=None) -> None:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
    info = api.repo_info(repo, repo_type=repo_type)
    if not getattr(info, "private", False):
        raise PublicRepoError(f"[hub] {repo} is PUBLIC but hub.private is true: refusing to push results. "
                              "Make it private on Hugging Face (Settings -> Change visibility).")


def hub_from_config(cfg, local_dir: Path, *, push: bool, only: Sequence[str] | None = None,
                    log: Callable[[str], None] = print, kaggle: bool | None = None, api=None) -> HubSync:
    """HubSync for `hub.runs_repo`. Pushing is off (results stay in `local_dir`) when `push` is
    False, the repo name is still a placeholder, or no HF token is available.

    On Kaggle, a placeholder repo or a missing token is an error (HubRequired), not a silent
    fallback: if a GPU session stops, notebook output may be lost and resuming from last.pt
    needs HF. An explicit --no-push is still honoured."""
    repo = cfg.hub.runs_repo
    token = hf_token(cfg.hub.token_env)
    if token is None and push:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    reason = ("--no-push or dry run" if not push else "hub.runs_repo is a placeholder" if "<" in repo
              else "no HF token (set Kaggle Secret HF_TOKEN or run `hf auth login`)" if not token else "")
    if reason and push and (on_kaggle() if kaggle is None else kaggle):
        raise HubRequired(f"[hub] on Kaggle, results must go to Hugging Face but {reason}. "
                          "Set hub.runs_repo in the config and the Kaggle Secret HF_TOKEN (enabled for this notebook).")
    if reason:
        log(f"[hub] pushing disabled: {reason}; results stay in {local_dir}")
    elif cfg.hub.private:
        assert_private_repo(repo, cfg.hub.runs_repo_type, token, api=api)
    lock = Path(cfg.path("artifacts")) / "lock.json"
    return HubSync(repo, local_dir, repo_type=cfg.hub.runs_repo_type, min_interval_s=cfg.hub.push_min_interval_seconds,
                   enabled=not reason, only=only, lock_path=lock, api=api, token=token, log=log,
                   strip_prefix="artifacts/")
