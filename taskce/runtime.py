"""Core domain types, atomic persistence, hashing, and isolated run workspaces."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Flag, auto
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import secrets
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Any, Literal, NewType
import uuid

if TYPE_CHECKING:
    from taskce.data import DataRoot
    from taskce.profiles import Profile


AnalysisName = Literal[
    "capture",
    "activation_analysis",
    "formula_search",
    "formula_comparison",
    "joint_search",
    "cross_layer_recurrence",
    "graph_analysis",
    "ablation",
]

Sha256Hex = NewType("Sha256Hex", str)
RunId = NewType("RunId", str)
LayerName = NewType("LayerName", str)

StageStatus = Literal["ok", "skipped", "failed"]
RunStatus = Literal["running", "validated", "complete", "failed"]
SearchBackendName = Literal[
    "vectorized_search_2a", "new_search", "search", "neural_search", "neural_weighted"
]


class ResourceFlags(Flag):
    """Explicit external capabilities required by an analysis stage."""

    NONE = 0
    CUDA = auto()
    HF_MODEL = auto()
    HF_DATASET = auto()


PINNED_2A_COMMIT: str = "6f2503fe78a05197943c7349d065224dd84648b0"
PINNED_2A_HASHES: Mapping[str, Sha256Hex] = {
    "__init__.py": Sha256Hex(
        "31940eb8d8fd563544b6a10603e77306e981077c23fe18aa92f40f527a34eead"
    ),
    "algorithm.py": Sha256Hex(
        "b6625658a1268ca07f96da6734da980eb43cfe43b39c38462fa09f52382d9683"
    ),
    "formula.py": Sha256Hex(
        "14d24e6ccefe62914a7d4462362caae99a698ba929be03d7a42303ac4b884f7b"
    ),
}
PINNED_2A_MODULE_KEYS: tuple[str, str, str] = (
    "vectorized_search",
    "vectorized_search.algorithm",
    "vectorized_search.formula",
)
PINNED_FORMULA_EVAL_SHA256: Sha256Hex = Sha256Hex(
    "c943c3ca49f2e8942fe53778cea8aa2b3d60837352e5ca46292b8cf9f52f9dcc"
)
PINNED_SNLI_JOINT_MANIFEST_SHA256: Sha256Hex = Sha256Hex(
    "cf80c8bc39462cfe846ed5741a7044387a06b5b5b78f698eb03f69238cd8e468"
)
PINNED_BASELINE_COMMIT: str = "57a1e74"

ALLOWED_PACKAGE_IMPORTS: Mapping[str, frozenset[str]] = {
    "taskce": frozenset(),
    "capture": frozenset({"taskce"}),
    "activation_analysis": frozenset({"taskce"}),
    "formula_search": frozenset({"taskce"}),
    "formula_comparison": frozenset({"taskce"}),
    "graph_analysis": frozenset({"taskce"}),
    "ablation": frozenset({"taskce"}),
    "joint_search": frozenset({"taskce", "formula_search"}),
    "cross_layer_recurrence": frozenset({"taskce", "formula_search"}),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path | str) -> Sha256Hex:
    """Stream-hash raw file bytes with SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return Sha256Hex(digest.hexdigest())


def sha256_json(value: Any) -> Sha256Hex:
    """Hash canonical compact sorted-key UTF-8 JSON (allow_nan=False)."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")
    return Sha256Hex(hashlib.sha256(encoded).hexdigest())


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    """Write bytes via unique sibling temp file, fsync, and atomic os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink(missing_ok=True)


def atomic_write_text(path: Path | str, text: str) -> None:
    """Atomically write UTF-8 text."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path | str, value: Any, *, canonical: bool = False) -> None:
    """Atomically write JSON (`canonical=True` uses sorted keys and compact separators)."""
    if canonical:
        rendered = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
                ensure_ascii=False,
            )
            + "\n"
        )
    else:
        rendered = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    atomic_write_text(path, rendered)


def mint_run_id() -> RunId:
    """Mint a collision-free UTC + host/pod + random-hex run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_host = os.environ.get("HOSTNAME") or socket.gethostname() or "host"
    host_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_host).strip("-")[:24] or "host"
    return RunId(f"{stamp}-{host_slug}-{secrets.token_hex(4)}")


def _git_metadata(repo_root: Path | None) -> tuple[str | None, bool | None]:
    if repo_root is None or not (repo_root / ".git").exists():
        return None, None
    try:
        commit_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        commit = commit_proc.stdout.strip() if commit_proc.returncode == 0 else None
        status_proc = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = bool(status_proc.stdout.strip()) if status_proc.returncode == 0 else None
        return commit, dirty
    except OSError:
        return None, None


@dataclass(frozen=True)
class StageRecord:
    name: str
    status: StageStatus
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class RunMeta:
    """Executor-owned envelope written atomically to `<output_dir>/run_meta.json`."""

    schema_version: int
    analysis: AnalysisName
    profile_name: str
    profile_sha256: Sha256Hex
    run_id: RunId
    status: RunStatus
    started_utc: str
    finished_utc: str | None
    git_commit: str | None
    git_dirty: bool | None
    data_dir: str
    baseline_dir: str | None
    hf_home: str | None
    python_version: str
    host: str
    requested_stages: tuple[str, ...]
    stages: tuple[StageRecord, ...]
    artifacts: Mapping[str, str]
    error: str | None = None


@dataclass(frozen=True)
class RunResult:
    analysis: AnalysisName
    profile_name: str
    run_id: RunId
    output_dir: Path
    status: RunStatus
    stages: tuple[StageRecord, ...]
    artifacts: Mapping[str, Path]
    meta_path: Path


class RunWorkspace:
    """Owns one isolated run directory, stage timing, and atomic `run_meta.json`.

    Fresh runs always use a unique run directory (or reject a non-empty explicit
    output directory unless `resume=True`). Resuming validates that prior
    `run_meta.json` matches the requested analysis and profile hash.
    """

    def __init__(
        self,
        *,
        analysis: AnalysisName,
        profile: Profile,
        data_root: DataRoot,
        run_id: RunId,
        output_dir: Path,
        requested_stages: tuple[str, ...],
        resume: bool,
        check_only: bool,
        started_utc: str,
        stages: list[StageRecord],
        artifacts: dict[str, Path],
        git_commit: str | None,
        git_dirty: bool | None,
    ) -> None:
        self.analysis = analysis
        self.profile = profile
        self.data_root = data_root
        self.run_id = run_id
        self.output_dir = output_dir
        self.requested_stages = requested_stages
        self.resume = resume
        self.check_only = check_only
        self._started_utc = started_utc
        self._stages = stages
        self._artifacts = artifacts
        self._git_commit = git_commit
        self._git_dirty = git_dirty
        self.meta_path = output_dir / "run_meta.json"

    @classmethod
    def open(
        cls,
        *,
        analysis: AnalysisName,
        profile: Profile,
        data_root: DataRoot,
        requested_stages: tuple[str, ...],
        run_id: RunId | None = None,
        output_dir: Path | None = None,
        resume: bool = False,
        check_only: bool = False,
    ) -> RunWorkspace:
        """Create or validate-for-resume one isolated output directory."""
        if output_dir is not None:
            target_dir = Path(output_dir)
            resolved_run_id = run_id or RunId(target_dir.name or mint_run_id())
        else:
            if resume and run_id is None:
                raise ValueError("resume=True requires either run_id or output_dir")
            resolved_run_id = run_id or mint_run_id()
            if (
                "/" in resolved_run_id
                or "\\" in resolved_run_id
                or resolved_run_id in {"", ".", ".."}
            ):
                raise ValueError(f"Invalid run_id {resolved_run_id!r}")
            target_dir = data_root.run_dir(analysis, profile.name, resolved_run_id)

        meta_path = target_dir / "run_meta.json"
        prior_stages: list[StageRecord] = []
        prior_artifacts: dict[str, Path] = {}
        started_utc = utc_now_iso()

        if resume:
            if not meta_path.is_file() and output_dir is None:
                raise ValueError(
                    f"Cannot resume missing run workspace (check --profile and --run-id): {meta_path}"
                )
            if meta_path.is_file():
                prior = json.loads(meta_path.read_text(encoding="utf-8"))
                if prior.get("analysis") != analysis:
                    raise ValueError(
                        f"Cannot resume {target_dir}: recorded analysis "
                        f"{prior.get('analysis')!r} != {analysis!r}"
                    )
                if prior.get("profile_sha256") != profile.profile_sha256:
                    raise ValueError(
                        f"Cannot resume {target_dir}: recorded profile_sha256 "
                        f"{prior.get('profile_sha256')!r} != {profile.profile_sha256!r}"
                    )
                if check_only and prior.get("status") == "complete":
                    raise ValueError(
                        f"Cannot resume completed run {target_dir} in --check-only mode"
                    )
                started_utc = str(prior.get("started_utc") or started_utc)
                for item in prior.get("stages", ()):
                    prior_stages.append(
                        StageRecord(
                            name=item["name"],
                            status=item["status"],
                            started_utc=item["started_utc"],
                            finished_utc=item["finished_utc"],
                            elapsed_seconds=float(item["elapsed_seconds"]),
                            metrics=dict(item.get("metrics") or {}),
                            error=item.get("error"),
                        )
                    )
                for key, value in (prior.get("artifacts") or {}).items():
                    prior_artifacts[key] = Path(value)
        elif target_dir.exists() and any(target_dir.iterdir()):
            raise FileExistsError(
                f"Output directory is not empty: {target_dir}; "
                "pass --resume or choose a fresh directory."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        git_commit, git_dirty = _git_metadata(data_root.repo_root)
        workspace = cls(
            analysis=analysis,
            profile=profile,
            data_root=data_root,
            run_id=resolved_run_id,
            output_dir=target_dir,
            requested_stages=tuple(requested_stages),
            resume=resume,
            check_only=check_only,
            started_utc=started_utc,
            stages=prior_stages,
            artifacts=prior_artifacts,
            git_commit=git_commit,
            git_dirty=git_dirty,
        )
        workspace._flush(status="running")
        return workspace

    def _build_meta(
        self,
        *,
        status: RunStatus,
        finished_utc: str | None = None,
        error: str | None = None,
    ) -> RunMeta:
        return RunMeta(
            schema_version=1,
            analysis=self.analysis,
            profile_name=self.profile.name,
            profile_sha256=self.profile.profile_sha256,
            run_id=self.run_id,
            status=status,
            started_utc=self._started_utc,
            finished_utc=finished_utc,
            git_commit=self._git_commit,
            git_dirty=self._git_dirty,
            data_dir=str(self.data_root.data_dir),
            baseline_dir=(
                str(self.data_root.baseline_dir)
                if self.data_root.baseline_dir is not None
                else None
            ),
            hf_home=os.environ.get("HF_HOME"),
            python_version=platform.python_version(),
            host=socket.gethostname(),
            requested_stages=self.requested_stages,
            stages=tuple(self._stages),
            artifacts={key: str(path) for key, path in self._artifacts.items()},
            error=error,
        )

    def _flush(
        self,
        *,
        status: RunStatus,
        finished_utc: str | None = None,
        error: str | None = None,
    ) -> None:
        meta = self._build_meta(status=status, finished_utc=finished_utc, error=error)
        atomic_write_json(self.meta_path, asdict(meta))

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        metrics_out: dict[str, Any] | None = None,
    ) -> Iterator[None]:
        """Time a stage, append StageRecord, and flush `run_meta.json` atomically."""
        started = utc_now_iso()
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            self._stages.append(
                StageRecord(
                    name=name,
                    status="failed",
                    started_utc=started,
                    finished_utc=utc_now_iso(),
                    elapsed_seconds=round(elapsed, 6),
                    metrics=dict(metrics_out) if metrics_out is not None else {},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            self._flush(status="running")
            raise
        else:
            elapsed = time.perf_counter() - t0
            self._stages.append(
                StageRecord(
                    name=name,
                    status="ok",
                    started_utc=started,
                    finished_utc=utc_now_iso(),
                    elapsed_seconds=round(elapsed, 6),
                    metrics=dict(metrics_out) if metrics_out is not None else {},
                )
            )
            self._flush(status="running")

    def record_artifact(self, key: str, path: Path | str) -> None:
        """Register a produced artifact path in run metadata."""
        self._artifacts[key] = Path(path)
        self._flush(status="running")

    def finalize(self, status: RunStatus, *, error: str | None = None) -> RunResult:
        """Persist terminal `run_meta.json` and return immutable `RunResult`."""
        self._flush(status=status, finished_utc=utc_now_iso(), error=error)
        return RunResult(
            analysis=self.analysis,
            profile_name=self.profile.name,
            run_id=self.run_id,
            output_dir=self.output_dir,
            status=status,
            stages=tuple(self._stages),
            artifacts=dict(self._artifacts),
            meta_path=self.meta_path,
        )
