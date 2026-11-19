"""Authoritative TASKCE_DATA_DIR path resolver separating code, data, and baselines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from taskce.runtime import AnalysisName, PINNED_BASELINE_COMMIT, RunId

if TYPE_CHECKING:
    from taskce.profiles import Profile


CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


@dataclass(frozen=True)
class CaptureRef:
    model_id: str
    dataset_name: str
    layer: str
    task_name: str | None = None
    checkpoint_name: str | None = None


@dataclass(frozen=True)
class PreparedInputBundle:
    """Validated prepared NPZ directory containing features.npz and layer NPZs."""

    directory: Path
    features_path: Path
    layer_paths: Mapping[str, Path]


@dataclass(frozen=True)
class DataRoot:
    """Path resolver for TASKCE_DATA_DIR captures, prepared inputs, runs, and baselines."""

    data_dir: Path
    baseline_dir: Path | None = None
    repo_root: Path | None = None

    @classmethod
    def from_env(
        cls,
        *,
        data_dir: Path | str | None = None,
        baseline_dir: Path | str | None = None,
        baseline_commit: str = PINNED_BASELINE_COMMIT,
        repo_root: Path | str | None = None,
    ) -> DataRoot:
        """Resolve DataRoot once at process or CLI boundary."""
        resolved_repo = (
            Path(repo_root).resolve()
            if repo_root is not None
            else Path(__file__).resolve().parents[1]
        )

        if data_dir is not None:
            resolved_data = Path(data_dir)
        elif env_data := os.environ.get("TASKCE_DATA_DIR"):
            resolved_data = Path(env_data)
        else:
            resolved_data = resolved_repo / "data"

        resolved_baseline: Path | None = None
        if baseline_dir is not None:
            resolved_baseline = Path(baseline_dir)
        elif env_baseline := os.environ.get("TASKCE_BASELINE_DIR"):
            resolved_baseline = Path(env_baseline)
        else:
            sibling = resolved_data.parent / "baselines" / baseline_commit
            nested = resolved_data / "baselines" / baseline_commit
            if sibling.is_dir():
                resolved_baseline = sibling
            elif nested.is_dir():
                resolved_baseline = nested

        return cls(
            data_dir=resolved_data,
            baseline_dir=resolved_baseline,
            repo_root=resolved_repo,
        )

    def activation_path(
        self,
        model_id: str,
        dataset_name: str,
        layer: str,
        *,
        task_name: str | None = None,
        checkpoint_name: str | None = None,
    ) -> Path:
        """Locate one existing `<dataset>_activation.pt` capture."""
        model_directory = self.data_dir.joinpath(*model_id.split("/"))

        if task_name is None:
            capture_directory = model_directory / "base"
        elif checkpoint_name is not None:
            capture_directory = model_directory / task_name / checkpoint_name
        else:
            task_directory = model_directory / task_name
            checkpoints = [
                (int(match.group(1)), path)
                for path in task_directory.iterdir()
                if path.is_dir()
                and (match := CHECKPOINT_PATTERN.fullmatch(path.name)) is not None
            ]
            if not checkpoints:
                raise FileNotFoundError(
                    f"No checkpoint-N directories found under {task_directory}."
                )
            capture_directory = max(checkpoints, key=lambda item: item[0])[1]

        activation_name = f"{dataset_name}_activation.pt"
        exact_path = capture_directory / layer / activation_name
        if exact_path.is_file():
            return exact_path

        suffix_matches = [
            directory / activation_name
            for directory in capture_directory.iterdir()
            if directory.is_dir()
            and directory.name.endswith(f".{layer}")
            and (directory / activation_name).is_file()
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if suffix_matches:
            raise ValueError(
                f"Layer {layer!r} matched multiple saved activation directories: "
                f"{[str(path.parent) for path in suffix_matches]}"
            )
        raise KeyError(layer)

    def capture_output_path(
        self,
        model_id: str,
        dataset_name: str,
        layer: str,
        *,
        task_name: str | None = None,
        checkpoint_name: str | None = None,
    ) -> Path:
        """Compute canonical write destination for a captured `.pt` tensor."""
        model_directory = self.data_dir.joinpath(*model_id.split("/"))
        if task_name is None:
            variant = Path("base")
        else:
            if checkpoint_name is None:
                raise ValueError(
                    "checkpoint_name is required when task_name is provided"
                )
            variant = Path(task_name) / checkpoint_name
        return model_directory / variant / layer / f"{dataset_name}_activation.pt"

    def load_activation(self, path: Path | str, *, as_numpy: bool = False) -> Any:
        """Lazily import torch and load one CPU float32 2D [examples, neurons] tensor."""
        import torch

        activation = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(activation, torch.Tensor):
            raise TypeError("Experimental activation data must be a tensor.")
        if activation.ndim != 2:
            raise ValueError(
                "Experimental activation data must have shape [examples, neurons], "
                f"got {tuple(activation.shape)}."
            )
        tensor = activation.detach().cpu().to(torch.float32)
        return tensor.numpy() if as_numpy else tensor

    def resolve_prepared_inputs(
        self,
        profile: Profile,
        *,
        override: Path | None = None,
    ) -> PreparedInputBundle:
        """Locate `features.npz` and `<layer>.npz` files under override or TASKCE_DATA_DIR."""
        candidates = (
            [Path(override)]
            if override is not None
            else [self.data_dir / rel for rel in profile.artifacts.prepared_inputs_relpaths]
        )
        for directory in candidates:
            features_path = directory / "features.npz"
            if features_path.is_file():
                layer_paths = {
                    path.stem: path
                    for path in sorted(directory.glob("*.npz"))
                    if path.name != "features.npz"
                }
                return PreparedInputBundle(
                    directory=directory,
                    features_path=features_path,
                    layer_paths=layer_paths,
                )
        searched = ", ".join(str(path) for path in candidates) or str(self.data_dir)
        raise FileNotFoundError(
            f"Prepared inputs (features.npz) not found for profile {profile.name!r}; "
            f"searched: {searched}"
        )

    def resolve_baseline_path(
        self,
        relpath: str,
        *,
        override: Path | None = None,
    ) -> Path:
        """Resolve a read-only reference file/dir from override or `baseline_dir`."""
        if override is not None:
            resolved = Path(override)
            if not resolved.exists():
                raise FileNotFoundError(f"Baseline override path not found: {resolved}")
            return resolved
        if self.baseline_dir is None:
            raise FileNotFoundError(
                f"No baseline directory resolved for {relpath!r}; "
                "set TASKCE_BASELINE_DIR or pass --baseline-dir."
            )
        candidate = self.baseline_dir / relpath
        if not candidate.exists():
            raise FileNotFoundError(f"Baseline artifact not found: {candidate}")
        return candidate

    def tokenizer_cache_dir(self, resource_name: str) -> Path:
        """Return `<data_dir>/tokenizers/<resource_name>` for newly trained tokenizers."""
        return self.data_dir / "tokenizers" / resource_name

    def run_dir(
        self,
        analysis: AnalysisName,
        profile_name: str,
        run_id: RunId,
    ) -> Path:
        """Return `<data_dir>/runs/<analysis>/<profile_name>/<run_id>`."""
        return self.data_dir / "runs" / analysis / profile_name / run_id

    def latest_completed_run(
        self,
        analysis: AnalysisName,
        profile_name: str,
    ) -> Path | None:
        """Scan immutable `run_meta.json` files for latest `status == 'complete'` run."""
        root = self.data_dir / "runs" / analysis / profile_name
        if not root.is_dir():
            return None
        completed: list[tuple[str, Path]] = []
        for entry in root.iterdir():
            meta_path = entry / "run_meta.json"
            if not entry.is_dir() or not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if meta.get("status") == "complete":
                stamp = str(meta.get("finished_utc") or meta.get("started_utc") or "")
                completed.append((stamp, entry))
        if not completed:
            return None
        return max(completed, key=lambda item: (item[0], item[1].name))[1]


def load_formula_table(path: Path | str) -> Any:
    """Load either legacy `(neuron, formula, iou, ...)` or sweep CSV into a DataFrame.

    For sweep CSVs without an `iou` column, select the highest-priority split
    (`test`, `validation`, `fit`) that actually has examples (`*_n > 0`), so
    all-example sweeps (`test_n == 0, validation_n == 0`) preserve their fit IoU
    while genuine held-out splits (`test_n > 0`) still use `test_iou`.
    """
    import pandas as pd

    frame = pd.read_csv(Path(path))
    if "iou" not in frame.columns:
        selected = False
        for split_name in ("test", "validation", "fit"):
            iou_col = f"{split_name}_iou"
            if iou_col not in frame.columns:
                continue
            n_col = f"{split_name}_n"
            if n_col in frame.columns:
                has_examples = bool(
                    (pd.to_numeric(frame[n_col], errors="coerce").fillna(0) > 0).any()
                )
                if not has_examples:
                    continue
            else:
                if not bool(pd.to_numeric(frame[iou_col], errors="coerce").notna().any()):
                    continue
            frame["iou"] = pd.to_numeric(frame[iou_col], errors="coerce").fillna(0.0)
            selected = True
            break
        if not selected and "backend_fit_iou" in frame.columns:
            frame["iou"] = pd.to_numeric(
                frame["backend_fit_iou"], errors="coerce"
            ).fillna(0.0)
    return frame
