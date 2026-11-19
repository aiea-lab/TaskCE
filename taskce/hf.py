"""Lazy-only Hugging Face cache policy, LoRA checkpoint resolution, and dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any

from taskce.data import DataRoot
from taskce.profiles import DatasetSpec


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


@dataclass(frozen=True)
class HfCachePolicy:
    """Enforces pod-local model/tokenizer cache separation from TASKCE_DATA_DIR."""

    hf_home: Path
    token: str | None = None


def configure_hf_cache(
    data_root: DataRoot,
    *,
    hf_home: Path | None = None,
    allow_under_data_dir: bool = False,
) -> HfCachePolicy:
    """Resolve and export HF_HOME outside TASKCE_DATA_DIR unless explicitly opted in."""
    if hf_home is not None:
        resolved_home = Path(hf_home)
    elif env_home := os.environ.get("HF_HOME"):
        resolved_home = Path(env_home)
    else:
        resolved_home = Path.home() / ".cache" / "huggingface"

    if not allow_under_data_dir:
        try:
            inside_data = resolved_home.resolve().is_relative_to(
                data_root.data_dir.resolve()
            )
        except OSError:
            inside_data = False
        if inside_data:
            raise ValueError(
                "HF_HOME must stay pod-local, separate from TASKCE_DATA_DIR: "
                f"{resolved_home} is inside {data_root.data_dir}"
            )

    os.environ["HF_HOME"] = str(resolved_home)
    return HfCachePolicy(hf_home=resolved_home, token=os.environ.get("HF_TOKEN"))


def resolve_lora_checkpoint(
    repo_id: str,
    task_name: str,
    *,
    checkpoint_name: str | None = None,
    policy: HfCachePolicy | None = None,
) -> Path:
    """Resolve a local LoRA directory or download a HF snapshot and select checkpoint-N."""
    local_candidate = Path(repo_id)
    if local_candidate.is_dir():
        lora_root = local_candidate
    else:
        from huggingface_hub import snapshot_download

        token = policy.token if policy is not None else os.environ.get("HF_TOKEN")
        lora_root = Path(
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
        )

    task_directory = lora_root / task_name
    if not task_directory.is_dir():
        raise FileNotFoundError(f"No LoRA task directory found at {task_directory}")

    if checkpoint_name is not None:
        selected = task_directory / checkpoint_name
        if not selected.is_dir():
            raise FileNotFoundError(f"Requested LoRA checkpoint not found: {selected}")
        return selected

    checkpoints = [path for path in task_directory.iterdir() if path.is_dir()]
    if not checkpoints:
        raise FileNotFoundError(f"No LoRA checkpoint found for {task_name} under {task_directory}")

    def sort_key(path: Path) -> tuple[int, str]:
        match = _CHECKPOINT_RE.fullmatch(path.name)
        return (int(match.group(1)) if match else -1, path.name)

    return max(checkpoints, key=sort_key)


def load_hf_dataset(
    dataset: DatasetSpec,
    *,
    dataset_path: Path | None = None,
) -> Any:
    """Load dataset from `datasets.load_from_disk` or `datasets.load_dataset`."""
    from datasets import load_dataset, load_from_disk

    if dataset_path is not None:
        return load_from_disk(str(dataset_path))
    return load_dataset(dataset.hf_id, split=dataset.split)
