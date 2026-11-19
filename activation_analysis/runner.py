"""Executor runner for the `activation_analysis` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskce.profiles import Profile
from taskce.runtime import RunWorkspace, atomic_write_json


@dataclass(frozen=True)
class ActivationAnalysisOptions:
    layer: str | None = None
    checkpoint_name: str | None = None


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layer",
        default=None,
        help="Module layer name (default: the layer.8 alias of the profile).",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Checkpoint directory name for the finetuned variant.",
    )


def options_from_args(args: argparse.Namespace) -> ActivationAnalysisOptions:
    return ActivationAnalysisOptions(
        layer=args.layer,
        checkpoint_name=args.checkpoint_name,
    )


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: ActivationAnalysisOptions | None,
    stages: tuple[str, ...],
) -> Mapping[str, Path]:
    """Run threshold coverage, neuron fate, or task separation over saved captures."""
    opts = options or ActivationAnalysisOptions()
    data_root = workspace.data_root
    layer = opts.layer or profile.model.resolve_module_path("layer.8")
    task_name = profile.model.lora_task or profile.dataset.name
    checkpoint_name = opts.checkpoint_name or profile.model.checkpoint_name

    artifacts: dict[str, Path] = {}
    for stage in stages:
        with workspace.stage(stage):
            if stage == "threshold_coverage":
                from activation_analysis.threshold_coverage import run as run_coverage

                activation_path = data_root.activation_path(
                    profile.model.model_id,
                    profile.dataset.name,
                    layer,
                    task_name=task_name,
                    checkpoint_name=checkpoint_name,
                )
                result = run_coverage(
                    activation_path,
                    task_name=task_name,
                    alpha=profile.search.alpha,
                    min_acts=profile.search.min_acts,
                )
                artifacts["threshold_coverage_json"] = _write_json(
                    workspace.output_dir / f"threshold_coverage_{task_name}.json",
                    result,
                )
            elif stage == "neuron_fate":
                from activation_analysis.neuron_fate import run as run_fate

                base_path = data_root.activation_path(
                    profile.model.model_id,
                    profile.dataset.name,
                    layer,
                )
                finetuned_path = data_root.activation_path(
                    profile.model.model_id,
                    profile.dataset.name,
                    layer,
                    task_name=task_name,
                    checkpoint_name=checkpoint_name,
                )
                result = run_fate(base_path, finetuned_path)
                artifacts["neuron_fate_json"] = _write_json(
                    workspace.output_dir / f"neuron_fate_{task_name}.json",
                    result,
                )
            elif stage == "task_separation":
                from activation_analysis.task_separation import run as run_separation

                activation_path = data_root.activation_path(
                    profile.model.model_id,
                    profile.dataset.name,
                    layer,
                    task_name=task_name,
                    checkpoint_name=checkpoint_name,
                )
                result = run_separation(task_name, activation_path, profile=profile)
                artifacts["task_separation_json"] = _write_json(
                    workspace.output_dir / f"task_separation_{task_name}.json",
                    result,
                )
            else:
                raise ValueError(f"Unknown activation_analysis stage {stage!r}")
    return artifacts


def _write_json(path: Path, result: Any) -> Path:
    atomic_write_json(path, _to_jsonable(result))
    return path


def _to_jsonable(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return value.tolist()
    if isinstance(value, Mapping):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
