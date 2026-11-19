"""Executor runner for the `ablation` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from taskce.data import DataRoot
from taskce.profiles import Profile
from taskce.runtime import RunWorkspace


@dataclass(frozen=True)
class AblationRunnerOptions:
    formula_path: Path | None = None
    output_directory: Path | None = None
    layer: str | None = None
    lora_repo: str | None = None


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--formula-path",
        type=Path,
        default=None,
        help="Formula CSV from formula_search (default: compositional path).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Output directory (default: TASKCE_DATA_DIR).",
    )
    parser.add_argument(
        "--layer",
        default=None,
        help="Ablation layer module path (default: the layer.8 alias of the profile).",
    )
    parser.add_argument(
        "--lora-repo",
        default=None,
        help="LoRA repository override (default: the profile lora_repo).",
    )


def options_from_args(args: argparse.Namespace) -> AblationRunnerOptions:
    return AblationRunnerOptions(
        formula_path=args.formula_path,
        output_directory=args.output_directory,
        layer=args.layer,
        lora_repo=args.lora_repo,
    )


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: AblationRunnerOptions | None,
    stages: tuple[str, ...],
) -> Mapping[str, Path]:
    """Run select and infer stages; HF imports stay lazy inside infer."""
    opts = options or AblationRunnerOptions()
    data_root = workspace.data_root
    task_name = profile.model.lora_task or profile.dataset.name
    formula_path = opts.formula_path or _default_formula_path(task_name)
    output_directory = opts.output_directory or data_root.data_dir
    layer = opts.layer or profile.model.resolve_module_path("layer.8")

    artifacts: dict[str, Path] = {}

    if "select" in stages:
        with workspace.stage("select"):
            from ablation.selection import run_ablation_analysis

            run_ablation_analysis(
                formula_path,
                output_dir=output_directory,
                weight_column_names=profile.dataset.weight_column_names,
            )
        artifacts["selection_report"] = output_directory / "ablation_analysis_report.txt"
        artifacts["selection_good"] = output_directory / "ablation_selection_good.csv"
        artifacts["selection_bad"] = output_directory / "ablation_selection_bad.csv"
        artifacts["selection_iou_ranked"] = output_directory / "ablation_selection_iou_ranked.csv"

    if "infer" in stages:
        with workspace.stage("infer"):
            from ablation import analysis as ablation_analysis

            result = ablation_analysis.run(
                task_name,
                formula_path=formula_path,
                output_directory=output_directory,
                model_id=profile.model.model_id,
                lora_repository=opts.lora_repo
                or profile.model.lora_repo
                or ablation_analysis.DEFAULT_LORA_REPOSITORY,
                layer=layer or ablation_analysis.DEFAULT_LAYER,
            )
        artifacts.update(result.result_paths)
        artifacts["ablation_good_vs_bad_plot"] = result.plot_paths["good_bad"]
        artifacts["ablation_combined_plot"] = result.plot_paths["combined"]
    return artifacts


def _default_formula_path(task_name: str) -> Path:
    root = DataRoot.from_env()
    return root.data_dir / "compositional_explanations" / f"{task_name}_beam_results.csv"
