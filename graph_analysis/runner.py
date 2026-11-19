"""Executor runner for the `graph_analysis` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from taskce.profiles import Profile
from taskce.runtime import RunWorkspace


@dataclass(frozen=True)
class GraphAnalysisOptions:
    layer: str | None = None
    formula_path: Path | None = None
    output_directory: Path | None = None


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layer",
        default=None,
        help="Module layer name (default: the layer.8 alias of the profile).",
    )
    parser.add_argument(
        "--formula-path",
        type=Path,
        default=None,
        help="Compositional formula CSV (default: profile compositional path).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Output directory for graphs and reports (default: run workspace output directory).",
    )


def options_from_args(args: argparse.Namespace) -> GraphAnalysisOptions:
    return GraphAnalysisOptions(
        layer=args.layer,
        formula_path=args.formula_path,
        output_directory=args.output_directory,
    )


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: GraphAnalysisOptions | None,
    stages: tuple[str, ...],
) -> Mapping[str, Path]:
    """Run the build_and_report stage; heavy plotting imports stay lazy."""
    from graph_analysis import analysis

    opts = options or GraphAnalysisOptions()
    data_root = workspace.data_root
    layer = opts.layer or profile.model.resolve_module_path("layer.8")
    task_name = profile.model.lora_task or profile.dataset.name
    formula_path = opts.formula_path or (
        data_root.data_dir / "compositional_explanations" / f"{task_name}_beam_results.csv"
    )
    output_directory = opts.output_directory or workspace.output_dir
    artifacts: dict[str, Path] = {}

    for stage in stages:
        with workspace.stage(stage):
            if stage != "build_and_report":
                raise ValueError(f"Unknown graph_analysis stage {stage!r}")
            outputs = analysis.run(
                task_name,
                str(data_root.activation_path(profile.model.model_id, profile.dataset.name, layer)),
                str(
                    data_root.activation_path(
                        profile.model.model_id,
                        profile.dataset.name,
                        layer,
                        task_name=task_name,
                        checkpoint_name=profile.model.checkpoint_name,
                    )
                ),
                formula_path=formula_path,
                output_directory=output_directory,
            )
            for key, bundle in outputs.items():
                for artifact_name, path in bundle.items():
                    artifacts[f"{key}_{artifact_name}"] = Path(path)
    return artifacts
