"""Executor runner for the `formula_comparison` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskce.profiles import Profile
from taskce.runtime import RunWorkspace


@dataclass(frozen=True)
class FormulaComparisonOptions:
    layer: str = "14th"
    tasks: tuple[str, ...] | None = None
    diff_dir: Path | None = None
    output_path: Path | None = None
    csv_paths: tuple[Path, ...] | None = None


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layer",
        default="14th",
        help="Layer directory name for the diff stage (default: %(default)s).",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated tasks for the diff stage (default: claim,snli).",
    )
    parser.add_argument(
        "--diff-dir",
        type=Path,
        default=None,
        help="Existing formula_diff directory for the diff_viewer stage.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output HTML path for the ast_viewer and verify_ast_viewer stages.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=None,
        help="Source CSVs for the ast_viewer and verify_ast_viewer stages.",
    )


def options_from_args(args: argparse.Namespace) -> FormulaComparisonOptions:
    tasks = tuple(args.tasks.split(",")) if args.tasks else None
    return FormulaComparisonOptions(
        layer=args.layer,
        tasks=tasks,
        diff_dir=args.diff_dir,
        output_path=args.output_path,
        csv_paths=tuple(args.csv) if args.csv else None,
    )


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: FormulaComparisonOptions | None,
    stages: tuple[str, ...],
) -> Mapping[str, Path]:
    """Run diff, diff_viewer, ast_viewer, and verify_ast_viewer stages."""
    opts = options or FormulaComparisonOptions()
    raw_layer = Path(opts.layer)
    layer_dir = (
        raw_layer
        if raw_layer.is_absolute() or raw_layer.exists()
        else workspace.data_root.data_dir / raw_layer
    )
    base_dir = layer_dir / "results_base"
    finetuned_dir = layer_dir / "results"
    diff_dir = opts.diff_dir or (workspace.output_dir / "formula_diff")
    artifacts: dict[str, Path] = {}

    for stage in stages:
        with workspace.stage(stage):
            if stage == "diff":
                from formula_comparison.formula_diff.analysis import (
                    write_formula_diff_generation,
                )

                for task in (opts.tasks or ("claim", "snli")):
                    write_formula_diff_generation(
                        task,
                        base_csv=base_dir / f"{task}_beam_results.csv",
                        finetuned_csv=finetuned_dir / f"{task}_beam_results.csv",
                        output_dir=diff_dir,
                    )
                artifacts["formula_diff_dir"] = diff_dir
            elif stage == "diff_viewer":
                from formula_comparison.formula_diff.viewer import build_viewer

                output_path = opts.output_path or (
                    workspace.output_dir / "formula_diff_viewer.html"
                )
                build_viewer(diff_dir, output_path)
                artifacts["formula_diff_viewer_html"] = output_path
            elif stage == "ast_viewer":
                from formula_comparison.formula_ast import viewer

                csv_paths = opts.csv_paths or (viewer.DEFAULT_CSV,)
                output_path = opts.output_path or (
                    workspace.output_dir / "formula_ast_viewer.html"
                )
                viewer.build_viewer(csv_paths, output_path)
                artifacts["formula_ast_viewer_html"] = output_path
            elif stage == "verify_ast_viewer":
                from formula_comparison.formula_ast import viewer
                from formula_comparison.formula_ast.verify_viewer import verify_viewer

                csv_paths = opts.csv_paths or (viewer.DEFAULT_CSV,)
                html_path = opts.output_path or (
                    workspace.output_dir / "formula_ast_viewer.html"
                )
                verify_viewer(html_path, list(csv_paths))
                artifacts["formula_ast_viewer_verified"] = html_path
            else:
                raise ValueError(f"Unknown formula_comparison stage {stage!r}")
    return artifacts
