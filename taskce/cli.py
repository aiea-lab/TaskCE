"""Unified CLI entry point: `uv run taskce <analysis> --profile <profile>`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys

from taskce.executor import ANALYSES, doctor, execute
from taskce.profiles import list_builtin_profiles
from taskce.runtime import RunId


def _add_common_analysis_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_profile: str,
    stage_names: tuple[str, ...],
) -> None:
    parser.add_argument(
        "--profile",
        default=default_profile,
        help="Study profile name or path to a .toml profile (default: %(default)s).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override TASKCE_DATA_DIR for this run.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="Override TASKCE_BASELINE_DIR for read-only reference artifacts.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run identifier (required when resuming without --output).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output directory instead of <TASKCE_DATA_DIR>/runs/.../<run_id>.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run directory after validating metadata consistency.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run read-only validation stages only and skip GPU/model stages.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        dest="stages",
        choices=stage_names,
        default=None,
        help="Select one or more stages to run (repeatable).",
    )


def _try_register_runner_args(
    runner_module: str,
    subparser: argparse.ArgumentParser,
) -> None:
    top_pkg = runner_module.split(".", 1)[0]
    try:
        runner = importlib.import_module(runner_module)
    except ModuleNotFoundError as exc:
        if exc.name in (runner_module, top_pkg):
            return
        raise
    runner.add_cli_arguments(subparser)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI for analysis runs, Bowman inference, and environment checks."""
    parser = argparse.ArgumentParser(
        prog="taskce",
        description="Unified TaskCE CLI executor for compositional neuron analyses.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="List all analyses, their stages, and bundled study profiles.",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Run read-only environment and cryptographic pin checks.",
    )
    doctor_parser.add_argument("--profile", default="snli")
    doctor_parser.add_argument("--data-dir", type=Path, default=None)
    doctor_parser.add_argument("--baseline-dir", type=Path, default=None)
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable JSON.",
    )

    bowman_parser = subparsers.add_parser(
        "bowman-repro",
        aliases=["bowman_repro"],
        help="Run Bowman inference on the SNLI validation split.",
    )
    bowman_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="PyTorch device for Bowman inference (default: %(default)s).",
    )

    for spec in ANALYSES.values():
        aliases = [spec.name] if spec.cli_name != spec.name else []
        subparser = subparsers.add_parser(
            spec.cli_name,
            aliases=aliases,
            help=spec.summary,
            description=spec.summary,
        )
        subparser.set_defaults(_analysis=spec.name)
        _add_common_analysis_arguments(
            subparser,
            default_profile=spec.default_profile,
            stage_names=tuple(stage.name for stage in spec.stages),
        )
        _try_register_runner_args(spec.runner_module, subparser)

    return parser


def _cmd_list(as_json: bool) -> int:
    profiles = list_builtin_profiles()
    if as_json:
        payload = {
            "analyses": [
                {
                    "name": spec.name,
                    "command": spec.cli_name,
                    "summary": spec.summary,
                    "default_stages": list(spec.default_stages),
                    "stages": [
                        {
                            "name": stage.name,
                            "description": stage.description,
                            "default": stage.default,
                        }
                        for stage in spec.stages
                    ],
                }
                for spec in ANALYSES.values()
            ],
            "profiles": list(profiles),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("Analyses:")
    for spec in ANALYSES.values():
        stages_summary = ", ".join(
            f"{stage.name}{'' if stage.default else ' (opt-in)'}"
            for stage in spec.stages
        )
        print(f"  {spec.cli_name:<24} {spec.summary}")
        print(f"  {'':<24} stages: {stages_summary}")
    print("\nBuilt-in profiles:")
    for name in profiles:
        print(f"  {name}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    checks = doctor(
        profile=args.profile,
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
    )
    if args.as_json:
        print(json.dumps([asdict(item) for item in checks], indent=2))
    else:
        for item in checks:
            status = "OK  " if item.ok else "WARN"
            print(f"[{status}] {item.name}: {item.detail}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point returning POSIX exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "list":
        return _cmd_list(args.as_json)
    if args.command == "doctor":
        return _cmd_doctor(args)
    if args.command in ("bowman-repro", "bowman_repro"):
        from bowman_repro.__main__ import main as run_bowman

        run_bowman(device=args.device)
        return 0

    analysis_name = args._analysis
    spec = ANALYSES[analysis_name]
    runner = importlib.import_module(spec.runner_module)
    options = runner.options_from_args(args)

    result = execute(
        analysis_name,
        profile=args.profile,
        data_dir=args.data_dir,
        baseline_dir=args.baseline_dir,
        options=options,
        stages=args.stages,
        run_id=RunId(args.run_id) if args.run_id else None,
        output_dir=args.output,
        resume=args.resume,
        check_only=args.check_only,
    )
    print(
        json.dumps(
            {
                "analysis": result.analysis,
                "profile": result.profile_name,
                "run_id": result.run_id,
                "status": result.status,
                "output_dir": str(result.output_dir),
                "meta_path": str(result.meta_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
