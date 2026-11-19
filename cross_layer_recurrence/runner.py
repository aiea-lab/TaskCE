"""Executor runner for the `cross_layer_recurrence` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from taskce.profiles import Profile
from taskce.runtime import RunWorkspace


@dataclass(frozen=True)
class CrossLayerRecurrenceOptions:
    locations: tuple[str, ...] | None = None
    results_dir: Path | None = None
    inputs_dir: Path | None = None
    sentence_arrow: Path | None = None
    neuron_ids: str = "all"
    device: str = "cuda"
    no_seed_layer8: bool = False


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--locations",
        default=None,
        help="Comma-separated layer names in ascending model order; "
        "defaults to the profile recurrence_layers.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing full/<layer>.csv formula outputs.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=None,
        help="Directory containing prepared features.npz and <layer>.npz files.",
    )
    parser.add_argument(
        "--sentence-arrow",
        type=Path,
        default=None,
        help="Explicit snli-validation.arrow path for full_sweep row-hash verification.",
    )
    parser.add_argument(
        "--ids",
        dest="neuron_ids",
        default="all",
        help="Comma-separated neuron IDs or 'all' for full_sweep (default: all).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for full_sweep (default: %(default)s).",
    )
    parser.add_argument(
        "--no-seed-layer8",
        action="store_true",
        help="Refit layer.8 directly instead of seeding from the historical reference trial.",
    )


def options_from_args(args: argparse.Namespace) -> CrossLayerRecurrenceOptions:
    return CrossLayerRecurrenceOptions(
        locations=tuple(args.locations.split(",")) if args.locations else None,
        results_dir=args.results_dir,
        inputs_dir=args.inputs,
        sentence_arrow=args.sentence_arrow,
        neuron_ids=args.neuron_ids,
        device=args.device,
        no_seed_layer8=bool(args.no_seed_layer8),
    )


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: CrossLayerRecurrenceOptions | None,
    stages: tuple[str, ...],
) -> Mapping[str, Path]:
    """Run index, chains, and match stages, or the opt-in full_sweep stage."""
    opts = options or CrossLayerRecurrenceOptions()
    artifacts: dict[str, Path] = {}
    locations = opts.locations or tuple(profile.artifacts.recurrence_layers)
    fresh_results_dir: Path | None = None

    def _resolve_results_dir() -> Path:
        if fresh_results_dir is not None:
            return fresh_results_dir
        if opts.results_dir is not None:
            return opts.results_dir
        if not profile.artifacts.recurrence_results_relpath:
            raise ValueError(
                f"Profile {profile.name!r} has no recurrence_results_relpath; pass --results-dir"
            )
        return workspace.data_root.resolve_baseline_path(
            profile.artifacts.recurrence_results_relpath
        )

    for stage in stages:
        with workspace.stage(stage):
            if stage == "index":
                from cross_layer_recurrence import index_subformulas as index_mod

                results_dir = _resolve_results_dir()
                formulas = index_mod.load_formulas(results_dir, locations=locations)
                occurrences, proper_locations = index_mod.index_subformulas(formulas)
                recurrence, summary = index_mod.summarize(
                    occurrences, proper_locations, locations=locations
                )
                counts_dir = workspace.output_dir / "subformula-counts"
                counts_dir.mkdir(parents=True, exist_ok=True)
                index_mod.write_index(workspace.output_dir / "recurrence.json.gz", recurrence)
                index_mod.write_index(
                    counts_dir / "all-compound-occurrences.json.gz",
                    dict(sorted(occurrences.items())),
                )
                (counts_dir / "counts.json").write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
                artifacts["index"] = workspace.output_dir / "recurrence.json.gz"
                artifacts["all_compound_occurrences"] = counts_dir / "all-compound-occurrences.json.gz"
                artifacts["counts"] = counts_dir / "counts.json"
            elif stage == "chains":
                from cross_layer_recurrence import preliminary_chains as chains_mod

                index_path = workspace.output_dir / "subformula-counts" / "all-compound-occurrences.json.gz"
                if not index_path.is_file():
                    raise FileNotFoundError(
                        f"Missing subformula index; run stage 'index' first: {index_path}"
                    )
                chains_mod.find_preliminary_chains(
                    index_path,
                    workspace.output_dir / "preliminary-chains.json",
                    locations,
                )
                artifacts["preliminary_chains"] = workspace.output_dir / "preliminary-chains.json"
            elif stage == "match":
                from cross_layer_recurrence import matching as matching_mod

                bundle = workspace.data_root.resolve_prepared_inputs(
                    profile, override=opts.inputs_dir
                )
                results_dir = _resolve_results_dir()
                argv: list[str] = [
                    "--csv-dir", str(results_dir / "full"),
                    "--activation-dir", str(bundle.directory),
                    "--features", str(bundle.features_path),
                    "--layers", *locations,
                    "--output", str(workspace.output_dir / "recurrence.json"),
                ]
                if profile.artifacts.recurrence_later_start is None:
                    raise ValueError("Profile has no recurrence_later_start")
                argv += ["--later-start", profile.artifacts.recurrence_later_start]
                if profile.artifacts.recurrence_later_end:
                    argv += ["--later-end", profile.artifacts.recurrence_later_end]
                matching_mod.main(argv)
                artifacts["match_json"] = workspace.output_dir / "recurrence.json"
            elif stage == "full_sweep":
                from cross_layer_recurrence import full_sweep

                full_sweep.run_full_sweep(
                    workspace.data_root,
                    profile,
                    run_root=workspace.output_dir,
                    inputs_dir=opts.inputs_dir,
                    reference_dir=opts.results_dir,
                    layers=locations,
                    sentence_path=opts.sentence_arrow,
                    neuron_ids=opts.neuron_ids,
                    device=opts.device,
                    seed_layer8=not opts.no_seed_layer8,
                )
                fresh_results_dir = workspace.output_dir
                artifacts["import_json"] = workspace.output_dir / "import.json"
                artifacts["selection_json"] = workspace.output_dir / "selection.json"
                artifacts["full_dir"] = workspace.output_dir / "full"
            else:
                raise ValueError(f"Unknown cross_layer_recurrence stage {stage!r}")
    return artifacts
