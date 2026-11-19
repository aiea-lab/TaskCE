"""Executor runner for the `formula_search` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taskce.profiles import Profile
from taskce.runtime import RunWorkspace, SearchBackendName


@dataclass(frozen=True)
class FormulaSearchOptions:
    layer: str = "layer.8"
    inputs_dir: Path | None = None
    capture_root: Path | None = None
    capture_map: Path | None = None
    dataset_path: Path | None = None
    row_hashes: Path | None = None
    assume_capture_order: bool = False
    frozen_features: Path | None = None
    neuron_ids: str = "all"
    pair: tuple[int, int] | None = None
    backend: SearchBackendName | None = None
    beam: int | None = None
    length: int | None = None
    quantile: float | None = None
    min_fit_fraction: float = 0.05
    all_examples: bool = False
    prefix_selection: bool = False
    chunk_size: int = 32
    max_chunks: int | None = None
    neuron_batch_size: int = 1
    device: str = "cuda"
    formulas_csv: Path | None = None
    activation_path: Path | None = None
    checkpoint_name: str | None = None
    alpha: float | None = None
    min_acts: int | None = None


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layer",
        default="layer.8",
        help="Prepared layer name to search (default: %(default)s).",
    )
    parser.add_argument(
        "--inputs-dir",
        type=Path,
        default=None,
        help="Directory containing prepared features.npz and <layer>.npz.",
    )
    parser.add_argument(
        "--capture-root",
        type=Path,
        default=None,
        help="Capture checkpoint directory for the prepare stage.",
    )
    parser.add_argument(
        "--capture-map",
        type=Path,
        default=None,
        help="JSON mapping output names to .pt paths relative to --capture-root.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional local datasets.save_to_disk path for the prepare stage.",
    )
    parser.add_argument(
        "--row-hashes",
        type=Path,
        default=None,
        help="Ordered capture row-hash JSON manifest for the prepare stage.",
    )
    parser.add_argument(
        "--assume-capture-order",
        action="store_true",
        help="Attest captures follow dataset order when --row-hashes is absent.",
    )
    parser.add_argument(
        "--frozen-features",
        type=Path,
        default=None,
        help="Historical inputs.pt to verify against the layer.8 capture.",
    )
    parser.add_argument(
        "--ids",
        dest="neuron_ids",
        default="all",
        help="Comma-separated neuron IDs or 'all' (default: all).",
    )
    parser.add_argument(
        "--pair",
        default=None,
        help="Two comma-separated neuron IDs whose conjunction is searched.",
    )
    parser.add_argument(
        "--backend",
        choices=("vectorized_search_2a", "new_search", "search"),
        default=None,
        help="Override search backend from profile.",
    )
    parser.add_argument("--beam", type=int, default=None)
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--quantile", type=float, default=None)
    parser.add_argument("--min-fit-fraction", type=float, default=0.05)
    parser.add_argument("--all-examples", action="store_true")
    parser.add_argument("--prefix-selection", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--neuron-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--formulas-csv",
        type=Path,
        default=None,
        help="Existing formula CSV input for the opt-in attach_weights stage.",
    )
    parser.add_argument(
        "--activation-path",
        type=Path,
        default=None,
        help="Explicit raw .pt capture path for the opt-in general stage.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="LoRA checkpoint directory name for the opt-in general stage.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override top-tail activation fraction for the opt-in general stage.",
    )
    parser.add_argument(
        "--min-acts",
        type=int,
        default=None,
        help="Override minimum active examples for pruning in the opt-in general stage.",
    )


def options_from_args(args: argparse.Namespace) -> FormulaSearchOptions:
    pair: tuple[int, int] | None = None
    if args.pair:
        parts = [int(item) for item in args.pair.split(",")]
        if len(parts) != 2 or parts[0] == parts[1]:
            raise ValueError("--pair requires two distinct comma-separated neuron IDs")
        pair = (parts[0], parts[1])
    return FormulaSearchOptions(
        layer=args.layer,
        inputs_dir=args.inputs_dir,
        capture_root=args.capture_root,
        capture_map=args.capture_map,
        dataset_path=args.dataset_path,
        row_hashes=args.row_hashes,
        assume_capture_order=bool(args.assume_capture_order),
        frozen_features=args.frozen_features,
        neuron_ids=args.neuron_ids,
        pair=pair,
        backend=args.backend,
        beam=args.beam,
        length=args.length,
        quantile=args.quantile,
        min_fit_fraction=args.min_fit_fraction,
        all_examples=bool(args.all_examples),
        prefix_selection=bool(args.prefix_selection),
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        neuron_batch_size=args.neuron_batch_size,
        device=args.device,
        formulas_csv=args.formulas_csv,
        activation_path=args.activation_path,
        checkpoint_name=args.checkpoint_name,
        alpha=args.alpha,
        min_acts=args.min_acts,
    )


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: FormulaSearchOptions | None,
    stages: tuple[str, ...],
) -> Mapping[str, Path]:
    """Execute selected `formula_search` stages (`prepare`, `search`, `attach_weights`)."""
    opts = options or FormulaSearchOptions()
    artifacts: dict[str, Path] = {}
    prepared_dir: Path | None = opts.inputs_dir
    search_csv_path: Path | None = opts.formulas_csv

    if "prepare" in stages:
        if opts.capture_root is None:
            raise ValueError("Stage 'prepare' requires --capture-root")
        from formula_search.prepare import prepare_inputs

        target_inputs = workspace.output_dir / "inputs"
        metrics: dict[str, Any] = {}
        with workspace.stage("prepare", metrics_out=metrics):
            provenance = prepare_inputs(
                profile=profile,
                capture_root=opts.capture_root,
                output_dir=target_inputs,
                data_root=workspace.data_root,
                capture_map=opts.capture_map,
                frozen_features=opts.frozen_features,
                dataset_path=opts.dataset_path,
                row_hashes_path=opts.row_hashes,
                assume_capture_order=opts.assume_capture_order,
            )
            metrics["examples"] = provenance["examples"]
            metrics["layers"] = sorted(provenance["captures"])
        prepared_dir = target_inputs
        artifacts["prepared_inputs"] = target_inputs
        artifacts["prepare_provenance"] = target_inputs / "provenance.json"

    if "search" in stages:
        from formula_search.sweep import run_sweep

        bundle = workspace.data_root.resolve_prepared_inputs(
            profile, override=prepared_dir
        )
        if opts.layer not in bundle.layer_paths:
            available = ", ".join(sorted(bundle.layer_paths))
            raise FileNotFoundError(
                f"Layer {opts.layer!r} not found in {bundle.directory}; "
                f"available layers: {available}"
            )
        output_csv = workspace.output_dir / f"{opts.layer}.csv"
        backend = opts.backend or profile.search.backend
        beam = opts.beam if opts.beam is not None else profile.search.beam
        length = opts.length if opts.length is not None else profile.search.length
        quantile = opts.quantile if opts.quantile is not None else profile.search.quantile
        metrics = {"layer": opts.layer, "backend": backend, "beam": beam, "length": length}
        with workspace.stage("search", metrics_out=metrics):
            run_sweep(
                features_path=bundle.features_path,
                activation_path=bundle.layer_paths[opts.layer],
                layer=opts.layer,
                output_path=output_csv,
                beam=beam,
                length=length,
                neuron_ids=opts.neuron_ids,
                pair=opts.pair,
                backend=backend,
                quantile=quantile,
                min_fit_fraction=opts.min_fit_fraction,
                seed=profile.search.seed,
                max_iterations=profile.search.max_iterations,
                chunk_size=opts.chunk_size,
                max_chunks=opts.max_chunks,
                neuron_batch_size=opts.neuron_batch_size,
                device=opts.device,
                resume=workspace.resume,
                all_examples=opts.all_examples,
                prefix_selection=opts.prefix_selection,
            )
        search_csv_path = output_csv
        artifacts["formulas_csv"] = output_csv
        artifacts["formulas_manifest"] = output_csv.with_suffix(
            output_csv.suffix + ".manifest.json"
        )

    if "attach_weights" in stages:
        if search_csv_path is None:
            raise ValueError(
                "Stage 'attach_weights' requires either stage 'search' or --formulas-csv"
            )
        from formula_search.analysis import attach_classification_weights

        weighted_csv = workspace.output_dir / f"{search_csv_path.stem}_with_weights.csv"
        with workspace.stage("attach_weights"):
            attach_classification_weights(profile, search_csv_path, weighted_csv)
        artifacts["weighted_formulas_csv"] = weighted_csv

    if "general" in stages:
        from formula_search import analysis as general_analysis

        task_name = profile.model.lora_task or profile.dataset.name
        checkpoint_name = opts.checkpoint_name or profile.model.checkpoint_name
        module_layer = str(profile.model.resolve_module_path(opts.layer))
        act_path = opts.activation_path or workspace.data_root.activation_path(
            profile.model.model_id,
            profile.dataset.name,
            module_layer,
            task_name=task_name,
            checkpoint_name=checkpoint_name,
        )
        general_backend = opts.backend or "search"
        search_fn, search_kwargs = _resolve_general_backend(
            general_backend,
            beam=opts.beam,
            length=opts.length,
            neuron_batch_size=opts.neuron_batch_size,
            device=opts.device,
        )
        general_csv = workspace.output_dir / f"{task_name}_beam_results.csv"
        metrics = {
            "layer": module_layer,
            "backend": general_backend,
            "alpha": opts.alpha if opts.alpha is not None else profile.search.alpha,
            "min_acts": opts.min_acts if opts.min_acts is not None else profile.search.min_acts,
        }
        with workspace.stage("general", metrics_out=metrics):
            general_analysis.run(
                task_name,
                act_path,
                search_fn=search_fn,
                output_path=general_csv,
                model_id=profile.model.model_id,
                lora_repository=profile.model.lora_repo
                or general_analysis.DEFAULT_LORA_REPOSITORY,
                alpha=opts.alpha if opts.alpha is not None else profile.search.alpha,
                min_acts=opts.min_acts if opts.min_acts is not None else profile.search.min_acts,
                checkpoint_name=checkpoint_name,
                dataset_path=opts.dataset_path,
                data_root=workspace.data_root,
                **search_kwargs,
            )
        artifacts["general_formulas_csv"] = general_csv

    return artifacts


def _resolve_general_backend(
    backend: str,
    *,
    beam: int | None,
    length: int | None,
    neuron_batch_size: int,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    search_kwargs: dict[str, Any] = {"device": device}
    if backend == "search":
        from formula_search.search.algorithm import SearchConfig, search_all

        cfg_kwargs: dict[str, Any] = {}
        if beam is not None:
            cfg_kwargs["beam_size"] = beam
        if length is not None:
            cfg_kwargs["maximum_formula_length"] = length
        if cfg_kwargs:
            search_kwargs["config"] = SearchConfig(**cfg_kwargs)
        return search_all, search_kwargs
    if backend == "new_search":
        from formula_search.new_search.algorithm import SearchConfig, search_all

        cfg_kwargs = {}
        if beam is not None:
            cfg_kwargs["beam_size"] = beam
        if length is not None:
            cfg_kwargs["maximum_formula_length"] = length
        if neuron_batch_size != 1:
            cfg_kwargs["neuron_batch_size"] = neuron_batch_size
        if cfg_kwargs:
            search_kwargs["config"] = SearchConfig(**cfg_kwargs)
        return search_all, search_kwargs
    if backend == "vectorized_search_2a":
        from formula_search.verify_2a import load_verified_2a

        verified = load_verified_2a(require_runtime=True)
        assert verified.search_all is not None and verified.search_config is not None
        cfg_kwargs = {}
        if beam is not None:
            cfg_kwargs["beam_size"] = beam
        if length is not None:
            cfg_kwargs["maximum_formula_length"] = length
        if cfg_kwargs:
            search_kwargs["config"] = verified.search_config(**cfg_kwargs)
        return verified.search_all, search_kwargs
    raise ValueError(f"Unsupported general formula_search backend: {backend!r}")
