"""Fit compositional explanations over prepared NPZ inputs and score frozen formulas.

Inputs: features.npz contains vectors[concept, example], names[concept], and
split[example] (0=fit, 1=validation, 2=test). Activation NPZ contains the raw
activation[example, original_neuron] matrix. No row is shuffled by this runner.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
import io
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from formula_search.formula_eval import FormulaEvaluator
from taskce.runtime import (
    SearchBackendName,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


METRICS = ("iou", "n", "support", "predictions", "intersection")
FIELDS = (
    "layer",
    "neuron",
    "neuron_ids",
    "target_type",
    "status",
    "formula",
    "backend_fit_iou",
    "thresholds",
    *(f"{split}_{metric}" for split in ("fit", "validation", "test") for metric in METRICS),
    "beam",
    "length",
    "min_fit_support",
)


def score(target: np.ndarray, prediction: np.ndarray, split: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, name in enumerate(("fit", "validation", "test")):
        mask = split == index
        active = target[mask]
        predicted = prediction[mask]
        intersection = int(np.count_nonzero(active & predicted))
        support = int(np.count_nonzero(active))
        predictions = int(np.count_nonzero(predicted))
        union = support + predictions - intersection
        result.update(
            {
                f"{name}_iou": intersection / union if union else 0.0,
                f"{name}_n": int(mask.sum()),
                f"{name}_support": support,
                f"{name}_predictions": predictions,
                f"{name}_intersection": intersection,
            }
        )
    return result


def persist_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def run_sweep(
    *,
    features_path: Path,
    activation_path: Path,
    layer: str,
    output_path: Path,
    beam: int,
    length: int,
    neuron_ids: str = "all",
    pair: tuple[int, int] | None = None,
    backend: SearchBackendName | None = None,
    quantile: float = 0.945,
    min_fit_fraction: float = 0.05,
    seed: int = 20261111,
    max_iterations: int = 48,
    chunk_size: int = 32,
    max_chunks: int | None = None,
    neuron_batch_size: int = 1,
    device: str = "cuda",
    resume: bool = False,
    all_examples: bool = False,
    prefix_selection: bool = False,
) -> Path:
    """Run resumable chunked compositional formula search over prepared NPZ inputs."""
    for key, val in (
        ("beam", beam),
        ("length", length),
        ("chunk_size", chunk_size),
        ("neuron_batch_size", neuron_batch_size),
    ):
        if val < 1:
            raise ValueError(f"{key} must be positive, got {val}")
    if max_chunks is not None and max_chunks < 1:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")
    if not (0.0 < quantile < 1.0) or not (0.0 <= min_fit_fraction <= 1.0):
        raise ValueError("Invalid quantile or minimum fit fraction")

    import torch
    from sympy import Symbol

    chosen_backend = backend or os.environ.get("TASKCE_BACKEND") or "new_search"
    pinned = chosen_backend == "vectorized_search_2a"

    if pinned:
        from formula_search.verify_2a import load_verified_2a

        verified = load_verified_2a(require_runtime=True)
        if prefix_selection:
            raise ValueError("2.a must use its native beam selector")
        assert verified.search_config is not None and verified.search_all is not None
        SearchConfig = verified.search_config
        search_all = verified.search_all
        backend_sha256: dict[str, str] = verified.proof.module_hashes()
        backend_commit: str | None = verified.proof.commit
        evaluator_sha256 = str(verified.evaluator_sha256)
    elif chosen_backend == "new_search":
        from formula_search.new_search import algorithm, kernels
        from formula_search.new_search.algorithm import SearchConfig, search_all

        if prefix_selection:
            from formula_search.prefix_selection import select_unique_topk

            algorithm.select_unique_topk = select_unique_topk
        backend_sha256 = {
            module.__name__: str(sha256_file(Path(module.__file__)))
            for module in (algorithm, kernels)
        }
        backend_commit = None
        evaluator_sha256 = str(sha256_file(Path(__file__).with_name("formula_eval.py")))
    elif chosen_backend == "search":
        from formula_search.search import algorithm
        from formula_search.search.algorithm import SearchConfig, search_all

        if prefix_selection:
            raise ValueError("prefix_selection is only supported with new_search")
        backend_sha256 = {algorithm.__name__: str(sha256_file(Path(algorithm.__file__)))}
        backend_commit = None
        evaluator_sha256 = str(sha256_file(Path(__file__).with_name("formula_eval.py")))
    else:
        raise ValueError(f"Unsupported search backend: {chosen_backend!r}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    with np.load(features_path, allow_pickle=False) as data:
        vectors = data["vectors"]
        names = data["names"].astype(str)
        split = data["split"]
    with np.load(activation_path, allow_pickle=False) as data:
        activation = data["activation"]

    if vectors.dtype != np.bool_ or vectors.ndim != 2:
        raise ValueError("Feature vectors must be a bool [concept, example] matrix")
    if names.shape != (vectors.shape[0],) or len(set(names)) != len(names):
        raise ValueError("Feature names must be unique and match concept count")
    if any(not name or name in ("AND", "OR", "NOT", "(", ")") for name in names):
        raise ValueError("Feature names must be nonempty and distinct from syntax")
    if activation.ndim != 2 or activation.shape[0] != vectors.shape[1]:
        raise ValueError("Activation and feature example dimensions differ")
    if split.shape != (vectors.shape[1],) or not np.isin(split, [0, 1, 2]).all():
        raise ValueError("split must have one 0/1/2 value per example")
    if all_examples:
        split = np.zeros_like(split)
    elif not all(np.any(split == index) for index in (0, 1, 2)):
        raise ValueError("Fit, validation, and test must each contain examples")
    if not np.isfinite(activation).all():
        raise ValueError("Activation matrix contains nonfinite values")
    if beam > len(names):
        raise ValueError("Beam size cannot exceed feature count in this backend")

    if pair is not None:
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError("pair requires two distinct neuron IDs")
        ids = [int(pair[0]), int(pair[1])]
        requested: list[tuple[int, ...]] = [(ids[0], ids[1])]
    else:
        ids = (
            list(range(activation.shape[1]))
            if neuron_ids == "all"
            else [int(value) for value in neuron_ids.split(",")]
        )
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("neuron_ids must contain distinct neuron IDs")
        requested = [(value,) for value in ids]
    if any(value < 0 or value >= activation.shape[1] for value in ids):
        raise ValueError("Original neuron ID is outside activation matrix")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    config: dict[str, Any] = {
        "schema": 1,
        "layer": layer,
        "targets": requested,
        "features_sha256": str(sha256_file(features_path)),
        "activation_sha256": str(sha256_file(activation_path)),
        "runner_sha256": str(sha256_file(Path(__file__))),
        "evaluator_sha256": evaluator_sha256,
        "backend_sha256": backend_sha256,
        "prefix_selector_sha256": (
            str(sha256_file(Path(__file__).with_name("prefix_selection.py")))
            if prefix_selection
            else None
        ),
        "beam": beam,
        "length": length,
        "seed": seed,
        "quantile": quantile,
        "eligibility_fraction": min_fit_fraction,
        "neuron_batch_size": neuron_batch_size,
        "device": device,
    }
    if pinned:
        config.update(backend_commit=backend_commit, max_iterations=max_iterations)
    if all_examples:
        config.update(example_scope="all", example_count=len(split))
    config = json.loads(json.dumps(config))

    rows: list[dict[str, Any]] = []
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["configuration"] != config:
            raise ValueError("Resume configuration or input contents changed")
        if output_path.exists():
            with output_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
    elif output_path.exists() or manifest_path.exists():
        raise FileExistsError("Output already exists; use --resume or another path")

    completed = {row["neuron"] for row in rows}
    expected = {"&".join(map(str, values)) for values in requested}
    if not completed <= expected or len(completed) != len(rows):
        raise ValueError("Saved results do not match requested targets")

    atomic_write_json(
        manifest_path,
        {
            "configuration": config,
            "coverage": {
                "requested": len(requested),
                "completed": len(rows),
                "eligible": sum(row["status"] != "ineligible" for row in rows),
                "explained": sum(row["status"] == "ok" for row in rows),
            },
        },
    )

    fit = split == 0
    minimum = max(1, math.ceil(int(fit.sum()) * min_fit_fraction))
    unique_ids = sorted(set(ids))
    raw = torch.as_tensor(activation[:, unique_ids])
    if not raw.is_floating_point():
        raw = raw.float()
    thresholds = torch.quantile(raw[torch.from_numpy(fit)], quantile, dim=0).numpy()
    binary = raw.numpy() > thresholds
    by_id = {value: column for column, value in enumerate(unique_ids)}

    evaluator = FormulaEvaluator(names, vectors)
    fit_features = [(Symbol(name), vector[fit]) for name, vector in zip(names, vectors)]
    if pinned:
        search_config = SearchConfig(
            maximum_formula_length=length,
            beam_size=beam,
            max_iterations=max_iterations,
        )
    elif chosen_backend == "new_search":
        search_config = SearchConfig(
            maximum_formula_length=length,
            beam_size=beam,
            neuron_batch_size=neuron_batch_size,
        )
    else:
        search_config = SearchConfig(
            maximum_formula_length=length,
            beam_size=beam,
        )

    pending = [
        values for values in requested if "&".join(map(str, values)) not in completed
    ]
    if max_chunks is not None:
        pending = pending[: max_chunks * chunk_size]

    for start in range(0, len(pending), chunk_size):
        chunk = pending[start : start + chunk_size]
        targets = [
            np.all(binary[:, [by_id[value] for value in values]], axis=1)
            for values in chunk
        ]
        searchable = [
            index
            for index, target in enumerate(targets)
            if pair is not None or int(target[fit].sum()) >= minimum
        ]
        results = {}
        if searchable:
            fit_targets = np.column_stack([targets[index][fit] for index in searchable])
            with open(os.devnull, "w", encoding="utf-8") as quiet, redirect_stdout(
                quiet if pinned else sys.stdout
            ):
                output = search_all(
                    torch.from_numpy(fit_targets),
                    fit_features,
                    device=device,
                    config=search_config,
                )
            if sorted(result.activation_index for result in output) != list(range(len(searchable))):
                raise RuntimeError("Backend returned missing or duplicate targets")
            results = {searchable[result.activation_index]: result for result in output}

        for index, (values, target) in enumerate(zip(chunk, targets)):
            result = results.get(index)
            valid = result is not None and result.best_formula not in ("", "LOW_ACTS_PRUNED")
            formula = result.best_formula if valid else ""
            prediction = evaluator.evaluate(formula) if valid else np.zeros_like(target)
            metrics = score(target, prediction, split)
            if valid and not math.isclose(metrics["fit_iou"], result.best_score, abs_tol=2e-6):
                raise RuntimeError(f"Backend/evaluator IoU disagreement: {values}")
            row: dict[str, Any] = {
                "layer": layer,
                "neuron": "&".join(map(str, values)),
                "neuron_ids": json.dumps(values),
                "target_type": "pair" if pair is not None else "neuron",
                "status": "ok" if valid else ("no_formula" if result else "ineligible"),
                "formula": formula,
                "backend_fit_iou": result.best_score if result else "",
                "thresholds": json.dumps([float(thresholds[by_id[value]]) for value in values]),
                **metrics,
                "beam": beam,
                "length": length,
                "min_fit_support": minimum,
            }
            if not valid:
                for split_name in ("fit", "validation", "test"):
                    for metric in ("iou", "predictions", "intersection"):
                        row[f"{split_name}_{metric}"] = ""
            rows.append(row)

        persist_rows(output_path, rows)
        coverage = {
            "requested": len(requested),
            "completed": len(rows),
            "eligible": sum(row["status"] != "ineligible" for row in rows),
            "explained": sum(row["status"] == "ok" for row in rows),
        }
        atomic_write_json(manifest_path, {"configuration": config, "coverage": coverage})
        print(json.dumps({"layer": layer, **coverage}), flush=True)

    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--ids", help="Comma separated original neuron IDs, or all")
    targets.add_argument("--pair", help="Two original IDs; target is their conjunction")
    parser.add_argument("--beam", type=int, required=True)
    parser.add_argument("--length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--quantile", type=float, default=0.945)
    parser.add_argument("--min-fit-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--neuron-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all-examples", action="store_true")
    parser.add_argument("--prefix-selection", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    pair = None
    if args.pair:
        parts = [int(value) for value in args.pair.split(",")]
        if len(parts) != 2:
            raise ValueError("--pair requires two comma-separated neuron IDs")
        pair = (parts[0], parts[1])
    run_sweep(
        features_path=args.features,
        activation_path=args.activation,
        layer=args.layer,
        output_path=args.output,
        beam=args.beam,
        length=args.length,
        neuron_ids=args.ids or "all",
        pair=pair,
        backend=args.backend,
        quantile=args.quantile,
        min_fit_fraction=args.min_fit_fraction,
        seed=args.seed,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        neuron_batch_size=args.neuron_batch_size,
        device=args.device,
        resume=args.resume,
        all_examples=args.all_examples,
        prefix_selection=args.prefix_selection,
    )


if __name__ == "__main__":
    main()
