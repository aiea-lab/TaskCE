"""Fit compositional explanations on SNLI fit rows, then score frozen formulas.

Inputs: features.npz contains vectors[concept, example], names[concept], and
split[example] (0=fit, 1=validation, 2=test). Activation NPZ contains the raw
activation[example, original_neuron] matrix. No row is shuffled by this runner.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from contextlib import redirect_stdout
from pathlib import Path
import sys

import numpy as np

from formula_eval import FormulaEvaluator


METRICS = ("iou", "n", "support", "predictions", "intersection")
FIELDS = (
    "layer", "neuron", "neuron_ids", "target_type", "status", "formula",
    "backend_fit_iou", "thresholds",
    *(f"{split}_{metric}" for split in ("fit", "validation", "test") for metric in METRICS),
    "beam", "length", "min_fit_support",
)


def score(target, prediction, split):
    result = {}
    for index, name in enumerate(("fit", "validation", "test")):
        mask = split == index
        active = target[mask]
        predicted = prediction[mask]
        intersection = int(np.count_nonzero(active & predicted))
        support = int(np.count_nonzero(active))
        predictions = int(np.count_nonzero(predicted))
        union = support + predictions - intersection
        result.update({
            f"{name}_iou": intersection / union if union else 0.0,
            f"{name}_n": int(mask.sum()),
            f"{name}_support": support,
            f"{name}_predictions": predictions,
            f"{name}_intersection": intersection,
        })
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def persist(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def arguments():
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
    parser.add_argument("--quantile", type=float, default=0.945)
    parser.add_argument("--min-fit-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-chunks", type=int, help="Checkpoint after this many chunks, for a bounded first run")
    parser.add_argument("--neuron-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all-examples", action="store_true",
                        help="Threshold, search, and score on every example in the supplied dataset")
    parser.add_argument("--prefix-selection", action="store_true", help="Use exact rank-prefix bitmap deduplication")
    args = parser.parse_args()
    for key in ("beam", "length", "chunk_size", "neuron_batch_size"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.max_chunks is not None and args.max_chunks < 1:
        parser.error("--max-chunks must be positive")
    if not 0 < args.quantile < 1 or not 0 <= args.min_fit_fraction <= 1:
        parser.error("Invalid quantile or minimum fit fraction")
    return args


def main():
    args = arguments()
    import torch
    from sympy import Symbol

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "theoretical" / "compositional_explanations"))
    pinned = os.environ.get("TASKCE_BACKEND") == "vectorized_search_2a"
    if pinned:
        from verify_2a import verify
        proof = verify()
        if args.prefix_selection:
            raise ValueError("2.a must use its native beam selector")
        from vectorized_search.algorithm import SearchConfig, search_all
    else:
        from new_search import algorithm, kernels
        from new_search.algorithm import SearchConfig, search_all
    if args.prefix_selection:
        from prefix_selection import select_unique_topk
        algorithm.select_unique_topk = select_unique_topk

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    with np.load(args.features, allow_pickle=False) as data:
        vectors = data["vectors"]
        names = data["names"].astype(str)
        split = data["split"]
    with np.load(args.activation, allow_pickle=False) as data:
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
    if args.all_examples:
        split = np.zeros_like(split)
    elif not all(np.any(split == index) for index in (0, 1, 2)):
        raise ValueError("Fit, validation, and test must each contain examples")
    if not np.isfinite(activation).all():
        raise ValueError("Activation matrix contains nonfinite values")
    if args.beam > len(names):
        raise ValueError("Beam size cannot exceed feature count in this backend")
    if args.pair:
        ids = [int(value) for value in args.pair.split(",")]
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError("--pair requires two distinct neuron IDs")
        requested = [tuple(ids)]
    else:
        ids = list(range(activation.shape[1])) if args.ids == "all" else [int(value) for value in args.ids.split(",")]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("--ids must contain distinct neuron IDs")
        requested = [(value,) for value in ids]
    if any(value < 0 or value >= activation.shape[1] for value in ids):
        raise ValueError("Original neuron ID is outside activation matrix")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    config = {
        "schema": 1, "layer": args.layer, "targets": requested,
        "features_sha256": digest(args.features), "activation_sha256": digest(args.activation),
        "runner_sha256": digest(Path(__file__)),
        "evaluator_sha256": digest(Path(__file__).with_name("formula_eval.py")),
        "backend_sha256": ({k: v["sha256"] for k, v in proof["modules"].items()} if pinned else
                           {module.__name__: digest(Path(module.__file__)) for module in (algorithm, kernels)}),
        "prefix_selector_sha256": digest(Path(__file__).with_name("prefix_selection.py")) if args.prefix_selection else None,
        "beam": args.beam, "length": args.length, "seed": args.seed,
        "quantile": args.quantile, "eligibility_fraction": args.min_fit_fraction,
        "neuron_batch_size": args.neuron_batch_size, "device": args.device,
    }
    if pinned:
        config.update(backend_commit=proof["commit"], max_iterations=48)
    if args.all_examples:
        config.update(example_scope="all", example_count=len(split))
    config = json.loads(json.dumps(config))
    rows = []
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["configuration"] != config:
            raise ValueError("Resume configuration or input contents changed")
        if args.output.exists():
            with args.output.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
    elif args.output.exists() or manifest_path.exists():
        raise FileExistsError("Output already exists; use --resume or another path")
    completed = {row["neuron"] for row in rows}
    expected = {"&".join(map(str, values)) for values in requested}
    if not completed <= expected or len(completed) != len(rows):
        raise ValueError("Saved results do not match requested targets")
    # Establish the resume contract before the first expensive GPU chunk.
    atomic_json(manifest_path, {"configuration": config, "coverage": {
        "requested": len(requested), "completed": len(rows),
        "eligible": sum(row["status"] != "ineligible" for row in rows),
        "explained": sum(row["status"] == "ok" for row in rows),
    }})
    fit = split == 0
    minimum = max(1, math.ceil(int(fit.sum()) * args.min_fit_fraction))
    unique_ids = sorted(set(ids))
    # torch.quantile matches the repository's threshold implementation exactly.
    raw = torch.as_tensor(activation[:, unique_ids])
    if not raw.is_floating_point():
        raw = raw.float()
    thresholds = torch.quantile(raw[torch.from_numpy(fit)], args.quantile, dim=0).numpy()
    binary = raw.numpy() > thresholds
    by_id = {value: column for column, value in enumerate(unique_ids)}
    evaluator = FormulaEvaluator(names, vectors)
    fit_features = [(Symbol(name), vector[fit]) for name, vector in zip(names, vectors)]
    search_config = SearchConfig(maximum_formula_length=args.length, beam_size=args.beam,
                                 **({"max_iterations": 48} if pinned else {"neuron_batch_size": args.neuron_batch_size}))
    pending = [values for values in requested if "&".join(map(str, values)) not in completed]
    if args.max_chunks is not None:
        pending = pending[:args.max_chunks * args.chunk_size]
    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start:start + args.chunk_size]
        targets = [np.all(binary[:, [by_id[value] for value in values]], axis=1) for values in chunk]
        searchable = [index for index, target in enumerate(targets)
                      if args.pair or int(target[fit].sum()) >= minimum]
        results = {}
        if searchable:
            fit_targets = np.column_stack([targets[index][fit] for index in searchable])
            with open(os.devnull, "w") as quiet, redirect_stdout(quiet if pinned else sys.stdout):
                output = search_all(torch.from_numpy(fit_targets), fit_features,
                                    device=args.device, config=search_config)
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
            row = {
                "layer": args.layer, "neuron": "&".join(map(str, values)),
                "neuron_ids": json.dumps(values), "target_type": "pair" if args.pair else "neuron",
                "status": "ok" if valid else ("no_formula" if result else "ineligible"),
                "formula": formula, "backend_fit_iou": result.best_score if result else "",
                "thresholds": json.dumps([float(thresholds[by_id[value]]) for value in values]),
                **metrics, "beam": args.beam, "length": args.length, "min_fit_support": minimum,
            }
            if not valid:
                for name in ("fit", "validation", "test"):
                    for metric in ("iou", "predictions", "intersection"):
                        row[f"{name}_{metric}"] = ""
            rows.append(row)
        persist(args.output, rows)
        coverage = {"requested": len(requested), "completed": len(rows),
                    "eligible": sum(row["status"] != "ineligible" for row in rows),
                    "explained": sum(row["status"] == "ok" for row in rows)}
        atomic_json(manifest_path, {"configuration": config, "coverage": coverage})
        print(json.dumps({"layer": args.layer, **coverage}), flush=True)


if __name__ == "__main__":
    main()
