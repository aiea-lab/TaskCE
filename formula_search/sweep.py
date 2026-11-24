"""Fit concept explanations over prepared NPZ inputs and score frozen predictors.

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

from taskce.runtime import (
    SearchBackendName,
    atomic_write_bytes,
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
FORMULA_NEURAL_FIELDS = (
    *FIELDS, "neural_concept_count", "neural_unique_concepts", "neural_fit_iou_at_48",
    "neural_trace_path", "neural_trace_sha256", "neural_table_path", "neural_table_sha256",
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


def persist_rows(
    path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...] = FIELDS
) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _table_path(output_path: Path, neuron: str) -> Path:
    return Path(f"{output_path.stem}.table") / f"{neuron}.npz"


def _trace_path(output_path: Path, neuron: str) -> Path:
    return Path(f"{output_path.stem}.trace") / f"{neuron}.json"



NEURAL_TABLE_COLUMNS = ("keep_logit", "order", "or_logit", "and_logit", "and_not_logit")
SELF_TABLE_COLUMNS = ("keep_logit", "or_logit", "and_logit", "and_not_logit")


def _decode_self_table(saved, table, mode):
    ids, positions, queries, keys = (saved[key] for key in ("ids", "positions", "queries", "keys"))
    expected_ids = np.r_[np.flatnonzero(table[:, 0] >= 0), len(table)]
    count = len(expected_ids)
    if (
        not np.array_equal(ids, expected_ids)
        or not np.issubdtype(positions.dtype, np.integer)
        or positions.shape != (count,) or positions[-1] != len(table)
        or len(np.unique(positions)) != count or np.any(positions < 0)
        or np.any(positions[:-1] >= len(table))
        or queries.dtype != np.float32 or keys.dtype != np.float32
        or queries.ndim != 2 or queries.shape != keys.shape
        or queries.shape[0] != count or queries.shape[1] < 1
        or not np.isfinite(queries).all() or not np.isfinite(keys).all()
    ):
        raise ValueError("Invalid neural self-attention factors")
    remaining = list(range(count - 1))
    current = count - 1
    indices = []
    while remaining:
        if mode == "causal":
            remaining = [i for i in remaining if positions[i] < positions[current]]
            if not remaining:
                break
        scores = keys[remaining] @ queries[current]
        current = remaining.pop(int(np.argmax(scores)))
        indices.append(int(ids[current]))
    return indices



def validate_neural_table(saved, names: np.ndarray, trace: dict[str, Any], mode: str = "cross") -> None:
    """Check the persisted table schema and independently decode the saved trace."""
    required = {"schema", "names", "table", "columns", "keep_threshold"}
    if mode != "cross":
        required |= {"mask_mode", "ids", "positions", "queries", "keys"}
    if set(saved.files) != required:
        raise ValueError("Invalid neural table fields")
    table = saved["table"]
    columns = NEURAL_TABLE_COLUMNS if mode == "cross" else SELF_TABLE_COLUMNS
    if (
        mode not in ("cross", "causal", "bidirectional")
        or saved["schema"].shape != ()
        or int(saved["schema"]) != (1 if mode == "cross" else 2)
        or not np.array_equal(saved["names"], names)
        or tuple(saved["columns"].tolist()) != columns
        or saved["keep_threshold"].shape != ()
        or float(saved["keep_threshold"]) != 0.0
        or table.dtype != np.float32 or table.shape != (len(names), len(columns))
        or not np.isfinite(table).all()
    ):
        raise ValueError("Invalid neural table schema or contents")
    if mode == "cross":
        selected = np.flatnonzero(table[:, 0] >= 0.0)
        indices = selected[np.argsort(-table[selected, 1], kind="stable")].tolist()
    else:
        if saved["mask_mode"].shape != () or str(saved["mask_mode"]) != mode:
            raise ValueError("Invalid neural self-attention mask mode")
        indices = _decode_self_table(saved, table, mode)
        if trace.get("kept_count") != int(np.count_nonzero(table[:, 0] >= 0)):
            raise ValueError("Neural self-attention kept count disagrees")
    op_start = 2 if mode == "cross" else 1
    operations = (["ATOM"] + [
        ("OR", "AND", "AND NOT")[int(table[i, op_start:].argmax())] for i in indices[1:]
    ]) if indices else []
    from formula_search.neural_search.instructions import render_formula

    if (list(trace["indices"]) != indices or list(trace["operations"]) != operations
            or trace["names"] != [str(names[i]) for i in indices]
            or trace.get("formula") != render_formula(names, indices, operations)
            or len(trace["prefix_ious"]) != len(indices)):
        raise ValueError("Neural table and formula trace disagree")


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
    attention_mode: str = "cross",
    layout_seed: int | None = None,
    encoder_width: int = 96,
    readout_width: int = 32,
    chunk_size: int = 32,
    max_chunks: int | None = None,
    neuron_batch_size: int = 1,
    device: str = "cuda",
    resume: bool = False,
    all_examples: bool = False,
    prefix_selection: bool = False,
) -> Path:
    """Run resumable concept search over prepared NPZ inputs."""
    chosen_backend = backend or os.environ.get("TASKCE_BACKEND") or "new_search"
    pinned = chosen_backend == "vectorized_search_2a"
    neural = chosen_backend == "neural_search"
    if attention_mode not in ("cross", "causal", "bidirectional"):
        raise ValueError("Invalid attention mode")
    if attention_mode == "cross" and layout_seed is not None:
        raise ValueError("layout_seed applies only to self-attention modes")
    positive_options = {
        "chunk_size": chunk_size,
        "neuron_batch_size": neuron_batch_size,
    }
    if neural:
        if neuron_batch_size != 1:
            raise ValueError("Neural backends require neuron_batch_size=1")
        if prefix_selection:
            raise ValueError("prefix_selection is only supported with new_search")
        positive_options.update(max_iterations=max_iterations, encoder_width=encoder_width, readout_width=readout_width)
    else:
        positive_options.update(beam=beam, length=length)
    for key, val in positive_options.items():
        if val < 1:
            raise ValueError(f"{key} must be positive, got {val}")
    if max_chunks is not None and max_chunks < 1:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")
    if not (0.0 < quantile < 1.0) or not (0.0 <= min_fit_fraction <= 1.0):
        raise ValueError("Invalid quantile or minimum fit fraction")

    import torch

    if neural:
        from formula_search.neural_search import attention_formula as neural_algorithm
        from formula_search.neural_search import instructions as neural_instructions

        modules = [neural_algorithm, neural_instructions]
        if attention_mode != "cross":
            from formula_search.neural_search import self_attention

            modules.append(self_attention)
        backend_sha256 = {
            module.__name__: str(sha256_file(Path(module.__file__)))
            for module in modules
        }
        evaluator_sha256 = str(sha256_file(Path(__file__).with_name("formula_eval.py")))
    elif pinned:
        from formula_search.verify_2a import load_verified_2a

        verified = load_verified_2a(require_runtime=True)
        if prefix_selection:
            raise ValueError("2.a must use its native beam selector")
        assert verified.search_config is not None and verified.search_all is not None
        SearchConfig = verified.search_config
        search_all = verified.search_all
        backend_sha256 = verified.proof.module_hashes()
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
    if not vectors.shape[0]:
        raise ValueError("Feature vectors must contain at least one concept")
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
    if not neural and beam > len(names):
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
        "beam": None if neural else beam,
        "length": None if neural else length,
        "seed": seed,
        "quantile": quantile,
        "eligibility_fraction": min_fit_fraction,
        "neuron_batch_size": neuron_batch_size,
        "device": device,
    }
    if pinned:
        config.update(backend_commit=backend_commit, max_iterations=max_iterations)
    if neural:
        config.update(backend=chosen_backend, max_iterations=max_iterations,
                      encoder_width=encoder_width, readout_width=readout_width,
                      attention_mode=attention_mode, layout_seed=layout_seed)
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
        elif previous.get("coverage", {}).get("completed", 0):
            raise ValueError("Resume results CSV is missing for completed targets")
    elif output_path.exists() or manifest_path.exists():
        raise FileExistsError("Output already exists; use --resume or another path")

    completed = {row["neuron"] for row in rows}
    expected = {"&".join(map(str, values)) for values in requested}
    if not completed <= expected or len(completed) != len(rows):
        raise ValueError("Saved results do not match requested targets")
    if neural:
        for row in rows:
            if row["status"] not in ("ok", "no_formula"):
                continue
            relative = _trace_path(output_path, row["neuron"])
            trace = output_path.parent / relative
            if (
                row.get("neural_trace_path") != relative.as_posix()
                or not trace.is_file()
                or row.get("neural_trace_sha256") != sha256_file(trace)
            ):
                raise ValueError(f"Missing or changed neural trace: {trace}")
            relative = _table_path(output_path, row["neuron"])
            table_path = output_path.parent / relative
            if (
                row.get("neural_table_path") != relative.as_posix()
                or not table_path.is_file()
                or row.get("neural_table_sha256") != sha256_file(table_path)
            ):
                raise ValueError(f"Missing or changed neural table: {table_path}")
            trace_data = json.loads(trace.read_text(encoding="utf-8"))
            if trace_data.get("formula") != row["formula"]:
                raise ValueError("Neural formula and trace disagree")
            with np.load(table_path, allow_pickle=False) as saved:
                validate_neural_table(saved, names, trace_data, attention_mode)

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

    if neural:
        fit_vectors = vectors[:, fit]
    from formula_search.formula_eval import FormulaEvaluator

    evaluator = FormulaEvaluator(names, vectors)
    if not neural:
        from sympy import Symbol

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
        if searchable and not neural:
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
            neuron = "&".join(map(str, values))
            formula = ""
            backend_fit_iou: float | str = ""
            status = "ineligible"
            fitted = None
            if neural:
                if index in searchable:
                    common = dict(steps=max_iterations, seed=seed, device=device,
                                  encoder_width=encoder_width, readout_width=readout_width)
                    if attention_mode == "cross":
                        fitted = neural_algorithm.fit_formula(fit_vectors, target[fit], names, **common)
                    else:
                        fitted = self_attention.fit_self_formula(
                            fit_vectors, target[fit], names, mask_mode=attention_mode,
                            layout_seed=layout_seed, **common,
                        )
                    formula = fitted.formula
                    backend_fit_iou = fitted.iou
                    status = "ok" if formula else "no_formula"
                    if formula:
                        # Alternating operations create nested expressions in the
                        # existing evaluator; give its parser room for this result.
                        previous_limit = sys.getrecursionlimit()
                        needed_limit = max(previous_limit, 2 * len(fitted.indices) + 100)
                        try:
                            sys.setrecursionlimit(needed_limit)
                            prediction = evaluator.evaluate(formula)
                        finally:
                            sys.setrecursionlimit(previous_limit)
            else:
                result = results.get(index)
                if result is not None:
                    backend_fit_iou = result.best_score
                    status = "no_formula"
                    if result.best_formula not in ("", "LOW_ACTS_PRUNED"):
                        formula = result.best_formula
                        prediction = evaluator.evaluate(formula)
                        status = "ok"
            if status != "ok":
                prediction = np.zeros_like(target)
            metrics = score(target, prediction, split)
            scored_formula = neural and fitted is not None
            if (status == "ok" or scored_formula) and not math.isclose(
                metrics["fit_iou"], float(backend_fit_iou), abs_tol=2e-6
            ):
                raise RuntimeError(f"Backend/scored fit IoU disagreement: {values}")
            row: dict[str, Any] = {
                "layer": layer,
                "neuron": neuron,
                "neuron_ids": json.dumps(values),
                "target_type": "pair" if pair is not None else "neuron",
                "status": status,
                "formula": formula,
                "backend_fit_iou": backend_fit_iou,
                "thresholds": json.dumps([float(thresholds[by_id[value]]) for value in values]),
                **metrics,
                "beam": "" if neural else beam,
                "length": "" if neural else length,
                "min_fit_support": minimum,
            }
            if status != "ok" and not scored_formula:
                for split_name in ("fit", "validation", "test"):
                    for metric in ("iou", "predictions", "intersection"):
                        row[f"{split_name}_{metric}"] = ""
            if neural:
                row.update(
                    neural_concept_count=len(fitted.indices) if fitted is not None else "",
                    neural_unique_concepts=len(set(fitted.indices)) if fitted is not None else "",
                    neural_fit_iou_at_48=(fitted.prefix_ious[min(48, len(fitted.prefix_ious)) - 1]
                                          if fitted is not None and fitted.prefix_ious else 0.0 if fitted is not None else ""),
                )
                if fitted is not None:
                    relative = _trace_path(output_path, neuron)
                    trace_path = output_path.parent / relative
                    prefix_48 = fitted.prefix_ious[min(48, len(fitted.prefix_ious)) - 1] if fitted.prefix_ious else 0.0
                    trace = {
                        "schema": 1,
                        "backend": chosen_backend,
                        "formula": fitted.formula,
                        "indices": fitted.indices,
                        "names": [str(names[i]) for i in fitted.indices],
                        "operations": fitted.operations,
                        "prefix_ious": fitted.prefix_ious,
                        "fit_iou_at_48": prefix_48,
                        "stop_reason": fitted.stop_reason,
                    }
                    buffer = io.BytesIO()
                    table_data = dict(schema=np.array(1), names=names, table=fitted.table,
                                      columns=np.array(NEURAL_TABLE_COLUMNS), keep_threshold=np.array(0.0))
                    if attention_mode != "cross":
                        ids = np.r_[np.flatnonzero(fitted.table[:, 0] >= 0), len(names)]
                        table_data.update(
                            schema=np.array(2), columns=np.array(SELF_TABLE_COLUMNS),
                            mask_mode=np.array(attention_mode), ids=ids,
                            positions=fitted.positions[ids], queries=fitted.queries[ids],
                            keys=fitted.keys[ids],
                        )
                        trace["kept_count"] = int(fitted.kept_count)
                    np.savez(buffer, **table_data)
                    payload = buffer.getvalue()
                    with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
                        validate_neural_table(saved, names, trace, attention_mode)
                    table_relative = _table_path(output_path, neuron)
                    atomic_write_bytes(output_path.parent / table_relative, payload)
                    atomic_write_json(trace_path, trace, canonical=True)
                    row.update(
                        neural_trace_path=relative.as_posix(),
                        neural_trace_sha256=sha256_file(trace_path),
                        neural_table_path=table_relative.as_posix(),
                        neural_table_sha256=sha256_file(output_path.parent / table_relative),
                    )
            rows.append(row)

        fields = FORMULA_NEURAL_FIELDS if neural else FIELDS
        persist_rows(output_path, rows, fields)
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
    parser.add_argument("--max-iterations", type=int, default=48)
    parser.add_argument("--attention-mode", choices=("cross", "causal", "bidirectional"), default="cross")
    parser.add_argument("--layout-seed", type=int, default=None)
    parser.add_argument("--encoder-width", type=int, default=96)
    parser.add_argument("--readout-width", type=int, default=32)
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
        max_iterations=args.max_iterations,
        attention_mode=args.attention_mode,
        layout_seed=args.layout_seed,
        encoder_width=args.encoder_width,
        readout_width=args.readout_width,
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
