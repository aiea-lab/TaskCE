"""Evaluate pointer-ordered self-attention on the same SNLI targets as cross-attention."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from formula_search.formula_eval import FormulaEvaluator
from formula_search.neural_search import self_attention
from formula_search.neural_search.benchmark_attention import (
    ACTIVATION_HASH, FEATURES_HASH, PILOT_IDS, SEED, STEPS, load_capture, measure, read_reference,
)
from formula_search.neural_search.instructions import evaluate_formula
from taskce.runtime import atomic_write_json, sha256_file


def check_path(result, vectors, names, target, evaluator):
    table = result.table
    size = len(names)
    if table.shape != (size, 4) or table.dtype != np.float32 or not np.isfinite(table).all():
        raise ValueError("Expected a finite keep/OR/AND/AND NOT table in vocabulary order")
    kept = np.flatnonzero(table[:, 0] >= 0)
    if len(kept) != result.kept_count:
        raise ValueError("Inclusion count differs from model output")
    ids = np.append(kept, size)
    queries = np.ascontiguousarray(result.queries[ids])
    keys = np.ascontiguousarray(result.keys[ids])
    positions = np.asarray(result.positions)[ids]
    # Only retained concepts and the neuron can participate in a decoded edge.
    current = len(kept)
    available = np.arange(len(kept))
    chosen = []
    while len(available):
        if result.mask_mode == "causal":
            available = available[positions[available] < positions[current]]
        if not len(available):
            break
        scores = keys[available] @ queries[current]
        best = np.argmax(scores)
        current = available[best]
        chosen.append(int(kept[current]))
        available = np.delete(available, best)
    if chosen != list(result.indices):
        raise ValueError("Saved attention factors do not reproduce the selected path")
    operators = np.asarray(("OR", "AND", "AND NOT"))

    def operations(indices):
        return ["ATOM", *operators[table[indices[1:], 1:].argmax(axis=1)].tolist()] if indices else []

    actions = operations(chosen)
    if actions != list(result.operations):
        raise ValueError("Four-column table and chosen operators disagree")
    limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(limit, 2 * len(chosen) + 100))
        parsed, metrics = measure(evaluator, result.formula, target, result.iou)
    finally:
        sys.setrecursionlimit(limit)
    if not np.array_equal(parsed, evaluate_formula(vectors, chosen, actions)):
        raise ValueError("Rendered formula and replayed attention path differ")

    def score(indices):
        mask = evaluate_formula(vectors, indices, operations(indices))
        return np.count_nonzero(mask & target) / max(1, np.count_nonzero(mask | target))

    strength_order = sorted(chosen, key=lambda i: (-table[i, 0], i))
    metrics.update(
        kept=len(kept), leaves=len(chosen), indices=chosen, operations=actions, formula=result.formula,
        prefix48_iou=score(chosen[:48]), same_ids_keep_order_iou=score(strength_order),
    )
    factors = dict(table=table, ids=ids, positions=positions, keys=keys, queries=queries)
    return metrics, factors


def run(inputs, output, reference_path, mode, layout_seed, encoder, readout, ids, steps, shard):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA TITAN Xp":
        raise RuntimeError(f"Expected Titan Xp, got {gpu}")
    x, names, raw = load_capture(inputs)
    archived = read_reference(reference_path)
    evaluator = FormulaEvaluator(names, x)
    rows = output / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "vocabulary.json", names.tolist())
    meta = dict(
        gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda, feature_sha256=FEATURES_HASH,
        activation_sha256=ACTIVATION_HASH, encoder_width=encoder, readout_width=readout,
        mask=mode, layout_seed=layout_seed, steps=steps, seed=SEED, shard=shard, ids=list(ids),
        model_sha256=sha256_file(self_attention.__file__), bench_sha256=sha256_file(__file__),
    )
    atomic_write_json(output / "worker.json", meta)
    for index, neuron in enumerate(ids):
        column = raw[:, neuron].copy()
        threshold = float(torch.quantile(torch.from_numpy(column), 0.945))
        target = column > threshold
        ref = archived[neuron]
        if threshold != float(ref["target_threshold"]) or int(target.sum()) != int(ref["support"]):
            raise ValueError("Target does not match frozen 2.a target")
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = self_attention.fit_self_formula(
            x, target, names, steps=steps, seed=SEED, device="cuda", encoder_width=encoder,
            readout_width=readout, mask_mode=mode, layout_seed=layout_seed,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        metrics, factors = check_path(result, x, names, target, evaluator)
        path = rows / f"{neuron:04d}.npz"
        np.savez_compressed(path, **factors)
        row = dict(neuron=neuron, n=len(target), support=int(target.sum()), threshold=threshold,
                   seconds=seconds, cold=index == 0, reference_iou=float(ref["reference_iou"]),
                   reference_seconds=float(ref["reference_seconds"]), factors_sha256=sha256_file(path), **metrics)
        atomic_write_json(path.with_suffix(".json"), row, canonical=True)
        print(json.dumps({k: row[k] for k in ("neuron", "iou", "leaves", "kept", "seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--encoder-width", type=int, required=True)
    parser.add_argument("--readout-width", type=int, required=True)
    parser.add_argument("--mode", choices=("causal", "bidirectional"), required=True)
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--ids", help="Comma-separated IDs, or 'all'. Defaults to the size-pilot cohort")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    neurons = tuple(range(2048)) if args.ids == "all" else (
        tuple(map(int, args.ids.split(","))) if args.ids else PILOT_IDS
    )
    if not 0 <= args.shard < args.shards:
        parser.error("shard index must be within shard count")
    run(args.inputs, args.output, args.reference, args.mode, args.layout_seed, args.encoder_width,
        args.readout_width, neurons[args.shard::args.shards], args.steps, args.shard)
