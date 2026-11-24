"""Run one attention-formula fit at a time on the frozen SNLI capture."""

import argparse
import csv
from itertools import product
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from formula_search.formula_eval import FormulaEvaluator
from formula_search.neural_search import attention_formula
from formula_search.neural_search.instructions import evaluate_formula, render_formula
from taskce.runtime import atomic_write_json, sha256_file


FEATURES_HASH = "276095165cc8b606046aff0dfd18d1f43e36a11d9f635fbffff3ad2867b3f5d5"
ACTIVATION_HASH = "057a9e3a76f5f0ac7779def258fbc0e439d4ac0fa94a8839048b9697e18da00f"
SEED = 20261111
STEPS = 256
PILOT_IDS = (0, 1, 2, 3, 4, 5, 127, 255, 383, 511, 639, 767, 895, 1023, 1279, 1535)
PILOT_WIDTHS = tuple(product((32, 96, 256), (32, 128)))


def load_capture(inputs):
    features, activations = inputs / "features.npz", inputs / "embedding_norm.npz"
    if (
        sha256_file(features) != FEATURES_HASH
        or sha256_file(activations) != ACTIVATION_HASH
    ):
        raise ValueError("Inputs differ from the frozen 10k SNLI capture")
    with np.load(features, allow_pickle=False) as file:
        vectors, names = file["vectors"], file["names"].astype(str)
    with np.load(activations, allow_pickle=False) as file:
        raw = file["activation"]
    if (
        vectors.shape != (3854, 10000)
        or vectors.dtype != np.bool_
        or raw.shape != (10000, 2048)
    ):
        raise ValueError("Unexpected capture dimensions or mask dtype")
    if names.shape != (3854,) or len(set(names)) != 3854:
        raise ValueError("The capture must have one distinct name per concept mask")
    return vectors, names, raw


def read_reference(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    by_id = {int(row["neuron"]): row for row in rows}
    if len(rows) != 2048 or sorted(by_id) != list(range(2048)):
        raise ValueError("Archived 2.a data must have one row per neuron")
    if any(row["n"] != "10000" or row["gpu"] != "NVIDIA TITAN Xp" for row in rows):
        raise ValueError("Archived 2.a data must come from the Titan Xp sweep")
    return by_id


def measure(evaluator, formula, target, expected):
    mask = evaluator.evaluate(formula) if formula else np.zeros_like(target)
    intersection = int(np.count_nonzero(mask & target))
    predictions = int(np.count_nonzero(mask))
    score = intersection / max(1, int(target.sum()) + predictions - intersection)
    if not math.isclose(score, expected, rel_tol=0, abs_tol=2e-6):
        raise ValueError(
            f"Rendered formula IoU {score} disagrees with reported IoU {expected}"
        )
    return mask, dict(iou=score, intersection=intersection, predictions=predictions)


def verify_formula(evaluator, learned, target, vectors, names):
    indices = tuple(int(index) for index in learned.indices)
    operations = tuple(str(operation) for operation in learned.operations)
    prefix_ious = tuple(float(score) for score in learned.prefix_ious)
    length = len(indices)
    if len(operations) != length or len(prefix_ious) != length:
        raise ValueError("The action and prefix score traces have different lengths")
    if any(not 0 <= index < len(names) for index in indices):
        raise ValueError("An action refers to a concept outside the vocabulary")

    table = learned.table
    if (
        table.dtype != np.float32
        or table.shape != (len(names), 5)
        or not np.isfinite(table).all()
    ):
        raise ValueError("Expected five finite float32 logits per concept")
    kept = [index for index, row in enumerate(table) if row[0] >= 0]
    kept.sort(key=lambda index: table[index, 1], reverse=True)
    joins = ("OR", "AND", "AND NOT")
    actions = tuple(joins[int(table[index, 2:].argmax())] for index in kept[1:])
    decoded_operations = ("ATOM", *actions) if kept else ()
    if tuple(kept) != indices or decoded_operations != operations:
        raise ValueError(
            "Independently decoding the five logits produced a different action trace"
        )

    # Alternating operators can produce deep expressions for the grammar parser.
    previous_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(max(previous_limit, 2 * length + 100))
        parsed_mask, metrics = measure(
            evaluator, learned.formula, target, float(learned.iou)
        )
        if not np.array_equal(
            parsed_mask, evaluate_formula(vectors, indices, operations)
        ):
            raise ValueError("Rendered formula and action replay disagree")
        prefix_length = min(length, 48)
        prefix_formula = render_formula(
            names, indices[:prefix_length], operations[:prefix_length]
        )
        expected = prefix_ious[prefix_length - 1] if prefix_length else 0.0
        _, prefix_metrics = measure(evaluator, prefix_formula, target, expected)
    finally:
        sys.setrecursionlimit(previous_limit)

    return dict(
        formula=learned.formula,
        indices=indices,
        operations=operations,
        prefix_ious=prefix_ious,
        leaves=length,
        unique=len(set(indices)),
        prefix48_formula=prefix_formula,
        prefix48_leaves=prefix_length,
        prefix48_iou=prefix_metrics["iou"],
        **metrics,
    )


def run(inputs, output, reference_path, shard, shards, widths, neurons):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    vectors, names, raw = load_capture(inputs)
    historical = read_reference(reference_path)
    evaluator = FormulaEvaluator(names, vectors)
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA TITAN Xp":
        raise RuntimeError(f"Expected NVIDIA TITAN Xp, got {gpu}")
    encoder_width, readout_width = widths
    rows = output / "rows"
    rows.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "vocabulary.json", names.tolist())
    atomic_write_json(
        output / f"worker-{shard}.json",
        dict(
            gpu=gpu,
            torch=torch.__version__,
            cuda=torch.version.cuda,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
            logit_columns=["keep", "order", "OR", "AND", "AND NOT"],
            shard=shard,
            shards=shards,
            neuron_ids=list(neurons),
            seed=SEED,
            steps=STEPS,
            encoder_width=encoder_width,
            readout_width=readout_width,
            feature_sha256=FEATURES_HASH,
            activation_sha256=ACTIVATION_HASH,
            backend_sha256=sha256_file(Path(attention_formula.__file__)),
            benchmark_sha256=sha256_file(Path(__file__)),
            historical_reference_sha256=sha256_file(reference_path),
        ),
        canonical=True,
    )
    first = True
    for neuron in neurons:
        row_path = rows / f"{neuron:04d}.json"
        logits_path = row_path.with_suffix(".npz")
        if row_path.is_file() and logits_path.is_file():
            continue
        column = raw[:, neuron].copy()
        threshold = float(torch.quantile(torch.from_numpy(column), 0.945))
        target = column > threshold
        reference = historical[neuron]
        if float(reference["target_threshold"]) != threshold or int(
            reference["support"]
        ) != int(target.sum()):
            raise ValueError(f"Neuron {neuron} does not match the archived 2.a target")

        def fit():
            return attention_formula.fit_formula(
                vectors,
                target,
                names,
                steps=STEPS,
                seed=SEED,
                device="cuda",
                encoder_width=encoder_width,
                readout_width=readout_width,
            )

        torch.cuda.synchronize()
        start = time.perf_counter()
        learned = fit()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        selected = verify_formula(evaluator, learned, target, vectors, names)
        table = learned.table
        temp = logits_path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, table=table)
        os.replace(temp, logits_path)
        atomic_write_json(
            row_path,
            dict(
                neuron=neuron,
                n=len(target),
                support=int(target.sum()),
                target_threshold=threshold,
                shard=shard,
                gpu=gpu,
                first_in_worker=first,
                encoder_width=encoder_width,
                readout_width=readout_width,
                selected=dict(
                    seconds=seconds,
                    logits=logits_path.name,
                    logits_sha256=sha256_file(logits_path),
                    **selected,
                ),
                reference=dict(
                    iou=float(reference["reference_iou"]),
                    leaves=int(reference["reference_leaves"]),
                    unique=int(reference["reference_unique"]),
                    seconds=float(reference["reference_seconds"]),
                    first_in_worker=reference["first_in_worker"] == "True",
                    timing_source="archived_titan_xp_run",
                ),
            ),
            canonical=True,
        )
        first = False
        print(
            f"shard {shard} neuron {neuron}: attention {selected['iou']:.6f}, {seconds:.3f}s, "
            f"{selected['leaves']} terms, archived 2.a {reference['reference_iou']}",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--encoder-width", type=int)
    parser.add_argument("--readout-width", type=int)
    args = parser.parse_args()
    if args.pilot:
        if not 0 <= args.shard < len(PILOT_WIDTHS) or args.shards != len(PILOT_WIDTHS):
            parser.error("Use one shard per pilot width pair")
        widths, neurons = PILOT_WIDTHS[args.shard], PILOT_IDS
    else:
        if (
            not 0 <= args.shard < args.shards
            or not args.encoder_width
            or not args.readout_width
        ):
            parser.error("A valid shard and both model widths are required")
        widths = (args.encoder_width, args.readout_width)
        neurons = range(args.shard, 2048, args.shards)
    run(
        args.inputs,
        args.output,
        args.reference,
        args.shard,
        args.shards,
        widths,
        neurons,
    )
