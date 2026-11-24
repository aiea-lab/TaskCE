"""Paired single-neuron benchmark on the frozen SNLI embedding_norm capture."""

import argparse
from collections.abc import Callable
from contextlib import redirect_stdout
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from sympy import Symbol
import torch

from formula_search.formula_eval import FormulaEvaluator
from formula_search.neural_search import formula as decoder
from formula_search.verify_2a import load_verified_2a
from taskce.runtime import atomic_write_json, sha256_file


FEATURES_HASH = "276095165cc8b606046aff0dfd18d1f43e36a11d9f635fbffff3ad2867b3f5d5"
ACTIVATION_HASH = "057a9e3a76f5f0ac7779def258fbc0e439d4ac0fa94a8839048b9697e18da00f"
SEED = 20261111
STEPS = 256


def timed(search: Callable):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = search()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def measure(evaluator, formula, target, expected):
    mask = evaluator.evaluate(formula) if formula else np.zeros_like(target)
    intersection = int(np.count_nonzero(mask & target))
    predictions = int(np.count_nonzero(mask))
    iou = intersection / max(1, int(target.sum()) + predictions - intersection)
    if not math.isclose(iou, expected, abs_tol=2e-6):
        raise ValueError(f"Rendered formula IoU {iou} disagrees with search IoU {expected}")
    return mask, dict(iou=iou, intersection=intersection, predictions=predictions)


def run(inputs: Path, output: Path, shard: int, shards: int):
    if not 0 <= shard < shards:
        raise ValueError("shard must be between zero and shards - 1")
    torch.set_num_threads(1)
    features, activations = inputs/"features.npz", inputs/"embedding_norm.npz"
    if sha256_file(features) != FEATURES_HASH or sha256_file(activations) != ACTIVATION_HASH:
        raise ValueError("Inputs differ from the frozen 10k SNLI capture")
    with np.load(features, allow_pickle=False) as file:
        vectors, names = file["vectors"], file["names"].astype(str)
    with np.load(activations, allow_pickle=False) as file:
        raw = file["activation"]
    assert vectors.shape == (3854,10000) and raw.shape == (10000,2048) and vectors.dtype == np.bool_
    ref = load_verified_2a(require_runtime=True)
    config = ref.search_config(maximum_formula_length=48, beam_size=40, max_iterations=48)
    vocabulary = [(Symbol(name), vector) for name, vector in zip(names, vectors)]
    evaluator = FormulaEvaluator(names, vectors)
    gpu = torch.cuda.get_device_name()
    rows = output/"rows"
    rows.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output/f"worker-{shard}.json", dict(
        gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda, shard=shard, shards=shards,
        seed=SEED, steps=STEPS, feature_sha256=FEATURES_HASH, activation_sha256=ACTIVATION_HASH,
        backend_sha256=sha256_file(Path(decoder.__file__)), reference_hashes=ref.proof.module_hashes(),
        benchmark_sha256=sha256_file(Path(__file__)),
    ))
    with open(os.devnull, "w") as quiet:
        cold = True
        for position, neuron in enumerate(range(shard,raw.shape[1],shards)):
            path = rows/f"{neuron:04d}.json"
            if path.is_file():
                continue
            column = raw[:,neuron].copy()
            threshold = float(torch.quantile(torch.from_numpy(column), 0.945))
            target = column > threshold
            def symbolic():
                with redirect_stdout(quiet):
                    result, = ref.search_all(torch.from_numpy(target[:,None]), vocabulary,
                                             device="cuda", config=config)
                return result

            def neural():
                return decoder.fit_formula(vectors, target, names, steps=STEPS, seed=SEED, device="cuda")

            neural_first = (position + shard) % 2 == 0
            if neural_first:
                learned, neural_seconds = timed(neural)
                baseline, reference_seconds = timed(symbolic)
            else:
                baseline, reference_seconds = timed(symbolic)
                learned, neural_seconds = timed(neural)

            expression = baseline.best_formula
            _, ref_metrics = measure(evaluator, expression, target, baseline.best_score)
            leaves = [m.group() for m in evaluator.lexer.finditer(expression) if m.group() in evaluator.features]
            if len(leaves) > 48:
                raise ValueError("2.a exceeded its formula length limit")
            reference = dict(formula=expression, leaves=len(leaves), unique=len(set(leaves)),
                             seconds=reference_seconds, **ref_metrics)
            previous_limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(max(previous_limit, 2 * len(learned.indices) + 100))
                mask, neural_metrics = measure(evaluator, learned.formula, target, learned.iou)
                if not np.array_equal(mask, learned.predict(vectors)):
                    raise ValueError("Selected actions and parsed formula differ on example masks")
                for length in {48, len(leaves)}:
                    prefix_length = min(length, len(learned.indices))
                    formula = decoder.render_formula(names, learned.indices[:length], learned.operations[:length])
                    expected = learned.prefix_ious[prefix_length - 1] if prefix_length else 0.
                    measure(evaluator, formula, target, expected)
            finally:
                sys.setrecursionlimit(previous_limit)
            selected = dict(formula=learned.formula, leaves=len(learned.indices), unique=len(set(learned.indices)),
                            seconds=neural_seconds, indices=learned.indices, operations=learned.operations,
                            prefix_ious=learned.prefix_ious, action_logits=learned.action_logits,
                            attention_logits=learned.attention_logits, action_ranks=learned.action_ranks,
                            stop=learned.stop_reason, **neural_metrics)
            atomic_write_json(path, dict(neuron=neuron, n=len(target), support=int(target.sum()),
                              target_threshold=threshold, shard=shard, gpu=gpu, first_in_worker=cold,
                              neural_first=neural_first, reference=reference, selected=selected), canonical=True)
            cold = False
            if position == 0 or (position + 1) % 8 == 0:
                print(f"shard {shard} finished {position+1} pairs; neuron {neuron}: "
                      f"neural {learned.iou:.4f} ({selected['leaves']} leaves), 2.a {reference['iou']:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    run(args.inputs, args.output, args.shard, args.shards)
