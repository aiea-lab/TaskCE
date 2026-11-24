"""Fit one attention configuration per shard on the frozen SNLI capture."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from formula_search.formula_eval import FormulaEvaluator
from formula_search.neural_search import attention_formula, self_attention
from formula_search.neural_search.benchmark_attention import (
    ACTIVATION_HASH,
    FEATURES_HASH,
    load_capture,
    read_reference,
    verify_formula,
)
from formula_search.neural_search.benchmark_path import check_path
from taskce.runtime import atomic_write_json, sha256_file


def read_trial(path, shard):
    trials = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(trials, list) or not 0 <= shard < len(trials):
        raise ValueError("Expected a list of trials and an in-range shard number")
    trial = trials[shard]
    fields = {"mode", "encoder_width", "readout_width", "learning_rate", "steps", "seed", "ids"}
    if not isinstance(trial, dict) or set(trial) != fields:
        raise ValueError(f"Each trial must specify {sorted(fields)}")
    if trial["mode"] not in ("cross", "bidirectional", "causal"):
        raise ValueError("Unknown attention mode")
    for name in ("encoder_width", "readout_width", "steps"):
        if type(trial[name]) is not int or trial[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(trial["seed"]) is not int or not 0 <= trial["seed"] < 2**63:
        raise ValueError("Expected a nonnegative 63-bit integer seed")
    rate = trial["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    ids = trial["ids"]
    if (
        not isinstance(ids, list) or not ids
        or any(type(n) is not int or not 0 <= n < 2048 for n in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("ids must be a nonempty list of distinct embedding_norm neuron IDs")
    return trial


def run(inputs, output, reference_path, trials_path, shard):
    trial = read_trial(trials_path, shard)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gpu = torch.cuda.get_device_name()
    if gpu != "NVIDIA TITAN Xp":
        raise RuntimeError(f"Expected NVIDIA TITAN Xp, got {gpu}")
    vectors, names, raw = load_capture(inputs)
    historical = read_reference(reference_path)
    evaluator = FormulaEvaluator(names, vectors)
    mode = trial["mode"]
    options = {
        field: trial[field]
        for field in ("encoder_width", "readout_width", "learning_rate", "steps", "seed")
    }
    fitter = attention_formula.fit_formula if mode == "cross" else self_attention.fit_self_formula
    fit_options = dict(options, device="cuda")
    if mode != "cross":
        fit_options["mask_mode"] = mode

    rows_dir = output / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(
        **trial, shard=shard, gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda,
        feature_sha256=FEATURES_HASH, activation_sha256=ACTIVATION_HASH,
        reference_sha256=sha256_file(reference_path), trials_sha256=sha256_file(trials_path),
        bench_sha256=sha256_file(__file__), cross_sha256=sha256_file(attention_formula.__file__),
        self_sha256=sha256_file(self_attention.__file__),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )
    worker_file = output / "worker.json"
    if worker_file.exists() and json.loads(worker_file.read_bytes()) != meta:
        raise ValueError("This output directory belongs to a different tuning worker")
    atomic_write_json(worker_file, meta, canonical=True)
    atomic_write_json(output / "vocabulary.json", names.tolist())

    for index, neuron in enumerate(trial["ids"]):
        ref = historical[neuron]
        column = raw[:, neuron].copy()
        threshold = float(torch.quantile(torch.from_numpy(column), 0.945))
        target = column > threshold
        support = int(target.sum())
        if threshold != float(ref["target_threshold"]) or support != int(ref["support"]):
            raise ValueError(f"Target for neuron {neuron} differs from the archived 2.a target")

        torch.cuda.synchronize()
        start = time.perf_counter()
        fitted = fitter(vectors, target, names, **fit_options)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if mode == "cross":
            checked = verify_formula(evaluator, fitted, target, vectors, names)
            kept = checked["leaves"]
            factors = {"table": fitted.table}
        else:
            checked, factors = check_path(fitted, vectors, names, target, evaluator)
            kept = checked["kept"]
        scores = {
            field: checked[field]
            for field in (
                "formula", "indices", "operations", "leaves", "prefix48_iou",
                "iou", "predictions", "intersection",
            )
        }
        path = rows_dir / f"{neuron:04d}.npz"
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **factors)
        temp.replace(path)
        row = dict(
            **options, mode=mode, neuron=neuron, n=len(target), threshold=threshold,
            support=support, kept=kept, seconds=seconds, cold=index == 0,
            reference_iou=float(ref["reference_iou"]), **scores, factors_sha256=sha256_file(path),
        )
        atomic_write_json(path.with_suffix(".json"), row, canonical=True)
        print(json.dumps({k: row[k] for k in ("mode", "neuron", "iou", "seconds", "leaves")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("params", type=Path)
    parser.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    run(args.inputs, args.output, args.reference, args.params, args.shard)
