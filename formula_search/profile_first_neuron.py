"""Profile the unchanged first-neuron compositional search."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch
from datasets import load_dataset

from formula_search.analysis import TASKS, load_activation
from formula_search.construction.tokenizer.orchestration import get_tokenizer
from formula_search.construction.vectors import construct_feature_vectors
from formula_search.postprocessing import prune_min_acts, threshold
from formula_search.verify_2a import load_verified_2a


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.profile:
        import pyinstrument
        from pyinstrument import Profiler
        from pyinstrument.renderers import JSONRenderer, SpeedscopeRenderer
    else:
        pyinstrument = None
        Profiler = None
        JSONRenderer = None
        SpeedscopeRenderer = None

    repository_root = Path(__file__).resolve().parents[1]
    backend = load_verified_2a(require_runtime=True)
    assert backend.search_config is not None and backend.search_all is not None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task = TASKS["snli"]
    source = Path(backend.proof.modules["vectorized_search.algorithm"].path).resolve()
    source_hash = hashlib.sha256(
        source.read_text(encoding="utf-8-sig").encode()
    ).hexdigest()
    head = subprocess.check_output(
        ["git", "-C", str(repository_root), "rev-parse", "HEAD"], text=True
    ).strip()
    print(f"search source: {source}", flush=True)
    print(f"git HEAD: {head}", flush=True)
    print(f"algorithm SHA256 (normalized line endings): {source_hash}", flush=True)

    print(f"[{utc_now()}] preparation started", flush=True)
    activations = load_activation(args.activation)
    dataset = load_dataset(task.dataset_name, split=task.split)
    feature_tokenizer = get_tokenizer(
        task.tokenizer_name,
        dataset,
        task.feature_columns,
        enable_pos=task.tokenizer_uses_pos,
    )
    features = construct_feature_vectors(
        dataset,
        feature_tokenizer,
        task.feature_columns,
    )
    binary_activations = threshold(activations, alpha=task.alpha)
    searchable = prune_min_acts(binary_activations, min_acts=task.min_acts)
    print(f"[{utc_now()}] preparation finished", flush=True)

    config = backend.search_config()
    device = torch.device("cuda")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    search_started = utc_now()
    search_start = time.perf_counter()
    profiler = Profiler(interval=0.001, async_mode="disabled") if args.profile else None
    print(f"[{search_started}] search started", flush=True)
    if profiler:
        profiler.start()
    search_results = backend.search_all(
        searchable.matrix,
        features,
        device=device,
        config=config,
    )
    torch.cuda.synchronize(device)
    if profiler:
        profiler.stop()
    search_elapsed = time.perf_counter() - search_start
    search_finished = utc_now()
    print(f"[{search_finished}] search finished ({search_elapsed:.3f}s)", flush=True)

    if profiler:
        profiler.write_html(str(args.output_dir / "first_neuron.html"))
        (args.output_dir / "first_neuron.json").write_text(
            profiler.output(renderer=JSONRenderer(show_all=True)), encoding="utf-8"
        )
        (args.output_dir / "first_neuron.txt").write_text(
            profiler.output_text(show_all=True, color=False, unicode=False),
            encoding="utf-8",
        )
        (args.output_dir / "first_neuron.speedscope.json").write_text(
            profiler.output(renderer=SpeedscopeRenderer(show_all=True)), encoding="utf-8"
        )

    result = {
        "search_started_utc": search_started,
        "search_finished_utc": search_finished,
        "search_elapsed_seconds": search_elapsed,
        "activation_path": str(args.activation.resolve()),
        "activation_shape": list(activations.shape),
        "searchable_shape": list(searchable.matrix.shape),
        "concept_count": len(features),
        "config": vars(config),
        "results": [
            {
                "activation_index": item.activation_index,
                "original_neuron_id": searchable.neuron_ids[item.activation_index],
                "best_formula": item.best_formula,
                "best_score": item.best_score,
            }
            for item in search_results
        ],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "algorithm_path": str(source),
        "algorithm_sha256_normalized": source_hash,
        "git_head": head,
    }
    if profiler and pyinstrument is not None:
        result["pyinstrument"] = {
            "version": pyinstrument.__version__,
            "interval_seconds": 0.001,
            "files": [
                "first_neuron.html",
                "first_neuron.json",
                "first_neuron.txt",
                "first_neuron.speedscope.json",
            ],
        }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"results: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
