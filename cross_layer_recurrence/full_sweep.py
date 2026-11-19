"""Run the selected 2.a search across all neurons and SNLI validation examples.

Parameters come from the run workspace `DataRoot` and study `Profile` instead of
hardcoded paths. The sentence-hash check reads the SNLI validation arrow file
located under the baseline or data root.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile

import numpy as np

from formula_search import sweep as formula_search_sweep
from formula_search.verify_2a import verify
from taskce.data import DataRoot
from taskce.profiles import Profile
from taskce.runtime import atomic_write_json


LAYERS = [f"layer.{n}" for n in range(8, 15)] + ["embedding_norm"]
BEST = (40, 48)
QUANTILE = .945


def find_sentence_path(data_root: DataRoot, profile: Profile) -> Path:
    """Locate the SNLI validation arrow file that verifies prepared features."""
    roots = []
    if data_root.baseline_dir is not None:
        roots.append(data_root.baseline_dir)
    roots.append(data_root.data_dir)
    for root in roots:
        matches = sorted(root.rglob("snli-validation.arrow"))
        if matches:
            return matches[0]
    searched = ", ".join(str(root) for root in roots)
    raise FileNotFoundError(f"No snli-validation.arrow under: {searched}")


def load_sentences(features, sentence_path: Path):
    """Verify sentence row hashes against prepared features and load the sentences."""
    try:
        from pyarrow.ipc import open_stream

        data = open_stream(str(sentence_path)).read_all().to_pylist()
    except (ImportError, OSError, FileNotFoundError) as error:
        return None, str(error)
    hashes = []
    for row in data:
        encoded = json.dumps(
            [row["premise"], row["hypothesis"], int(row["label"])],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        hashes.append(hashlib.sha256(encoded.encode("utf-8")).hexdigest())
    if not np.array_equal(np.asarray(hashes), features["row_hashes"]):
        raise ValueError("sentence row hashes do not match prepared features")
    return data, None


def search(root, output, layer, setting, *, profile, example_count=10000, neurons=2048):
    """Run one all-neuron sweep via `formula_search.sweep` and validate its rows."""
    beam, length = setting
    print(f"SEARCH {layer} all beam={beam} length={length}", flush=True)
    formula_search_sweep.run_sweep(
        features_path=root / "inputs" / "features.npz",
        activation_path=root / "inputs" / f"{layer}.npz",
        layer=layer,
        output_path=output,
        beam=beam,
        length=length,
        quantile=QUANTILE,
        min_fit_fraction=0,
        seed=profile.search.seed,
        backend="vectorized_search_2a",
        chunk_size=16,
        device="cuda",
        all_examples=True,
        resume=True,
    )
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [[n] for n in range(neurons)]
    if [json.loads(r["neuron_ids"]) for r in rows] != ids:
        raise ValueError(f"Incomplete neuron coverage: {output}")
    if any(r["status"] != "ok" or int(r["fit_n"]) != example_count for r in rows):
        raise ValueError(f"Missing formula or incomplete dataset: {output}")
    return rows


def statistics(rows):
    ious = np.asarray([float(row["backend_fit_iou"]) for row in rows])
    return dict(
        peak=float(ious.max()),
        mean=float(ious.mean()),
        median=float(np.median(ious)),
        peak_neuron=int(rows[int(ious.argmax())]["neuron"]),
    )


def stage_reference_trial(trials, reference, setting):
    """Seed trials and the layer 8 parent trial from the reference results when missing."""
    trials.mkdir(exist_ok=True)
    seeds = {
        "scores.json": reference / "trials/scores.json",
        f"{setting[0]}-{setting[1]}.csv.manifest.json": reference / "full/layer.8.csv.manifest.json",
        f"{setting[0]}-{setting[1]}.csv": reference / "full/layer.8.csv",
    }
    for name, source in seeds.items():
        destination = trials / name
        if destination.exists():
            continue
        with source.open("rb") as stream, NamedTemporaryFile(dir=trials, delete=False) as temp:
            temporary = Path(temp.name)
            try:
                shutil.copyfileobj(stream, temp)
                temp.close()
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)


def run_full_sweep(data_root: DataRoot, profile: Profile, run_root: Path):
    """Run the pinned 2.a sweep across all layers into `run_root`."""
    if os.environ.get("TASKCE_BACKEND") != "vectorized_search_2a":
        raise ValueError("Expected the pinned 2.a backend")
    proof = verify()
    bundle = data_root.resolve_prepared_inputs(profile)
    sentence_path = find_sentence_path(data_root, profile)
    with np.load(bundle.features_path) as features:
        sentences, error = load_sentences(features, sentence_path)
        if error or len(sentences) != 10000 or not np.array_equal(features["example_ids"], np.arange(10000)):
            raise ValueError(f"Expected 10,000 aligned SNLI validation examples: {error}")
    with np.load(bundle.layer_paths["layer.8"]) as activation:
        if activation["activation"].shape != (10000, 2048):
            raise ValueError("Unexpected layer 8 dimensions")
    proof.update(dataset="stanfordnlp/snli:validation", examples=10000, neurons=2048, alpha=.055,
                 objective="arithmetic mean of search IoU across neurons")
    atomic_write_json(run_root / "import.json", proof)
    print("VERIFIED", json.dumps(proof), flush=True)
    trials = run_root / "trials"
    reference = data_root.resolve_baseline_path(profile.artifacts.recurrence_results_relpath)
    stage_reference_trial(trials, reference, BEST)
    best = BEST
    history = json.loads((trials / "scores.json").read_text())
    rows = search(run_root, trials / "40-48.csv", "layer.8", best, profile=profile)
    scores = statistics(rows)
    recorded = next(row for row in history if row["setting"] == list(best))
    if any(scores[key] != recorded[key] for key in scores):
        raise ValueError("Saved 40/48 results disagree with trial history")
    atomic_write_json(run_root / "selection.json", dict(best=best, method="user-selected", **scores, trials=history))
    full = run_root / "full"
    full.mkdir(exist_ok=True)
    for ext in (".csv", ".csv.manifest.json"):
        shutil.copyfile(trials / f"{best[0]}-{best[1]}{ext}", full / ("layer.8" + ext))
    for layer in LAYERS[1:]:
        search(run_root, full / (layer + ".csv"), layer, best, profile=profile)
    print("COMPLETE", flush=True)
