"""Run the selected 2.a search across all neurons and SNLI validation examples.

Set TASKCE_BACKEND=vectorized_search_2a and TASKCE_RUN_ROOT to the run directory.
Prepare inputs/ with ../shared/prepare.py. A fresh run starts from the recorded
Layer 8 trial in the adjacent results directory. Set TASKCE_REFERENCE_RESULTS
when those results are mounted separately from the source.
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

SHARED = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED))

from run_search import atomic_json
from verify_2a import verify

LAYERS = [f"layer.{n}" for n in range(8, 15)] + ["embedding_norm"]
BEST = (40, 48)
QUANTILE = .945
SENTENCES = Path('/~ /TaskCE-validation/huggingface/datasets/stanfordnlp___snli/plain_text/0.0.0/cdb5c3d5eed6ead6e5a341c8e56e669bb666725b/snli-validation.arrow')


def load_sentences(features):
    try:
        from pyarrow.ipc import open_stream
        data = open_stream(str(SENTENCES)).read_all().to_pylist()
        hashes = []
        for row in data:
            encoded = json.dumps([row['premise'], row['hypothesis'], int(row['label'])], ensure_ascii=False, separators=(',', ':'))
            hashes.append(hashlib.sha256(encoded.encode('utf-8')).hexdigest())
        if not np.array_equal(np.asarray(hashes), features['row_hashes']):
            raise ValueError('sentence row hashes do not match prepared features')
        return data, None
    except (ImportError, OSError, FileNotFoundError) as error:
        return None, str(error)


def search(root, output, layer, setting):
    beam, length = setting
    print(f"SEARCH {layer} all beam={beam} length={length}", flush=True)
    subprocess.run([
        sys.executable, str(SHARED / "run_search.py"), "--layer", layer, "--ids", "all",
        "--features", str(root / "inputs/features.npz"),
        "--activation", str(root / "inputs" / (layer + ".npz")), "--output", str(output),
        "--all-examples", "--beam", str(beam), "--length", str(length), "--quantile", str(QUANTILE),
        "--min-fit-fraction", "0", "--chunk-size", "16", "--device", "cuda", "--resume",
    ], check=True)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [[n] for n in range(2048)]
    if [json.loads(r["neuron_ids"]) for r in rows] != ids:
        raise ValueError(f"Incomplete neuron coverage: {output}")
    if any(r["status"] != "ok" or int(r["fit_n"]) != 10000 for r in rows):
        raise ValueError(f"Missing formula or incomplete dataset: {output}")
    return rows


def statistics(rows):
    ious = np.asarray([float(row["backend_fit_iou"]) for row in rows])
    return dict(peak=float(ious.max()), mean=float(ious.mean()), median=float(np.median(ious)),
                peak_neuron=int(rows[int(ious.argmax())]["neuron"]))


def stage_reference_trial(trials, reference, setting):
    beam, length = setting
    trials.mkdir(exist_ok=True)
    seeds = {
        "scores.json": reference / "trials/scores.json",
        f"{beam}-{length}.csv.manifest.json": reference / "full/layer.8.csv.manifest.json",
        f"{beam}-{length}.csv": reference / "full/layer.8.csv",
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


def main(root):
    if os.environ.get("TASKCE_BACKEND") != "vectorized_search_2a":
        raise ValueError("Expected the pinned 2.a backend")
    proof = verify()
    with np.load(root / "inputs/features.npz") as features:
        sentences, error = load_sentences(features)
        if error or len(sentences) != 10000 or not np.array_equal(features["example_ids"], np.arange(10000)):
            raise ValueError(f"Expected 10,000 aligned SNLI validation examples: {error}")
    with np.load(root / "inputs/layer.8.npz") as activation:
        if activation["activation"].shape != (10000, 2048):
            raise ValueError("Unexpected layer 8 dimensions")
    proof.update(dataset="stanfordnlp/snli:validation", examples=10000, neurons=2048, alpha=.055,
                 objective="arithmetic mean of search IoU across neurons")
    atomic_json(root / "import.json", proof)
    print("VERIFIED", json.dumps(proof), flush=True)
    trials = root / "trials"
    reference = Path(os.environ.get("TASKCE_REFERENCE_RESULTS", str(Path(__file__).with_name("results"))))
    stage_reference_trial(trials, reference, BEST)
    best = BEST
    history = json.loads((trials / "scores.json").read_text())
    rows = search(root, trials / "40-48.csv", "layer.8", best)
    scores = statistics(rows)
    recorded = next(row for row in history if row["setting"] == list(best))
    if any(scores[key] != recorded[key] for key in scores):
        raise ValueError("Saved 40/48 results disagree with trial history")
    atomic_json(root / "selection.json", dict(best=best, method="user-selected", **scores, trials=history))
    full = root / "full"
    full.mkdir(exist_ok=True)
    for ext in (".csv", ".csv.manifest.json"):
        shutil.copyfile(trials / f"{best[0]}-{best[1]}{ext}", full / ("layer.8" + ext))
    for layer in LAYERS[1:]:
        search(root, full / (layer + ".csv"), layer, best)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main(Path(os.environ["TASKCE_RUN_ROOT"]))
