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
from formula_search.verify_2a import load_verified_2a
from taskce.data import DataRoot, PreparedInputBundle
from taskce.profiles import Profile
from taskce.runtime import atomic_write_json, sha256_file


LAYERS = [f"layer.{n}" for n in range(8, 15)] + ["embedding_norm"]
BEST = (40, 48)
QUANTILE = .945


def find_sentence_path(
    data_root: DataRoot,
    profile: Profile,
    *,
    override: Path | None = None,
) -> Path:
    """Locate the SNLI validation arrow file that verifies prepared features."""
    if override is not None:
        resolved = Path(override)
        if not resolved.is_file():
            raise FileNotFoundError(f"Sentence arrow file not found: {resolved}")
        return resolved
    roots: list[Path] = []
    if data_root.baseline_dir is not None:
        roots.append(data_root.baseline_dir)
    roots.append(data_root.data_dir)
    roots.append(data_root.data_dir.parent)
    validation_sibling = data_root.data_dir.parent.parent / "TaskCE-validation"
    if validation_sibling.is_dir():
        roots.append(validation_sibling)
    for root in roots:
        if not root.is_dir():
            continue
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


def _expected_ids(neuron_ids: str, neurons: int) -> list[list[int]]:
    if neuron_ids == "all":
        return [[n] for n in range(neurons)]
    return [[int(item)] for item in neuron_ids.split(",")]


def search(
    bundle: PreparedInputBundle,
    output: Path,
    layer: str,
    setting: tuple[int, int],
    *,
    profile: Profile,
    example_count: int = 10000,
    neurons: int = 2048,
    neuron_ids: str = "all",
    device: str = "cuda",
):
    """Run one layer sweep via `formula_search.sweep` and validate its rows."""
    beam, length = setting
    if layer not in bundle.layer_paths:
        raise FileNotFoundError(
            f"Prepared activation for layer {layer!r} not found in {bundle.directory}"
        )
    print(f"SEARCH {layer} {neuron_ids} beam={beam} length={length}", flush=True)
    formula_search_sweep.run_sweep(
        features_path=bundle.features_path,
        activation_path=bundle.layer_paths[layer],
        layer=layer,
        output_path=output,
        beam=beam,
        length=length,
        neuron_ids=neuron_ids,
        quantile=QUANTILE,
        min_fit_fraction=0,
        seed=profile.search.seed,
        max_iterations=profile.search.max_iterations,
        backend="vectorized_search_2a",
        chunk_size=16,
        device=device,
        all_examples=True,
        resume=True,
    )
    with output.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    ids = _expected_ids(neuron_ids, neurons)
    if [json.loads(r["neuron_ids"]) for r in rows] != ids:
        raise ValueError(f"Incomplete neuron coverage: {output}")
    if any(r["status"] != "ok" or int(r["fit_n"]) != example_count for r in rows):
        raise ValueError(f"Missing formula or incomplete dataset: {output}")
    return rows


def verify_historical_layer8_seed(
    csv_path: Path,
    manifest_path: Path,
    *,
    bundle: PreparedInputBundle,
    setting: tuple[int, int],
    profile: Profile,
    verified_2a,
    example_count: int = 10000,
    neurons: int = 2048,
    device: str = "cuda",
) -> tuple[list[dict[str, str]], str | None]:
    """Verify scientific provenance of an archived layer.8 seed explicitly.

    Allows only `runner_sha256` to differ from `formula_search/sweep.py`
    because the archived seed was written by `experiments/shared/run_search.py`.
    Ordinary resume in `formula_search/sweep.py` keeps strict full-config matching.
    """
    beam, length = setting
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = previous.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError(f"Invalid historical seed manifest: {manifest_path}")
    expected = {
        "schema": 1,
        "layer": "layer.8",
        "targets": [[n] for n in range(neurons)],
        "features_sha256": str(sha256_file(bundle.features_path)),
        "activation_sha256": str(sha256_file(bundle.layer_paths["layer.8"])),
        "evaluator_sha256": str(verified_2a.evaluator_sha256),
        "backend_sha256": verified_2a.proof.module_hashes(),
        "backend_commit": verified_2a.proof.commit,
        "prefix_selector_sha256": None,
        "beam": beam,
        "length": length,
        "seed": profile.search.seed,
        "quantile": QUANTILE,
        "eligibility_fraction": 0.0,
        "max_iterations": profile.search.max_iterations,
        "example_scope": "all",
        "example_count": example_count,
        "device": device,
    }
    mismatches = [
        key for key, value in expected.items() if configuration.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Historical layer.8 seed provenance mismatch on fields: "
            + ", ".join(mismatches)
        )
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if [json.loads(r["neuron_ids"]) for r in rows] != [[n] for n in range(neurons)]:
        raise ValueError(f"Incomplete neuron coverage in historical seed: {csv_path}")
    if any(r["status"] != "ok" or int(r["fit_n"]) != example_count for r in rows):
        raise ValueError(f"Missing formula or incomplete dataset in historical seed: {csv_path}")
    return rows, configuration.get("runner_sha256")


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


def run_full_sweep(
    data_root: DataRoot,
    profile: Profile,
    run_root: Path,
    *,
    inputs_dir: Path | None = None,
    reference_dir: Path | None = None,
    layers: tuple[str, ...] | list[str] | None = None,
    sentence_path: Path | None = None,
    neuron_ids: str = "all",
    device: str = "cuda",
    seed_layer8: bool = True,
):
    """Run the pinned 2.a sweep across configured layers into `run_root`."""
    backend = os.environ.get("TASKCE_BACKEND") or profile.search.backend
    if backend != "vectorized_search_2a":
        raise ValueError(
            f"Expected the pinned 2.a backend ('vectorized_search_2a'), got {backend!r}"
        )
    verified_2a = load_verified_2a(require_runtime=True)
    proof = verified_2a.proof.to_manifest_dict()
    bundle = data_root.resolve_prepared_inputs(profile, override=inputs_dir)
    selected_layers = list(layers) if layers else list(profile.artifacts.recurrence_layers or LAYERS)
    if not selected_layers:
        raise ValueError("No layers configured for full_sweep")
    first_layer = "layer.8" if "layer.8" in selected_layers else selected_layers[0]
    if first_layer not in bundle.layer_paths:
        raise FileNotFoundError(
            f"Prepared activation for layer {first_layer!r} not found in {bundle.directory}"
        )
    with np.load(bundle.layer_paths[first_layer], allow_pickle=False) as activation:
        example_count, neurons = map(int, activation["activation"].shape)
    if seed_layer8 and neuron_ids == "all" and (example_count, neurons) != (10000, 2048):
        raise ValueError(
            f"Unexpected layer 8 dimensions: {(example_count, neurons)} != (10000, 2048)"
        )
    resolved_sentences = find_sentence_path(data_root, profile, override=sentence_path)
    with np.load(bundle.features_path, allow_pickle=False) as features:
        sentences, error = load_sentences(features, resolved_sentences)
        if (
            error
            or len(sentences) != example_count
            or not np.array_equal(features["example_ids"], np.arange(example_count))
        ):
            raise ValueError(
                f"Expected {example_count} aligned SNLI validation examples: {error}"
            )
    proof.update(
        dataset=f"{profile.dataset.hf_id}:{profile.dataset.split}",
        examples=example_count,
        neurons=neurons,
        alpha=profile.search.alpha,
        objective="arithmetic mean of search IoU across neurons",
    )
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "import.json", proof)
    print("VERIFIED", json.dumps(proof), flush=True)

    best = (profile.search.beam, profile.search.length)
    full = run_root / "full"
    full.mkdir(exist_ok=True)
    remaining_layers = list(selected_layers)

    if seed_layer8 and neuron_ids == "all" and "layer.8" in selected_layers:
        trials = run_root / "trials"
        if reference_dir is not None:
            reference = reference_dir
        else:
            if not profile.artifacts.recurrence_results_relpath:
                raise ValueError(
                    f"Profile {profile.name!r} has no recurrence_results_relpath"
                )
            reference = data_root.resolve_baseline_path(
                profile.artifacts.recurrence_results_relpath
            )
        stage_reference_trial(trials, reference, best)
        history = json.loads((trials / "scores.json").read_text(encoding="utf-8"))
        seed_csv = trials / f"{best[0]}-{best[1]}.csv"
        seed_manifest = trials / f"{best[0]}-{best[1]}.csv.manifest.json"
        rows, seed_runner_sha256 = verify_historical_layer8_seed(
            seed_csv,
            seed_manifest,
            bundle=bundle,
            setting=best,
            profile=profile,
            verified_2a=verified_2a,
            example_count=example_count,
            neurons=neurons,
            device=device,
        )
        scores = statistics(rows)
        recorded = next(row for row in history if row["setting"] == list(best))
        if any(scores[key] != recorded[key] for key in scores):
            raise ValueError("Saved 40/48 results disagree with trial history")
        atomic_write_json(
            run_root / "selection.json",
            dict(
                best=best,
                method="user-selected",
                seed_source="historical_layer8_reference",
                seed_runner_sha256=seed_runner_sha256,
                **scores,
                trials=history,
            ),
        )
        for ext in (".csv", ".csv.manifest.json"):
            shutil.copyfile(trials / f"{best[0]}-{best[1]}{ext}", full / ("layer.8" + ext))
        remaining_layers = [layer for layer in selected_layers if layer != "layer.8"]
    else:
        atomic_write_json(
            run_root / "selection.json",
            dict(
                best=best,
                method="direct_refit",
                seed_source=None,
                layers=selected_layers,
                neuron_ids=neuron_ids,
            ),
        )

    for layer in remaining_layers:
        search(
            bundle,
            full / f"{layer}.csv",
            layer,
            best,
            profile=profile,
            example_count=example_count,
            neurons=neurons,
            neuron_ids=neuron_ids,
            device=device,
        )
    print("COMPLETE", flush=True)
