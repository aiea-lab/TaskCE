"""Compare symbolic parent conjunctions with direct joint-firing search on SNLI."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / "experiments/shared"
sys.path.insert(0, str(SHARED))
import verify_2a


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@dataclass(frozen=True)
class Pair:
    id: str
    neuron_ids: tuple
    parent_formulas: tuple
    parent_lengths: tuple
    thresholds: tuple
    shared_branch: str
    conjunction: str
    conjunction_length: int
    expected_joint_support: int
    sweep_lengths: tuple


@dataclass(frozen=True)
class Manifest:
    text: bytes
    sha256: str
    pairs: tuple
    seed: int
    beam: int
    max_iterations: int
    quantile: float
    tolerance: float


def load_and_validate_manifest(path):
    text = path.read_bytes()
    value = json.loads(text)
    require(value["schema_version"] == 1, "Unsupported manifest schema")
    dataset = value["dataset"]
    require(all(dataset.get(k) == v for k, v in dict(name="stanfordnlp/snli", split="validation",
            example_scope="all", example_count=10000).items()), "Expected all 10000 SNLI validation rows")
    require(value["layer"] == "layer.8", "Expected layer.8")
    backend = value["backend"]
    require(backend == dict(name="vectorized_search_2a", commit=verify_2a.COMMIT,
                           module_hashes=verify_2a.HASHES), "Backend identity differs from pinned 2.a")
    search, parent = value["search"], value["parent_search"]
    require(all(search.get(k) == v for k, v in dict(beam=40, max_iterations=128,
            quantile=.945, seed=20261111, device="cuda").items()), "Unexpected search configuration")
    require(parent == dict(beam=40, maximum_formula_length=48, max_iterations=48,
            quantile=.945, seed=20261111, example_count=10000, example_scope="all"), "Parent configuration differs")
    require(search["length_definition"] == "SymPy expression.count(Symbol), after automatic construction normalization",
            "Unexpected formula length definition")
    require(search["target"] == "AND of the two measured activation columns after strict > thresholding over all 10000 examples",
            "Unexpected target definition")
    acceptance = value["acceptance"]
    require(acceptance == dict(input_hashes_match=True, thresholds_match_parents=True, all_examples=10000,
            exact_shared_branch=True, include_conjunction_length=True, iteration_cap_covers_all_lengths=True,
            independent_score_tolerance=2e-6, do_not_force_recovery=True), "Unexpected acceptance contract")
    require(value["inputs"]["features_file"] == "features.npz" and
            value["inputs"]["activation_file"] == "layer.8.npz", "Unexpected input names")
    pairs = tuple(Pair(**{k: tuple(v) if isinstance(v, list) else v for k, v in row.items()})
                  for row in value["pairs"])
    require(pairs and len({p.id for p in pairs}) == len(pairs), "Missing or duplicate pairs")
    for pair in pairs:
        require(len(pair.neuron_ids) == len(set(pair.neuron_ids)) == 2 and
                all(type(n) is int and n >= 0 for n in pair.neuron_ids) and
                pair.id == "-".join(map(str, pair.neuron_ids)), "Invalid pair identity")
        require(len(pair.parent_formulas) == len(pair.parent_lengths) == len(pair.thresholds) == 2,
                "Expected two complete parents")
        require(pair.parent_lengths == (48, 48) and all(math.isfinite(t) for t in pair.thresholds), "Invalid parent metadata")
        require(len(pair.sweep_lengths) >= 2 and len(set(pair.sweep_lengths)) == len(pair.sweep_lengths)
                and all(type(n) is int and n > 0 for n in pair.sweep_lengths), "Invalid explicit sweep")
        require(pair.conjunction_length in pair.sweep_lengths and max(pair.sweep_lengths) <= search["max_iterations"],
                "Sweep must include conjunction length and fit iteration cap")
        require(type(pair.expected_joint_support) is int and pair.expected_joint_support > 0, "Empty measured target")
    return Manifest(text, hashlib.sha256(text).hexdigest(), pairs, search["seed"], search["beam"],
                    search["max_iterations"], search["quantile"], acceptance["independent_score_tolerance"])


def score(target, prediction):
    intersection = int((target & prediction).sum())
    union = int((target | prediction).sum())
    return dict(n=int(target.size), support=int(target.sum()), predictions=int(prediction.sum()),
                intersection=intersection, union=union, iou=intersection / union if union else 0.0)


def main():
    cli = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""From the repository root, run the frozen sweep with:
  python experiments/joint-search/run.py --manifest experiments/joint-search/manifest.json
    --inputs INPUT_DIR --parents experiments/joint-search/results/parents/layer.8.csv
    --output NEW_OUTPUT_DIR

Enter the command on one line. INPUT_DIR must contain the original full SNLI
validation capture inputs matching the frozen manifest. Search requires CUDA.
Use the source at commit 33a4034 to resume or replay the archived checkpoint
with its original provenance.""",
    )
    cli.add_argument("--manifest", required=True, type=Path)
    cli.add_argument("--inputs", required=True, type=Path, help="Directory containing features.npz and layer.8.npz")
    cli.add_argument("--parents", required=True, type=Path, help="Parent CSV; adjacent .csv.manifest.json is required")
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--resume", action="store_true", help="Resume an output with matching source and manifest hashes")
    cli.add_argument("--check-only", action="store_true", help="Validate inputs and saved cells without starting searches")
    args = cli.parse_args()
    manifest = load_and_validate_manifest(args.manifest)
    contract = json.loads(manifest.text)
    source_files = dict(runner=Path(__file__), evaluator=SHARED / "formula_eval.py",
                        recurrence=SHARED / "recurrence.py", verify=Path(verify_2a.__file__))
    provenance = dict(manifest_sha256=manifest.sha256, source_sha256={k: digest(v) for k, v in source_files.items()},
                      backend=contract["backend"])
    provenance_sha = hashlib.sha256(encoded(provenance)).hexdigest()
    identity = dict(manifest_sha256=manifest.sha256, provenance_sha256=provenance_sha)
    output = args.output
    if output.exists() and any(output.iterdir()):
        require(args.resume, "Output exists; use --resume or a new directory")
        require(json.loads((output / "provenance.json").read_bytes()) == provenance, "Resume provenance drift")
        require((output / "manifest.json").read_bytes() == manifest.text, "Resume manifest snapshot drift")
    paths = dict(features_sha256=args.inputs / "features.npz", activation_sha256=args.inputs / "layer.8.npz",
                 parent_csv_sha256=args.parents, parent_manifest_sha256=Path(str(args.parents) + ".manifest.json"))
    for key, path in paths.items():
        require(digest(path) == contract["inputs"][key], f"Input hash mismatch: {key}")
    proof = verify_2a.verify()
    import numpy as np
    import torch
    from sympy import Symbol
    from sympy.logic import And, Or, Not
    import recurrence
    from formula_eval import FormulaEvaluator
    from vectorized_search.algorithm import SearchConfig, search_all

    configuration = json.loads(paths["parent_manifest_sha256"].read_bytes())["configuration"]
    expected = dict(schema=1, layer="layer.8", features_sha256=contract["inputs"]["features_sha256"],
                    activation_sha256=contract["inputs"]["activation_sha256"], beam=40, length=48,
                    seed=manifest.seed, quantile=manifest.quantile, max_iterations=48, example_scope="all",
                    example_count=10000, backend_commit=verify_2a.COMMIT, device="cuda", prefix_selector_sha256=None,
                    evaluator_sha256=provenance["source_sha256"]["evaluator"])
    expected["backend_sha256"] = {k: v["sha256"] for k, v in proof["modules"].items()}
    require(all(configuration.get(k) == v for k, v in expected.items()), "Parent provenance/configuration mismatch")
    with np.load(paths["features_sha256"], allow_pickle=False) as data:
        vectors, names, split = data["vectors"], data["names"].astype(str), data["split"]
    with np.load(paths["activation_sha256"], allow_pickle=False) as data:
        activation = data["activation"]
    require(vectors.dtype == np.bool_ and vectors.ndim == 2 and vectors.shape[1] == 10000,
            "Features must be bool [concept,10000]")
    require(names.shape == (vectors.shape[0],) and len(set(names)) == len(names) and len(names) >= manifest.beam,
            "Invalid feature names/count")
    require(all(name and name not in ("AND", "OR", "NOT", "(", ")") for name in names), "Invalid feature atoms")
    require(split.shape == (10000,) and np.isin(split, [0, 1, 2]).all(), "Invalid recorded split")
    require(activation.ndim == 2 and activation.shape[0] == 10000 and np.isfinite(activation).all(),
            "Activations must be finite [10000,neuron]")
    require(configuration["targets"] == [[n] for n in range(activation.shape[1])], "Parent target population differs")
    with args.parents.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == activation.shape[1] and len({r["neuron"] for r in rows}) == len(rows), "Parent CSV population differs")
    by_id = {int(r["neuron"]): r for r in rows}
    evaluator = FormulaEvaluator(names.tolist(), vectors)
    lexer = recurrence.make_lexer(names.tolist())

    def parse(text):
        require(bool(text) and text != "LOW_ACTS_PRUNED", "Backend returned an empty/pruned formula")
        return recurrence.parse(text, lexer, evaluator.features)

    def symbolic(node):
        if node[0] == "atom":
            return Symbol(node[1])
        return {"AND": And, "OR": Or, "NOT": Not}[node[0]](*(symbolic(n) for n in node[1:]))

    prepared, baselines, masks = {}, [], {}
    for pair in manifest.pairs:
        require(max(pair.neuron_ids) < activation.shape[1], "Parent ID outside activation columns")
        roots = [parse(text) for text in pair.parent_formulas]
        branch = parse(pair.shared_branch)
        require(branch[0] in ("AND", "OR") and all(branch in set(recurrence.subtrees(r)) for r in roots),
                f"No exact shared compound: {pair.id}")
        require(tuple(int(symbolic(r).count(Symbol)) for r in roots) == pair.parent_lengths, "Parent length mismatch")
        combined = parse(pair.conjunction)
        require(pair.conjunction == f"({pair.parent_formulas[0]} AND {pair.parent_formulas[1]})" and
                symbolic(combined) == And(*(symbolic(r) for r in roots)) and
                int(symbolic(combined).count(Symbol)) == pair.conjunction_length, "Conjunction structure/length mismatch")
        raw = torch.as_tensor(activation[:, list(pair.neuron_ids)])
        if not raw.is_floating_point():
            raw = raw.float()
        thresholds = torch.quantile(raw, manifest.quantile, dim=0).numpy()
        require(tuple(map(float, thresholds)) == pair.thresholds, "Recomputed raw quantile thresholds differ")
        for index, neuron in enumerate(pair.neuron_ids):
            row = by_id[neuron]
            require(row["status"] == "ok" and row["layer"] == "layer.8" and row["target_type"] == "neuron"
                    and json.loads(row["neuron_ids"]) == [neuron] and row["formula"] == pair.parent_formulas[index]
                    and json.loads(row["thresholds"]) == [pair.thresholds[index]] and int(row["beam"]) == manifest.beam
                    and int(row["length"]) == 48 and int(row["fit_n"]) == 10000
                    and int(row["validation_n"]) == int(row["test_n"]) == 0, "Parent CSV row differs")
        target = np.all(raw.numpy() > thresholds, axis=1)
        require(int(target.sum()) == pair.expected_joint_support, "Measured joint support differs")
        baseline = evaluator.evaluate(pair.conjunction)
        require(np.array_equal(baseline, evaluator.evaluate(pair.parent_formulas[0]) &
                               evaluator.evaluate(pair.parent_formulas[1])), "Conjunction evaluation inconsistency")
        prepared[pair.id] = (pair, target)
        masks[pair.id + "_measured_joint"] = target
        masks[pair.id + "_symbolic_baseline"] = baseline
        baselines.append(dict(**identity, pair_id=pair.id, neuron_ids=list(pair.neuron_ids), thresholds=list(pair.thresholds),
                              formula=pair.conjunction, normalized_length=pair.conjunction_length, **score(target, baseline)))

    cells = output / "cells"
    expected_cells = {f"{p.id}-L{n}.json": (p, n) for p in manifest.pairs for n in p.sweep_lengths}

    def verify_cell(result, pair, budget):
        require(all(result.get(k) == v for k, v in dict(**identity, pair_id=pair.id, maximum_formula_length=budget,
                neuron_ids=list(pair.neuron_ids), thresholds=list(pair.thresholds), status="ok").items()), "Saved cell identity mismatch")
        formula = result["formula"]
        length = int(symbolic(parse(formula)).count(Symbol))
        require(0 < length <= budget and result["normalized_length"] == length, "Returned formula exceeds budget")
        actual = score(prepared[pair.id][1], evaluator.evaluate(formula))
        require(all(result.get(k) == v for k, v in actual.items()), "Independent saved cell counts/score mismatch")
        require(math.isfinite(result["backend_iou"]) and abs(actual["iou"] - result["backend_iou"]) <= manifest.tolerance,
                "Backend/evaluator IoU disagreement")

    completed = {}
    if cells.exists():
        for path in cells.iterdir():
            if path.name.endswith(".tmp"):
                continue
            require(path.name in expected_cells, f"Unexpected saved cell: {path.name}")
            result = json.loads(path.read_bytes())
            verify_cell(result, *expected_cells[path.name])
            completed[path.name] = result
    output.mkdir(parents=True, exist_ok=True)
    cells.mkdir(exist_ok=True)
    if not (output / "provenance.json").exists():
        atomic_json(output / "provenance.json", provenance)
        (output / "manifest.json.tmp").write_bytes(manifest.text)
        (output / "manifest.json.tmp").replace(output / "manifest.json")
    if (output / "baselines.json").exists():
        require(json.loads((output / "baselines.json").read_bytes()) == baselines, "Saved baseline drift")
    else:
        atomic_json(output / "baselines.json", baselines)
    if (output / "masks.npz").exists():
        with np.load(output / "masks.npz", allow_pickle=False) as saved:
            require(set(saved.files) == set(masks) and all(np.array_equal(saved[k], v) for k, v in masks.items()), "Saved masks drift")
    else:
        with (output / "masks.npz.tmp").open("wb") as stream:
            np.savez_compressed(stream, **masks)
        (output / "masks.npz.tmp").replace(output / "masks.npz")
    atomic_json(output / "validation.json", dict(**identity, backend_verification=proof, all_examples=10000,
                threshold_comparison=">", population="all SNLI validation rows in capture order", validated_cells=len(completed)))

    def persist(state, error=None):
        atomic_json(output / "results.json", [completed[name] for name in expected_cells if name in completed])
        atomic_json(output / "status.json", dict(**identity, state=state, completed=len(completed), requested=len(expected_cells), error=error))

    if args.check_only:
        persist("validated")
        print(f"Validated all inputs, {len(baselines)} baselines and {len(completed)} saved cells", flush=True)
        return
    persist("running")
    try:
        require(torch.cuda.is_available(), "CUDA unavailable")
        features = [(Symbol(name), vector) for name, vector in zip(names, vectors)]
        for filename, (pair, budget) in expected_cells.items():
            if filename in completed:
                continue
            np.random.seed(manifest.seed)
            torch.manual_seed(manifest.seed)
            print(f"Starting {pair.id} maximum length {budget}", flush=True)
            returned = search_all(torch.from_numpy(prepared[pair.id][1][:, None]), features, device="cuda",
                                 config=SearchConfig(maximum_formula_length=budget, beam_size=manifest.beam,
                                                     max_iterations=manifest.max_iterations))
            require(len(returned) == 1 and returned[0].activation_index == 0, "Backend returned missing/duplicate targets")
            found = returned[0]
            length = int(symbolic(parse(found.best_formula)).count(Symbol))
            result = dict(**identity, pair_id=pair.id, maximum_formula_length=budget, neuron_ids=list(pair.neuron_ids),
                          thresholds=list(pair.thresholds), status="ok", formula=found.best_formula, normalized_length=length,
                          backend_iou=float(found.best_score), **score(prepared[pair.id][1], evaluator.evaluate(found.best_formula)))
            verify_cell(result, pair, budget)
            atomic_json(cells / filename, result)
            completed[filename] = result
            persist("running")
        persist("complete")
    except Exception as exc:
        persist("failed", f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
