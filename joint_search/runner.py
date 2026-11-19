"""Compare symbolic parent conjunctions with direct joint-firing search on SNLI."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from formula_search import formula_eval, syntax, verify_2a
from formula_search.syntax import make_lexer, parse, subtrees
from taskce.profiles import Profile
from taskce.runtime import RunWorkspace, atomic_write_text


SWEEP_CSV_COLUMNS = (
    "pair_id",
    "maximum_formula_length",
    "normalized_length",
    "n",
    "support",
    "predictions",
    "intersection",
    "union",
    "iou",
    "formula",
)


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


def score(target, prediction):
    intersection = int((target & prediction).sum())
    union = int((target | prediction).sum())
    return dict(n=int(target.size), support=int(target.sum()), predictions=int(prediction.sum()),
                intersection=intersection, union=union, iou=intersection / union if union else 0.0)


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
    dataset_name: str
    split: str
    example_count: int
    layer: str
    seed: int
    beam: int
    max_iterations: int
    quantile: float
    tolerance: float
    search: Mapping[str, Any]
    parent_search: Mapping[str, Any]
    pairs: tuple[Pair, ...]
    inputs: Mapping[str, Any]


def load_and_validate_manifest(path):
    """Validate every acceptance and hash invariant recorded in the manifest."""
    text = Path(path).read_bytes()
    value = json.loads(text)
    require(value["schema_version"] == 1, "Unsupported manifest schema")

    dataset = value["dataset"]
    require(
        isinstance(dataset, dict)
        and isinstance(dataset.get("name"), str)
        and bool(dataset["name"])
        and isinstance(dataset.get("split"), str)
        and bool(dataset["split"])
        and dataset.get("example_scope") == "all",
        "Unexpected dataset contract",
    )
    example_count = dataset.get("example_count")
    require(
        type(example_count) is int and example_count > 0,
        "Invalid dataset example_count",
    )

    layer = value.get("layer")
    require(isinstance(layer, str) and layer, "Missing manifest layer")

    backend = value["backend"]
    require(
        backend == dict(name="vectorized_search_2a", commit=verify_2a.COMMIT, module_hashes=verify_2a.HASHES),
        "Backend identity differs from pinned 2.a",
    )

    search = value.get("search")
    parent = value.get("parent_search")
    require(isinstance(search, dict) and isinstance(parent, dict), "Missing search configuration")
    for key in ("beam", "max_iterations", "quantile", "seed", "device"):
        require(key in search, f"Missing search.{key}")
    for key in ("beam", "maximum_formula_length", "max_iterations", "quantile", "seed", "example_count", "example_scope"):
        require(key in parent, f"Missing parent_search.{key}")

    require(search.get("length_definition") == "SymPy expression.count(Symbol), after automatic construction normalization",
            "Unexpected formula length definition")
    expected_target = (
        "AND of the two measured activation columns after strict > thresholding "
        f"over all {example_count} examples"
    )
    require(search.get("target") == expected_target, "Unexpected target definition")

    inputs = value.get("inputs")
    require(
        isinstance(inputs, dict)
        and inputs.get("features_file") == "features.npz"
        and inputs.get("activation_file") == f"{layer}.npz",
        "Unexpected input names",
    )
    for key in ("features_sha256", "activation_sha256", "parent_csv_sha256", "parent_manifest_sha256"):
        require(isinstance(inputs.get(key), str), f"Missing inputs.{key}")

    acceptance = value.get("acceptance")
    require(isinstance(acceptance, dict), "Missing acceptance contract")
    for key in ("input_hashes_match", "thresholds_match_parents", "exact_shared_branch",
                "include_conjunction_length", "iteration_cap_covers_all_lengths", "do_not_force_recovery"):
        require(acceptance.get(key) is True, f"Unexpected acceptance.{key}")
    require(acceptance.get("all_examples") == example_count, "Unexpected acceptance.all_examples")
    tolerance = acceptance.get("independent_score_tolerance")
    require(
        isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool) and tolerance >= 0,
        "Unexpected acceptance.independent_score_tolerance",
    )

    pairs = tuple(
        Pair(
            id=row["id"],
            neuron_ids=tuple(int(value) for value in row["neuron_ids"]),
            parent_formulas=tuple(str(value) for value in row["parent_formulas"]),
            parent_lengths=tuple(int(value) for value in row["parent_lengths"]),
            thresholds=tuple(float(value) for value in row["thresholds"]),
            shared_branch=str(row["shared_branch"]),
            conjunction=str(row["conjunction"]),
            conjunction_length=int(row["conjunction_length"]),
            expected_joint_support=int(row["expected_joint_support"]),
            sweep_lengths=tuple(int(value) for value in row["sweep_lengths"]),
        )
        for row in value["pairs"]
    )
    require(pairs and len({pair.id for pair in pairs}) == len(pairs), "Missing or duplicate pairs")
    for pair in pairs:
        require(
            len(pair.neuron_ids) == len(set(pair.neuron_ids)) == 2
            and all(type(n) is int and n >= 0 for n in pair.neuron_ids)
            and pair.id == "-".join(map(str, pair.neuron_ids)),
            "Invalid pair identity",
        )
        require(
            len(pair.parent_formulas) == len(pair.parent_lengths) == len(pair.thresholds) == 2,
            "Expected two complete parents",
        )
        require(
            all(type(length) is int and length > 0 for length in pair.parent_lengths)
            and all(math.isfinite(t) for t in pair.thresholds),
            "Invalid parent metadata",
        )
        require(
            len(pair.sweep_lengths) >= 2
            and len(set(pair.sweep_lengths)) == len(pair.sweep_lengths)
            and all(type(n) is int and n > 0 for n in pair.sweep_lengths),
            "Invalid explicit sweep",
        )
        require(
            pair.conjunction_length in pair.sweep_lengths
            and max(pair.sweep_lengths) <= search["max_iterations"],
            "Sweep must include conjunction length and fit iteration cap",
        )
        require(type(pair.expected_joint_support) is int and pair.expected_joint_support > 0, "Empty measured target")

    return Manifest(
        text=text,
        sha256=hashlib.sha256(text).hexdigest(),
        dataset_name=dataset["name"],
        split=dataset["split"],
        example_count=example_count,
        layer=layer,
        seed=int(search["seed"]),
        beam=int(search["beam"]),
        max_iterations=int(search["max_iterations"]),
        quantile=float(search["quantile"]),
        tolerance=float(tolerance),
        search=search,
        parent_search=parent,
        pairs=pairs,
        inputs=inputs,
    )


@dataclass(frozen=True)
class JointSearchOptions:
    manifest: Path | None = None
    inputs_dir: Path | None = None
    parents_csv: Path | None = None
    verify_baseline: bool = False


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest path (default: profile's joint_manifest_name under joint_search/manifests/).",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        default=None,
        help="Directory containing features.npz and the manifest activation file.",
    )
    parser.add_argument(
        "--parents",
        type=Path,
        default=None,
        help="Parent CSV; adjacent .csv.manifest.json is required.",
    )
    parser.add_argument(
        "--verify-baseline",
        action="store_true",
        help="Compare the completed sweep against the baseline after validation and sweep.",
    )


def options_from_args(args: argparse.Namespace) -> JointSearchOptions:
    return JointSearchOptions(
        manifest=args.manifest,
        inputs_dir=args.inputs,
        parents_csv=args.parents,
        verify_baseline=bool(args.verify_baseline),
    )


def _manifest_path(data_root, profile: Profile, override: Path | None) -> Path:
    if override is not None:
        return override
    name = profile.artifacts.joint_manifest_name
    if not name:
        raise ValueError(f"Profile {profile.name!r} has no joint_manifest_name")
    repo_root = data_root.repo_root or Path(__file__).resolve().parents[1]
    return repo_root / "joint_search" / "manifests" / name


@dataclass
class _Validation:
    manifest: Manifest
    verified: Any
    evaluator: Any
    symbolic: Callable
    parse_formula: Callable
    names: list
    vectors: Any
    prepared: dict
    baselines: list
    masks: dict
    identity: dict
    expected_cells: dict
    completed: dict
    output: Path
    cells: Path
    verify_cell: Callable


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: JointSearchOptions | None,
    stages: tuple[str, ...],
):
    """Run joint_search stages (`validate`, `sweep`) and optional baseline reproduction."""
    opts = options or JointSearchOptions()
    data_root = workspace.data_root

    manifest_path = _manifest_path(data_root, profile, opts.manifest)
    manifest = load_and_validate_manifest(manifest_path)
    contract = json.loads(manifest.text)

    bundle = data_root.resolve_prepared_inputs(profile, override=opts.inputs_dir)
    if manifest.layer not in bundle.layer_paths:
        raise FileNotFoundError(
            f"Manifest layer {manifest.layer!r} not found in {bundle.directory}; "
            f"available layers: {sorted(bundle.layer_paths)}"
        )
    activation_path = bundle.layer_paths[manifest.layer]
    parents_path = opts.parents_csv or data_root.resolve_baseline_path(
        profile.artifacts.joint_parent_csv_relpath
    )
    output = workspace.output_dir

    artifacts: dict[str, Path] = {}
    validation: _Validation | None = None

    if "validate" in stages or "sweep" in stages:
        with workspace.stage("validate"):
            validation = _validate(manifest, contract, bundle, parents_path, output, workspace)
        artifacts.update(
            provenance=output / "provenance.json",
            manifest=output / "manifest.json",
            baselines=output / "baselines.json",
            masks=output / "masks.npz",
            validation=output / "validation.json",
        )

    if "sweep" in stages:
        assert validation is not None
        with workspace.stage("sweep"):
            _sweep(validation)
        artifacts.update(
            results=output / "results.json",
            sweep_csv=output / "sweep.csv",
            status=output / "status.json",
        )

    if opts.verify_baseline:
        baseline_dir = data_root.resolve_baseline_path(profile.artifacts.joint_baseline_results_relpath)
        verify_against_baseline(output, baseline_dir, tolerance=manifest.tolerance)
        artifacts["reproduction_report"] = output / "reproduction_report.json"
    return artifacts


def _validate(manifest, contract, bundle, parents_path, output, workspace):
    import numpy as np
    from sympy import And, Not, Or, Symbol

    import torch

    paths = dict(
        features=bundle.features_path,
        activation=bundle.layer_paths[manifest.layer],
        parent_csv=parents_path,
        parent_manifest=Path(str(parents_path) + ".manifest.json"),
    )
    for key, path in paths.items():
        require(digest(path) == contract["inputs"][f"{key}_sha256"], f"Input hash mismatch: {key}")

    verified = verify_2a.load_verified_2a(require_runtime=False)

    source_files = dict(
        runner=Path(__file__),
        evaluator=Path(formula_eval.__file__),
        syntax=Path(syntax.__file__),
        verify=Path(verify_2a.__file__),
    )
    provenance = dict(
        manifest_sha256=manifest.sha256,
        source_sha256={key: digest(path) for key, path in source_files.items()},
        backend=contract["backend"],
    )
    provenance_sha = hashlib.sha256(encoded(provenance)).hexdigest()
    identity = dict(manifest_sha256=manifest.sha256, provenance_sha256=provenance_sha)

    output.mkdir(parents=True, exist_ok=True)
    if not (output / "provenance.json").exists():
        atomic_json(output / "provenance.json", provenance)
        (output / "manifest.json.tmp").write_bytes(manifest.text)
        (output / "manifest.json.tmp").replace(output / "manifest.json")

    configuration = json.loads(paths["parent_manifest"].read_bytes())["configuration"]
    expected = dict(
        schema=1,
        layer=manifest.layer,
        features_sha256=contract["inputs"]["features_sha256"],
        activation_sha256=contract["inputs"]["activation_sha256"],
        beam=manifest.parent_search["beam"],
        length=manifest.parent_search["maximum_formula_length"],
        seed=manifest.parent_search["seed"],
        quantile=manifest.parent_search["quantile"],
        max_iterations=manifest.parent_search["max_iterations"],
        example_scope=manifest.parent_search["example_scope"],
        example_count=manifest.parent_search["example_count"],
        backend_commit=verified.proof.commit,
        device=manifest.search["device"],
        prefix_selector_sha256=None,
        evaluator_sha256=provenance["source_sha256"]["evaluator"],
    )
    expected["backend_sha256"] = verified.proof.module_hashes()
    require(all(configuration.get(k) == v for k, v in expected.items()), "Parent provenance/configuration mismatch")

    with np.load(paths["features"], allow_pickle=False) as data:
        vectors, names, split = data["vectors"], data["names"].astype(str), data["split"]
    with np.load(paths["activation"], allow_pickle=False) as data:
        activation = data["activation"]
    example_count = manifest.example_count
    require(vectors.dtype == np.bool_ and vectors.ndim == 2 and vectors.shape[1] == example_count,
            "Features must be bool [concept,example_count]")
    require(names.shape == (vectors.shape[0],) and len(set(names)) == len(names) and len(names) >= manifest.beam,
            "Invalid feature names/count")
    require(all(name and name not in ("AND", "OR", "NOT", "(", ")") for name in names), "Invalid feature atoms")
    require(split.shape == (example_count,) and np.isin(split, [0, 1, 2]).all(), "Invalid recorded split")
    require(activation.ndim == 2 and activation.shape[0] == example_count and np.isfinite(activation).all(),
            f"Activations must be finite [{example_count},neuron]")
    require(configuration["targets"] == [[n] for n in range(activation.shape[1])], "Parent target population differs")

    with paths["parent_csv"].open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == activation.shape[1] and len({r["neuron"] for r in rows}) == len(rows), "Parent CSV population differs")
    by_id = {int(r["neuron"]): r for r in rows}

    evaluator = formula_eval.FormulaEvaluator(names.tolist(), vectors)
    lexer = make_lexer(names.tolist())

    def parse_formula(text):
        require(bool(text) and text != "LOW_ACTS_PRUNED", "Backend returned an empty/pruned formula")
        return parse(text, lexer, evaluator.features)

    def symbolic(node):
        if node[0] == "atom":
            return Symbol(node[1])
        return {"AND": And, "OR": Or, "NOT": Not}[node[0]](*(symbolic(n) for n in node[1:]))

    prepared, baselines, masks = {}, [], {}
    for pair in manifest.pairs:
        require(max(pair.neuron_ids) < activation.shape[1], "Parent ID outside activation columns")
        roots = [parse_formula(text) for text in pair.parent_formulas]
        branch = parse_formula(pair.shared_branch)
        require(branch[0] in ("AND", "OR") and all(branch in set(subtrees(r)) for r in roots),
                f"No exact shared compound: {pair.id}")
        require(tuple(int(symbolic(r).count(Symbol)) for r in roots) == pair.parent_lengths, "Parent length mismatch")
        combined = parse_formula(pair.conjunction)
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
            require(row["status"] == "ok" and row["layer"] == manifest.layer and row["target_type"] == "neuron"
                    and json.loads(row["neuron_ids"]) == [neuron] and row["formula"] == pair.parent_formulas[index]
                    and json.loads(row["thresholds"]) == [pair.thresholds[index]] and int(row["beam"]) == manifest.beam
                    and int(row["length"]) == manifest.parent_search["maximum_formula_length"] and int(row["fit_n"]) == example_count
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
        _require_match(result, dict(
            pair_id=pair.id,
            maximum_formula_length=budget,
            neuron_ids=list(pair.neuron_ids),
            thresholds=list(pair.thresholds),
            status="ok",
            manifest_sha256=identity["manifest_sha256"],
        ), "Saved cell identity mismatch")
        formula = result["formula"]
        length = int(symbolic(parse_formula(formula)).count(Symbol))
        require(0 < length <= budget and result["normalized_length"] == length, "Returned formula exceeds budget")
        actual = score(prepared[pair.id][1], evaluator.evaluate(formula))
        _require_match(result, actual, "Independent saved cell counts/score mismatch")
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
    atomic_json(output / "validation.json", dict(**identity, backend_verification=verified.proof.to_manifest_dict(), all_examples=example_count,
                threshold_comparison=">", population=f"all {manifest.dataset_name} {manifest.split} rows in capture order", validated_cells=len(completed)))

    return _Validation(
        manifest=manifest,
        verified=verified,
        evaluator=evaluator,
        symbolic=symbolic,
        parse_formula=parse_formula,
        names=names.tolist(),
        vectors=vectors,
        prepared=prepared,
        baselines=baselines,
        masks=masks,
        identity=identity,
        expected_cells=expected_cells,
        completed=completed,
        output=output,
        cells=cells,
        verify_cell=verify_cell,
    )


def _require_match(result, expected, message):
    """Compare saved values; `provenance_sha256` differs by design across runners."""
    for key, value in expected.items():
        if key == "provenance_sha256":
            continue
        require(result.get(key) == value, message)


def _write_sweep_csv(path, completed, expected_cells):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SWEEP_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for filename in expected_cells:
        result = completed.get(filename)
        if result is None:
            continue
        writer.writerow({column: result.get(column) for column in SWEEP_CSV_COLUMNS})
    atomic_write_text(path, buffer.getvalue())


def _sweep(validation):
    import numpy as np
    from sympy import Symbol

    import torch

    manifest = validation.manifest
    verified = verify_2a.load_verified_2a(require_runtime=True)
    require(torch.cuda.is_available(), "CUDA unavailable")
    features = [(Symbol(name), vector) for name, vector in zip(validation.names, validation.vectors)]

    def persist(state, error=None):
        _write_sweep_csv(validation.output / "sweep.csv", validation.completed, validation.expected_cells)
        atomic_json(validation.output / "results.json", [validation.completed[name] for name in validation.expected_cells if name in validation.completed])
        atomic_json(validation.output / "status.json", dict(**validation.identity, state=state, completed=len(validation.completed), requested=len(validation.expected_cells), error=error))

    persist("running")
    try:
        for filename, (pair, budget) in validation.expected_cells.items():
            if filename in validation.completed:
                continue
            np.random.seed(manifest.seed)
            torch.manual_seed(manifest.seed)
            print(f"Starting {pair.id} maximum length {budget}", flush=True)
            returned = verified.search_all(
                torch.from_numpy(validation.prepared[pair.id][1][:, None]),
                features,
                device="cuda",
                config=verified.search_config(
                    maximum_formula_length=budget,
                    beam_size=manifest.beam,
                    max_iterations=manifest.max_iterations,
                ),
            )
            require(len(returned) == 1 and returned[0].activation_index == 0, "Backend returned missing/duplicate targets")
            found = returned[0]
            length = int(validation.symbolic(validation.parse_formula(found.best_formula)).count(Symbol))
            result = dict(**validation.identity, pair_id=pair.id, maximum_formula_length=budget,
                          neuron_ids=list(pair.neuron_ids), thresholds=list(pair.thresholds), status="ok",
                          formula=found.best_formula, normalized_length=length, backend_iou=float(found.best_score),
                          **score(validation.prepared[pair.id][1], validation.evaluator.evaluate(found.best_formula)))
            validation.verify_cell(result, pair, budget)
            atomic_json(validation.cells / filename, result)
            validation.completed[filename] = result
            persist("running")
        persist("complete")
    except Exception as exc:
        persist("failed", f"{type(exc).__name__}: {exc}")
        raise


def verify_against_baseline(output_dir, baseline_results_dir, tolerance=0.0):
    """Compare a completed joint-search output against the reference baseline.

    Ignores `provenance_sha256` (it reflects the refactored runner's source
    hash); `manifest_sha256` must match. Raises ValueError on any mismatch;
    writes `reproduction_report.json` atomically.
    """
    output_dir = Path(output_dir)
    baseline_dir = Path(baseline_results_dir)
    mismatches: list[str] = []

    current_baselines = _load_json(output_dir / "baselines.json")
    baseline_baselines = _load_json(baseline_dir / "baselines.json")
    _compare_values(
        current_baselines,
        baseline_baselines,
        "baselines.json",
        mismatches,
        ignore_keys=("provenance_sha256",),
        numeric_tolerance=tolerance,
    )

    current_masks = _load_masks(output_dir / "masks.npz")
    baseline_masks = _load_masks(baseline_dir / "masks.npz")
    _compare_mappings(current_masks, baseline_masks, "masks.npz", mismatches)

    current_sweep = _load_csv(output_dir / "sweep.csv")
    baseline_sweep = _load_csv(baseline_dir / "sweep.csv")
    _compare_sweep_rows(current_sweep, baseline_sweep, "sweep.csv", tolerance, mismatches)

    current_cells = _load_cells(output_dir / "cells")
    baseline_cells = _load_cells(baseline_dir / "cells")
    _compare_cells(current_cells, baseline_cells, "cells", tolerance, mismatches)

    report = {
        "baseline_dir": str(baseline_dir),
        "output_dir": str(output_dir),
        "tolerance": tolerance,
        "checked": ["baselines.json", "masks.npz", "sweep.csv", "cells/*.json"],
        "cell_counts": {"current": len(current_cells), "baseline": len(baseline_cells)},
        "mismatches": mismatches,
        "verified": not mismatches,
    }
    atomic_json(output_dir / "reproduction_report.json", report)
    if mismatches:
        raise ValueError("Joint-search output differs from the baseline: " + "; ".join(mismatches))
    return report


def _load_json(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError):
        return None


def _load_masks(path):
    import numpy as np

    masks: dict[str, np.ndarray] = {}
    path = Path(path)
    if path.is_file():
        try:
            with np.load(path, allow_pickle=False) as data:
                masks = {name: data[name] for name in data.files}
        except (OSError, ValueError):
            masks = {}
    return masks


def _load_cells(cells_dir):
    cells: dict[str, dict] = {}
    path = Path(cells_dir)
    if not path.is_dir():
        return cells
    for item in sorted(path.glob("*.json")):
        try:
            cells[item.name] = json.loads(item.read_bytes())
        except (OSError, ValueError):
            continue
    return cells


def _load_csv(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _compare_values(current, baseline, label, mismatches, *, ignore_keys=(), numeric_tolerance=0.0):
    import numpy as np

    if isinstance(current, np.ndarray) or isinstance(baseline, np.ndarray):
        if not np.array_equal(current, baseline):
            mismatches.append(f"{label}: arrays differ")
    elif isinstance(current, Mapping) and isinstance(baseline, Mapping):
        _compare_mappings(current, baseline, label, mismatches, ignore_keys=ignore_keys, numeric_tolerance=numeric_tolerance)
    elif isinstance(current, (list, tuple)) or isinstance(baseline, (list, tuple)):
        current_items = list(current) if isinstance(current, (list, tuple)) else None
        baseline_items = list(baseline) if isinstance(baseline, (list, tuple)) else None
        if current_items is None or baseline_items is None or len(current_items) != len(baseline_items):
            mismatches.append(f"{label}: lengths differ")
            return
        for index, (current_item, baseline_item) in enumerate(zip(current_items, baseline_items)):
            _compare_values(current_item, baseline_item, f"{label}[{index}]", mismatches, ignore_keys=ignore_keys, numeric_tolerance=numeric_tolerance)
    else:
        current_number = _number_or_none(current)
        baseline_number = _number_or_none(baseline)
        if current_number is not None and baseline_number is not None:
            if not math.isclose(current_number, baseline_number, rel_tol=0.0, abs_tol=numeric_tolerance):
                mismatches.append(f"{label}: {current!r} != {baseline!r}")
        elif current != baseline:
            mismatches.append(f"{label}: {current!r} != {baseline!r}")


def _number_or_none(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _compare_mappings(current, baseline, label, mismatches, *, ignore_keys=(), numeric_tolerance=0.0):
    if set(current) != set(baseline):
        mismatches.append(
            f"{label}: key sets differ (current-only: {sorted(set(current) - set(baseline))}, "
            f"baseline-only: {sorted(set(baseline) - set(current))})"
        )
        return
    for key in sorted(current):
        if key in ignore_keys:
            continue
        _compare_values(current[key], baseline[key], f"{label}[{key!r}]", mismatches, ignore_keys=ignore_keys, numeric_tolerance=numeric_tolerance)


def _compare_cells(current_cells, baseline_cells, label, tolerance, mismatches):
    if set(current_cells) != set(baseline_cells):
        mismatches.append(
            f"{label}: cell file sets differ (current-only: {sorted(set(current_cells) - set(baseline_cells))}, "
            f"baseline-only: {sorted(set(baseline_cells) - set(current_cells))})"
        )
        return
    for name in sorted(current_cells):
        _compare_mappings(
            current_cells[name],
            baseline_cells[name],
            f"{label}/{name}",
            mismatches,
            ignore_keys=("provenance_sha256",),
            numeric_tolerance=tolerance,
        )


def _compare_sweep_rows(current_rows, baseline_rows, label, tolerance, mismatches):
    if len(current_rows) != len(baseline_rows):
        mismatches.append(
            f"{label}: row counts differ (current={len(current_rows)}, baseline={len(baseline_rows)})"
        )
        return
    for index, (current, baseline) in enumerate(zip(current_rows, baseline_rows)):
        for key in sorted(set(current) | set(baseline)):
            current_value = current.get(key)
            baseline_value = baseline.get(key)
            try:
                current_number = float(current_value)
                baseline_number = float(baseline_value)
            except (TypeError, ValueError):
                equal = current_value == baseline_value
            else:
                equal = math.isclose(
                    current_number,
                    baseline_number,
                    rel_tol=0.0,
                    abs_tol=tolerance if key == "iou" else 0.0,
                )
            if not equal:
                mismatches.append(f"{label}[{index}][{key!r}]: {current_value!r} != {baseline_value!r}")
