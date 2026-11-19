"""Parity regression tests against pre-refactor main (f0949a0)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from ablation.selection import run_ablation_analysis
from cross_layer_recurrence import full_sweep
from formula_search import sweep as formula_search_sweep
from formula_search.runner import FormulaSearchOptions
from formula_search.verify_2a import load_verified_2a
from graph_analysis.analysis import neuron_report_frame
from taskce.data import DataRoot, PreparedInputBundle, load_formula_table
from taskce.executor import execute
from taskce.profiles import load_profile
from taskce.runtime import sha256_file


def test_load_formula_table_preserves_legacy_heldout_and_all_example_scores(
    tmp_path: Path,
) -> None:
    legacy_csv = tmp_path / "legacy.csv"
    legacy_csv.write_text(
        "neuron,formula,iou,weight_entailment,weight_neutral,weight_contradiction\n"
        "0,(premise:cat AND hypothesis:cat),0.61,0.5,-0.1,0.2\n"
        "1,LOW_ACTS_PRUNED,0.0,0.1,0.2,0.3\n",
        encoding="utf-8",
    )
    legacy = load_formula_table(legacy_csv)
    assert legacy["iou"].tolist() == pytest.approx([0.61, 0.0])

    all_example_csv = tmp_path / "layer.8.csv"
    all_example_csv.write_text(
        "layer,neuron,neuron_ids,target_type,status,formula,backend_fit_iou,thresholds,"
        "fit_iou,fit_n,fit_support,fit_predictions,fit_intersection,"
        "validation_iou,validation_n,validation_support,validation_predictions,validation_intersection,"
        "test_iou,test_n,test_support,test_predictions,test_intersection,"
        "beam,length,min_fit_support,weight_entailment,weight_neutral,weight_contradiction\n"
        "layer.8,0,[0],neuron,ok,(premise:dog AND hypothesis:dog),0.55,[1.2],"
        "0.55,10000,550,550,385,0.0,0,0,0,0,0.0,0,0,0,0,40,48,1,0.4,-0.2,0.1\n"
        "layer.8,1,[1],neuron,ok,(premise:cat AND hypothesis:cat),0.31,[0.9],"
        "0.31,10000,550,500,240,0.0,0,0,0,0,0.0,0,0,0,0,40,48,1,-0.3,0.5,0.2\n",
        encoding="utf-8",
    )
    all_ex = load_formula_table(all_example_csv)
    assert all_ex["iou"].tolist() == pytest.approx([0.55, 0.31])

    ablation_res = run_ablation_analysis(all_example_csv, output_dir=tmp_path / "ablation_all")
    assert len(ablation_res.good_neurons) == 2
    assert ablation_res.good_neurons["iou"].tolist() == pytest.approx([0.55, 0.31])

    communities = pd.DataFrame({"node": [0, 1], "community": [0, 0]})
    degrees = pd.DataFrame(
        {
            "node": [0, 1],
            "degree": [1, 1],
            "positive_edge_count": [1, 1],
            "negative_edge_count": [0, 0],
            "positive_weighted_degree": [1.0, 1.0],
            "negative_weighted_degree": [0.0, 0.0],
        }
    )
    report_df = neuron_report_frame(communities, all_ex, degrees)
    assert report_df.sort_values("node")["iou"].tolist() == pytest.approx([0.55, 0.31])

    heldout_csv = tmp_path / "heldout.csv"
    heldout_csv.write_text(
        "layer,neuron,neuron_ids,target_type,status,formula,backend_fit_iou,thresholds,"
        "fit_iou,fit_n,fit_support,fit_predictions,fit_intersection,"
        "validation_iou,validation_n,validation_support,validation_predictions,validation_intersection,"
        "test_iou,test_n,test_support,test_predictions,test_intersection,"
        "beam,length,min_fit_support\n"
        "layer.8,0,[0],neuron,ok,(premise:dog AND hypothesis:dog),0.80,[1.2],"
        "0.80,6000,300,300,260,0.65,2000,100,100,75,0.42,2000,100,100,59,40,48,300\n",
        encoding="utf-8",
    )
    heldout = load_formula_table(heldout_csv)
    assert heldout["iou"].tolist() == pytest.approx([0.42])

    val_only_csv = tmp_path / "val_only.csv"
    val_only_csv.write_text(
        "layer,neuron,neuron_ids,target_type,status,formula,backend_fit_iou,thresholds,"
        "fit_iou,fit_n,validation_iou,validation_n,test_iou,test_n\n"
        "layer.8,0,[0],neuron,ok,(premise:dog AND hypothesis:dog),0.80,[1.2],"
        "0.80,8000,0.65,2000,0.0,0\n",
        encoding="utf-8",
    )
    val_only = load_formula_table(val_only_csv)
    assert val_only["iou"].tolist() == pytest.approx([0.65])


def test_formula_search_general_stage_matches_direct_analysis_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("datasets")
    from formula_search import analysis as general_analysis
    from formula_search import runner as fs_runner

    data_dir = tmp_path / "data"
    act_path = (
        data_dir
        / "LiquidAI"
        / "LFM2.5-1.2B-Thinking"
        / "snli"
        / "checkpoint-10"
        / "model.layers.8.feed_forward"
        / "snli_activation.pt"
    )
    act_path.parent.mkdir(parents=True)
    activations = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 0.0],
            [4.0, 3.0, 0.0],
            [5.0, 4.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.save(activations, act_path)

    fake_dataset = [{"premise": f"p{i}", "hypothesis": f"h{i}"} for i in range(6)]
    monkeypatch.setattr(general_analysis, "load_dataset", lambda *a, **kw: fake_dataset)
    monkeypatch.setattr(general_analysis, "get_tokenizer", lambda *a, **kw: object())
    monkeypatch.setattr(
        general_analysis,
        "construct_feature_vectors",
        lambda *a, **kw: [("feat_a", np.array([0, 0, 0, 1, 1, 1], dtype=bool))],
    )
    monkeypatch.setattr(
        general_analysis,
        "load_classification_weights",
        lambda *a, **kw: torch.tensor(
            [[0.5, -0.1, 0.2], [-0.4, 0.3, 0.1], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )

    def fake_search_all(matrix, features, **kwargs):
        return [
            SimpleNamespace(
                activation_index=idx,
                best_formula=f"(premise:f{idx} AND hypothesis:f{idx})",
                best_score=0.75 - 0.2 * idx,
            )
            for idx in range(matrix.shape[1])
        ]

    monkeypatch.setattr(
        fs_runner,
        "_resolve_general_backend",
        lambda *a, **kw: (fake_search_all, {}),
    )
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    direct_csv = tmp_path / "direct_snli_beam_results.csv"
    general_analysis.run(
        "snli",
        act_path,
        search_fn=fake_search_all,
        output_path=direct_csv,
        alpha=0.5,
        min_acts=2,
        data_root=DataRoot(data_dir=data_dir),
    )

    result = execute(
        "formula_search",
        profile="snli",
        data_dir=data_dir,
        stages=("general",),
        options=FormulaSearchOptions(
            layer="layer.8",
            backend="search",
            alpha=0.5,
            min_acts=2,
            device="cpu",
        ),
    )
    assert result.status == "complete"
    runner_csv = result.artifacts["general_formulas_csv"]
    assert runner_csv.read_text(encoding="utf-8") == direct_csv.read_text(encoding="utf-8")


def test_full_sweep_historical_seed_explicit_and_ordinary_resume_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs_dir = tmp_path / "explicit_inputs"
    inputs_dir.mkdir(parents=True)
    features_path = inputs_dir / "features.npz"
    layer8_path = inputs_dir / "layer.8.npz"
    np.savez(
        features_path,
        vectors=np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=bool),
        names=np.array(["premise:a", "hypothesis:b"]),
        split=np.array([0, 0, 1, 2]),
        example_ids=np.arange(4),
        row_hashes=np.array(["h0", "h1", "h2", "h3"]),
    )
    np.savez(
        layer8_path,
        activation=np.arange(8, dtype=np.float32).reshape(4, 2),
    )

    bundle = PreparedInputBundle(
        directory=inputs_dir,
        features_path=features_path,
        layer_paths={"layer.8": layer8_path},
    )
    profile = load_profile("snli")
    verified = load_verified_2a(require_runtime=False)

    seed_csv = tmp_path / "40-48.csv"
    seed_csv.write_text(
        "layer,neuron,neuron_ids,target_type,status,formula,backend_fit_iou,thresholds,"
        "fit_iou,fit_n,fit_support,fit_predictions,fit_intersection,"
        "validation_iou,validation_n,validation_support,validation_predictions,validation_intersection,"
        "test_iou,test_n,test_support,test_predictions,test_intersection,"
        "beam,length,min_fit_support\n"
        "layer.8,0,[0],neuron,ok,premise:a,0.6,[1.0],0.6,4,2,2,1,0.0,0,0,0,0,0.0,0,0,0,0,40,48,1\n"
        "layer.8,1,[1],neuron,ok,hypothesis:b,0.4,[1.0],0.4,4,2,2,1,0.0,0,0,0,0,0.0,0,0,0,0,40,48,1\n",
        encoding="utf-8",
    )
    seed_manifest = tmp_path / "40-48.csv.manifest.json"
    historical_config = {
        "schema": 1,
        "layer": "layer.8",
        "targets": [[0], [1]],
        "features_sha256": str(sha256_file(features_path)),
        "activation_sha256": str(sha256_file(layer8_path)),
        "runner_sha256": "historical_run_search_py_sha256",
        "evaluator_sha256": str(verified.evaluator_sha256),
        "backend_sha256": verified.proof.module_hashes(),
        "backend_commit": verified.proof.commit,
        "prefix_selector_sha256": None,
        "beam": 40,
        "length": 48,
        "seed": profile.search.seed,
        "quantile": 0.945,
        "eligibility_fraction": 0.0,
        "max_iterations": profile.search.max_iterations,
        "example_scope": "all",
        "example_count": 4,
        "device": "cuda",
    }
    seed_manifest.write_text(json.dumps({"configuration": historical_config}), encoding="utf-8")

    rows, seed_runner_sha = full_sweep.verify_historical_layer8_seed(
        seed_csv,
        seed_manifest,
        bundle=bundle,
        setting=(40, 48),
        profile=profile,
        verified_2a=verified,
        example_count=4,
        neurons=2,
        device="cuda",
    )
    assert len(rows) == 2
    assert seed_runner_sha == "historical_run_search_py_sha256"

    # Mutating any scientific field must still be rejected.
    bad_config = dict(historical_config, seed=999)
    seed_manifest.write_text(json.dumps({"configuration": bad_config}), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        full_sweep.verify_historical_layer8_seed(
            seed_csv,
            seed_manifest,
            bundle=bundle,
            setting=(40, 48),
            profile=profile,
            verified_2a=verified,
            example_count=4,
            neurons=2,
            device="cuda",
        )

    # Ordinary sweep resume keeps strict runner_sha256 / configuration matching.
    seed_manifest.write_text(json.dumps({"configuration": historical_config}), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume configuration or input contents changed"):
        formula_search_sweep.run_sweep(
            features_path=features_path,
            activation_path=layer8_path,
            layer="layer.8",
            output_path=seed_csv,
            beam=2,
            length=4,
            backend="search",
            quantile=0.945,
            min_fit_fraction=0.0,
            seed=profile.search.seed,
            device="cpu",
            all_examples=True,
            resume=True,
        )

    # Non-2.a TASKCE_BACKEND override is rejected; unset TASKCE_BACKEND uses profile.
    monkeypatch.setenv("TASKCE_BACKEND", "new_search")
    with pytest.raises(ValueError, match="pinned 2.a backend"):
        full_sweep.run_full_sweep(
            DataRoot(data_dir=tmp_path),
            profile,
            run_root=tmp_path / "fresh_out",
            inputs_dir=inputs_dir,
        )


def test_combined_full_sweep_index_and_match_consume_fresh_formulas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import gzip

    from cross_layer_recurrence import index_subformulas as index_mod
    from cross_layer_recurrence import matching as matching_mod
    from cross_layer_recurrence.runner import CrossLayerRecurrenceOptions

    data_dir = tmp_path / "data"
    inputs_dir = data_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    baseline_dir = tmp_path / "baselines" / "57a1e74"
    archived_full = baseline_dir / "experiments" / "cross-layer-recurrence" / "results" / "full"
    archived_full.mkdir(parents=True)

    # Prepared inputs for 2 layers, 2 neurons, 12 examples (fit/val/test all populated).
    rng = np.random.default_rng(11)
    vectors = np.zeros((4, 12), dtype=bool)
    vectors[0, :6] = True
    vectors[1, :6] = True
    vectors[2, 6:] = True
    vectors[3, 6:] = True
    names = np.array(["premise:arch_a", "hypothesis:arch_b", "premise:fresh_x", "hypothesis:fresh_y"])
    split = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    np.savez(inputs_dir / "features.npz", vectors=vectors, names=names, split=split)
    for layer in ("layer.8", "layer.9"):
        acts = rng.standard_normal((12, 2)).astype(np.float32)
        np.savez(inputs_dir / f"{layer}.npz", activation=acts)

    def _write_layer_csv(directory: Path, layer: str, formula: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{layer}.csv").write_text(
            "layer,neuron,neuron_ids,target_type,status,formula,backend_fit_iou,thresholds,"
            "fit_iou,fit_n,validation_iou,validation_n,test_iou,test_n,beam,length,min_fit_support\n"
            f"{layer},0,[0],neuron,ok,{formula},0.5,[0.0],0.5,12,0.0,0,0.0,0,40,48,1\n"
            f"{layer},1,[1],neuron,ok,{formula},0.5,[0.0],0.5,12,0.0,0,0.0,0,40,48,1\n",
            encoding="utf-8",
        )

    archived_formula = "(hypothesis:arch_b AND premise:arch_a)"
    fresh_formula = "(hypothesis:fresh_y AND premise:fresh_x)"
    for layer in ("layer.8", "layer.9"):
        _write_layer_csv(archived_full, layer, archived_formula)

    # Bind small-fixture dimensions for index_subformulas helpers.
    orig_load_formulas = index_mod.load_formulas
    orig_summarize = index_mod.summarize
    monkeypatch.setattr(
        index_mod,
        "load_formulas",
        lambda results, *, locations=index_mod.LOCATIONS: orig_load_formulas(
            results, locations=locations, examples=12, neurons=2
        ),
    )
    monkeypatch.setattr(
        index_mod,
        "summarize",
        lambda occ, prop, *, locations=index_mod.LOCATIONS: orig_summarize(
            occ, prop, locations=locations, examples=12, neurons=2
        ),
    )

    # Stub full_sweep to write deliberately distinct fresh formulas into <run>/full.
    def fake_run_full_sweep(data_root, profile, run_root, **kwargs):
        fresh_full = run_root / "full"
        for layer in ("layer.8", "layer.9"):
            _write_layer_csv(fresh_full, layer, fresh_formula)
        (run_root / "import.json").write_text("{}\n", encoding="utf-8")
        (run_root / "selection.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(full_sweep, "run_full_sweep", fake_run_full_sweep)

    # Capture what matching.main reads and also let it run on the real CSVs.
    seen_match_csv_dirs: list[Path] = []
    orig_match_main = matching_mod.main

    def spy_match_main(argv=None):
        argv_list = list(argv or [])
        csv_idx = argv_list.index("--csv-dir") + 1
        seen_match_csv_dirs.append(Path(argv_list[csv_idx]))
        return orig_match_main(argv_list)

    monkeypatch.setattr(matching_mod, "main", spy_match_main)

    # Build a 2-layer profile pointing at our archived baseline.
    profiles_dir = data_dir / "profiles"
    profiles_dir.mkdir(parents=True)
    (profiles_dir / "snli_two_layer.toml").write_text(
        'name = "snli_two_layer"\n'
        'extends = "snli"\n\n'
        "[artifacts]\n"
        'recurrence_layers = ["layer.8", "layer.9"]\n'
        'recurrence_later_start = "layer.9"\n'
        'recurrence_later_end = "layer.9"\n',
        encoding="utf-8",
    )

    # 1. Combined run: full_sweep -> index -> match must consume fresh formulas.
    combined = execute(
        "cross_layer_recurrence",
        profile="snli_two_layer",
        data_dir=data_dir,
        baseline_dir=baseline_dir,
        stages=("full_sweep", "index", "match"),
        options=CrossLayerRecurrenceOptions(locations=("layer.8", "layer.9")),
    )
    assert combined.status == "complete"
    with gzip.open(combined.artifacts["index"], "rt", encoding="utf-8") as stream:
        combined_index = json.load(stream)
    assert fresh_formula in combined_index
    assert archived_formula not in combined_index

    assert seen_match_csv_dirs[-1] == combined.output_dir / "full"
    combined_match = json.loads(combined.artifacts["match_json"].read_text(encoding="utf-8"))
    match_exprs = {item["expression"] for item in combined_match["expressions"]}
    assert fresh_formula in match_exprs
    assert archived_formula not in match_exprs

    # 2. Standalone index + match must still read archived formulas.
    standalone = execute(
        "cross_layer_recurrence",
        profile="snli_two_layer",
        data_dir=data_dir,
        baseline_dir=baseline_dir,
        stages=("index", "match"),
        options=CrossLayerRecurrenceOptions(locations=("layer.8", "layer.9")),
    )
    assert standalone.status == "complete"
    with gzip.open(standalone.artifacts["index"], "rt", encoding="utf-8") as stream:
        standalone_index = json.load(stream)
    assert archived_formula in standalone_index
    assert fresh_formula not in standalone_index
    assert seen_match_csv_dirs[-1] == archived_full
    standalone_match = json.loads(standalone.artifacts["match_json"].read_text(encoding="utf-8"))
    standalone_exprs = {item["expression"] for item in standalone_match["expressions"]}
    assert archived_formula in standalone_exprs
    assert fresh_formula not in standalone_exprs
