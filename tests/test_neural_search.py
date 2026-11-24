import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")


def _bits():
    return np.array(list(itertools.product((False, True), repeat=3)), dtype=bool).T


def test_fit_neuron_learns_interpretable_exact_match_and_composite():
    from formula_search.neural_search.algorithm import fit_neuron

    vectors = np.tile(_bits(), (1, 4))
    for target in (vectors[0], vectors[0] | vectors[1]):
        result = fit_neuron(vectors, target, steps=48, seed=7, device="cpu")
        weights = result.weights
        prediction = result.predict(vectors)
        assert weights.shape == (3,)
        assert np.all(weights >= 0)
        assert np.isclose(weights.sum(), 1, atol=1e-5)
        assert result.fit_iou == pytest.approx(1.0)
        np.testing.assert_array_equal(prediction, target)
        if np.array_equal(target, vectors[0]):
            assert weights[0] > weights[1] and weights[0] > weights[2]


def test_attention_reads_the_neuron_and_concept_masks_before_training():
    from formula_search.neural_search import fit_neuron

    a, b, _ = _bits()
    concepts = np.stack([a, b, a])
    explanation_a = fit_neuron(concepts, a, steps=0, seed=7, device="cpu")
    explanation_b = fit_neuron(concepts, b, steps=0, seed=7, device="cpu")
    assert explanation_a.weights[0] == pytest.approx(explanation_a.weights[2])
    assert explanation_a.weights[0] > explanation_a.weights[1]
    assert explanation_b.weights[1] > explanation_b.weights[0]


def _inputs(directory, perturb=False):
    vectors = np.tile(_bits(), (1, 3))
    split = np.repeat(np.arange(3), 8)
    activation = np.column_stack(
        (np.tile(_bits()[0], 3), np.tile(_bits()[1], 3))
    ).astype(np.float32)
    if perturb:
        vectors[:, split != 0] = ~vectors[:, split != 0]
        activation[split != 0] = 0.0
    features = directory / "features.npz"
    activations = directory / "embedding_norm.npz"
    np.savez(features, vectors=vectors, names=np.array(["A", "B", "C"]), split=split)
    np.savez(activations, activation=activation)
    return vectors, activation, split


def _run(inputs, output, *extra, backend="neural_weighted"):
    from taskce.cli import main

    assert (
        main(
            [
                "formula-search",
                "--inputs-dir",
                str(inputs),
                "--ids",
                "0,1",
                "--backend",
                backend,
                "--max-iterations",
                "48",
                "--quantile",
                "0.5",
                "--device",
                "cpu",
                "--chunk-size",
                "1",
                "--output",
                str(output.parent),
                *extra,
            ]
        )
        == 0
    )


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _artifact(output, row):
    path = Path(row["attention_path"])
    assert not path.is_absolute()
    path = output.parent / path
    with np.load(path, allow_pickle=False) as saved:
        return saved["weights"].copy(), saved["threshold"].copy()


def test_cli_replays_predictions_and_heldout_data_does_not_change_fit(tmp_path):
    from formula_search.neural_search import AttentionExplanation

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    a = _inputs(first)
    b = _inputs(second, perturb=True)
    out_a, out_b = (
        first / "run" / "embedding_norm.csv",
        second / "run" / "embedding_norm.csv",
    )
    _run(first, out_a)
    _run(second, out_b)
    rows_a, rows_b = _rows(out_a), _rows(out_b)
    assert len(rows_a) == len(rows_b) == 2
    for row_a, row_b in zip(rows_a, rows_b):
        assert row_a["layer"] == "embedding_norm" and row_a["status"] == "ok"
        weights, threshold = _artifact(out_a, row_a)
        other_weights, other_threshold = _artifact(out_b, row_b)
        np.testing.assert_allclose(weights, other_weights)
        np.testing.assert_allclose(threshold, other_threshold)
        assert float(row_a["fit_iou"]) == pytest.approx(1.0)
        for row, (vectors, activation, split), output in (
            (row_a, a, out_a),
            (row_b, b, out_b),
        ):
            weights, threshold = _artifact(output, row)
            prediction = weights.astype(np.float64) @ vectors >= threshold
            restored = AttentionExplanation.load(
                output.parent / row["attention_path"], ["A", "B", "C"]
            )
            np.testing.assert_array_equal(restored.predict(vectors), prediction)
            with pytest.raises(ValueError, match="order"):
                AttentionExplanation.load(
                    output.parent / row["attention_path"], ["B", "A", "C"]
                )
            target = activation[:, int(row["neuron"])] > 0.5
            for index, name in enumerate(("fit", "validation", "test")):
                mask = split == index
                intersection = np.count_nonzero(prediction[mask] & target[mask])
                union = np.count_nonzero(prediction[mask] | target[mask])
                assert int(row[f"{name}_predictions"]) == np.count_nonzero(
                    prediction[mask]
                )
                assert float(row[f"{name}_iou"]) == pytest.approx(
                    intersection / union if union else 0
                )
    assert float(rows_a[0]["test_iou"]) != float(rows_b[0]["test_iou"])
    alone = tmp_path / "alone" / "embedding_norm.csv"
    _run(first, alone, "--ids", "1")
    together_weights, together_cutoff = _artifact(out_a, rows_a[1])
    alone_weights, alone_cutoff = _artifact(alone, _rows(alone)[0])
    np.testing.assert_array_equal(together_weights, alone_weights)
    np.testing.assert_array_equal(together_cutoff, alone_cutoff)


def test_resume_preserves_results_and_rejects_missing_model(tmp_path):
    _inputs(tmp_path)
    output = tmp_path / "run" / "embedding_norm.csv"
    _run(tmp_path, output, "--max-chunks", "1")
    first = _rows(output)
    assert len(first) == 1
    weights, threshold = _artifact(output, first[0])
    _run(tmp_path, output, "--resume")
    resumed = _rows(output)
    assert len(resumed) == 2 and resumed[0] == first[0]
    replay_weights, replay_threshold = _artifact(output, resumed[0])
    np.testing.assert_array_equal(weights, replay_weights)
    np.testing.assert_array_equal(threshold, replay_threshold)
    (output.parent / resumed[0]["attention_path"]).unlink()
    with pytest.raises(ValueError, match="attention artifact"):
        _run(tmp_path, output, "--resume")


def test_all_example_search_scores_the_entire_context(tmp_path):
    vectors, activation, _ = _inputs(tmp_path)
    output = tmp_path / "all" / "embedding_norm.csv"
    _run(tmp_path, output, "--all-examples", "--ids", "0")
    (row,) = _rows(output)
    assert int(row["fit_n"]) == vectors.shape[1] == 24
    assert int(row["fit_intersection"]) == np.count_nonzero(activation[:, 0]) == 12
    assert int(row["validation_n"]) == int(row["test_n"]) == 0
    assert float(row["fit_iou"]) == 1.0


@pytest.mark.parametrize("operation", ["ATOM", "OR", "AND", "AND NOT"])
def test_fit_formula_selects_and_replays_boolean_operations(operation):
    from formula_search.formula_eval import FormulaEvaluator
    from formula_search.neural_search.formula import fit_formula

    vectors = _bits()
    names = ["A", "B", "distractor"]
    a, b, _ = vectors
    target = {
        "ATOM": a,
        "OR": a | b,
        "AND": a & b,
        "AND NOT": a & ~b,
    }[operation]
    result = fit_formula(vectors, target, names, steps=128, seed=7, device="cpu")

    expected_names = {"A"} if operation == "ATOM" else {"A", "B"}
    assert len(result.indices) == len(expected_names)
    assert {names[index] for index in result.indices} == expected_names
    assert list(result.operations) == (["ATOM"] if operation == "ATOM" else ["ATOM", operation])
    assert result.iou == pytest.approx(1.0)
    prediction = vectors[result.indices[0]].copy()
    prefixes = [prediction.copy()]
    for index, action in zip(result.indices[1:], result.operations[1:]):
        if action == "OR":
            prediction |= vectors[index]
        elif action == "AND":
            prediction &= vectors[index]
        elif action == "AND NOT":
            prediction &= ~vectors[index]
        else:
            pytest.fail(f"Unexpected action: {action}")
        prefixes.append(prediction.copy())
    assert len(result.prefix_ious) == len(prefixes)
    for prefix, reported in zip(prefixes, result.prefix_ious):
        intersection = np.count_nonzero(prefix & target)
        union = np.count_nonzero(prefix | target)
        assert reported == pytest.approx(intersection / union if union else 0.0)
    np.testing.assert_array_equal(prediction, target)
    np.testing.assert_array_equal(FormulaEvaluator(names, vectors).evaluate(result.formula), target)


def test_discrete_cli_all_examples_replays_named_formula(tmp_path):
    from formula_search.formula_eval import FormulaEvaluator

    vectors, activation, _ = _inputs(tmp_path)
    output = tmp_path / "discrete" / "embedding_norm.csv"
    _run(tmp_path, output, "--all-examples", backend="neural_search")
    rows = _rows(output)
    assert len(rows) == 2
    evaluator = FormulaEvaluator(["A", "B", "C"], vectors)
    for index, row in enumerate(rows):
        assert row["status"] == "ok"
        assert row["formula"] == ("A", "B")[index]
        prediction = evaluator.evaluate(row["formula"])
        target = activation[:, index] > 0.5
        np.testing.assert_array_equal(prediction, target)
        assert int(row["fit_n"]) == vectors.shape[1] == 24
        assert int(row["fit_intersection"]) == np.count_nonzero(target)
        assert float(row["fit_iou"]) == pytest.approx(1.0)
        assert float(row["backend_fit_iou"]) == pytest.approx(1.0)
        assert int(row["neural_concept_count"]) == 1
        assert int(row["neural_unique_concepts"]) == 1
        assert int(row["validation_n"]) == int(row["test_n"]) == 0

    alone = tmp_path / "alone_discrete" / "embedding_norm.csv"
    _run(tmp_path, alone, "--all-examples", "--ids", "1", backend="neural_search")
    assert _rows(alone)[0]["formula"] == rows[1]["formula"]


def test_formula_recovers_disjoint_concepts_beyond_48_and_audits_replay():
    from formula_search.formula_eval import FormulaEvaluator
    from formula_search.neural_search.formula import fit_formula

    # Each positive example has its own concept; extra negative examples make
    # the mask context include examples that must remain false throughout.
    vectors = np.concatenate((np.eye(55, dtype=bool), np.zeros((55, 5), dtype=bool)), axis=1)
    target = np.array([True] * 55 + [False] * 5)
    names = [f"concept_{index}" for index in range(55)]
    result = fit_formula(vectors, target, names, steps=48, seed=7, device="cpu")

    assert len(result.indices) == 55
    assert set(result.indices) == set(range(55))
    assert result.operations == ("ATOM",) + ("OR",) * 54
    assert result.iou == pytest.approx(1.0)
    assert result.best_at_48.iou == pytest.approx(48 / 55)
    assert result.truncated_48.iou == pytest.approx(48 / 55)
    assert result.iou - result.best_at_48.iou == pytest.approx(7 / 55)
    assert result.best_at_48.indices == result.indices[:48]
    assert result.truncated_48.indices == result.indices[:48]
    assert result.stop_reason == "perfect_iou"

    for record in (result, result.best_at_48, result.truncated_48):
        count = len(record.indices)
        assert len(record.operations) == len(record.prefix_ious) == count
        assert len(record.action_logits) == len(record.action_ranks) == count
        assert len(record.attention_logits) == count
        assert np.isfinite(record.action_logits).all()
        assert np.isfinite(record.attention_logits).all()
        assert all(1 <= rank <= 16 for rank in record.action_ranks)
        expected = vectors[list(record.indices)].any(axis=0)
        np.testing.assert_array_equal(record.predict(vectors), expected)
        np.testing.assert_array_equal(FormulaEvaluator(names, vectors).evaluate(record.formula), expected)
        assert record.iou == pytest.approx(np.count_nonzero(expected & target) / np.count_nonzero(expected | target))


def test_discrete_trace_contains_actions_and_is_checked_on_resume(tmp_path):
    _inputs(tmp_path)
    output = tmp_path / "trace" / "embedding_norm.csv"
    _run(tmp_path, output, "--all-examples", "--max-chunks", "1", backend="neural_search")
    (row,) = _rows(output)
    trace_path = output.parent / row["neural_trace_path"]
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["indices"] == [0]
    assert trace["names"] == ["A"]
    assert trace["operations"] == ["ATOM"]
    assert trace["prefix_ious"] == [1.0]
    assert trace["fit_iou_at_48"] == float(row["neural_fit_iou_at_48"]) == 1.0
    assert len(trace["action_logits"]) == len(trace["attention_logits"]) == 1
    assert 1 <= trace["action_ranks"][0] <= 16
    _run(tmp_path, output, "--all-examples", "--resume", backend="neural_search")
    trace_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed neural trace"):
        _run(tmp_path, output, "--all-examples", "--resume", backend="neural_search")
