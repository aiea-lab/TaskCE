import csv
import itertools
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


def _run(inputs, output, *extra):
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
                "neural_search",
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
