import csv
import itertools
import json

import numpy as np
import pytest

pytest.importorskip("torch")


def _inputs(directory, perturb=False):
    vectors = np.tile(np.array(list(itertools.product((False, True), repeat=3)), dtype=bool).T, (1, 3))
    split = np.repeat(np.arange(3), 8)
    activation = np.column_stack((vectors[0], vectors[1])).astype(np.float32)
    if perturb:
        vectors[:, split != 0] = ~vectors[:, split != 0]
        activation[split != 0] = 0
    np.savez(directory / "features.npz", vectors=vectors, names=np.array(["A", "B", "C"]), split=split)
    np.savez(directory / "embedding_norm.npz", activation=activation)
    return vectors, activation, split


def _run(inputs, output, *extra):
    from taskce.cli import main

    assert main([
        "formula-search", "--inputs-dir", str(inputs), "--ids", "0,1",
        "--backend", "neural_search", "--max-iterations", "2",
        "--encoder-width", "8", "--readout-width", "4", "--quantile", "0.5",
        "--device", "cpu", "--chunk-size", "1", "--output", str(output.parent),
        *extra,
    ]) == 0


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_table_contract_and_decode_rejects_tampering():
    from formula_search.sweep import validate_neural_table
    import io

    names = np.array(["A", "B", "C"])
    table = np.array([[1, 2, 0, 0, 0], [0, 2, 0, 2, 0], [-1, 9, 0, 0, 0]], dtype=np.float32)
    stream = io.BytesIO()
    np.savez(stream, schema=np.array(1), names=names, table=table,
             columns=np.array(["keep_logit", "order", "or_logit", "and_logit", "and_not_logit"]),
             keep_threshold=np.array(0.0))
    trace = {"indices": [0, 1], "names": ["A", "B"], "operations": ["ATOM", "AND"],
             "formula": "(A AND B)", "prefix_ious": [0.5, 0.75]}
    with np.load(io.BytesIO(stream.getvalue()), allow_pickle=False) as saved:
        validate_neural_table(saved, names, trace)
        with pytest.raises(ValueError, match="disagree"):
            validate_neural_table(saved, names, {**trace, "indices": [1, 0]})
        with pytest.raises(ValueError, match="schema"):
            validate_neural_table(saved, names[::-1], trace)


def test_cli_tables_replay_and_resume(tmp_path):
    from formula_search.formula_eval import FormulaEvaluator

    vectors, activation, split = _inputs(tmp_path)
    output = tmp_path / "run" / "embedding_norm.csv"
    _run(tmp_path, output, "--max-chunks", "1")
    first = _rows(output)
    assert len(first) == 1
    _run(tmp_path, output, "--resume")
    rows = _rows(output)
    assert len(rows) == 2 and rows[0] == first[0]
    manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert manifest["configuration"]["encoder_width"] == 8
    assert manifest["configuration"]["readout_width"] == 4
    evaluator = FormulaEvaluator(["A", "B", "C"], vectors)
    for row in rows:
        trace = json.loads((output.parent / row["neural_trace_path"]).read_text())
        with np.load(output.parent / row["neural_table_path"], allow_pickle=False) as saved:
            assert saved["table"].shape == (3, 5)
            assert saved["table"].dtype == np.float32
        prediction = evaluator.evaluate(row["formula"]) if row["formula"] else np.zeros(vectors.shape[1], dtype=bool)
        target = activation[:, int(row["neuron"])] > 0.5
        fit = split == 0
        union = np.count_nonzero(prediction[fit] | target[fit])
        expected = np.count_nonzero(prediction[fit] & target[fit]) / union if union else 0
        assert float(row["fit_iou"]) == pytest.approx(expected)
        assert float(row["neural_fit_iou_at_48"]) == pytest.approx(trace["prefix_ious"][min(48, len(trace["prefix_ious"])) - 1] if trace["prefix_ious"] else 0)
    with pytest.raises(ValueError, match="configuration"):
        _run(tmp_path, output, "--resume", "--encoder-width", "9")
    (output.parent / rows[0]["neural_table_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed neural table"):
        _run(tmp_path, output, "--resume")


def test_fit_ignores_heldout_and_resets_for_each_neuron(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _inputs(a)
    _inputs(b, perturb=True)
    out_a, out_b = a / "run" / "embedding_norm.csv", b / "run" / "embedding_norm.csv"
    _run(a, out_a)
    _run(b, out_b)
    rows_a, rows_b = _rows(out_a), _rows(out_b)
    for row_a, row_b in zip(rows_a, rows_b):
        with np.load(out_a.parent / row_a["neural_table_path"], allow_pickle=False) as x, np.load(out_b.parent / row_b["neural_table_path"], allow_pickle=False) as y:
            np.testing.assert_array_equal(x["table"], y["table"])
    alone = a / "alone" / "embedding_norm.csv"
    _run(a, alone, "--ids", "1")
    with np.load(out_a.parent / rows_a[1]["neural_table_path"], allow_pickle=False) as x, np.load(alone.parent / _rows(alone)[0]["neural_table_path"], allow_pickle=False) as y:
        np.testing.assert_array_equal(x["table"], y["table"])


def test_pair_without_positives_reports_zero_iou(tmp_path):
    _inputs(tmp_path)
    activation_path = tmp_path / "embedding_norm.npz"
    np.savez(activation_path, activation=np.zeros((24, 2), dtype=np.float32))
    output = tmp_path / "empty" / "embedding_norm.csv"
    _run(tmp_path, output, "--pair", "0,1", "--all-examples")
    (row,) = _rows(output)
    assert int(row["fit_support"]) == 0
    assert float(row["fit_iou"]) == 0.0
    assert float(row["backend_fit_iou"]) == 0.0


@pytest.mark.parametrize("mode", ["causal", "bidirectional"])
def test_self_cli_saves_replayable_factors_and_resumes(tmp_path, mode):
    _inputs(tmp_path)
    output = tmp_path / mode / "embedding_norm.csv"
    _run(tmp_path, output, "--ids", "0", "--attention-mode", mode, "--layout-seed", "42")
    (row,) = _rows(output)
    trace = json.loads((output.parent / row["neural_trace_path"]).read_text())
    with np.load(output.parent / row["neural_table_path"], allow_pickle=False) as saved:
        assert saved["table"].shape == (3, 4)
        assert str(saved["mask_mode"]) == mode
        np.testing.assert_array_equal(saved["ids"], np.r_[np.flatnonzero(saved["table"][:, 0] >= 0), 3])
        assert saved["queries"].dtype == saved["keys"].dtype == np.float32
        assert trace["kept_count"] == len(saved["ids"]) - 1
    _run(tmp_path, output, "--ids", "0", "--attention-mode", mode, "--layout-seed", "42", "--resume")
    with pytest.raises(ValueError, match="configuration"):
        _run(tmp_path, output, "--ids", "0", "--attention-mode", mode, "--layout-seed", "43", "--resume")


def test_self_replay_breaks_ties_by_id_and_causal_path_can_stop_early():
    from formula_search.sweep import validate_neural_table
    import io

    names = np.array(["A", "B", "C"])
    table = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    stream = io.BytesIO()
    np.savez(stream, schema=np.array(2), names=names, table=table,
             columns=np.array(["keep_logit", "or_logit", "and_logit", "and_not_logit"]),
             keep_threshold=np.array(0.0), mask_mode=np.array("causal"),
             ids=np.arange(4), positions=np.array([0, 2, 1, 3]),
             queries=np.ones((4, 2), dtype=np.float32),
             keys=np.ones((4, 2), dtype=np.float32))
    trace = {"indices": [0], "names": ["A"], "operations": ["ATOM"],
             "formula": "A", "prefix_ious": [0.5], "kept_count": 3}
    with np.load(io.BytesIO(stream.getvalue()), allow_pickle=False) as saved:
        validate_neural_table(saved, names, trace, "causal")
        with pytest.raises(ValueError, match="disagree"):
            validate_neural_table(saved, names, {**trace, "indices": [2]}, "causal")
