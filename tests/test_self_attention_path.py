import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sa = pytest.importorskip("formula_search.neural_search.self_attention")


def test_directed_attention_walk_and_causal_stranding():
    # The neuron's ranking is A, B, C. After choosing A, its row points to C.
    queries = np.eye(4, dtype=np.float32)
    keys = np.zeros_like(queries)
    keys[:3, 3] = [3, 2, 1]
    keys[:3, 0] = [0, 1, 3]
    table = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]], dtype=np.float32)
    natural = np.arange(4)
    assert sa.decode_path(table, queries, keys, natural, "bidirectional") == (
        (0, 2, 1),
        ("ATOM", "OR", "AND NOT"),
    )
    assert sa.decode_path(table, queries, keys, natural, "causal") == ((0,), ("ATOM",))
    reordered = np.array([2, 0, 1, 3])
    assert sa.decode_path(table, queries, keys, reordered, "causal")[0] == (0, 2, 1)
    assert sa.decode_path(
        table, queries, np.zeros_like(keys), reordered, "bidirectional"
    )[0] == (0, 1, 2)


def test_causal_context_cannot_read_a_later_token_and_lazy_rows_equal_full_attention():
    model = sa._SelfPolicy(4, 2, 8, 9, seed=5)
    x = torch.eye(3, 4)
    positions = torch.arange(3)
    rows = torch.arange(2)
    changed = x.clone()
    changed[-1] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    with torch.no_grad():
        for causal in (True, False):
            factors = model.project(x)
            before = model.operators(*factors, rows, positions, causal)
            single = model.operators(*factors, rows[1:], positions, causal)
            after = model.operators(*model.project(changed), rows, positions, causal)
            torch.testing.assert_close(single, before[1:])
            if causal:
                torch.testing.assert_close(before, after)
            else:
                assert not torch.allclose(before, after)


@pytest.mark.parametrize("causal", [False, True])
def test_exact_path_distribution_and_terminal_reward_gradient(causal):
    keep = torch.tensor([0.7, -0.4], dtype=torch.float64, requires_grad=True)
    edges = torch.tensor(
        [[0.1, 0.3], [0.6, -0.4], [-0.1, 0.4]], dtype=torch.float64, requires_grad=True
    )
    ops = torch.tensor(
        [[0.0, 0.6, 0.3], [0.9, 0.8, -0.4]], dtype=torch.float64, requires_grad=True
    )
    p = keep.sigmoid()
    rows = []
    expected = []
    for flags in itertools.product([False, True], repeat=2):
        gate_prob = torch.where(torch.tensor(flags), p, 1 - p).prod()
        chosen = [i for i, enabled in enumerate(flags) if enabled]
        # Enumerate each legal terminating path for these exact gates.
        if len(chosen) == 2:
            paths = [(0,), (1, 0)] if causal else [(0, 1), (1, 0)]
        else:
            paths = [tuple(chosen)]
        for path in paths:
            for joins in itertools.product(range(3), repeat=max(0, len(path) - 1)):
                prob = gate_prob
                if len(chosen) == 2:
                    prob = prob * edges[2].softmax(0)[path[0]]
                if joins:
                    prob = prob * ops[path[1]].softmax(0)[joins[0]]
                expected.append(prob)
                ids = chosen + [0] * (2 - len(chosen))
                visit = [ids.index(node) for node in path] + [0] * (2 - len(path))
                op = [0, joins[0] if joins else 0]
                rows.append((ids, len(chosen), visit, op, len(path)))
    ids, lengths, visit, joins, path_lengths = map(torch.tensor, zip(*rows))
    parents = torch.cat((ids, torch.full_like(ids[:, :1], 2)), dim=1)
    restricted_edges = edges[parents[:, :, None], ids[:, None, :]]
    actual = sa._path_log_prob(
        keep,
        ids,
        lengths,
        restricted_edges,
        ops[ids],
        visit,
        joins,
        path_lengths,
        torch.arange(3),
        causal,
    ).exp()
    expected = torch.stack(expected)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(), keep.new_tensor(1.0))
    reward = torch.linspace(-0.5, 1.0, len(rows), dtype=keep.dtype)
    gradients = torch.autograd.grad(actual @ reward, (keep, edges, ops))
    reference = torch.autograd.grad(expected @ reward, (keep, edges, ops))
    for observed, independent in zip(gradients, reference, strict=True):
        torch.testing.assert_close(observed, independent)


@pytest.mark.parametrize("mode", ["bidirectional", "causal"])
def test_saved_factors_and_operations_replay_model_output(mode):
    bits = np.array(list(itertools.product([False, True], repeat=3)), dtype=bool).T
    target = (bits[0] | bits[1]) & ~bits[2]
    result = sa.fit_self_formula(
        bits,
        target,
        list("ABC"),
        steps=4,
        seed=20261111,
        device="cpu",
        encoder_width=8,
        readout_width=6,
        mask_mode=mode,
        layout_seed=39,
    )
    assert result.table.shape == (3, 4)
    assert result.positions[3] == 3
    np.testing.assert_allclose(
        result.keys[:3] @ result.queries[3] + result.keep_bias,
        result.table[:, 0],
        atol=1e-5,
    )
    assert sa.decode_path(
        result.table, result.queries, result.keys, result.positions, mode
    ) == (result.indices, result.operations)
    if mode == "bidirectional":
        assert result.kept_count == len(result.indices)
    pred = result.predict(bits)
    assert result.iou == np.count_nonzero(pred & target) / np.count_nonzero(
        pred | target
    )


def test_token_layout_is_independent_of_example_subsetting():
    rng = np.random.default_rng(27)
    vectors = rng.random((23, 101)) < 0.35
    target = vectors[0] | vectors[4]
    options = dict(steps=4, seed=24, device="cpu", encoder_width=16, readout_width=12,
                   mask_mode="bidirectional")
    names = [str(i) for i in range(len(vectors))]
    direct = sa.fit_self_formula(vectors, target, names, **options)
    subset = vectors[:, np.ones(target.size, dtype=bool)]
    assert subset.flags.f_contiguous
    prepared = sa.fit_self_formula(subset, target, names, **options)
    np.testing.assert_array_equal(prepared.table, direct.table)
    np.testing.assert_array_equal(prepared.queries, direct.queries)
    np.testing.assert_array_equal(prepared.keys, direct.keys)
