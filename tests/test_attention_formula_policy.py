import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")
attention = pytest.importorskip("formula_search.neural_search.attention_formula")


def test_value_readout_changes_the_operator_scores_while_the_keep_gate_uses_qk():
    torch.manual_seed(38)
    masks = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [1, 1, 1, 0]], dtype=bool)
    device = torch.device("cpu")
    sparse = attention._csr(masks, device)
    transpose = attention._csr(masks.T, device)
    dense = torch.tensor(masks, dtype=torch.float32)
    rate = dense.mean(dim=1, keepdim=True)
    norm = (dense.sum(dim=1, keepdim=True) * (1 - rate)).clamp_min(1).sqrt()
    neuron = torch.tensor([0.5, -0.5, 0.5, -0.5])
    width = 8
    model = attention._Policy(4, len(masks), width, 9, seed=5)
    with torch.no_grad():
        original = model(sparse, transpose, rate, norm, neuron)
        model.key_value_weight[:, width:].neg_()
        changed = model(sparse, transpose, rate, norm, neuron)
        q = torch.nn.functional.layer_norm(neuron @ model.query_weight, (width,))
        projected = ((dense - rate) @ model.key_value_weight[:, :width]) / norm
        keys = torch.nn.functional.layer_norm(projected, (width,))
        raw_keep = keys @ q / width**0.5

    assert original.shape == (len(masks), 5)
    torch.testing.assert_close(original[:, 0], raw_keep + model.keep_bias)
    torch.testing.assert_close(original[:, 0], changed[:, 0])
    assert not torch.allclose(original[:, 1], changed[:, 1])
    assert not torch.allclose(original[:, 2:], changed[:, 2:])


def test_complete_formula_log_prob_matches_probability_of_every_possible_two_token_formula():
    # Two gates produce one empty, two singleton, and six two-concept formulas.
    logits = torch.tensor(
        [[0.6, -1.0, 0.3, -0.4, 1.0], [-0.5, 0.2, 0.7, -0.3, -0.8]],
        dtype=torch.float64, requires_grad=True,
    )
    keep = logits[:, 0].sigmoid()
    rows = []
    probabilities = []
    for gates in itertools.product((False, True), repeat=2):
        selection = [i for i, value in enumerate(gates) if value]
        gate_prob = torch.where(torch.tensor(gates), keep, 1 - keep).prod()
        for order in itertools.permutations(selection):
            for operators in itertools.product(range(3), repeat=max(0, len(order) - 1)):
                probability = gate_prob
                instructions = [[0, 0, 0], [0, 0, 0]]
                for j, i in enumerate(order):
                    if j == 0 and len(order) == 2:
                        probability = probability * logits[:, 1].softmax(0)[i]
                    op = operators[j - 1] if j else 0
                    if j:
                        probability = probability * logits[i, 2:].softmax(0)[op]
                    instructions[j] = [i, op, 1]
                rows.append(instructions)
                probabilities.append(probability)

    actual = attention._log_prob(logits, torch.tensor(rows)).exp()
    expected = torch.stack(probabilities)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(), logits.new_tensor(1.0))
    # Distinct terminal rewards exercise the score-function gradient for all five columns.
    rewards = torch.linspace(0, 1, len(rows), dtype=logits.dtype)
    actual_gradient, = torch.autograd.grad(actual @ rewards, logits)
    expected_gradient, = torch.autograd.grad(expected @ rewards, logits)
    torch.testing.assert_close(actual_gradient, expected_gradient)
