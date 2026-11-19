from cross_layer_recurrence.index_subformulas import summarize


def test_summarize_preserves_integer_neuron_formula_count():
    occurrences = {"(a AND b)": [["layer.8", 0]]}
    proper_locations = {"(a AND b)": {"layer.8"}}

    recurrence, summary = summarize(
        occurrences,
        proper_locations,
        locations=("layer.8", "layer.9"),
        examples=10,
        neurons=4,
    )

    assert recurrence == {}
    assert summary["scope"]["neuron_formulas"] == 8
    assert isinstance(summary["scope"]["neuron_formulas"], int)
