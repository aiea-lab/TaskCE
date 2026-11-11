# How a beam candidate survives

[`run.py`](../run.py) passes an `[example, neuron]` activation matrix and
one boolean mask per concept to [`search_all`](algorithm.py). It packs 32
examples per word. For each neuron, `LevelSearch` ranks concepts by intersection
over union (IoU) with the neuron's mask. It discards zero scores and seeds the
beam with the first `beam_size` concepts. Equal scores keep input order.

A generated candidate is a recipe: take one eligible parent from the beam,
one of the neuron's positive-scoring concepts, and one operation (AND, OR, or
AND NOT, in that order). Its bitmap says which examples satisfy the recipe.
PyTorch generates the bitmaps in parent, concept, then operation order.
Each bitmap gets one Triton IoU score, even if several recipes produced it.
The earliest recipe for each bitmap competes for the next beam. Higher scores
win; on a tie the earlier recipe wins. Only survivors need a SymPy formula.

For example, let the neuron fire on both examples (`11`), and let concepts
`A = 10` and `B = 01` seed a beam of width two in that order. Here are all
four distinct generated bitmaps, in selection order:

| Bitmap | Earliest recipe (zero-based generated row) | IoU | Next beam |
| --- | --- | ---: | --- |
| `11` | 4: A OR B | 1 | First |
| `10` | 0: A AND A | ½ | Second (simplifies to A) |
| `01` | 8: B AND NOT A | ½ | Loses the tie to row 0 |
| `00` | 2: A AND NOT A | 0 | Below the cutoff |

Row 7 (B OR A) also makes `11`. Row 4 supplies the formula because it came
first. Row numbers restart at each expansion. A parent may expand only when
its current SymPy formula has fewer than `maximum_formula_length` symbols.
The search records the best **entering** beam at each iteration, replacing
the incumbent only on a strict improvement. With `max_iterations=1`, this
example returns A at ½; the newly generated A OR B is never visited.

See the [bounded statechart](verification/README.md) for the selection rules
and the cases checked. [`analysis.py`](../analysis.py) defaults to a different
search implementation. [`legacy/`](legacy/) holds the older vectorized search.

To profile the existing first-neuron search, run from the repository root:

```sh
python theoretical/compositional_explanations/profile_first_neuron.py --activation <activation.pt> --output-dir <profile-output> --profile
```

The CLI prepares the configured SNLI inputs, then profiles the complete `search_all` call on CUDA, including host-to-device conversion and the final CUDA synchronization. It writes the source hash and Git HEAD with `result.json`, plus HTML, JSON, text, and Speedscope profiles. Install the optional sampler with `uv pip install pyinstrument==5.1.3`; without `--profile`, Pyinstrument is not imported. This is a host wall-clock profile and does not break down individual GPU kernels.
