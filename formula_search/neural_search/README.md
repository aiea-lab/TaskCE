# Search for a Boolean explanation of a neuron

`neural_search` trains a fresh attention policy per neuron to select named concepts and combine them with OR, AND, and AND NOT. It thresholds the neuron's activations at the configured quantile and trains against that binary target. Every update sees the same target and every concept mask across every fit example. With `--all-examples`, the fit context covers the entire prepared dataset.

The policy builds eight signed projections of the aligned concept masks and target, including a centered target correlation. It learns separate attention queries for the initial atom and the three later operators. The final action logit combines this attention score with a learned score derived from each candidate mask's true and false positive counts and the current formula's state.

Training uses Adam to increase expected hard IoU under the action probabilities. A second loss term favors the ordering implied by exact candidate IoUs. The policy explores prefixes by perturbing action logits during training. After training, it orders actions by their unperturbed logits, examines up to 16 per prefix, and accepts the first that strictly improves hard IoU. It stops at perfect IoU or when all 16 proposals fail to improve it. A concept can occur multiple times.

Formula length counts concept occurrences, including the initial atom and repeated names. The CSV reports the final fit IoU and `neural_fit_iou_at_48`. The latter evaluates the first 48 actions of the same trained policy, or the whole formula when it stops earlier.

## Run a comparison

Use the same prepared input directory, layer, neuron, threshold and device. The neural command trains for 256 updates. Pinned 2.a uses length and iteration limits of 48 and a beam of 40.

```sh
uv run taskce formula-search --profile snli --stage search --inputs-dir inputs --layer embedding_norm --ids 0 --all-examples --chunk-size 1 --device cuda --backend neural_search --max-iterations 256 --output runs/neural_0
uv run taskce formula-search --profile snli --stage search --inputs-dir inputs --layer embedding_norm --ids 0 --all-examples --chunk-size 1 --device cuda --backend vectorized_search_2a --beam 40 --length 48 --max-iterations 48 --output runs/2a_0
```

Use `--ids all --min-fit-fraction 0` to include every coordinate with a positive target. With `--all-examples`, fit IoU measures overlap on all prepared examples. Omit that flag to train on the fit split and score the frozen formula on validation and test. The full 2,048-neuron comparison is in the [SNLI layer sweep](snli_layer_sweep.md).

## Inspect and replay

The CSV contains the rendered `formula`, the total and unique concept counts, and a relative `neural_trace_path` with a checksum. The JSON trace records selected indices, names and operators, prefix IoUs, `action_logits`, `attention_logits`, one-based `action_ranks`, `fit_iou_at_48`, and the stopping reason. Each logit is the selected action's score at that prefix. Rank 1 means the policy's first proposal passed the IoU check.

```python
import csv
import json
from pathlib import Path
import numpy as np
from formula_search.formula_eval import FormulaEvaluator

csv_path = Path("runs/neural_0/embedding_norm.csv")
with csv_path.open(newline="", encoding="utf-8") as stream:
    row = next(csv.DictReader(stream))
trace = json.loads((csv_path.parent / row["neural_trace_path"]).read_text(encoding="utf-8"))
with np.load("inputs/features.npz", allow_pickle=False) as data:
    prediction = FormulaEvaluator(data["names"].astype(str), data["vectors"]).evaluate(row["formula"])
for action in zip(trace["names"], trace["operations"], trace["action_logits"], trace["attention_logits"], trace["action_ranks"], trace["prefix_ious"]):
    print(action)
```

`neural_weighted` fits the dense thresholded attention predictor used in the [historical six-neuron pilot](snli_pilot.md).
