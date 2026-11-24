# Reconstruct a neuron from concept masks

The context for one search is the measured firing mask of one neuron followed by every binary concept mask. Each token is a vector indexed by the same examples. The neuron supplies the attention query. The concept masks supply the keys and values. This starter model has one attention readout and zero transformer blocks.

The sweep thresholds that neuron's final-token activations at the configured quantile. Attention compares its centered firing mask with each centered concept mask by cosine similarity. The model learns one gain per example in the query and a shared scale for the softmax. Each concept's resulting probability is its contribution wherever that concept is present.

Every update sees the same tokens and all of the same examples. The model finds the firing cutoff with the best hard intersection over union (IoU) and minimizes `1 - IoU`. A straight-through estimator supplies gradients. The search retains the highest training IoU it sees, recalibrates the cutoff for the saved weights, and resets the parameters and optimizer for the next neuron.

## Run a comparison

Use the same prepared inputs, layer, IDs and device for both commands. On the SNLI profile the neural search defaults to `embedding_norm` and 48 updates. Override the training budget with `--max-iterations`.

```sh
uv run taskce formula-search --profile snli --stage search --layer embedding_norm --ids 0 --all-examples --chunk-size 1 --neuron-batch-size 1 --device cuda --backend neural_search
uv run taskce formula-search --profile snli --stage search --layer embedding_norm --ids 0 --all-examples --chunk-size 1 --neuron-batch-size 1 --device cuda --backend vectorized_search_2a
```

With `--all-examples`, `fit_iou` in `embedding_norm.csv` is the final overlap across all 10,000 SNLI examples. Omit the flag to fit on the prepared fit rows and score the frozen predictor on validation and test. The stage record in `run_meta.json` includes input reads, fitting, scoring and artifact writes. An invocation with one target gives a single-neuron stage time.

## Replay a saved neuron

The neural CSV identifies the predictor in its `attention_path` column. Its NPZ contains every concept name and probability in the prepared feature order. The predicted neuron fires when the sum of the present concepts' probabilities reaches the stored cutoff. Positive additive weights can express some conjunctions and disjunctions. They cannot encode inhibition by concept presence.

```python
import csv
from pathlib import Path

import numpy as np
from formula_search.neural_search import AttentionExplanation

csv_path = Path("run/embedding_norm.csv")
with csv_path.open(newline="", encoding="utf-8") as stream:
    row = next(r for r in csv.DictReader(stream) if r["status"] == "ok")

with np.load("inputs/features.npz", allow_pickle=False) as features:
    names = features["names"]
    model = AttentionExplanation.load(csv_path.parent / row["attention_path"], names)
    predicted_mask = model.predict(features["vectors"])
    for index in np.argsort(-model.weights)[:10]:
        print(names[index], float(model.weights[index]))
```
