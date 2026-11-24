# Generating neuron formulas with attention

`--attention-mode cross` produces five scores per concept. With `--attention-mode bidirectional` or `causal`, the model uses a self-attention path for the order and produces four scores per concept.

The [SNLI experiment](attention_experiment.md) compares decoder choices, model widths, and all 2,048 final-layer neurons against pinned vectorized search 2.a on Titan Xp GPUs.

## Cross-attention

The context contains the firing mask of one neuron and a binary mask for each named concept, all aligned to the same examples. On the SNLI capture these inputs have 10,000 coordinates per token. Each neuron gets a fresh model and optimizer.

A learned query from the neuron attends to learned keys and values from the concepts. A learned scalar adjusts the query's contribution to the attention scores. For each concept, its raw attention score plus a learned bias supplies the keep logit. The shared readout consumes its key, the neuron query and the attention-weighted value summary. It predicts an order score and three operator logits. Encoder width controls the dimension of Q, K and V. Readout width controls the hidden dimension used to produce the last four columns.

The concept projections sum learned weights at the active example coordinates in a fixed order. Backpropagation uses the same sum over the transposed concept masks.

| Column | Meaning at inference |
| --- | --- |
| 0 | Include if the keep logit is at least zero |
| 1 | Sort included concepts by descending order score, breaking ties by input ID |
| 2–4 | Pick the largest of the OR, AND and AND NOT logits in that order |

The first included ID starts the expression. For each subsequent ID, fold its chosen operator and its original concept mask into the running expression. An empty selection predicts an all-false mask. For example, IDs for dog, cat and leash in that order with OR at cat and AND NOT at leash yield `((dog OR cat) AND (NOT leash))`. Each ID occurs at most once. The selected count determines the length.

## Self-attention paths

The window contains every concept, followed by the neuron. All tokens use shared learned Q, K and V projections. The neuron-to-concept logit plus a learned bias determines inclusion. Each concept attends to the legal values in the window and uses the resulting context for its three operator logits. The columns are keep, OR, AND and AND NOT.

Start the path at the neuron token. Among the kept, unvisited concepts reachable from the current token, take the strongest Q/K attention edge. Move to that concept and read its own attention row to choose the next one. The first concept seeds the formula. Each later concept contributes its own highest-scoring operator. Ties use original concept ID. The saved keys, scaled queries, IDs and positions reproduce the walk.

Bidirectional attention can visit every kept concept. Causal attention permits an edge to a previous token. Once the path moves early in the window, it may strand kept concepts in later positions. The neuron occupies the final position so its initial query can see the whole vocabulary. Use `--layout-seed 1729` to permute the concept positions while preserving their original IDs and masks. The models use content projections and a visibility mask. Bidirectional scoring is invariant to this position permutation.

## Training

Each update samples 24 complete formulas. Independent Bernoulli decisions choose the kept concepts. Cross-attention samples a weighted ordering of the retained IDs. Self-attention samples the next edge from the current token among the remaining reachable IDs. Categorical decisions supply the operators for the visited concepts after the first. The exact IoU of each completed binary formula supplies the reward. A policy gradient uses the mean reward of the other 23 samples as its baseline. Final evaluation decodes the trained outputs deterministically.

## Search and replay

Pass the same prepared inputs and target IDs to both searches. These commands search neuron 0 in the last embedding normalization layer. Use `--ids all --min-fit-fraction 0` for all nonempty target masks.

```sh
uv run taskce formula-search --profile snli --stage search --inputs-dir inputs --layer embedding_norm --ids 0 --backend neural_search --encoder-width 96 --readout-width 32 --all-examples --max-iterations 256 --chunk-size 1 --device cuda --output runs/attention
uv run taskce formula-search --profile snli --stage search --inputs-dir inputs --layer embedding_norm --ids 0 --backend vectorized_search_2a --beam 40 --length 48 --max-iterations 48 --all-examples --chunk-size 1 --device cuda --output runs/2a
```

The layer CSV records the rendered formula, its actual IoU and the score of its first 48 terms. The trace identifies the selected concepts and operators. The accompanying NPZ preserves the outputs for every concept in vocabulary order, including skipped concepts. Self-attention runs also retain the factors and positions for replaying every edge choice. The artifact checksums validate resume.

The [historical greedy search](snli_layer_sweep.md) and [historical weighted predictor](snli_pilot.md) have their own results and pinned source revisions.
