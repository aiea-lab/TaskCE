# SNLI embedding normalization weighted-predictor pilot

These six-neuron results use the dense weighted predictor, now named `neural_weighted`. The discrete `neural_search` backend selects named Boolean formulas. See the [current method](README.md) and the [full-layer comparison](snli_layer_sweep.md).

On six neurons, neural search reached mean IoU 0.2221 after 256 updates on all 10,000 examples. Pinned vectorized search 2.a reached 0.1758. Neural search led on four neurons. Neuron 3 favored 2.a by 0.0952 IoU.

## All-example search results

| Original neuron ID | Initial attention | Neural, 48 updates | Neural, 256 updates | Vectorized 2.a |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0543 | 0.1562 | 0.2149 | 0.1567 |
| 1 | 0.0517 | 0.2507 | 0.2938 | 0.1622 |
| 2 | 0.1232 | 0.1240 | 0.1240 | 0.1462 |
| 3 | 0.0610 | 0.1469 | 0.1609 | 0.2561 |
| 4 | 0.0907 | 0.2394 | 0.2706 | 0.1752 |
| 5 | 0.0615 | 0.2069 | 0.2686 | 0.1582 |
| **Mean IoU** | **0.0737** | **0.1873** | **0.2221** | **0.1758** |

The capture comes from the SNLI LoRA adapter at checkpoint 882 for `LiquidAI/LFM2.5-1.2B-Thinking`. The measured layer is `model.embedding_norm` at the final token. IDs 0 through 5 were chosen before scoring. Each neural context contained one neuron token and 3,854 concept vectors (1,971 premise, 1,883 hypothesis). Both backends used those concepts and the same 94.5th-percentile firing threshold. IoU is the intersection of the predicted and measured masks divided by their union.

Initial attention used the untrained query projection and a calibrated cutoff. Each 48- or 256-update search began from the same initialization and saw the full context at every update. The pinned reference used the SNLI profile's beam of 40, formula length limit of 48 and iteration limit of 48. Replaying the saved neural weights or evaluating the saved symbolic formula reproduced every reported intersection and prediction count.

## Single-neuron wall time

| Search | Median across six neurons | Range |
| --- | ---: | ---: |
| Neural, 48 updates | 0.74 s | 0.72 to 0.80 s |
| Neural, 256 updates | 1.11 s | 1.10 to 1.17 s |
| Vectorized 2.a | 3.91 s | 3.07 to 4.15 s |

All runs used the same Nautilus Kubernetes pod and NVIDIA GeForce RTX 3090 with PyTorch 2.6.0, CUDA 12.4 and Triton 3.2.0. After a warm-up on neuron 6, each timed executor call processed one neuron. The comparison alternated backend order across the six IDs. Timing included reading node-local inputs, fitting, scoring and writing artifacts, with CUDA synchronization at both ends. The independent replay ran outside the timer. Five runs of neuron 0 gave medians of 0.71 s for 48 neural updates and 3.77 s for 2.a. A separate profile showed that input decoding and fitting accounted for much of the neural run. Deduplicating candidate masks accounted for 3.0 s of the profiled 2.a run.

## Held-out check

In a separate run, both searches used the prepared premise-grouped split of 6,001 fit, 1,917 validation and 2,082 test examples. The 48-update neural model averaged 0.2508 fit IoU and 0.0635 test IoU. Pinned 2.a averaged 0.2258 fit and 0.0668 test. The all-example table measures search quality on this capture. The six IDs cover a small part of the layer's 2,048 coordinates. The capture provenance infers row alignment from the capture pipeline and a layer-8 mask check, with no independent per-example IDs in the raw activation tensor.
