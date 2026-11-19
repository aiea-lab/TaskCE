# Vectorized search

```mermaid
---
title: "Which recipes enter the next beam?"
---
stateDiagram-v2
  s0: Concept masks
  s1: Current beam
  s2: Search finished
  s3: Parents below the length limit
  s4: All candidate recipes
  s5: First recipe for each bitmap
  s6: Winners by score and generation order
  s7: Formulas and masks for the next beam
  [*] --> s0
  s0 --> s1: seed with positive-scoring concepts
  s1 --> s2: stop [empty beam or iteration limit]
  s1 --> s3: visit the beam and filter parents
  s3 --> s4: combine each parent with each concept
  s3 --> s1: clear the beam [no eligible parent]
  s4 --> s5: keep the earliest row for each bitmap
  s5 --> s6: retain at most the beam width
  s6 --> s7: decode each winning recipe
  s7 --> s1: advance one iteration
```

To profile the existing first-neuron search, run from the repository root:

```sh
python formula_search/profile_first_neuron.py --activation <activation.pt> --output-dir <profile-output> --profile
```

The CLI prepares the configured SNLI inputs, then profiles the complete `search_all` call on CUDA, including host-to-device conversion and the final CUDA synchronization. It writes the source hash and Git HEAD with `result.json`, plus HTML, JSON, text, and Speedscope profiles. Install the optional sampler with `uv pip install pyinstrument==5.1.3`; without `--profile`, Pyinstrument is not imported. This is a host wall-clock profile and does not break down individual GPU kernels.
