"""Optional exact prefix selector for the pinned new_search backend.

Assign ``new_search.algorithm.select_unique_topk = select_unique_topk`` before
searching to opt in. No search configuration or scoring semantics change.
Fingerprints only filter equality checks; every matching fingerprint is checked
against the complete packed bitmap before a candidate is excluded.
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'theoretical/compositional_explanations'))
from new_search.algorithm import select_unique_topk as reference_select


def _fingerprints(vectors):
    words = torch.arange(vectors.shape[-1], dtype=torch.int64, device=vectors.device)
    weights = (words + 1) * 6364136223846793005 + 1442695040888963407
    return (vectors.to(torch.int64) * weights).sum(dim=-1)


def _exclude_equal(candidates, references, possible, excluded):
    """Mark candidates with an exact reference match, bounding temporary words."""
    matches = possible.nonzero(as_tuple=False)
    for start in range(0, matches.shape[0], 8192):
        batch, candidate, reference = matches[start:start + 8192].unbind(dim=1)
        equal = (candidates[batch, candidate] == references[batch, reference]).all(dim=-1)
        # Duplicate indices only ever write True, including with CUDA indexing.
        excluded[batch[equal], candidate[equal]] = True


def select_unique_topk(candidate_scores, parent_vectors, packed_features,
                       retained_ids, beam_size):
    """Return exactly the pinned selector's scores, flat indices, and validity.

    Flat order is parent, retained feature, operation (AND, OR, AND_NOT).
    Finite candidates qualify only if their mask differs from zero, every
    atomic feature, every parent, and every earlier candidate in stable
    descending score order. Earlier nonfinite candidates also block duplicates,
    matching the original. Invalid slots use -inf and the last sorted index.
    """
    flat_scores = candidate_scores.flatten(start_dim=1)
    scores, indices = torch.sort(flat_scores, dim=1, descending=True, stable=True)
    neuron_count, candidate_count = flat_scores.shape
    feature_count = retained_ids.shape[1]
    width = parent_vectors.shape[-1]
    device = candidate_scores.device
    references = torch.cat((
        torch.zeros((neuron_count, 1, width), dtype=parent_vectors.dtype, device=device),
        packed_features.unsqueeze(0).expand(neuron_count, -1, -1),
        parent_vectors,
    ), dim=1)
    reference_hashes = _fingerprints(references)
    output_positions = torch.full((neuron_count, beam_size), candidate_count - 1,
                                  dtype=torch.int64, device=device)
    output_valid = torch.zeros_like(output_positions, dtype=torch.bool)
    pending = torch.ones(neuron_count, dtype=torch.bool, device=device)
    prefix_limit = max(512, beam_size)
    prefix = min(candidate_count, prefix_limit, max(32, beam_size * 4))
    while bool(pending.any()):
        active = pending.nonzero(as_tuple=False).flatten()
        ordered = indices[active, :prefix]
        operations = ordered % 3
        feature_slots = (ordered // 3) % feature_count
        parent_slots = ordered // (3 * feature_count)
        features = packed_features[retained_ids[active[:, None], feature_slots]]
        parents = parent_vectors[active[:, None], parent_slots]
        candidates = torch.where(
            (operations == 0).unsqueeze(-1), parents & features,
            torch.where((operations == 1).unsqueeze(-1), parents | features, parents & ~features),
        )
        hashes = _fingerprints(candidates)
        excluded = torch.zeros((len(active), prefix), dtype=torch.bool, device=device)
        possible = hashes[:, :, None] == reference_hashes[active, None, :]
        _exclude_equal(candidates, references[active], possible, excluded)
        positions = torch.arange(prefix, device=device)
        earlier = positions[:, None] > positions[None, :]
        possible = (hashes[:, :, None] == hashes[:, None, :]) & earlier
        _exclude_equal(candidates, candidates, possible, excluded)
        eligible = ~excluded & torch.isfinite(scores[active, :prefix])
        winners = torch.topk(torch.where(eligible, positions, candidate_count),
                             beam_size, dim=1, largest=False, sorted=True).values
        valid = winners < candidate_count
        # Once no finite candidates remain outside the prefix, unseen masks
        # cannot become winners, even if fewer than beam_size survived.
        more_finite = torch.isfinite(scores[active, prefix:]).any(dim=1)
        finished = valid.all(dim=1) | ~more_finite
        finished_active = active[finished]
        output_positions[finished_active] = winners[finished].clamp_max(candidate_count - 1)
        output_valid[finished_active] = valid[finished]
        pending[finished_active] = False
        if prefix == prefix_limit and bool(pending.any()):
            # Saturated or heavily duplicated scores can exhaust the prefix.
            # Fall back one neuron at a time rather than materializing a giant
            # prefix-by-prefix equality matrix across the entire batch.
            selected_scores = scores.gather(1, output_positions).masked_fill(~output_valid, -torch.inf)
            selected_indices = indices.gather(1, output_positions)
            for neuron in pending.nonzero(as_tuple=False).flatten().tolist():
                fallback = reference_select(candidate_scores[neuron:neuron + 1],
                                     parent_vectors[neuron:neuron + 1], packed_features,
                                     retained_ids[neuron:neuron + 1], beam_size)
                selected_scores[neuron], selected_indices[neuron], output_valid[neuron] = (
                    value[0] for value in fallback
                )
            return selected_scores, selected_indices, output_valid
        prefix = min(candidate_count, prefix_limit, prefix * 2)
    selected_scores = scores.gather(1, output_positions).masked_fill(~output_valid, -torch.inf)
    selected_indices = indices.gather(1, output_positions)
    return selected_scores, selected_indices, output_valid


def verify(device, cases):
    """Compare candidate indices, validity and scores with the reference selector."""
    reference = reference_select
    for seed in range(cases):
        generator = torch.Generator(device=device).manual_seed(seed)
        neurons, parents_count = 1 + seed % 4, 1 + seed % 5
        features_count, width = 2 + seed % 11, 1 + seed % 9
        retained_count = 1 + seed % features_count
        atoms = torch.randint(-(2**31), 2**31 - 1, (features_count, width),
                              dtype=torch.int32, device=device, generator=generator)
        parents = atoms[torch.randint(features_count, (neurons, parents_count),
                                      device=device, generator=generator)].clone()
        retained = torch.randint(features_count, (neurons, retained_count),
                                 device=device, generator=generator)
        scores = torch.randint(0, 5, (neurons, parents_count, retained_count, 3),
                               device=device, generator=generator).float() / 5
        if seed % 4 == 0:
            atoms.zero_()
        if seed % 5 == 0:
            parents.zero_()
        if seed % 7 == 0:
            scores.flatten()[::3] = -torch.inf
        if seed % 11 == 0:
            scores.flatten()[::5] = torch.inf
        if seed % 13 == 0:
            scores.flatten()[::7] = torch.nan
        expected = reference(scores, parents, atoms, retained, parents_count)
        observed = select_unique_topk(scores, parents, atoms, retained, parents_count)
        for index, (original, replacement) in enumerate(zip(expected, observed)):
            if not torch.equal(original, replacement):
                raise RuntimeError(f"Selector mismatch: seed={seed}, output={index}")
    # These distinct masks deliberately share the linear fingerprint zero.
    collision = torch.tensor([[[1, -2, 1]]], dtype=torch.int32, device=device)
    zero = torch.zeros_like(collision)
    excluded = torch.zeros((1, 1), dtype=torch.bool, device=device)
    possible = _fingerprints(collision)[:, :, None] == _fingerprints(zero)[:, None, :]
    if not bool(possible.all()):
        raise RuntimeError("Collision fixture no longer matches the fingerprint")
    _exclude_equal(collision, zero, possible, excluded)
    if bool(excluded.any()):
        raise RuntimeError("Fingerprint collision incorrectly excluded a distinct mask")
    saturated_scores = torch.ones((1, 2, 90, 3), device=device)
    saturated_parents = torch.zeros((1, 2, 1), dtype=torch.int32, device=device)
    saturated_atoms = torch.zeros((90, 1), dtype=torch.int32, device=device)
    saturated_ids = torch.arange(90, device=device).unsqueeze(0)
    expected = reference(saturated_scores, saturated_parents, saturated_atoms, saturated_ids, 2)
    observed = select_unique_topk(saturated_scores, saturated_parents, saturated_atoms, saturated_ids, 2)
    if not all(torch.equal(original, replacement) for original, replacement in zip(expected, observed)):
        raise RuntimeError("Saturated prefix fallback differs from original")
    print(f"Matched all three pinned outputs for {cases} synthetic cases on {device}; "
          "collision rejected; saturated fallback matched")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cases", type=int, default=160)
    options = parser.parse_args()
    verify(options.device, options.cases)
