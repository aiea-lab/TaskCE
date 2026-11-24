"""Generate a left-folded Boolean formula by walking a self-attention graph."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .attention_formula import _TerminalIoU, _prefix_ious
from .instructions import bool_vectors, evaluate_formula, render_formula

_OPS = ("OR", "AND", "AND NOT")
_ROLLOUTS = 24


@dataclass(frozen=True)
class SelfFormulaResult:
    table: np.ndarray
    queries: np.ndarray
    keys: np.ndarray
    positions: np.ndarray
    keep_bias: float
    mask_mode: str
    kept_count: int
    indices: tuple[int, ...]
    operations: tuple[str, ...]
    formula: str
    iou: float
    prefix_ious: tuple[float, ...]
    stop_reason: str

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        return evaluate_formula(vectors, self.indices, self.operations)


def _layout(concepts, seed):
    positions = np.arange(concepts + 1)
    if seed is not None:
        ordering = np.random.default_rng(seed).permutation(concepts)
        positions[ordering] = np.arange(concepts)
    return positions


def decode_path(table, queries, keys, positions, mask_mode):
    """Walk strongest reachable edges. Arrays use original concept IDs, neuron last."""
    if mask_mode not in ("causal", "bidirectional"):
        raise ValueError("mask_mode must be causal or bidirectional")
    remaining = np.flatnonzero(table[:, 0] >= 0)
    current = len(table)
    indices = []
    while remaining.size:
        if mask_mode == "causal":
            remaining = remaining[positions[remaining] < positions[current]]
            if not remaining.size:
                break
        # remaining stays in original ID order, including after a layout permutation.
        winner = int(np.argmax(keys[remaining] @ queries[current]))
        current = int(remaining[winner])
        indices.append(current)
        remaining = np.delete(remaining, winner)
    joins = table[indices[1:], 1:].argmax(axis=1)
    operations = ("ATOM", *(_OPS[j] for j in joins)) if indices else ()
    return tuple(indices), operations


class _SelfPolicy(nn.Module):
    def __init__(
        self, examples: int, concepts: int, width: int, hidden: int, seed: int
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        query_rng = torch.Generator().manual_seed(seed + 92849)
        weights = [
            torch.randn((examples, width), generator=g)
            for g in (query_rng, generator, generator)
        ]
        self.projections = nn.Parameter(torch.cat(weights, dim=1))
        self.log_query_gain = nn.Parameter(torch.zeros(()))
        p = min(0.5, 12 / concepts)
        self.keep_bias = nn.Parameter(torch.tensor(math.log(p) - math.log1p(-p)))
        self.hidden = nn.Linear(3 * width, hidden)
        self.operator = nn.Linear(hidden, 3)
        nn.init.xavier_uniform_(self.hidden.weight, generator=generator)
        nn.init.normal_(self.operator.weight, std=0.04, generator=generator)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.operator.bias)
        self.width = width

    def project(self, tokens):
        raw_q, raw_k, values = (tokens @ self.projections).split(self.width, dim=1)
        q = F.layer_norm(raw_q, (self.width,))
        keys = F.layer_norm(raw_k, (self.width,))
        queries = q * (self.log_query_gain.exp() / math.sqrt(self.width))
        return q, queries, keys, values

    def operators(self, q, queries, keys, values, rows, positions, causal):
        # An operator logit only needs this token's attention row for one layer.
        # Each requested row still attends to every legal key in the context.
        scores = queries[rows] @ keys.T
        if causal:
            scores = scores.masked_fill(
                positions[rows, None] < positions[None, :], -torch.inf
            )
        context = F.layer_norm(scores.softmax(1) @ values, (self.width,))
        features = torch.cat((q[rows], keys[rows], context), dim=1)
        return self.operator(F.gelu(self.hidden(features)))


def _draw_paths(ids, lengths, edge_logits, op_logits, positions, causal, rng, reward):
    """Sample from each row over unvisited reachable members of that rollout's gate set."""
    batch_size, width = ids.shape
    visited = np.zeros_like(ids)
    joins = np.zeros_like(ids)
    path_lengths = np.zeros(batch_size, dtype=np.int64)
    rewards = np.zeros(batch_size, dtype=np.float32)
    for b, length in enumerate(lengths):
        remaining = np.arange(length)
        parent = width  # The neuron's factors occupy the appended parent slot.
        while remaining.size:
            logits = edge_logits[b, parent, remaining]
            winner = int(np.argmax(logits + rng.gumbel(size=len(logits))))
            parent = remaining[winner]
            visited[b, path_lengths[b]] = parent
            path_lengths[b] += 1
            remaining = np.delete(remaining, winner)
            if causal:
                remaining = remaining[
                    positions[ids[b, remaining]] < positions[ids[b, parent]]
                ]
        selected = visited[b, : path_lengths[b]]
        if len(selected) > 1:
            op_scores = op_logits[b, selected[1:]]
            joins[b, 1 : len(selected)] = (
                op_scores + rng.gumbel(size=op_scores.shape)
            ).argmax(1)
        rewards[b] = reward.score(ids[b, selected], joins[b, : len(selected)])
    return visited, joins, path_lengths, rewards


def _path_log_prob(
    keep, ids, lengths, edges, ops, visited, joins, path_lengths, positions, causal
):
    """Log probability of the sampled gates (including stranded ones), edges and used joins."""
    batch, width = ids.shape
    slots = torch.arange(width, device=ids.device)
    membership = slots < lengths[:, None]
    log_gate = F.logsigmoid(-keep).sum() + (keep[ids] * membership).sum(1)
    if not width:
        return log_gate
    which = torch.arange(batch, device=ids.device)[:, None]
    active = slots < path_lengths[:, None]
    start = torch.full_like(visited[:, :1], width)
    parent = torch.cat((start, visited[:, :-1]), dim=1)
    score = edges[which, parent]

    # Candidates visited earlier are ineligible at this step. Unvisited gated tokens
    # retain the sentinel rank, even when the causal path ends before reaching them.
    rank = torch.full_like(ids, width)
    valid_batch, valid_step = active.nonzero(as_tuple=True)
    rank[valid_batch, visited[valid_batch, valid_step]] = valid_step
    allowed = membership[:, None, :] & (rank[:, None, :] >= slots[None, :, None])
    if causal:
        candidate_positions = positions[ids]
        parent_positions = torch.cat(
            (
                positions[-1].expand(batch, 1),
                candidate_positions[which, visited[:, :-1]],
            ),
            1,
        )
        allowed = allowed & (
            candidate_positions[:, None, :] < parent_positions[:, :, None]
        )
    # For padding-only steps give the unused softmax a finite denominator.
    score = score.masked_fill(~(allowed | ~active[:, :, None]), -torch.inf)
    edge = score.gather(2, visited[:, :, None]).squeeze(2) - score.logsumexp(2)
    join = ops[which, visited].log_softmax(2).gather(2, joins[:, :, None]).squeeze(2)
    return (
        log_gate
        + torch.where(active, edge, 0.0).sum(1)
        + (join[:, 1:] * active[:, 1:]).sum(1)
    )


def _draw_update(policy, factors, keep, positions, layout, causal, rng, reward):
    q, queries, keys, values = factors
    keep_cpu = keep.detach().cpu().numpy()
    gates = rng.logistic(size=(_ROLLOUTS, len(keep))) < keep_cpu
    lengths = gates.sum(1)
    width = int(lengths.max())
    if not width:
        # Every sampled formula is empty, so every leave-one-out advantage is zero.
        return keep.sum() * 0.0
    ids = np.zeros((_ROLLOUTS, width), dtype=np.int64)
    for b, row in enumerate(gates):
        ids[b, : lengths[b]] = np.flatnonzero(row)
    rows, inverse = np.unique(ids, return_inverse=True)
    device = keep.device

    def as_long(array):
        return torch.as_tensor(array, dtype=torch.long, device=device)

    chosen = as_long(ids)
    # Only a gated token can be visited and receive operator credit. Compute the
    # union of needed attention rows once for all rollouts without altering logits.
    ops = policy.operators(q, queries, keys, values, as_long(rows), positions, causal)
    ops = ops[as_long(inverse.reshape(ids.shape))]
    parents = torch.cat((queries[chosen], queries[-1].expand(_ROLLOUTS, 1, -1)), dim=1)
    edges = parents @ keys[chosen].transpose(1, 2)
    visited, joins, path_lengths, rewards = _draw_paths(
        ids,
        lengths,
        edges.detach().cpu().numpy(),
        ops.detach().cpu().numpy(),
        layout,
        causal,
        rng,
        reward,
    )
    log_prob = _path_log_prob(
        keep,
        chosen,
        as_long(lengths),
        edges,
        ops,
        as_long(visited),
        as_long(joins),
        as_long(path_lengths),
        positions,
        causal,
    )
    baseline = (rewards.sum() - rewards) / (_ROLLOUTS - 1)
    advantage = torch.as_tensor(rewards - baseline, device=device)
    return -(advantage * log_prob).mean()


def fit_self_formula(
    vectors: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    *,
    steps: int,
    seed: int,
    device: str,
    encoder_width: int,
    readout_width: int,
    mask_mode: str,
    layout_seed: int | None = None,
    learning_rate: float = 0.01,
) -> SelfFormulaResult:
    """Train on completed formula IoUs, then follow one deterministic attention path."""
    vectors = bool_vectors(vectors)
    target = np.asarray(target)
    if target.shape != (vectors.shape[1],) or target.dtype != np.bool_:
        raise ValueError("target must be a boolean vector aligned with vectors")
    if len(names) != vectors.shape[0] or not len(names) or not target.size:
        raise ValueError("names must match a nonempty concept-by-example matrix")
    if steps < 0 or encoder_width < 1 or readout_width < 1:
        raise ValueError("invalid training budget or model width")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if mask_mode not in ("causal", "bidirectional"):
        raise ValueError("mask_mode must be causal or bidirectional")
    fit_device = torch.device(device)
    layout = _layout(len(names), layout_seed)
    positions = torch.as_tensor(layout, device=fit_device)
    # Column subsetting can return a Fortran-ordered array. Normalize strides
    # before any reductions so equivalent inputs follow the same GPU numerics.
    tokens = torch.as_tensor(
        np.vstack((vectors, target)), device=fit_device, dtype=torch.float32
    ).contiguous()
    tokens -= tokens.mean(dim=1, keepdim=True)
    tokens = F.normalize(tokens, dim=1)
    model = _SelfPolicy(target.size, len(names), encoder_width, readout_width, seed).to(
        fit_device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(seed)
    reward = _TerminalIoU(vectors, target)
    causal = mask_mode == "causal"
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        factors = model.project(tokens)
        _, queries, keys, _ = factors
        keep = keys[:-1] @ queries[-1] + model.keep_bias
        loss = _draw_update(
            model, factors, keep, positions, layout, causal, rng, reward
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        q, queries, keys, values = model.project(tokens)
        keep = keys[:-1] @ queries[-1] + model.keep_bias
        rows = torch.arange(len(names), device=fit_device)
        ops = torch.cat(
            [
                model.operators(q, queries, keys, values, chunk, positions, causal)
                for chunk in rows.split(256)
            ]
        )
        table = torch.cat((keep[:, None], ops), dim=1).cpu().numpy()
        queries, keys = queries.cpu().numpy(), keys.cpu().numpy()
        bias = model.keep_bias.item()
    indices, operations = decode_path(table, queries, keys, layout, mask_mode)
    prefix_ious = _prefix_ious(vectors, target, indices, operations)
    kept_count = int(np.count_nonzero(table[:, 0] >= 0))
    reason = (
        "no_concepts_kept"
        if not kept_count
        else (
            "no_reachable_kept_concept"
            if len(indices) < kept_count
            else "all_kept_visited"
        )
    )
    return SelfFormulaResult(
        table,
        queries,
        keys,
        layout,
        bias,
        mask_mode,
        kept_count,
        indices,
        operations,
        render_formula(names, indices, operations),
        prefix_ious[-1] if prefix_ious else 0.0,
        prefix_ious,
        reason,
    )
