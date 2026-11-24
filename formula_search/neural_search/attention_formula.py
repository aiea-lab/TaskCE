"""Fit a one-pass attention policy for left-folded Boolean formulas."""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .instructions import bool_vectors, evaluate_formula, render_formula


_OPERATIONS = ("OR", "AND", "AND NOT")
_ROLLOUTS = 24


@dataclass(frozen=True)
class AttentionFormulaResult:
    table: np.ndarray
    indices: tuple[int, ...]
    operations: tuple[str, ...]
    formula: str
    iou: float
    prefix_ious: tuple[float, ...]
    stop_reason: str

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        return evaluate_formula(vectors, self.indices, self.operations)


def decode_table(table: np.ndarray) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Read keep, order and joins directly from the learned logits."""
    kept = np.flatnonzero(table[:, 0] >= 0)
    ordered = kept[np.argsort(-table[kept, 1], kind="stable")]
    if not ordered.size:
        return (), ()
    joins = table[ordered[1:], 2:].argmax(axis=1)
    return tuple(ordered.tolist()), ("ATOM", *(_OPERATIONS[join] for join in joins))


def _csr(vectors: np.ndarray, device: torch.device) -> torch.Tensor:
    rows, columns = vectors.nonzero()
    pointers = np.empty(vectors.shape[0] + 1, dtype=np.int32)
    pointers[0] = 0
    np.cumsum(np.bincount(rows, minlength=vectors.shape[0]), out=pointers[1:])
    return torch.sparse_csr_tensor(
        torch.from_numpy(pointers).to(device),
        torch.from_numpy(columns.astype(np.int32)).to(device),
        torch.ones(columns.size, device=device),
        size=vectors.shape,
    )


def _sum_binary_rows(masks: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    # Each stored column denotes a one, so an ordered embedding-bag sum is
    # exactly the binary matrix product without atomic sparse accumulations.
    return F.embedding_bag(
        masks.col_indices(),
        weights,
        masks.crow_indices(),
        mode="sum",
        include_last_offset=True,
    )


class _ProjectMasks(torch.autograd.Function):
    @staticmethod
    def forward(ctx, masks, masks_transposed, weights):
        ctx.save_for_backward(masks_transposed)
        return _sum_binary_rows(masks, weights)

    @staticmethod
    def backward(ctx, projected_gradient):
        (masks_transposed,) = ctx.saved_tensors
        # The adjoint is the same ordered sum over transposed binary rows.
        return (
            None,
            None,
            _sum_binary_rows(masks_transposed, projected_gradient.contiguous()),
        )


class _Policy(nn.Module):
    def __init__(
        self,
        example_count: int,
        concept_count: int,
        width: int,
        readout_width: int,
        seed: int,
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        # Keep query initialization independent of the key/value initialization.
        query_generator = torch.Generator().manual_seed(seed + 92849)
        query_start = torch.randn((example_count, width), generator=query_generator)
        keys_start = torch.randn((example_count, width), generator=generator)
        values_start = torch.randn((example_count, width), generator=generator)
        self.query_weight = nn.Parameter(query_start)
        self.key_value_weight = nn.Parameter(
            torch.cat((keys_start, values_start), dim=1)
        )
        self.log_query_gain = nn.Parameter(torch.zeros(()))
        self.hidden = nn.Linear(3 * width, readout_width)
        self.output = nn.Linear(readout_width, 4)
        nn.init.xavier_uniform_(self.hidden.weight, generator=generator)
        nn.init.normal_(self.output.weight, std=0.04, generator=generator)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)
        initial_keep = min(0.5, 12 / concept_count)
        self.keep_bias = nn.Parameter(
            torch.tensor(math.log(initial_keep) - math.log1p(-initial_keep))
        )
        self.width = width

    def forward(
        self,
        masks: torch.Tensor,
        masks_transposed: torch.Tensor,
        concept_rate: torch.Tensor,
        concept_norm: torch.Tensor,
        neuron: torch.Tensor,
    ) -> torch.Tensor:
        q = F.layer_norm(neuron @ self.query_weight, (self.width,))
        projected = _ProjectMasks.apply(masks, masks_transposed, self.key_value_weight)
        projected = (
            projected - concept_rate * self.key_value_weight.sum(dim=0)
        ) / concept_norm
        k, v = projected.split(self.width, dim=1)
        k = F.layer_norm(k, (self.width,))
        attention = (k @ q / math.sqrt(self.width)) * self.log_query_gain.exp()
        context = F.layer_norm(attention.softmax(dim=0) @ v, (self.width,))
        readout = torch.cat((q.expand_as(k), k, context.expand_as(k)), dim=1)
        order_and_joins = self.output(F.gelu(self.hidden(readout)))
        return torch.cat(
            ((attention + self.keep_bias)[:, None], order_and_joins), dim=1
        )


class _TerminalIoU:
    """Score exact, complete sampled actions using packed Boolean inputs on CPU."""

    def __init__(self, vectors: np.ndarray, target: np.ndarray):
        self.concepts = np.packbits(vectors, axis=1)
        self.target = np.packbits(target)

    def score(self, indices: np.ndarray, joins: np.ndarray) -> float:
        if not indices.size:
            return 0.0
        mask = self.concepts[indices[0]].copy()
        for index, join in zip(indices[1:], joins[1:], strict=True):
            concept = self.concepts[index]
            if join == 0:
                mask |= concept
            elif join == 1:
                mask &= concept
            else:
                mask &= ~concept
        union = int.from_bytes(mask | self.target, byteorder="little").bit_count()
        intersection = int.from_bytes(
            mask & self.target, byteorder="little"
        ).bit_count()
        return intersection / union if union else 0.0


def _sample(
    table: np.ndarray, rng: np.random.Generator, reward: _TerminalIoU
) -> tuple[np.ndarray, np.ndarray]:
    kept = rng.logistic(size=(_ROLLOUTS, len(table))) < table[:, 0]
    width = int(kept.sum(axis=1).max())
    draws = np.zeros((_ROLLOUTS, width, 3), dtype=np.int64)
    scores = np.empty(_ROLLOUTS, dtype=np.float32)
    for slot, row in enumerate(kept):
        ids = np.flatnonzero(row)
        order = np.argsort(-(table[ids, 1] + rng.gumbel(size=ids.size)), kind="stable")
        indices = ids[order]
        joins = (table[indices, 2:] + rng.gumbel(size=(ids.size, 3))).argmax(axis=1)
        draws[slot, : ids.size, 0] = indices
        draws[slot, : ids.size, 1] = joins
        draws[slot, : ids.size, 2] = 1
        scores[slot] = reward.score(indices, joins)
    return draws, scores


def _log_prob(table: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    # Independent Bernoulli factors. Outside the sampled rows, every gate is off.
    indices, joins, valid = draws.unbind(dim=2)
    selected = table[indices]
    keep = F.logsigmoid(-table[:, 0]).sum() + (selected[:, :, 0] * valid).sum(dim=1)
    if not selected.shape[1]:
        return keep

    # Exact PL probability over the retained subset only. Each denominator
    # sums the scores of the selected concepts at or after that position.
    order = selected[:, :, 1].masked_fill(valid == 0, torch.finfo(table.dtype).min)
    denominator = order.flip(dims=(1,)).logcumsumexp(dim=1).flip(dims=(1,))
    permutation = ((order - denominator) * valid).sum(dim=1)

    # The first selected concept is the atom, so its sampled operator is unused.
    join = selected[:, 1:, 2:].log_softmax(dim=2)
    join = join.gather(2, joins[:, 1:, None]).squeeze(2)
    return keep + permutation + (join * valid[:, 1:]).sum(dim=1)


def _prefix_ious(
    vectors: np.ndarray,
    target: np.ndarray,
    indices: tuple[int, ...],
    operations: tuple[str, ...],
) -> tuple[float, ...]:
    prediction = np.zeros_like(target)
    scores = []
    for index, operation in zip(indices, operations, strict=True):
        concept = vectors[index]
        if operation in ("ATOM", "OR"):
            prediction |= concept
        elif operation == "AND":
            prediction &= concept
        else:
            prediction &= ~concept
        union = np.count_nonzero(prediction | target)
        scores.append(
            float(np.count_nonzero(prediction & target) / union) if union else 0.0
        )
    return tuple(scores)


def fit_formula(
    vectors: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    *,
    steps: int,
    seed: int,
    device: str,
    encoder_width: int = 96,
    readout_width: int = 32,
) -> AttentionFormulaResult:
    """Train one neuron from exact terminal rewards, then decode the final table."""
    vectors = bool_vectors(vectors)
    target = np.asarray(target)
    if target.shape != (vectors.shape[1],) or target.dtype != np.bool_:
        raise ValueError("target must be a boolean vector aligned with vectors")
    if len(names) != vectors.shape[0] or not len(names) or not target.size:
        raise ValueError("names must match a nonempty concept-by-example matrix")
    if steps < 0 or encoder_width < 1 or readout_width < 1:
        raise ValueError(
            "steps must be nonnegative and encoder and readout widths must be positive"
        )

    fit_device = torch.device(device)
    masks = _csr(vectors, fit_device)
    masks_transposed = _csr(vectors.T, fit_device)
    concept_count = torch.as_tensor(
        vectors.sum(axis=1, dtype=np.float32)[:, None], device=fit_device
    )
    concept_rate = concept_count / target.size
    concept_norm = (concept_count * (1 - concept_rate)).clamp_min(1).sqrt()
    neuron = torch.as_tensor(target.astype(np.float32), device=fit_device)
    neuron = neuron - neuron.mean()
    neuron = F.normalize(neuron, dim=0)
    policy = _Policy(
        target.size, vectors.shape[0], encoder_width, readout_width, seed
    ).to(fit_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    rng = np.random.default_rng(seed)
    reward = _TerminalIoU(vectors, target)
    policy_inputs = masks, masks_transposed, concept_rate, concept_norm, neuron
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = policy(*policy_inputs)
        draws, scores = _sample(logits.detach().cpu().numpy(), rng, reward)
        baseline = (scores.sum() - scores) / (_ROLLOUTS - 1)
        advantage = torch.from_numpy(scores - baseline).to(fit_device)
        log_prob = _log_prob(logits, torch.from_numpy(draws).to(fit_device))
        loss = -(advantage * log_prob).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

    with torch.no_grad():
        table = policy(*policy_inputs).cpu().numpy()
    indices, operations = decode_table(table)
    prefix_ious = _prefix_ious(vectors, target, indices, operations)
    return AttentionFormulaResult(
        table,
        indices,
        operations,
        render_formula(names, indices, operations),
        prefix_ious[-1] if prefix_ious else 0.0,
        prefix_ious,
        "decoded" if indices else "no_concepts_kept",
    )
