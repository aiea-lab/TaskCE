"""Learn an attention policy for discrete, left-folded Boolean formulas."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .algorithm import _binary_vectors


_OPERATIONS = ("OR", "AND", "AND NOT")
_PROPOSALS = 16
_TOKEN_WIDTH = 8


def render_formula(
    names: Sequence[str], indices: Sequence[int], operations: Sequence[str]
) -> str:
    """Render an action prefix in the formula grammar used by the pinned evaluator."""
    if len(indices) == 0:
        return ""

    groups = []
    for index, operation in zip(indices[1:], operations[1:], strict=True):
        operator = "AND" if operation == "AND NOT" else operation
        atom = str(names[index])
        if operation == "AND NOT":
            atom = f"(NOT {atom})"
        if not groups or groups[-1][0] != operator:
            groups.append((operator, []))
        groups[-1][1].append(f" {operator} {atom}")

    parts = ["(" * len(groups), str(names[indices[0]])]
    for _, suffixes in groups:
        parts.extend(suffixes)
        parts.append(")")
    return "".join(parts)


def evaluate_formula(
    vectors: np.ndarray, indices: Sequence[int], operations: Sequence[str]
) -> np.ndarray:
    """Replay an action prefix on any examples in the same concept order."""
    vectors = _binary_vectors(vectors)
    prediction = np.zeros(vectors.shape[1], dtype=np.bool_)
    for index, operation in zip(indices, operations, strict=True):
        if operation in ("ATOM", "OR"):
            prediction |= vectors[index]
        elif operation == "AND":
            prediction &= vectors[index]
        elif operation == "AND NOT":
            prediction &= ~vectors[index]
        else:
            raise ValueError(f"Unknown formula operation {operation!r}")
    return prediction


@dataclass(frozen=True)
class FormulaResult:
    formula: str
    iou: float
    indices: tuple[int, ...]
    operations: tuple[str, ...]
    prefix_ious: tuple[float, ...]
    action_logits: tuple[float, ...]
    action_ranks: tuple[int, ...]
    attention_logits: tuple[float, ...]

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        return evaluate_formula(vectors, self.indices, self.operations)


@dataclass(frozen=True)
class FormulaSearchResult(FormulaResult):
    best_at_48: FormulaResult
    truncated_48: FormulaResult
    stop_reason: str


@dataclass(frozen=True)
class _Prefix:
    mask: torch.Tensor
    hits: int
    false_positives: int


@dataclass(frozen=True)
class _Candidates:
    keys: torch.Tensor
    state: torch.Tensor
    hits: torch.Tensor
    false_positives: torch.Tensor
    ious: torch.Tensor
    allowed: torch.Tensor
    is_atom: bool


@dataclass(frozen=True)
class _Action:
    index: int
    operation: str
    iou: float
    logit: float
    rank: int
    attention_logit: float


class _Policy(nn.Module):
    """Project the target/prefix query and attend to concept/operator keys."""

    def __init__(self, token_keys: torch.Tensor, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn((5, 4), generator=generator) * 0.05
        self.projection = nn.Parameter(projection.to(token_keys.device))
        self.register_buffer("token_keys", token_keys)
        self.token_query = nn.Parameter(
            torch.zeros((_TOKEN_WIDTH, 4), device=token_keys.device)
        )

    def forward(self, candidates: _Candidates) -> tuple[torch.Tensor, torch.Tensor]:
        query = candidates.state @ self.projection
        token_query = (
            self.token_query[:, :1] if candidates.is_atom else self.token_query[:, 1:]
        )
        attention_logits = (self.token_keys @ token_query.tanh()).flatten()
        logits = candidates.keys @ query + attention_logits
        return logits.masked_fill(~candidates.allowed, -1e9), attention_logits


class _Context:
    def __init__(
        self, vectors: np.ndarray, target: np.ndarray, device: torch.device, seed: int
    ):
        self.device = device
        self.target_count = int(target.sum())
        self.example_count = target.size
        self.masks = torch.from_numpy(np.ascontiguousarray(vectors)).to(device)
        self.target = torch.from_numpy(np.ascontiguousarray(target)).to(device)
        self.values = self.masks.float()
        self.partition = torch.stack((self.target, ~self.target), dim=1).float()
        self.concept_hits, self.concept_false_positives = (
            self.values @ self.partition
        ).unbind(dim=1)
        self.token_keys = self._project_tokens(seed)
        self.atoms = self._candidates(
            self.concept_hits[:, None],
            self.concept_false_positives[:, None],
            (self.concept_hits + self.concept_false_positives > 0)[:, None],
            0,
            0,
            is_atom=True,
        )

    def _project_tokens(self, seed: int) -> torch.Tensor:
        # Share a small signed basis over aligned example coordinates. The first
        # column also preserves the full centered neuron-to-concept correlation.
        generator = torch.Generator().manual_seed(seed)
        basis = torch.randint(
            2, (self.example_count, _TOKEN_WIDTH), generator=generator
        )
        basis = (2 * basis - 1).to(device=self.device, dtype=self.values.dtype)
        basis[:, 0] = 1.0
        centered_target = self.partition[:, :1] - self.target_count / self.example_count
        neuron_basis = basis * centered_target
        neuron_basis = neuron_basis - neuron_basis.mean(dim=0)
        neuron_basis = neuron_basis / neuron_basis.norm(dim=0).clamp_min(1e-6)
        count = self.concept_hits + self.concept_false_positives
        norm = (count * (1 - count / self.example_count)).sqrt().clamp_min(1.0)
        return (self.values @ neuron_basis) / norm[:, None]

    def _candidates(
        self, hits, false_positives, allowed, prefix_hits, prefix_fp, *, is_atom
    ):
        positive_rate = hits / self.target_count
        negative_rate = false_positives / self.target_count
        baseline_recall = prefix_hits / self.target_count
        baseline_fp = prefix_fp / self.target_count
        keys = torch.stack(
            (
                (positive_rate + 0.5 / self.target_count).log(),
                negative_rate.log1p(),
                positive_rate - baseline_recall,
                (negative_rate - baseline_fp) / (1 + baseline_fp),
            ),
            dim=-1,
        )
        state = torch.tensor(
            (
                1.0,
                self.target_count / self.example_count,
                baseline_recall,
                baseline_fp / (1 + baseline_fp),
                float(is_atom),
            ),
            device=self.device,
            dtype=keys.dtype,
        )
        return _Candidates(
            keys.flatten(0, 1),
            state,
            hits.flatten(),
            false_positives.flatten(),
            (hits / (self.target_count + false_positives)).flatten(),
            allowed.flatten(),
            is_atom,
        )

    def candidates(self, prefix: _Prefix | None) -> _Candidates:
        if prefix is None:
            return self.atoms
        shared_hits, shared_fp = (
            self.values @ (prefix.mask[:, None] * self.partition)
        ).unbind(dim=1)
        hits = torch.stack(
            (
                prefix.hits + self.concept_hits - shared_hits,
                shared_hits,
                prefix.hits - shared_hits,
            ),
            dim=1,
        )
        fp = torch.stack(
            (
                prefix.false_positives + self.concept_false_positives - shared_fp,
                shared_fp,
                prefix.false_positives - shared_fp,
            ),
            dim=1,
        )
        changed = (hits != prefix.hits) | (fp != prefix.false_positives)
        return self._candidates(
            hits, fp, changed, prefix.hits, prefix.false_positives, is_atom=False
        )

    def extend(
        self,
        prefix: _Prefix | None,
        index: int,
        operation: str,
        hits: int,
        false_positives: int,
    ) -> _Prefix:
        concept = self.masks[index]
        if prefix is None:
            mask = concept
        elif operation == "OR":
            mask = prefix.mask | concept
        elif operation == "AND":
            mask = prefix.mask & concept
        else:
            mask = prefix.mask & ~concept
        return _Prefix(mask, hits, false_positives)

    def choose(
        self,
        prefix: _Prefix | None,
        candidates: _Candidates,
        logits: torch.Tensor,
        attention_logits: torch.Tensor,
    ) -> tuple[_Prefix, _Action] | None:
        # Test only the policy's bounded proposals in descending logit order.
        # Stable sorting breaks exact ties by vocabulary order and then operator order.
        proposals = logits.argsort(descending=True, stable=True)[:_PROPOSALS]
        proposals_data = torch.stack(
            (
                proposals.to(torch.float64),
                candidates.hits[proposals],
                candidates.false_positives[proposals],
                logits[proposals],
                attention_logits[proposals],
            ),
            dim=1,
        ).tolist()
        previous_hits = prefix.hits if prefix is not None else 0
        previous_union = self.target_count + (
            prefix.false_positives if prefix is not None else 0
        )
        for rank, (action_id, hit_count, fp_count, logit, attention_logit) in enumerate(
            proposals_data, 1
        ):
            action, hits, false_positives = (
                int(action_id),
                int(hit_count),
                int(fp_count),
            )
            union = self.target_count + false_positives
            if hits * previous_union <= previous_hits * union:
                continue
            index, operation = (
                (action, "ATOM")
                if prefix is None
                else (
                    action // len(_OPERATIONS),
                    _OPERATIONS[action % len(_OPERATIONS)],
                )
            )
            return (
                self.extend(prefix, index, operation, hits, false_positives),
                _Action(index, operation, hits / union, logit, rank, attention_logit),
            )
        return None


def _loss(logits: torch.Tensor, candidates: _Candidates) -> torch.Tensor:
    log_prob = logits.log_softmax(dim=0)
    reward = candidates.ious
    # The first term maximizes expected hard IoU. The second supplies a
    # ranking signal even after the policy starts concentrating on a few moves.
    reward_scale = reward.max().clamp_min(0.01)
    scaled_reward = reward / reward_scale
    preferences = (
        (12 * scaled_reward).masked_fill(~candidates.allowed, -1e9).softmax(dim=0)
    )
    return (
        -(log_prob.exp() * scaled_reward).sum() - 0.25 * (preferences * log_prob).sum()
    )


def _result(names, actions: list[_Action], reason: str) -> FormulaSearchResult:
    def at(length):
        prefix = actions[:length]
        indices = tuple(action.index for action in prefix)
        operations = tuple(action.operation for action in prefix)
        ious = tuple(action.iou for action in prefix)
        return FormulaResult(
            render_formula(names, indices, operations),
            ious[-1] if ious else 0.0,
            indices,
            operations,
            ious,
            tuple(action.logit for action in prefix),
            tuple(action.rank for action in prefix),
            tuple(action.attention_logit for action in prefix),
        )

    full = at(len(actions))
    # Accepted moves always strictly improve IoU, so these are the same prefix.
    at_48 = at(48)
    return FormulaSearchResult(
        full.formula,
        full.iou,
        full.indices,
        full.operations,
        full.prefix_ious,
        full.action_logits,
        full.action_ranks,
        full.attention_logits,
        at_48,
        at_48,
        reason,
    )


def fit_formula(
    vectors: np.ndarray,
    target: np.ndarray,
    names: Sequence[str],
    *,
    steps: int,
    seed: int,
    device: str,
) -> FormulaSearchResult:
    """Fit a fresh policy online on one neuron, then decode one improving trajectory.

    Each step uses all concepts and fit examples. The exact same target is also
    the start state's query for every update. The policy considers up to sixteen
    logit-ranked proposals per state and accepts the first strict IoU improvement.
    Concept names can recur with any operator if they improve the current mask.
    Decoding ends at perfect IoU or when none of the proposed moves improves.
    """
    vectors = _binary_vectors(vectors)
    target = np.asarray(target)
    if target.shape != (vectors.shape[1],) or target.dtype != np.bool_:
        raise ValueError("target must be a boolean vector aligned with vectors")
    if len(names) != vectors.shape[0]:
        raise ValueError("names must contain one name per concept mask")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if not target.any():
        return _result(names, [], "no_positive_target")
    if not len(names):
        return _result(names, [], "no_concepts")

    fit_device = torch.device(device)
    context = _Context(vectors, target, fit_device, seed)
    policy = _Policy(context.token_keys, seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.075)
    generator = torch.Generator(device=fit_device).manual_seed(seed)
    prefix = None
    for _ in range(steps):
        candidates = context.candidates(prefix)
        logits, attention_logits = policy(candidates)
        loss = _loss(logits, candidates)
        if prefix is not None:
            atom_logits, _ = policy(context.atoms)
            loss = loss + 0.25 * _loss(atom_logits, context.atoms)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            noise = (
                torch.rand(logits.shape, generator=generator, device=fit_device) * 0.15
            )
            move = context.choose(prefix, candidates, logits + noise, attention_logits)
            prefix = move[0] if move is not None else None
            if (
                prefix is not None
                and prefix.hits == context.target_count
                and prefix.false_positives == 0
            ):
                prefix = None

    prefix = None
    actions = []
    with torch.no_grad():
        while True:
            candidates = context.candidates(prefix)
            logits, attention_logits = policy(candidates)
            move = context.choose(prefix, candidates, logits, attention_logits)
            if move is None:
                reason = "no_improving_proposal"
                break
            prefix, action = move
            actions.append(action)
            if prefix.hits == context.target_count and prefix.false_positives == 0:
                reason = "perfect_iou"
                break

    return _result(names, actions, reason)
