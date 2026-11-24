"""Fit a thresholded attention readout of binary concept presences."""

from dataclasses import dataclass
import io
import math
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class AttentionExplanation:
    """A frozen attention distribution and cutoff for replay on new examples."""

    weights: np.ndarray
    threshold: float
    fit_iou: float

    def to_npz(self, names: np.ndarray) -> bytes:
        """Serialize the frozen predictor with its ordered concept vocabulary."""
        stream = io.BytesIO()
        np.savez(
            stream,
            names=names,
            weights=self.weights,
            threshold=self.threshold,
            fit_iou=self.fit_iou,
        )
        return stream.getvalue()

    @classmethod
    def load(cls, path: Path, names: np.ndarray) -> "AttentionExplanation":
        """Restore an explanation for a feature matrix with these concept names."""
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(saved["names"], names):
                raise ValueError(
                    "Concept names or order differ from the fitted explanation"
                )
            return cls(
                saved["weights"], float(saved["threshold"]), float(saved["fit_iou"])
            )

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """Predict with the original concept order, regardless of example order."""
        vectors = _binary_vectors(vectors)
        if vectors.shape[0] != self.weights.size:
            raise ValueError("vectors must contain one row per attention weight")
        return _scores(self.weights, vectors) >= self.threshold


def _binary_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.dtype != np.bool_:
        raise ValueError("vectors must be a two-dimensional boolean array")
    return vectors


def _scores(weights: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    scores = np.empty(vectors.shape[1], dtype=np.float64)
    for start in range(0, scores.size, 256):
        batch = vectors[:, start : start + 256].T
        terms = np.multiply(batch, weights, dtype=np.float64, order="C")
        scores[start : start + len(batch)] = terms.sum(axis=1)
    return scores


def _fit_cutoff(scores: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    true_positives = np.cumsum(target[order])
    union = target.sum() + np.arange(1, target.size + 1) - true_positives
    group_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    ious = true_positives[group_ends] / np.maximum(union[group_ends], 1)
    best = int(ious.argmax())
    end = group_ends[best]
    cutoff = sorted_scores[end]
    if end + 1 < sorted_scores.size:
        lower = sorted_scores[end + 1]
        cutoff = max((lower + cutoff) / 2, np.nextafter(lower, cutoff))
    return float(cutoff), float(ious[best])


def _straight_through_iou(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        sorted_scores, order = scores.sort(descending=True)
        positives = target[order].cumsum(dim=0)
        count = torch.arange(1, scores.numel() + 1, device=scores.device)
        ious = positives / (target.sum() + count - positives)
        ties = sorted_scores[:-1] == sorted_scores[1:]
        ious[:-1].masked_fill_(ties, -1.0)
        end = ious.argmax().reshape(1)
        upper = sorted_scores.gather(0, end)
        lower = sorted_scores.gather(0, (end + 1).clamp_max(scores.numel() - 1))
        midpoint = (lower + upper) * 0.5
        cutoff = torch.where(midpoint > lower, midpoint, upper)
        scale = (0.25 * scores.std(unbiased=False)).clamp_min(1e-6)
        hard = (scores >= cutoff).to(scores.dtype)

    logits = (scores - cutoff) / scale
    soft = torch.where(
        target.bool(),
        -torch.nn.functional.softplus(-logits),
        torch.nn.functional.softplus(logits),
    )
    prediction = hard + (soft - soft.detach())
    intersection = (prediction * target).sum()
    union = (prediction + target - prediction * target).sum()
    return intersection / union


def fit_neuron(
    vectors: np.ndarray,
    target: np.ndarray,
    *,
    steps: int,
    seed: int,
    device: str,
) -> AttentionExplanation:
    """Attend from the neuron's firing mask to the complete concept masks.

    Learn a diagonal query projection on the same fit examples at every step.
    Concept mask similarities determine the weights, and the original concept
    masks supply the values. Each update uses the best hard cutoff for forward
    IoU, with logistic gradients in the backward pass.
    """
    vectors = _binary_vectors(vectors)
    target = np.asarray(target)
    concept_count, example_count = vectors.shape
    if target.shape != (example_count,) or target.dtype != np.bool_:
        raise ValueError("target must be a boolean vector aligned with vectors")
    if concept_count == 0 or example_count == 0:
        raise ValueError("at least one concept and one fit example are required")
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if not target.any() or target.all():
        weights = np.full(concept_count, 1 / concept_count, dtype=np.float32)
        if not target.any():
            return AttentionExplanation(weights, threshold=2.0, fit_iou=0.0)
        return AttentionExplanation(weights, threshold=0.0, fit_iou=1.0)

    rng = np.random.default_rng(seed)
    fit_device = torch.device(device)
    gain_init = 1.0 + 0.001 * rng.standard_normal(example_count, dtype=np.float32)
    query_gain = torch.nn.Parameter(torch.as_tensor(gain_init, device=fit_device))
    log_scale = torch.nn.Parameter(
        torch.tensor(math.log(math.sqrt(min(example_count, 64))), device=fit_device)
    )
    optimizer = torch.optim.Adam([query_gain, log_scale], lr=0.05)

    concept_masks = torch.tensor(
        np.ascontiguousarray(vectors), dtype=torch.float32, device=fit_device
    )
    firing_mask = torch.tensor(
        np.ascontiguousarray(target), dtype=torch.float32, device=fit_device
    )
    centered_firing = firing_mask - firing_mask.mean()
    concept_totals = concept_masks.sum(dim=1)
    concept_norms = (concept_totals * (1.0 - concept_totals / example_count)).sqrt()
    constant_concepts = concept_norms == 0
    concept_norms = concept_norms.masked_fill(constant_concepts, 1.0)

    def attention_weights() -> torch.Tensor:
        query = centered_firing * query_gain
        query = query - query.mean()
        query = query / query.norm().clamp_min(1e-6)
        similarities = (concept_masks @ query) / concept_norms
        similarities = similarities.masked_fill(constant_concepts, 0.0)
        scale = log_scale.clamp(max=math.log(128.0)).exp()
        return (scale * similarities).softmax(dim=0)

    best_iou = -1.0
    for step in range(steps + 1):
        optimizer.zero_grad(set_to_none=True)
        weights = attention_weights()
        scores = concept_masks.T @ weights
        iou = _straight_through_iou(scores, firing_mask)
        achieved = float(iou.detach())
        if achieved > best_iou:
            best_iou = achieved
            best_weights = weights.detach()
        if step == steps:
            break
        loss = 1.0 - iou
        loss.backward()
        optimizer.step()

    fitted_weights = best_weights.cpu().numpy()
    threshold, fit_iou = _fit_cutoff(_scores(fitted_weights, vectors), target)
    return AttentionExplanation(fitted_weights, threshold, fit_iou)
