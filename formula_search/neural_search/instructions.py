"""Render and replay ordered concept IDs and Boolean operations."""

from collections.abc import Sequence

import numpy as np


def bool_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.dtype != np.bool_:
        raise ValueError("vectors must be a two-dimensional boolean array")
    return vectors


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
    vectors = bool_vectors(vectors)
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
