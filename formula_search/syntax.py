"""Canonical Boolean formula grammar: lexer, parser, renderer, subtrees, and evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import re
from typing import Any

import numpy as np


def make_lexer(names: Iterable[str]) -> re.Pattern[str]:
    """Compile a longest-first atom lexer for the formula grammar."""
    atoms = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(f"(?:{atoms})(?=$|[\\s)])|AND\\b|OR\\b|NOT\\b|[()]")


def parse(text: str, lexer: re.Pattern[str], names: Mapping[str, Any] | set[str]) -> tuple[Any, ...]:
    """Parse a formula string into a canonical sorted-children AST tuple."""
    tokens: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor].isspace():
            cursor += 1
            continue
        match = lexer.match(text, cursor)
        if match is None:
            raise ValueError(f"unknown formula text at {text[cursor:]!r}")
        tokens.append(match.group())
        cursor = match.end()
    pos = 0

    def expression() -> tuple[Any, ...]:
        nonlocal pos
        if pos >= len(tokens):
            raise ValueError("unexpected end of formula")
        token = tokens[pos]
        pos += 1
        if token != "(":
            if token not in names:
                raise ValueError(f"unexpected token {token!r}")
            return ("atom", token)
        if pos < len(tokens) and tokens[pos] == "NOT":
            pos += 1
            child = expression()
            if pos >= len(tokens) or tokens[pos] != ")":
                raise ValueError("expected closing parenthesis after NOT")
            pos += 1
            return ("NOT", child)
        children = [expression()]
        if pos >= len(tokens) or tokens[pos] not in {"AND", "OR"}:
            raise ValueError("expected AND or OR")
        op = tokens[pos]
        while pos < len(tokens) and tokens[pos] == op:
            pos += 1
            children.append(expression())
        if pos >= len(tokens) or tokens[pos] != ")":
            raise ValueError("expected closing parenthesis")
        pos += 1
        return (op, *sorted(children, key=render))

    root = expression()
    if pos != len(tokens):
        raise ValueError(f"unexpected token {tokens[pos]!r}")
    return root


def render(node: tuple[Any, ...]) -> str:
    """Render a canonical AST tuple back to formula text."""
    if node[0] == "atom":
        return str(node[1])
    if node[0] == "NOT":
        return f"(NOT {render(node[1])})"
    return "(" + f" {node[0]} ".join(map(render, node[1:])) + ")"


def subtrees(node: tuple[Any, ...]) -> Iterator[tuple[Any, ...]]:
    """Yield every subtree of a formula AST rooted at `node`."""
    yield node
    if node[0] != "atom":
        for child in node[1:]:
            yield from subtrees(child)


def evaluate(node: tuple[Any, ...], features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Evaluate a parsed formula AST against named Boolean concept vectors."""
    op = node[0]
    if op == "atom":
        if node[1] not in features:
            raise ValueError(f"formula atom missing from features: {node[1]}")
        return features[node[1]]
    if op == "NOT":
        return ~evaluate(node[1], features)
    values = [evaluate(child, features) for child in node[1:]]
    return (np.logical_and if op == "AND" else np.logical_or).reduce(values)


def iou(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Compute Boolean intersection-over-union."""
    union = np.count_nonzero(prediction | truth)
    return float(np.count_nonzero(prediction & truth) / union) if union else 0.0
