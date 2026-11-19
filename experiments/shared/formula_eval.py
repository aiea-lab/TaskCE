"""Evaluate rendered formulas against the named Boolean concept vectors."""

import re


class FormulaEvaluator:
    """Evaluate the search grammar with one cached lexer for all formulas."""

    def __init__(self, names, vectors):
        self.features = dict(zip(names, vectors))
        atoms = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
        self.lexer = re.compile(f"(?:{atoms})(?=$|[\\s)])|AND\\b|OR\\b|NOT\\b|[()]")

    def evaluate(self, text):
        tokens = []
        cursor = 0
        while cursor < len(text):
            if text[cursor].isspace():
                cursor += 1
                continue
            match = self.lexer.match(text, cursor)
            if match is None:
                raise ValueError(f"Unknown formula text at {text[cursor:]!r}")
            tokens.append(match.group())
            cursor = match.end()
        position = 0

        def expression():
            nonlocal position
            if position >= len(tokens):
                raise ValueError(f"Truncated formula: {text}")
            token = tokens[position]
            position += 1
            if token != "(":
                if token not in self.features:
                    raise ValueError(f"Unknown atom {token!r}")
                return self.features[token]
            if position < len(tokens) and tokens[position] == "NOT":
                position += 1
                value = ~expression()
            else:
                value = expression()
                operator = None
                while position < len(tokens) and tokens[position] != ")":
                    current = tokens[position]
                    position += 1
                    if current not in ("AND", "OR") or operator not in (None, current):
                        raise ValueError(f"Invalid operation in {text!r}")
                    operator = current
                    right = expression()
                    value = value & right if current == "AND" else value | right
            if position >= len(tokens) or tokens[position] != ")":
                raise ValueError(f"Unclosed formula: {text}")
            position += 1
            return value

        result = expression()
        if position != len(tokens):
            raise ValueError(f"Trailing formula tokens: {text}")
        return result
