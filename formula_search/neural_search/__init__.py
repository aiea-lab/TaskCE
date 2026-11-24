"""Learn a Boolean formula from five attention-model outputs per concept."""

from .attention_formula import AttentionFormulaResult, decode_table, fit_formula

__all__ = ["AttentionFormulaResult", "decode_table", "fit_formula"]
