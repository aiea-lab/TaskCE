"""Reproduce CoEx inference and expose a minimal library interface for Bowman SNLI."""

from .api import (
    DEFAULT_LAYERS,
    LABELS,
    MODEL_ID,
    PAD_ID,
    BowmanBatch,
    BowmanBundle,
    BowmanTokenizer,
    EncodedPair,
    HookableLayer,
    capture_activations,
    classifier_weights,
    download_checkpoint,
    extract_activations,
    inference_precision,
    load_bowman,
)
from .model import BowmanEntailmentClassifier, TextEncoder

__all__ = [
    "BowmanBatch",
    "BowmanBundle",
    "BowmanEntailmentClassifier",
    "BowmanTokenizer",
    "DEFAULT_LAYERS",
    "EncodedPair",
    "HookableLayer",
    "LABELS",
    "MODEL_ID",
    "PAD_ID",
    "TextEncoder",
    "capture_activations",
    "classifier_weights",
    "download_checkpoint",
    "extract_activations",
    "inference_precision",
    "load_bowman",
]
