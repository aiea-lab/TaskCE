"""Flat library interface adapting the Bowman SNLI classifier to TaskCE."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile, gettempdir
from typing import Any, Literal, TypedDict
from urllib.request import urlopen

import spacy
import torch
from torch.nn.utils.rnn import pad_sequence

from .model import BowmanEntailmentClassifier


# =============================================================================
# Shared types and constants
# =============================================================================

CHECKPOINT_URL = "https://downloads.cs.stanford.edu/nlp/data/muj/bowman_snli/6.pth"
CHECKPOINT_SHA256 = "a0408617c50f2f75af0a3f7be320082d642eb1f3e3ec9eff14f50159bdee2cb1"

MODEL_ID = "bowman/snli-lstm"
LABELS: tuple[str, str, str] = ("entailment", "neutral", "contradiction")
PAD_ID: int = 1

# Modules whose forward output is a single Tensor compatible with HookRecorder.
# Excludes encoder.rnn, which returns (PackedSequence, (h_n, c_n)).
HookableLayer = Literal[
    "encoder",
    "bn",
    "dropout",
    "mlp.0",
    "mlp.1",
    "mlp.2",
    "mlp.3",
]

_TWO_DIM_LAYERS = frozenset({"bn", "dropout", "mlp.0", "mlp.1", "mlp.2", "mlp.3"})
DEFAULT_LAYERS: tuple[HookableLayer, ...] = ("mlp.1",)

EncodedPair = tuple[list[int], list[int]]


class BowmanBatch(TypedDict):
    """Keyword arguments matching BowmanEntailmentClassifier.forward."""

    premise: torch.Tensor  # [seq_len, batch], long, padded with PAD_ID
    premise_lengths: torch.Tensor  # [batch], long, CPU
    hypothesis: torch.Tensor  # [seq_len, batch], long, padded with PAD_ID
    hypothesis_lengths: torch.Tensor  # [batch], long, CPU


# =============================================================================
# Tokenization and padding adapter (plugs directly into capture.capture)
# =============================================================================


class BowmanTokenizer:
    """Adapter exposing `.pad(...)` so Bowman plugs into untouched `capture.capture`."""

    def __init__(self, spacy_tokenizer: Any, stoi: Mapping[str, int]) -> None:
        self._tokenizer = spacy_tokenizer
        self.stoi = stoi

    def encode_text(self, text: str) -> list[int]:
        unk_id = self.stoi["UNK"]
        ids = [self.stoi.get(token.lower_, unk_id) for token in self._tokenizer(text)]
        if not ids:
            # pack_padded_sequence rejects zero-length inputs.
            return [unk_id]
        return ids

    def encode_pair(self, premise: str, hypothesis: str) -> EncodedPair:
        return self.encode_text(premise), self.encode_text(hypothesis)

    def prepare_dataset(self, dataset: Any) -> dict[str, list[EncodedPair]]:
        """Pretokenize SNLI rows into `{'input_ids': [(premise_ids, hypothesis_ids), ...]}`."""
        if isinstance(dataset, Mapping):
            pairs = zip(dataset["premise"], dataset["hypothesis"], strict=True)
            return {"input_ids": [self.encode_pair(p, h) for p, h in pairs]}
        return {
            "input_ids": [
                self.encode_pair(row["premise"], row["hypothesis"]) for row in dataset
            ]
        }

    def pad(
        self,
        encoded_inputs: Mapping[str, Sequence[EncodedPair]],
        *,
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> BowmanBatch:
        """Pad a slice of `(premise_ids, hypothesis_ids)` into `BowmanBatch` kwargs."""
        if padding is not True:
            raise ValueError("BowmanTokenizer.pad requires padding=True.")
        if return_tensors != "pt":
            raise ValueError(
                f"BowmanTokenizer.pad only supports return_tensors='pt', got {return_tensors!r}"
            )
        pairs = encoded_inputs["input_ids"]
        if not pairs:
            raise ValueError("Cannot pad an empty batch.")

        premise_seqs = [torch.tensor(p, dtype=torch.long) for p, _ in pairs]
        hypothesis_seqs = [torch.tensor(h, dtype=torch.long) for _, h in pairs]
        return {
            "premise": pad_sequence(premise_seqs, batch_first=False, padding_value=PAD_ID),
            "premise_lengths": torch.tensor(
                [len(seq) for seq in premise_seqs], dtype=torch.long
            ),
            "hypothesis": pad_sequence(
                hypothesis_seqs, batch_first=False, padding_value=PAD_ID
            ),
            "hypothesis_lengths": torch.tensor(
                [len(seq) for seq in hypothesis_seqs], dtype=torch.long
            ),
        }


@dataclass(frozen=True)
class BowmanBundle:
    """Loaded eval-mode model and capture-compatible tokenizer sharing one checkpoint."""

    model: BowmanEntailmentClassifier
    tokenizer: BowmanTokenizer
    labels: tuple[str, str, str] = LABELS

    @property
    def stoi(self) -> Mapping[str, int]:
        return self.tokenizer.stoi


# =============================================================================
# Checkpoint and model loading (CLI, capture, ablation, formula_search)
# =============================================================================


def _checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download_checkpoint() -> Path:
    """Download and SHA-256 verify the Stanford Bowman SNLI checkpoint."""
    path = Path(gettempdir()) / "bowman_snli" / "6.pth"
    if path.is_file() and _checksum(path) == CHECKPOINT_SHA256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as partial:
        temporary_path = Path(partial.name)
    try:
        with (
            urlopen(CHECKPOINT_URL, timeout=60) as response,
            temporary_path.open("wb") as output,
        ):
            copyfileobj(response, output)
        if _checksum(temporary_path) != CHECKPOINT_SHA256:
            raise ValueError(
                "Stanford checkpoint did not match the expected SHA-256 digest"
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_bowman(
    device: str | torch.device = "cpu",
    *,
    checkpoint_path: Path | None = None,
) -> BowmanBundle:
    """Load the eval-mode Bowman classifier, checkpoint vocab, and BowmanTokenizer."""
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but PyTorch cannot access a CUDA device.")

    resolved_path = checkpoint_path if checkpoint_path is not None else download_checkpoint()
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=True)
    stoi: dict[str, int] = checkpoint["stoi"]
    model = BowmanEntailmentClassifier(len(stoi))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(target_device).eval()
    tokenizer = BowmanTokenizer(spacy.load("en_core_web_sm").tokenizer, stoi)
    return BowmanBundle(model=model, tokenizer=tokenizer)


@contextmanager
def inference_precision(device: torch.device) -> Iterator[None]:
    """Use full FP32 for Bowman on CUDA and restore the caller's settings."""
    if device.type != "cuda":
        yield
        return

    rnn = torch.backends.cudnn.rnn
    matmul = torch.backends.cuda.matmul
    previous_rnn = rnn.fp32_precision
    previous_matmul = matmul.fp32_precision
    try:
        rnn.fp32_precision = "ieee"
        matmul.fp32_precision = "ieee"
        yield
    finally:
        rnn.fp32_precision = previous_rnn
        matmul.fp32_precision = previous_matmul


# =============================================================================
# Capture / HookRecorder activation extraction (capture)
# =============================================================================


def extract_activations(
    layer: HookableLayer,
    invocations: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Extract a 2D [batch, neurons] activation tensor from HookRecorder invocations."""
    if layer == "encoder":
        if len(invocations) != 2:
            raise ValueError(
                f"Expected 2 encoder invocations (premise, hypothesis), got {len(invocations)}"
            )
        return torch.cat([invocations[0], invocations[1]], dim=1).clone()

    if layer in _TWO_DIM_LAYERS:
        if len(invocations) != 1:
            raise ValueError(f"Expected 1 invocation for {layer!r}, got {len(invocations)}")
        return invocations[0].clone()

    raise ValueError(f"Unsupported Bowman hook layer: {layer!r}")


def capture_activations(
    dataset: Any,
    *,
    layers: Sequence[str] = DEFAULT_LAYERS,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    checkpoint_path: Path | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, torch.Tensor]:
    """Capture 2D [examples, neurons] float32 activations through untouched `capture.capture`."""
    from capture.capture import capture

    target_device = torch.device(device)
    bundle = load_bowman(device=target_device, checkpoint_path=checkpoint_path)
    encoded = bundle.tokenizer.prepare_dataset(dataset)

    chunks_by_layer: dict[str, list[torch.Tensor]] = {}
    with (
        inference_precision(target_device),
        capture(
            bundle.model,
            encoded,
            tokenizer=bundle.tokenizer,
            layers=list(layers),
            batch_size=batch_size,
            on_progress=on_progress,
        ) as batches,
    ):
        for _indices, outputs in batches:
            for layer, invocations in outputs.items():
                chunks_by_layer.setdefault(layer, []).append(
                    extract_activations(layer, invocations)  # type: ignore[arg-type]
                )

    return {layer: torch.cat(chunks) for layer, chunks in chunks_by_layer.items()}


# =============================================================================
# Classifier head weight extraction (formula_search: attach_weights)
# =============================================================================


def classifier_weights(model: BowmanEntailmentClassifier) -> dict[str, torch.Tensor]:
    """Return detached CPU [1024] output-head weight vectors keyed by LABELS from model.mlp[3]."""
    output_linear = model.mlp[3]
    weight = output_linear.weight.detach().to(device="cpu", dtype=torch.float32)
    return {label: weight[idx].clone() for idx, label in enumerate(LABELS)}
