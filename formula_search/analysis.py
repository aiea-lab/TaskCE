"""Compositional explanation helpers and optional LoRA lm_head weight attachment."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM

from formula_search.construction.tokenizer.orchestration import get_tokenizer
from formula_search.construction.vectors import construct_feature_vectors
from formula_search.postprocessing import prune_min_acts, threshold
from formula_search.search.algorithm import search_all
from taskce.data import DataRoot, load_formula_table
from taskce.hf import HfCachePolicy, resolve_lora_checkpoint
from taskce.profiles import Profile
from taskce.runtime import atomic_write_text


DEFAULT_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking"
DEFAULT_LORA_REPOSITORY = "Heroi/multitune-lora-backup"
ACTIVATION_DATA_DIRECTORY = DataRoot.from_env().data_dir
DATA_DIRECTORY = Path("data")


def activation_path(
    model_id: str,
    dataset_name: str,
    layer: str,
    *,
    task_name: str | None = None,
    checkpoint_name: str | None = None,
) -> Path:
    """Return one generated activation file."""
    return DataRoot(data_dir=ACTIVATION_DATA_DIRECTORY).activation_path(
        model_id,
        dataset_name,
        layer,
        task_name=task_name,
        checkpoint_name=checkpoint_name,
    )


def result_path(task_name: str) -> Path:
    """Return the compositional data file owned by this analysis."""
    return DATA_DIRECTORY / f"{task_name}_beam_results.csv"


@dataclass(frozen=True)
class CompositionalTask:
    name: str
    dataset_name: str
    split: str
    feature_columns: tuple[str, ...]
    tokenizer_name: str
    tokenizer_uses_pos: bool
    alpha: float
    min_acts: int
    class_token_ids: dict[str, int]

    @property
    def weight_column_names(self) -> tuple[str, ...]:
        return tuple(
            f"weight_{label.replace(' ', '_')}"
            for label in self.class_token_ids
        )


TASKS = {
    "snli": CompositionalTask(
        name="snli",
        dataset_name="stanfordnlp/snli",
        split="validation",
        feature_columns=("premise", "hypothesis"),
        tokenizer_name="spacy-pos-snli-features",
        tokenizer_uses_pos=False,
        alpha=0.055,
        min_acts=500,
        class_token_ids={
            "entailment": 806,
            "neutral": 25919,
            "contradiction": 10913,
        },
    ),
    "claim": CompositionalTask(
        name="claim",
        dataset_name="tals/vitaminc",
        split="validation[:10_000]",
        feature_columns=("claim", "evidence"),
        tokenizer_name="spacy-pos-claim-features",
        tokenizer_uses_pos=False,
        alpha=0.052,
        min_acts=500,
        class_token_ids={
            "supports": 56744,
            "refutes": 1891,
            "not enough info": 2897,
        },
    ),
}


def load_activation(path: str | Path) -> torch.Tensor:
    """Load one raw experimental activation matrix from disk."""
    return DataRoot(data_dir=ACTIVATION_DATA_DIRECTORY).load_activation(path)


def classification_weights_from_model(
    model: Any,
    class_token_ids: dict[str, int] | Any,
) -> torch.Tensor:
    """Select analysis-supporting class weights from a loaded model."""
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise ValueError("Model does not expose weighted output embeddings.")
    class_ids = list(class_token_ids.values())
    return (
        output_embeddings.weight.detach()
        .to(device="cpu", dtype=torch.float32)[class_ids]
        .T
    )


def latest_task_checkpoint(
    lora_repository: str,
    task_name: str,
    *,
    token: str | bool | None,
    checkpoint_name: str | None = None,
) -> Path:
    policy = HfCachePolicy(
        hf_home=Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
        token=token if isinstance(token, str) else None,
    )
    return resolve_lora_checkpoint(
        lora_repository,
        task_name,
        checkpoint_name=checkpoint_name,
        policy=policy,
    )


def load_classification_weights(
    task: CompositionalTask,
    *,
    model_id: str,
    lora_repository: str,
    lora_token: str | bool | None,
    checkpoint_name: str | None = None,
) -> torch.Tensor:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    checkpoint = latest_task_checkpoint(
        lora_repository,
        task.name,
        token=lora_token,
        checkpoint_name=checkpoint_name,
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    try:
        return classification_weights_from_model(model, task.class_token_ids)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def attach_classification_weights(
    profile: Profile,
    formulas_csv: Path,
    output_csv: Path,
) -> Path:
    """Enrich an existing formula CSV with LoRA lm_head class weights."""
    if not profile.dataset.class_token_ids:
        raise ValueError(f"Profile {profile.name!r} defines no class_token_ids")
    if not profile.model.lora_repo or not profile.model.lora_task:
        raise ValueError(
            f"Profile {profile.name!r} requires model.lora_repo and model.lora_task"
        )
    frame = load_formula_table(formulas_csv)
    task = CompositionalTask(
        name=profile.model.lora_task,
        dataset_name=profile.dataset.hf_id,
        split=profile.dataset.split,
        feature_columns=profile.dataset.feature_columns,
        tokenizer_name=profile.dataset.tokenizer_name or profile.dataset.name,
        tokenizer_uses_pos=profile.dataset.tokenizer_uses_pos,
        alpha=profile.search.alpha,
        min_acts=profile.search.min_acts,
        class_token_ids=dict(profile.dataset.class_token_ids),
    )
    weights = load_classification_weights(
        task,
        model_id=profile.model.model_id,
        lora_repository=profile.model.lora_repo,
        lora_token=os.environ.get("HF_TOKEN"),
        checkpoint_name=profile.model.checkpoint_name,
    )
    weight_frame = pd.DataFrame(
        weights.tolist(),
        columns=profile.dataset.weight_column_names,
    )
    weight_frame.insert(0, "neuron", range(len(weight_frame)))

    enriched = frame.copy()
    enriched["neuron"] = pd.to_numeric(enriched["neuron"], errors="raise").astype(int)
    for col in profile.dataset.weight_column_names:
        if col in enriched.columns:
            enriched = enriched.drop(columns=[col])
    enriched = enriched.merge(weight_frame, on="neuron", how="left")
    atomic_write_text(output_csv, enriched.to_csv(index=False))
    return output_csv


def run(
    task_name: str,
    activation_path: str | Path,
    search_fn=search_all,
    output_path: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    lora_repository: str = DEFAULT_LORA_REPOSITORY,
    lora_token: str | bool | None = None,
    **kwargs,
) -> Path:
    """Build and save compositional explanations for one configured task."""
    task = TASKS[task_name]
    activations = load_activation(activation_path)
    dataset = load_dataset(task.dataset_name, split=task.split)
    if len(dataset) != activations.shape[0]:
        raise ValueError(
            f"{task.name} dataset rows do not match activation rows: "
            f"{len(dataset)} != {activations.shape[0]}."
        )

    feature_tokenizer = get_tokenizer(
        task.tokenizer_name,
        dataset,
        task.feature_columns,
        enable_pos=task.tokenizer_uses_pos,
    )
    features = construct_feature_vectors(
        dataset,
        feature_tokenizer,
        task.feature_columns,
    )

    binary_activations = threshold(activations, alpha=task.alpha)
    searchable_activations = prune_min_acts(
        binary_activations,
        min_acts=task.min_acts,
    )
    search_results = search_fn(
        searchable_activations.matrix,
        features,
        **kwargs,
    )

    classification_weights = load_classification_weights(
        task,
        model_id=model_id,
        lora_repository=lora_repository,
        lora_token=(
            os.environ.get("HF_TOKEN")
            if lora_token is None
            else lora_token
        ),
    )
    if classification_weights.shape[0] != activations.shape[1]:
        raise ValueError(
            "Classification weight and activation neuron counts differ: "
            f"{classification_weights.shape[0]} != {activations.shape[1]}."
        )

    results = pd.DataFrame(
        classification_weights.tolist(),
        columns=task.weight_column_names,
    )
    results.insert(0, "iou", 0.0)
    results.insert(0, "formula", "LOW_ACTS_PRUNED")
    results.insert(0, "neuron", range(len(results)))
    results = results.set_index("neuron", drop=False)

    for result in search_results:
        neuron_id = searchable_activations.neuron_ids[result.activation_index]
        results.at[neuron_id, "formula"] = result.best_formula
        results.at[neuron_id, "iou"] = result.best_score

    results = results.sort_values(
        "iou",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    output_path = (
        Path(output_path)
        if output_path is not None
        else result_path(task_name)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return output_path
