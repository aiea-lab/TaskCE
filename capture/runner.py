"""Executor runner for the `capture` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from taskce.profiles import Profile
from taskce.runtime import RunWorkspace


@dataclass(frozen=True)
class CaptureOptions:
    batch_size: int = 32
    layers: tuple[str, ...] | None = None


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Capture batch size (default: %(default)s).",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma-separated module layer names; defaults to the profile layers.",
    )


def options_from_args(args: argparse.Namespace) -> CaptureOptions:
    layers = tuple(args.layers.split(",")) if args.layers else None
    return CaptureOptions(batch_size=args.batch_size, layers=layers)


def run(
    workspace: RunWorkspace,
    profile: Profile,
    options: CaptureOptions | None,
    stages: tuple[str, ...],
) -> Mapping[str, Path]:
    """Run capture stages (`base`, `finetuned`) for the requested variants."""
    opts = options or CaptureOptions()
    artifacts: dict[str, Path] = {}
    for stage in stages:
        with workspace.stage(stage):
            variant = _stage_variant(profile, stage)
            for key, path in _capture_variant(workspace, profile, opts, variant).items():
                artifacts[f"{stage}_{key}"] = path
    return artifacts


def _stage_variant(profile: Profile, stage: str) -> str | None:
    if stage == "base":
        return None
    if stage == "finetuned":
        if profile.model.lora_task is None:
            raise ValueError("Profile has no lora_task for the 'finetuned' stage")
        return profile.model.lora_task
    raise ValueError(f"Unknown capture stage {stage!r}")


def _capture_variant(
    workspace: RunWorkspace,
    profile: Profile,
    opts: CaptureOptions,
    variant: str | None,
) -> dict[str, Path]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from capture.capture import capture
    from capture.prepare import prepare_dataset

    data_root = workspace.data_root
    layers = opts.layers or tuple(str(layer) for layer in profile.model.layers)
    if not layers:
        raise ValueError("Profile defines no layers to capture")

    tokenizer = AutoTokenizer.from_pretrained(profile.model.model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = prepare_dataset(profile, tokenizer)

    checkpoint_name: str | None = None
    model = AutoModelForCausalLM.from_pretrained(
        profile.model.model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if variant is not None:
        from peft import PeftModel
        from taskce.hf import resolve_lora_checkpoint

        checkpoint = resolve_lora_checkpoint(
            profile.model.lora_repo or "",
            variant,
            checkpoint_name=profile.model.checkpoint_name,
        )
        checkpoint_name = checkpoint.name
        model = PeftModel.from_pretrained(model, str(checkpoint))

    final_tokens: dict[str, list[Any]] = {}
    with capture(
        model,
        dataset,
        tokenizer=tokenizer,
        layers=list(layers),
        batch_size=opts.batch_size,
    ) as batches:
        for _indices, outputs in batches:
            for layer, invocations in outputs.items():
                # Copy the final tokens so their views do not retain the full batch.
                final_tokens.setdefault(layer, []).append(
                    invocations[0][:, -1].clone()
                )

    artifacts: dict[str, Path] = {}
    for layer, chunks in final_tokens.items():
        activations = torch.cat(chunks)
        output_path = data_root.capture_output_path(
            profile.model.model_id,
            profile.dataset.name,
            layer,
            task_name=variant,
            checkpoint_name=checkpoint_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f"{output_path.name}.tmp")
        torch.save(activations, temp_path)
        os.replace(temp_path, output_path)
        artifacts[f"capture_{layer.replace('.', '_')}"] = output_path
    return artifacts
