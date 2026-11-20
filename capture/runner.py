"""Executor runner for the `capture` analysis."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import uuid

from taskce.profiles import Profile
from taskce.runtime import RunWorkspace


@dataclass(frozen=True)
class CaptureOptions:
    batch_size: int = 32
    layers: tuple[str, ...] | None = None
    device: str = "cpu"


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
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="PyTorch device for Bowman capture (default: %(default)s).",
    )


def options_from_args(args: argparse.Namespace) -> CaptureOptions:
    layers = tuple(args.layers.split(",")) if args.layers else None
    return CaptureOptions(batch_size=args.batch_size, layers=layers, device=args.device)


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
            if (
                profile.model.model_id == "bowman/snli-lstm"
                and stage == "base"
                and "finetuned" in stages
            ):
                continue
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
    data_root = workspace.data_root
    layers = opts.layers or tuple(str(layer) for layer in profile.model.layers)
    if not layers:
        raise ValueError("Profile defines no layers to capture")

    if profile.model.model_id == "bowman/snli-lstm":
        from bowman_repro import capture_activations
        from taskce.hf import load_hf_dataset

        checkpoint_name = profile.model.checkpoint_name if variant is not None else None
        activations_by_layer = capture_activations(
            load_hf_dataset(profile.dataset),
            layers=layers,
            batch_size=opts.batch_size,
            device=opts.device,
        )
        return _save_layer_activations(
            data_root,
            profile,
            activations_by_layer,
            variant=variant,
            checkpoint_name=checkpoint_name,
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from capture.capture import capture
    from capture.prepare import prepare_dataset

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

    return _save_layer_activations(
        data_root,
        profile,
        {layer: torch.cat(chunks) for layer, chunks in final_tokens.items()},
        variant=variant,
        checkpoint_name=checkpoint_name,
    )


def _save_layer_activations(
    data_root: Any,
    profile: Profile,
    activations_by_layer: Mapping[str, Any],
    *,
    variant: str | None,
    checkpoint_name: str | None,
) -> dict[str, Path]:
    import torch

    artifacts: dict[str, Path] = {}
    for layer, activations in activations_by_layer.items():
        output_path = data_root.capture_output_path(
            profile.model.model_id,
            profile.dataset.name,
            layer,
            task_name=variant,
            checkpoint_name=checkpoint_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(
            f"{output_path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        try:
            torch.save(activations, temp_path)
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
        artifacts[f"capture_{layer.replace('.', '_')}"] = output_path
    return artifacts
