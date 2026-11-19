"""Prepare SNLI chat-template datasets for activation capture."""

from __future__ import annotations

from typing import Any

from taskce.profiles import Profile


def format_snli(example):
    example["prompt"] = [
        {
            "role": "system",
            "content": (
                "Determine the relationship between the `premise` and "
                "`hypothesis` and respond with `entailment`, `neutral`, "
                "or `contradiction`."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Premise: {example['premise']}\n"
                f"Hypothesis: {example['hypothesis']}"
            ),
        },
    ]
    return example


def get_prepared_snli(tokenizer, split="validation"):
    """Load SNLI and tokenize each example with the bundled chat prompt."""
    from datasets import load_dataset
    from taskce.profiles import load_profile

    profile = load_profile("snli")
    dataset = load_dataset(profile.dataset.hf_id, split=split)
    if profile.dataset.prompt is None:
        raise ValueError("The snli profile has no [dataset.prompt] table")
    return _tokenize_with_prompt(dataset, profile.dataset.prompt, tokenizer)


def prepare_dataset(
    profile: Profile,
    tokenizer: Any,
    *,
    dataset_path: Any = None,
):
    """Load the profile dataset and tokenize it with `dataset.prompt`.

    `dataset_path` is an optional `datasets.save_to_disk` directory loaded
    through `taskce.hf.load_hf_dataset` instead of the HF Hub dataset.
    """
    from taskce.hf import load_hf_dataset

    dataset = load_hf_dataset(profile.dataset, dataset_path=dataset_path)
    prompt_spec = profile.dataset.prompt
    if prompt_spec is None:
        raise ValueError(f"Profile {profile.name!r} has no [dataset.prompt] table")
    return _tokenize_with_prompt(dataset, prompt_spec, tokenizer)


def _tokenize_with_prompt(dataset, prompt_spec, tokenizer):
    return dataset.map(
        lambda example: {
            "input_ids": tokenizer.apply_chat_template(
                prompt_spec.render(example),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        }
    )
