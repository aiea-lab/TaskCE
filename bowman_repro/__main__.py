"""Run SNLI validation inference with the original Bowman checkpoint."""

from datasets import load_dataset
import torch

from .api import LABELS, inference_precision, load_bowman


def main(device: str = "cpu"):
    try:
        bundle = load_bowman(device=device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    target_device = torch.device(device)
    snli = load_dataset("stanfordnlp/snli", split="validation").to_pandas()
    with inference_precision(target_device), torch.inference_mode():
        for premise, hypothesis, label in snli[
            ["premise", "hypothesis", "label"]
        ].to_numpy():
            if label == -1:
                print(premise, hypothesis, "is unlabeled")
                continue

            encoded = bundle.tokenizer.encode_pair(premise, hypothesis)
            batch = bundle.tokenizer.pad({"input_ids": [encoded]})
            logits = bundle.model(
                premise=batch["premise"].to(target_device),
                premise_lengths=batch["premise_lengths"],
                hypothesis=batch["hypothesis"].to(target_device),
                hypothesis_lengths=batch["hypothesis_lengths"],
            )
            prediction = LABELS[logits.argmax(dim=1).item()]
            print(f"Premise: {premise}\nHypothesis: {hypothesis}")
            gold = LABELS[label]
            print(f"Gold: {gold} | Predicted: {prediction}\n")


if __name__ == "__main__":
    main()
