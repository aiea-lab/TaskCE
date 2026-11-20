"""Run two SNLI development examples with the original Bowman checkpoint."""

import spacy
import torch

from .model import BowmanEntailmentClassifier

import hashlib
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile, gettempdir
from urllib.request import urlopen
from datasets import load_dataset

URL = "https://downloads.cs.stanford.edu/nlp/data/muj/bowman_snli/6.pth"
SHA256 = "a0408617c50f2f75af0a3f7be320082d642eb1f3e3ec9eff14f50159bdee2cb1"


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def download_checkpoint() -> Path:
    path = Path(gettempdir()) / "bowman_snli" / "6.pth"
    if path.is_file() and checksum(path) == SHA256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as partial:
        temporary_path = Path(partial.name)
    try:
        with urlopen(URL, timeout=60) as response, temporary_path.open("wb") as output:
            copyfileobj(response, output)
        if checksum(temporary_path) != SHA256:
            raise ValueError(
                "Stanford checkpoint did not match the expected SHA-256 digest"
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


LABELS = ["entailment", "neutral", "contradiction"]
snli = load_dataset("stanfordnlp/snli", split="validation").to_pandas()


# SNLI dev pair IDs 4705552913.jpg#2r1e and 4705552913.jpg#2r1c.


def tokenize(text, tokenizer, stoi):
    ids = [stoi.get(token.lower_, stoi["UNK"]) for token in tokenizer(text)]
    return torch.tensor(ids, dtype=torch.long).reshape(-1, 1), torch.tensor([len(ids)])


def main():
    checkpoint = torch.load(
        download_checkpoint(), map_location="cpu", weights_only=True
    )
    stoi = checkpoint["stoi"]
    model = BowmanEntailmentClassifier(len(stoi))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    with torch.inference_mode():
        for premise, hypothesis, label in snli[
            ["premise", "hypothesis", "label"]
        ].to_numpy():
            if label == -1:
                print(premise, hypothesis, "is unlabeled")
                continue

            premise_ids, premise_length = tokenize(premise, tokenizer, stoi)
            hypothesis_ids, hypothesis_length = tokenize(hypothesis, tokenizer, stoi)
            logits = model(
                premise_ids, premise_length, hypothesis_ids, hypothesis_length
            )
            prediction = LABELS[logits.argmax(dim=1).item()]
            print(f"Premise: {premise}\nHypothesis: {hypothesis}")
            gold = LABELS[label]
            print(f"Gold: {gold} | Predicted: {prediction}\n")


if __name__ == "__main__":
    main()
