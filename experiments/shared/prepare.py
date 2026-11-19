"""Prepare aligned SNLI features and captures for the Nautilus experiment.

Capture maps are JSON objects mapping output names ("embedding_norm" or
"layer.N") to .pt paths relative to --capture-root. An optional alignment
manifest is a JSON list of SHA-256 row hashes in capture order. The hash for
one row is SHA-256 of UTF-8 JSON [premise, hypothesis, label], serialized with
ensure_ascii=False and separators=(',', ':').
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def row_hash(row):
    payload = json.dumps(
        [row["premise"], row["hypothesis"], int(row["label"])],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_paths(root, mapping_path):
    if mapping_path:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        paths = {name: root / path for name, path in mapping.items()}
    else:
        paths = {}
        for path in root.rglob("snli_activation.pt"):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.feed_forward$", path.parent.name)
            name = f"layer.{match.group(1)}" if match else (
                "embedding_norm" if path.parent.name.endswith("embedding_norm") else None
            )
            if name:
                if name in paths:
                    raise ValueError(f"Ambiguous {name} captures: {paths[name]} and {path}; use --capture-map")
                paths[name] = path
    if not paths or "embedding_norm" not in paths:
        raise ValueError("No captures or missing embedding_norm; check --capture-root or use --capture-map")
    for name, path in paths.items():
        if not (name == "embedding_norm" or re.fullmatch(r"layer\.\d+", name)):
            raise ValueError(f"Invalid capture name: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(sorted(paths.items()))


def grouped_split(dataset):
    # Hashing premises keeps duplicate premises together and is stable if rows move.
    split = []
    for premise in dataset["premise"]:
        value = int.from_bytes(hashlib.sha256(premise.encode("utf-8")).digest()[:8], "big") / 2**64
        split.append(0 if value < 0.6 else 1 if value < 0.8 else 2)
    return np.asarray(split, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Repository root containing theoretical/compositional_explanations")
    parser.add_argument("--capture-root", type=Path, required=True, help="Fine-tuned SNLI checkpoint capture directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-map", type=Path, help="JSON mapping output names to .pt paths")
    parser.add_argument("--frozen-features", type=Path, help="Historical inputs.pt; verify its layer 8 threshold mask against the capture")
    parser.add_argument("--dataset-path", type=Path, help="Dataset saved with datasets.save_to_disk; otherwise use cached HF SNLI validation")
    alignment = parser.add_mutually_exclusive_group(required=True)
    alignment.add_argument("--row-hashes", type=Path, help="JSON list of ordered capture row hashes")
    alignment.add_argument("--assume-capture-order", action="store_true", help="Explicitly attest captures follow the repository's unfiltered SNLI validation order; not independently verified")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset, load_from_disk

    repo = args.source.resolve()
    sys.path.insert(0, str(repo / "theoretical" / "compositional_explanations"))



    dataset = load_from_disk(str(args.dataset_path)) if args.dataset_path else load_dataset("stanfordnlp/snli", split="validation")
    if not all(column in dataset.column_names for column in ("premise", "hypothesis", "label")):
        raise ValueError("Dataset must contain premise, hypothesis, and label columns")
    hashes = [row_hash(row) for row in dataset]
    if args.row_hashes:
        expected = json.loads(args.row_hashes.read_text(encoding="utf-8"))
        if hashes != expected:
            raise ValueError("Capture row hashes do not match dataset rows in order")
    paths = capture_paths(args.capture_root, args.capture_map)
    captures = {}
    for name, path in paths.items():
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != len(dataset):
            raise ValueError(f"{path}: expected tensor [examples, neurons] with {len(dataset)} examples")
        captures[name] = tensor.detach().to(torch.float32).numpy()

    # Match TASKS['snli']: the persisted tokenizer has POS disabled.
    tokenizer_path = repo / "theoretical/compositional_explanations/resources/tokenizers/spacy-pos-snli-features/tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Saved tokenizer required: {tokenizer_path}")
    columns = ("premise", "hypothesis")
    if args.frozen_features:
        frozen = torch.load(args.frozen_features, map_location="cpu", weights_only=True)
        names = np.asarray(frozen["feature_names"], dtype=str)
        frozen_vectors = frozen["feature_vectors"]
        if not isinstance(frozen_vectors, torch.Tensor) or frozen_vectors.dtype != torch.bool:
            raise ValueError("Frozen feature_vectors must be a boolean tensor")
        vectors = frozen_vectors.numpy()
        if vectors.shape != (len(names), len(dataset)):
            raise ValueError("Frozen feature matrix does not match names and dataset size")
        if "layer.8" not in captures:
            raise ValueError("Frozen inputs require a layer.8 capture for comparison")
        raw = torch.from_numpy(captures["layer.8"])
        ids = torch.as_tensor(frozen["neuron_ids"], dtype=torch.long)
        matrix = frozen["matrix"]
        if matrix.shape != (len(dataset), len(ids)) or matrix.dtype != torch.bool:
            raise ValueError("Unexpected frozen activation matrix")
        if len(ids) == 0 or torch.any(ids < 0) or torch.any(ids >= raw.shape[1]):
            raise ValueError("Invalid frozen neuron IDs")
        for start in range(0, len(ids), 64):
            selected = raw[:, ids[start:start + 64]]
            actual = selected > torch.quantile(selected, 0.945, dim=0)
            if not torch.equal(actual, matrix[:, start:start + 64]):
                raise ValueError("Frozen layer 8 activation masks differ from capture")
    else:
        from construction.tokenizer.orchestration import get_tokenizer
        from construction.vectors import construct_feature_vectors
        tokenizer = get_tokenizer("spacy-pos-snli-features", dataset, columns, enable_pos=False)
        features = construct_feature_vectors(dataset, tokenizer, columns)
        names = np.asarray([str(name) for name, _ in features], dtype=str)
        vectors = np.asarray([vector for _, vector in features], dtype=np.bool_)
    split = grouped_split(dataset)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "features.npz", vectors=vectors, names=names, split=split,
                        row_hashes=np.asarray(hashes), labels=np.asarray(dataset["label"], dtype=np.int64),
                        example_ids=np.arange(len(dataset), dtype=np.int64))
    for name, activation in captures.items():
        np.savez_compressed(args.output / f"{name}.npz", activation=activation)

    counts = {}
    for name, activation in captures.items():
        if activation.shape[1] <= 1052:
            continue
        # Same strict quantile threshold as the repository's SNLI configuration.
        targets = torch.from_numpy(activation[:, [467, 1052]])
        active = targets > torch.quantile(targets, 0.945, dim=0)
        counts[name] = {str(neuron): [int(active[split == part, col].sum()) for part in range(3)]
                        for col, neuron in enumerate((467, 1052))}
        counts[name]["both"] = [int(active[split == part].all(dim=1).sum()) for part in range(3)]
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    provenance = {
        "source": str(repo), "source_commit": commit.stdout.strip() if commit.returncode == 0 else None, "dataset": str(args.dataset_path.resolve()) if args.dataset_path else "stanfordnlp/snli:validation",
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None), "examples": len(dataset),
        "ordered_row_hashes_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
        "alignment": "ordered row hashes matched" if args.row_hashes else "repository capture order inferred from validation capture code; no per-example IDs stored",
        "row_hash_manifest": str(args.row_hashes.resolve()) if args.row_hashes else None,
        "tokenizer_sha256": file_hash(tokenizer_path), "pos_enabled": False,
        "frozen_features": {"path": str(args.frozen_features.resolve()), "sha256": file_hash(args.frozen_features), "layer8_masks": "matched", "dataset_order": "not independently verified by mask comparison"} if args.frozen_features else None,
        "split": "sha256(premise) buckets 60/20/20", "split_counts": np.bincount(split, minlength=3).tolist(),
        "captures": {name: {"path": str(path.resolve()), "sha256": file_hash(path), "shape": list(captures[name].shape)} for name, path in paths.items()},
        "threshold": "strict > per-neuron global 0.945 quantile", "positive_counts_fit_validation_test": counts,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
