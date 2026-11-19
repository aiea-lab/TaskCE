"""Prepare aligned concept features and layer activation NPZ bundles.

Capture maps are JSON objects mapping output names ("embedding_norm" or
"layer.N") to .pt paths relative to capture_root. An optional alignment
manifest is a JSON list of SHA-256 row hashes in capture order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np

from taskce.data import DataRoot
from taskce.profiles import Profile, load_profile
from taskce.runtime import atomic_write_json, sha256_file


def row_hash(row: dict[str, Any], *, columns: tuple[str, ...], label_column: str) -> str:
    values: list[Any] = [row[col] for col in columns]
    label_val = row[label_column]
    try:
        values.append(int(label_val))
    except (TypeError, ValueError):
        values.append(str(label_val))
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def capture_paths(
    root: Path,
    mapping_path: Path | None,
    *,
    dataset_name: str = "snli",
) -> dict[str, Path]:
    if mapping_path:
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        paths = {name: root / path for name, path in mapping.items()}
    else:
        paths = {}
        activation_filename = f"{dataset_name}_activation.pt"
        for path in root.rglob(activation_filename):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.feed_forward$", path.parent.name)
            name = (
                f"layer.{match.group(1)}"
                if match
                else ("embedding_norm" if path.parent.name.endswith("embedding_norm") else None)
            )
            if name:
                if name in paths:
                    raise ValueError(
                        f"Ambiguous {name} captures: {paths[name]} and {path}; use --capture-map"
                    )
                paths[name] = path
    if not paths or "embedding_norm" not in paths:
        raise ValueError(
            "No captures or missing embedding_norm; check --capture-root or use --capture-map"
        )
    for name, path in paths.items():
        if not (name == "embedding_norm" or re.fullmatch(r"layer\.\d+", name)):
            raise ValueError(f"Invalid capture name: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(sorted(paths.items()))


def grouped_split(dataset: Any, *, group_column: str) -> np.ndarray:
    """Hash primary text column into stable 60/20/20 fit/validation/test buckets."""
    split = []
    for text in dataset[group_column]:
        value = int.from_bytes(
            hashlib.sha256(str(text).encode("utf-8")).digest()[:8], "big"
        ) / 2**64
        split.append(0 if value < 0.6 else 1 if value < 0.8 else 2)
    return np.asarray(split, dtype=np.uint8)


def prepare_inputs(
    *,
    profile: Profile,
    capture_root: Path,
    output_dir: Path,
    data_root: DataRoot | None = None,
    capture_map: Path | None = None,
    frozen_features: Path | None = None,
    dataset_path: Path | None = None,
    row_hashes_path: Path | None = None,
    assume_capture_order: bool = False,
) -> dict[str, Any]:
    """Build aligned features.npz, <layer>.npz, and provenance.json."""
    if not row_hashes_path and not assume_capture_order:
        raise ValueError("Specify either row_hashes_path or assume_capture_order=True")

    import torch
    from taskce.hf import load_hf_dataset
    from formula_search.construction.tokenizer.orchestration import (
        TOKENIZER_DIRECTORY,
        get_tokenizer,
    )
    from formula_search.construction.vectors import construct_feature_vectors

    root = data_root or DataRoot.from_env()
    dataset = load_hf_dataset(profile.dataset, dataset_path=dataset_path)
    columns = profile.dataset.feature_columns
    label_column = profile.dataset.label_column
    required_cols = (*columns, label_column)
    if not all(column in dataset.column_names for column in required_cols):
        raise ValueError(f"Dataset must contain columns: {', '.join(required_cols)}")

    hashes = [
        row_hash(row, columns=columns, label_column=label_column) for row in dataset
    ]
    if row_hashes_path:
        expected = json.loads(row_hashes_path.read_text(encoding="utf-8"))
        if hashes != expected:
            raise ValueError("Capture row hashes do not match dataset rows in order")

    paths = capture_paths(
        capture_root, capture_map, dataset_name=profile.dataset.name
    )
    captures: dict[str, np.ndarray] = {}
    for name, path in paths.items():
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != len(dataset):
            raise ValueError(
                f"{path}: expected tensor [examples, neurons] with {len(dataset)} examples"
            )
        captures[name] = tensor.detach().to(torch.float32).numpy()

    tokenizer_name = profile.dataset.tokenizer_name or f"spacy-{profile.dataset.name}-features"
    uses_pos = profile.dataset.tokenizer_uses_pos
    resource_name = f"{tokenizer_name}_pos" if uses_pos else tokenizer_name
    bundled_tokenizer_json = TOKENIZER_DIRECTORY / resource_name / "tokenizer.json"
    cache_root = root.data_dir / "tokenizers"

    if frozen_features:
        frozen = torch.load(frozen_features, map_location="cpu", weights_only=True)
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
        quantile = profile.search.quantile
        for start in range(0, len(ids), 64):
            selected = raw[:, ids[start : start + 64]]
            actual = selected > torch.quantile(selected, quantile, dim=0)
            if not torch.equal(actual, matrix[:, start : start + 64]):
                raise ValueError("Frozen layer 8 activation masks differ from capture")
        tokenizer_json_path = bundled_tokenizer_json if bundled_tokenizer_json.is_file() else None
    else:
        tokenizer = get_tokenizer(
            tokenizer_name,
            dataset,
            columns,
            enable_pos=uses_pos,
            cache_directory=cache_root,
        )
        features = construct_feature_vectors(dataset, tokenizer, columns)
        names = np.asarray([str(name) for name, _ in features], dtype=str)
        vectors = np.asarray([vector for _, vector in features], dtype=np.bool_)
        if bundled_tokenizer_json.is_file():
            tokenizer_json_path = bundled_tokenizer_json
        else:
            tokenizer_json_path = cache_root / resource_name / "tokenizer.json"

    split = grouped_split(dataset, group_column=columns[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_labels = dataset[label_column]
    try:
        labels_array = np.asarray([int(v) for v in raw_labels], dtype=np.int64)
    except (TypeError, ValueError):
        labels_array = np.asarray([str(v) for v in raw_labels], dtype=str)

    np.savez_compressed(
        output_dir / "features.npz",
        vectors=vectors,
        names=names,
        split=split,
        row_hashes=np.asarray(hashes),
        labels=labels_array,
        example_ids=np.arange(len(dataset), dtype=np.int64),
    )
    for name, activation in captures.items():
        np.savez_compressed(output_dir / f"{name}.npz", activation=activation)

    repo_root = root.repo_root or Path(__file__).resolve().parents[1]
    commit_proc = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    provenance = {
        "profile": profile.name,
        "source": str(repo_root),
        "source_commit": commit_proc.stdout.strip() if commit_proc.returncode == 0 else None,
        "dataset": (
            str(dataset_path.resolve())
            if dataset_path
            else f"{profile.dataset.hf_id}:{profile.dataset.split}"
        ),
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "examples": len(dataset),
        "ordered_row_hashes_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
        "alignment": (
            "ordered row hashes matched"
            if row_hashes_path
            else "repository capture order attested; no per-example IDs stored"
        ),
        "row_hash_manifest": str(row_hashes_path.resolve()) if row_hashes_path else None,
        "tokenizer_sha256": (
            str(sha256_file(tokenizer_json_path))
            if tokenizer_json_path is not None and tokenizer_json_path.is_file()
            else None
        ),
        "pos_enabled": uses_pos,
        "frozen_features": (
            {
                "path": str(frozen_features.resolve()),
                "sha256": str(sha256_file(frozen_features)),
                "layer8_masks": "matched",
                "dataset_order": "not independently verified by mask comparison",
            }
            if frozen_features
            else None
        ),
        "split": f"sha256({columns[0]}) buckets 60/20/20",
        "split_counts": np.bincount(split, minlength=3).tolist(),
        "captures": {
            name: {
                "path": str(path.resolve()),
                "sha256": str(sha256_file(path)),
                "shape": list(captures[name].shape),
            }
            for name, path in paths.items()
        },
        "threshold": f"strict > per-neuron global {profile.search.quantile} quantile",
    }
    atomic_write_json(output_dir / "provenance.json", provenance)
    return provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="snli")
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-map", type=Path, default=None)
    parser.add_argument("--frozen-features", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=None)
    alignment = parser.add_mutually_exclusive_group(required=True)
    alignment.add_argument("--row-hashes", type=Path, default=None)
    alignment.add_argument("--assume-capture-order", action="store_true")
    args = parser.parse_args(argv)

    root = DataRoot.from_env()
    profile = load_profile(args.profile, data_root=root)
    provenance = prepare_inputs(
        profile=profile,
        capture_root=args.capture_root,
        output_dir=args.output,
        data_root=root,
        capture_map=args.capture_map,
        frozen_features=args.frozen_features,
        dataset_path=args.dataset_path,
        row_hashes_path=args.row_hashes,
        assume_capture_order=args.assume_capture_order,
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
