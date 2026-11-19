"""Inspect whether task labels organize one activation representation."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DATA_DIRECTORY = Path("data")
SNLI_LABELS = ("entailment", "neutral", "contradiction")


def load_activation(path):
    """Load one capture as a CPU float tensor shaped [examples, neurons]."""
    from taskce.data import DataRoot

    return DataRoot.from_env().load_activation(path)


@dataclass(frozen=True)
class TaskLabels:
    dataset_name: str
    split: str
    label_field: str
    label_names: tuple[str, ...] | None = None

    def from_dataset(self, dataset):
        values = dataset[self.label_field]
        if self.label_names is None:
            return [str(value).strip().lower() for value in values]
        return [self.label_names[int(value)] for value in values]


TASKS = {
    "snli": TaskLabels(
        "stanfordnlp/snli",
        "validation",
        "label",
        SNLI_LABELS,
    ),
    "claim": TaskLabels("tals/vitaminc", "validation[:10_000]", "label"),
    "fallacy": TaskLabels(
        "tasksource/logical-fallacy",
        "dev",
        "logical_fallacies",
    ),
}


def run(task_name, activation_path, *, profile=None, data_root=None):
    """Load one capture and its aligned labels, then return the PCA result."""
    if task_name in TASKS:
        task = TASKS[task_name]
    elif profile is not None:
        task = TaskLabels(
            profile.dataset.hf_id,
            profile.dataset.split,
            profile.dataset.label_column,
            tuple(profile.dataset.labels) or None,
        )
    else:
        raise KeyError(task_name)
    states = (
        data_root.load_activation(activation_path)
        if data_root is not None
        else load_activation(activation_path)
    )
    labels = None
    if profile is not None:
        from taskce.data import DataRoot

        root = data_root or DataRoot.from_env()
        try:
            bundle = root.resolve_prepared_inputs(profile)
        except FileNotFoundError:
            bundle = None
        if bundle is not None:
            labels = _labels_from_features(task, bundle.features_path)
            if labels is not None and len(labels) != states.shape[0]:
                labels = None
    if labels is None:
        labels = task.from_dataset(_load_dataset(task.dataset_name, task.split))
    return analyze_task_separation(states, labels)


def _labels_from_features(task, features_path):
    """Read prepared labels from features.npz when available."""
    if not Path(features_path).is_file():
        return None
    try:
        with np.load(features_path, allow_pickle=False) as data:
            if "labels" not in data:
                return None
            raw_labels = data["labels"]
    except (OSError, ValueError):
        return None
    if task.label_names is not None:
        return [task.label_names[int(value)] for value in raw_labels]
    return [str(value).strip().lower() for value in raw_labels]


def analyze_task_separation(states, labels):
    """Return the complete two-component label-projection result."""
    from sklearn.decomposition import PCA

    states = np.asarray(states, dtype=float)
    labels = [str(label) for label in labels]
    if states.ndim != 2:
        raise ValueError(f"expected 2D activation matrix, got {states.shape}")
    if len(labels) != states.shape[0]:
        raise ValueError(
            "dataset rows do not match activation rows: "
            f"{len(labels)} != {states.shape[0]}"
        )
    if states.shape[0] < 2 or states.shape[1] < 2:
        raise ValueError(
            "task separation requires at least 2 examples and 2 activation dimensions"
        )

    pca = PCA(n_components=2)
    points = pca.fit_transform(states)
    return {
        "points": points,
        "labels": labels,
        "label_counts": dict(Counter(labels)),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "components": pca.components_,
    }


def _load_dataset(dataset_name, split):
    from datasets import load_dataset

    return load_dataset(dataset_name, split=split)
