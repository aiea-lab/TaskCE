import numpy as np
import pytest

import activation_analysis.task_separation as task_separation_mod


class _FakeDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values


def test_task_separation_returns_projection_data_before_plotting():
    pytest.importorskip("sklearn", exc_type=ImportError)
    states = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0],
        ]
    )
    labels = ["left", "left", "right", "right"]

    result = task_separation_mod.analyze_task_separation(states, labels)

    assert set(result) == {
        "points",
        "labels",
        "label_counts",
        "explained_variance_ratio",
        "components",
    }
    assert result["points"].shape == (4, 2)
    assert result["components"].shape == (2, 3)
    assert result["label_counts"] == {"left": 2, "right": 2}


def test_task_separation_requires_aligned_labels():
    pytest.importorskip("sklearn", exc_type=ImportError)
    with pytest.raises(ValueError, match="dataset rows do not match"):
        task_separation_mod.analyze_task_separation(
            np.zeros((3, 2)),
            ["first", "second"],
        )


def test_task_separation_reloads_labels_for_raw_activation_rows(
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("datasets")
    activation_path = tmp_path / "snli.pt"
    torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4), activation_path)
    monkeypatch.setattr(
        task_separation_mod,
        "_load_dataset",
        lambda *_args, **_kwargs: _FakeDataset([0, 1]),
    )

    with pytest.raises(ValueError, match="dataset rows do not match activation rows"):
        task_separation_mod.run("snli", activation_path)
