import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("datasets")

import theoretical.activations.task_separation.analysis as task_separation


class FakeDataset:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values


def test_task_separation_reloads_labels_for_raw_activation_rows(
    tmp_path,
    monkeypatch,
):
    activation_path = tmp_path / "snli.pt"
    torch.save(torch.arange(12, dtype=torch.float32).reshape(3, 4), activation_path)
    monkeypatch.setattr(
        task_separation,
        "_load_dataset",
        lambda *_args, **_kwargs: FakeDataset([0, 1]),
    )

    with pytest.raises(ValueError, match="dataset rows do not match activation rows"):
        task_separation.run("snli", activation_path)

