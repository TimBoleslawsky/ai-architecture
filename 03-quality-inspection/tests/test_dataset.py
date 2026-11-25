"""Tests for synthetic defect generation and the dataset loader."""
from pathlib import Path

import numpy as np

from src.data.synthetic import CLASSES, generate_dataset, render_class_image


def test_render_class_image_produces_valid_array():
    rng = np.random.default_rng(0)
    for cls in CLASSES:
        arr = render_class_image(cls, rng)
        assert arr.shape == (200, 200)
        assert arr.dtype == np.uint8
        assert arr.min() >= 0 and arr.max() <= 255


def test_classes_are_distinct_and_expected():
    assert len(CLASSES) == 6
    assert "crazing" in CLASSES and "scratches" in CLASSES


def test_generate_dataset_writes_tree(tmp_path: Path):
    generate_dataset(tmp_path, per_class=5, seed=1)
    for cls in CLASSES:
        imgs = list((tmp_path / cls).glob("*.png"))
        assert len(imgs) == 5


def test_dataset_loader_shapes_and_labels(tmp_path: Path):
    from torch.utils.data import DataLoader

    from src.data.dataset import SteelDefectDataset

    generate_dataset(tmp_path, per_class=6, seed=2)
    ds = SteelDefectDataset(tmp_path)
    assert len(ds) == 6 * len(CLASSES)
    assert ds.classes == sorted(CLASSES)

    loader = DataLoader(ds, batch_size=4)
    x, y = next(iter(loader))
    assert x.shape == (4, 1, 96, 96)
    assert set(y.tolist()) <= set(range(len(CLASSES)))


def test_image_bytes_to_tensor_matches_dataset_preprocess(tmp_path: Path):
    from PIL import Image

    from src.data.dataset import TRANSFORM, SteelDefectDataset, image_bytes_to_tensor

    generate_dataset(tmp_path, per_class=2, seed=3)
    ds = SteelDefectDataset(tmp_path)
    path, _ = ds.samples[0]

    from_file = TRANSFORM(Image.open(path).convert("L"))
    from_bytes = image_bytes_to_tensor(path.read_bytes())
    np.testing.assert_allclose(from_file.numpy(), from_bytes.numpy(), rtol=1e-6)
