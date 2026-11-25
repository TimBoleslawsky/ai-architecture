"""PyTorch Dataset and preprocessing.

The preprocessing contract (resize size, per-channel mean/std) is *shared*
between training and serving: the serving API applies the identical transforms
before handing pixels to the ONNX model. These values are also persisted in the
model registry so the artifact is self-describing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

SIZE = 96
MEAN = 0.5
STD = 0.5

TRANSFORM = T.Compose(
    [
        T.Resize((SIZE, SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[MEAN], std=[STD]),
    ]
)

PREPROCESS_META = {
    "size": SIZE,
    "mean": [MEAN],
    "std": [STD],
    "grayscale": True,
}


class SteelDefectDataset(Dataset):
    """Reads ``<root>/<class>/img_*.png`` images.

    Class order is the alphabetical directory order; the same order is recorded
    in the model registry so serving can map logits to names.
    """

    def __init__(self, root: Path, transform=TRANSFORM) -> None:
        self.root = Path(root)
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if not self.classes:
            raise ValueError(f"no class directories under {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            for img in sorted((self.root / cls).glob("*.png")) + sorted(
                (self.root / cls).glob("*.jpg")
            ) + sorted((self.root / cls).glob("*.bmp")):
                self.samples.append((img, self.class_to_idx[cls]))
        if not self.samples:
            raise ValueError(f"no images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return transform_image(img), label


def transform_image(img: Image.Image) -> torch.Tensor:
    return TRANSFORM(img)


def image_bytes_to_tensor(data: bytes) -> torch.Tensor:
    """Preprocess raw uploaded bytes exactly like a training sample."""
    img = Image.open(__import__("io").BytesIO(data)).convert("L")
    return transform_image(img)


def tensor_to_input_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.numpy().astype(np.float32)[None, ...]
