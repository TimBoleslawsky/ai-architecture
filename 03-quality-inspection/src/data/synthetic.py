"""Procedural synthetic defect image generator.

Generates 200x200 grayscale images that mimic the six NEU steel-surface
defect classes so the whole pipeline runs offline and is fully reproducible:

- crazing:      networks of fine cracks
- inclusion:    dark, irregular embedded particles
- patches:      large irregular brighter/darker areas
- pitted_surface: many small dark pits
- rolled-in_scale: elongated dark strips
- scratches:    straight thin lines

Each image starts from a textured "steel" background; the defect is drawn on
top with random parameters. Run ``generate_dataset()`` to write a
``<out>/<class>/img_<n>.png`` tree, or use the optional NEU downloader for the
real dataset.

Run with:  uv run python -m src.data.synthetic --out data/raw/synthetic --per-class 150
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 200
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def _background(rng: np.random.Generator) -> Image.Image:
    noise = rng.normal(125.0, 14.0, (SIZE, SIZE)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(noise, mode="L").filter(ImageFilter.GaussianBlur(2.2))
    # coarse-grained brightness variations, like rolled steel
    draw = ImageDraw.Draw(img)
    for _ in range(3):
        x, y = rng.integers(0, SIZE, 2)
        r = int(rng.integers(20, 60))
        shade = int(rng.integers(-12, 13))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=max(0, min(255, 125 + shade)))
    return img.filter(ImageFilter.GaussianBlur(1.0))


def render_class_image(cls: str, rng: np.random.Generator) -> np.ndarray:
    img = _background(rng)
    draw = ImageDraw.Draw(img)

    if cls == "crazing":
        for _ in range(rng.integers(14, 22)):
            x, y = float(rng.integers(0, SIZE)), float(rng.integers(0, SIZE))
            pts = [(x, y)]
            for _ in range(rng.integers(8, 16)):
                x = min(SIZE - 1, max(0.0, x + rng.normal(0, 14)))
                y = min(SIZE - 1, max(0.0, y + rng.normal(0, 14)))
                pts.append((x, y))
            draw.line(pts, fill=int(rng.integers(55, 95)), width=1, joint="curve")

    elif cls == "inclusion":
        for _ in range(rng.integers(6, 11)):
            cx, cy = rng.integers(0, SIZE, 2)
            n = int(rng.integers(5, 9))
            r = int(rng.integers(4, 11))
            pts = [
                (
                    int(cx + r * math.cos(2 * math.pi * i / n + rng.random())),
                    int(cy + r * math.sin(2 * math.pi * i / n + rng.random())),
                )
                for i in range(n)
            ]
            draw.polygon(pts, fill=int(rng.integers(45, 90)))

    elif cls == "patches":
        for _ in range(rng.integers(3, 6)):
            cx, cy = rng.integers(0, SIZE, 2)
            r = int(rng.integers(22, 45))
            n = int(rng.integers(8, 14))
            pts = [
                (
                    int(cx + r * math.cos(2 * math.pi * i / n) + rng.integers(-14, 15)),
                    int(cy + r * math.sin(2 * math.pi * i / n) + rng.integers(-14, 15)),
                )
                for i in range(n)
            ]
            dark = rng.integers(0, 2) == 0
            fill = int(rng.integers(70, 105)) if dark else int(rng.integers(150, 185))
            draw.polygon(pts, fill=fill)

    elif cls == "pitted_surface":
        for _ in range(rng.integers(70, 110)):
            x, y = rng.integers(0, SIZE, 2)
            r = max(1, int(rng.normal(1.6, 0.6)))
            draw.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.integers(30, 70)))

    elif cls == "rolled-in_scale":
        for _ in range(rng.integers(4, 8)):
            cx, cy = rng.integers(0, SIZE, 2)
            a = float(rng.random() * math.pi)
            lx, ly = int(rng.integers(28, 55)), int(rng.integers(8, 18))
            cos_a, sin_a = math.cos(a), math.sin(a)
            corners = []
            for dx, dy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
                corners.append(
                    (
                        int(cx + dx * lx * cos_a - dy * ly * sin_a),
                        int(cy + dx * lx * sin_a + dy * ly * cos_a),
                    )
                )
            draw.polygon(corners, fill=int(rng.integers(40, 85)))

    elif cls == "scratches":
        for _ in range(rng.integers(3, 7)):
            x0, y0 = rng.integers(0, SIZE, 2)
            ang = rng.random() * math.pi
            length = float(rng.integers(60, 150))
            x1 = x0 + length * math.cos(ang)
            y1 = y0 + length * math.sin(ang)
            draw.line([(x0, y0), (x1, y1)], fill=int(rng.integers(40, 90)), width=1)

    else:
        raise ValueError(f"unknown class {cls!r}")

    return np.asarray(img)


def generate_dataset(out: Path, per_class: int, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    for cls in CLASSES:
        cls_dir = out / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = render_class_image(cls, rng)
            Image.fromarray(arr, mode="L").save(cls_dir / f"img_{i:04d}.png")
    print(f"wrote {per_class} images x {len(CLASSES)} classes -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/synthetic"))
    parser.add_argument("--per-class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.out, args.per_class, args.seed)


if __name__ == "__main__":
    main()
