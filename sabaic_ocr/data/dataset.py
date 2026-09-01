from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(images_dir: str | Path) -> List[Path]:
    images_dir = Path(images_dir)
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_yolo_labels(path: str | Path, num_classes: int) -> torch.Tensor:
    path = Path(path)
    if not path.exists():
        return torch.empty((0, 5), dtype=torch.float32)

    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 values, got {len(parts)}")
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        values = (cx, cy, w, h)
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"{path}:{line_no}: non-finite box value")
        if not 0 <= cls < num_classes:
            raise ValueError(f"{path}:{line_no}: class {cls} outside [0,{num_classes-1}]")
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{path}:{line_no}: invalid normalized box")
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        eps = 1e-6
        if x1 < -eps or y1 < -eps or x2 > 1 + eps or y2 > 1 + eps:
            raise ValueError(f"{path}:{line_no}: box extends outside image bounds")
        rows.append([cls, cx, cy, w, h])

    return torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, 5), dtype=torch.float32)


def letterbox(
    image: Image.Image,
    targets: torch.Tensor,
    size: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> tuple[Image.Image, torch.Tensor, dict]:
    image = image.convert("RGB")
    old_w, old_h = image.size
    scale = min(size / old_w, size / old_h)
    new_w = max(1, int(round(old_w * scale)))
    new_h = max(1, int(round(old_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    out = targets.clone()
    if out.numel() > 0:
        out[:, 1] = (out[:, 1] * old_w * scale + pad_x) / size
        out[:, 2] = (out[:, 2] * old_h * scale + pad_y) / size
        out[:, 3] = (out[:, 3] * old_w * scale) / size
        out[:, 4] = (out[:, 4] * old_h * scale) / size

    meta = {
        "old_size": (old_w, old_h),
        "new_size": (new_w, new_h),
        "scale": scale,
        "pad": (pad_x, pad_y),
        "input_size": size,
    }
    return canvas, out, meta


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class YoloCharacterDataset(Dataset):
    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        num_classes: int,
        image_size: int = 640,
        augment: Callable | None = None,
        return_meta: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.augment = augment
        self.return_meta = return_meta
        self.images = list_images(self.images_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        targets = read_yolo_labels(label_path, self.num_classes)
        if self.augment is not None:
            image = self.augment(image)
        image, targets, meta = letterbox(image, targets, self.image_size)
        tensor = pil_to_tensor(image)
        if self.return_meta:
            meta["image_path"] = str(image_path)
            meta["label_path"] = str(label_path)
            return tensor, targets, meta
        return tensor, targets


def yolo_collate(batch):
    if len(batch[0]) == 3:
        images, targets, metas = zip(*batch)
        return torch.stack(images, 0), list(targets), list(metas)
    images, targets = zip(*batch)
    return torch.stack(images, 0), list(targets)
