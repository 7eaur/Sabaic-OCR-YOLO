from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw

from .dataset import list_images, read_yolo_labels


def validate_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    num_classes: int,
    require_nonempty: bool = False,
    check_images: bool = True,
) -> dict:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    images = list_images(images_dir)

    errors: List[str] = []
    class_counts: Counter[int] = Counter()
    labeled_images = 0
    box_count = 0
    empty_label_images: List[str] = []
    corrupt_images: List[str] = []

    for image_path in images:
        if check_images:
            try:
                with Image.open(image_path) as probe:
                    probe.verify()
            except Exception as exc:
                corrupt_images.append(f"{image_path}: {exc}")
                errors.append(f"corrupt image: {image_path}")
                continue

        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"missing label: {label_path}")
            continue
        try:
            targets = read_yolo_labels(label_path, num_classes)
        except Exception as exc:
            errors.append(str(exc))
            continue

        labeled_images += 1
        box_count += int(targets.shape[0])
        if targets.shape[0] == 0:
            empty_label_images.append(str(image_path))
            if require_nonempty:
                errors.append(f"empty label: {label_path}")
        for cls in targets[:, 0].tolist():
            class_counts[int(cls)] += 1

    image_stems = {p.stem for p in images}
    orphan_labels = []
    if labels_dir.exists():
        orphan_labels = [str(p) for p in labels_dir.glob("*.txt") if p.stem not in image_stems]

    return {
        "images": len(images),
        "labeled_images": labeled_images,
        "boxes": box_count,
        "class_counts": dict(sorted(class_counts.items())),
        "errors": errors,
        "orphan_labels": orphan_labels,
        "empty_label_images": empty_label_images,
        "corrupt_images": corrupt_images,
        "valid": not errors and not orphan_labels and len(images) == labeled_images,
    }


def draw_label_preview(
    image_path: str | Path,
    label_path: str | Path,
    output_path: str | Path,
    class_names: Dict[int, str],
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    targets = read_yolo_labels(label_path, len(class_names))

    for row in targets.tolist():
        cls, cx, cy, bw, bh = row
        cls = int(cls)
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        draw.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), class_names.get(cls, str(cls)), fill=(255, 0, 0))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
