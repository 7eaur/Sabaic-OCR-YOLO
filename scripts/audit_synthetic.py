#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sabaic_ocr.config import load_classes, load_json
from sabaic_ocr.data.dataset import list_images, read_yolo_labels
from sabaic_ocr.data.labels import validate_dataset


def quantiles(values, qs=(0.0, 0.1, 0.5, 0.9, 1.0)):
    if not values:
        return {str(q): None for q in qs}
    arr = np.asarray(values, dtype=np.float32)
    vals = np.quantile(arr, qs)
    return {str(q): float(v) for q, v in zip(qs, vals)}


def main() -> None:
    p = argparse.ArgumentParser(description="Audit generated detector data before training.")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--classes", default="config/classes.json")
    p.add_argument("--model-config", default="config/model.json")
    p.add_argument("--output", default="")
    args = p.parse_args()

    classes = load_classes(args.classes)
    model = load_json(args.model_config)
    size = int(model["image_size"])
    basic = validate_dataset(args.images, args.labels, classes["num_classes"])

    counts = Counter()
    boxes_per_image = []
    widths_px = []
    heights_px = []
    center_x = []
    center_y = []

    for image_path in list_images(args.images):
        target = read_yolo_labels(
            Path(args.labels) / f"{image_path.stem}.txt", classes["num_classes"]
        )
        boxes_per_image.append(int(target.shape[0]))
        for row in target.tolist():
            cls_id, cx, cy, w, h = row
            counts[int(cls_id)] += 1
            widths_px.append(float(w) * size)
            heights_px.append(float(h) * size)
            center_x.append(float(cx))
            center_y.append(float(cy))

    nonzero = [counts[i] for i in range(classes["num_classes"]) if counts[i] > 0]
    report = {
        "valid_labels": basic["valid"],
        "images": basic["images"],
        "boxes": basic["boxes"],
        "empty_images": int(sum(1 for n in boxes_per_image if n == 0)),
        "boxes_per_image": quantiles(boxes_per_image),
        "width_px": quantiles(widths_px),
        "height_px": quantiles(heights_px),
        "center_x": quantiles(center_x),
        "center_y": quantiles(center_y),
        "class_counts": {str(i): counts[i] for i in range(classes["num_classes"])},
        "missing_classes": [i for i in range(classes["num_classes"]) if counts[i] == 0],
        "class_imbalance_max_min": (
            float(max(nonzero) / max(1, min(nonzero))) if nonzero else None
        ),
        "errors": basic["errors"],
        "orphan_labels": basic["orphan_labels"],
    }
    report["ready_for_training"] = (
        report["valid_labels"]
        and report["empty_images"] == 0
        and not report["missing_classes"]
    )

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if not report["ready_for_training"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
