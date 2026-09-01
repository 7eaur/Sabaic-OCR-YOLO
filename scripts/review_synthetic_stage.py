#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sabaic_ocr.config import load_classes, load_json
from sabaic_ocr.data.dataset import list_images, read_yolo_labels
from sabaic_ocr.data.labels import validate_dataset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_split(root: Path, split: str, num_classes: int, image_size: int) -> dict:
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split
    transcripts_dir = root / "transcripts" / split
    basic = validate_dataset(images_dir, labels_dir, num_classes, require_nonempty=True, check_images=True)

    counts = Counter()
    boxes_per_image = []
    wh_px = []
    missing_transcripts = []
    transcript_empty = []
    hashes = {}

    for img in list_images(images_dir):
        labels = read_yolo_labels(labels_dir / f"{img.stem}.txt", num_classes)
        boxes_per_image.append(int(labels.shape[0]))
        for row in labels.tolist():
            counts[int(row[0])] += 1
            wh_px.append([float(row[3]) * image_size, float(row[4]) * image_size])
        tp = transcripts_dir / f"{img.stem}.txt"
        if not tp.exists():
            missing_transcripts.append(str(tp))
        elif not tp.read_text(encoding="utf-8").strip():
            transcript_empty.append(str(tp))
        hashes[img.stem] = sha256(img)

    wh = np.asarray(wh_px, dtype=np.float32) if wh_px else np.empty((0, 2), np.float32)
    return {
        "split": split,
        "images": len(boxes_per_image),
        "boxes": int(sum(boxes_per_image)),
        "valid_labels": basic["valid"],
        "label_errors": basic["errors"],
        "corrupt_images": basic.get("corrupt_images", []),
        "empty_label_images": basic.get("empty_label_images", []),
        "missing_transcripts": missing_transcripts,
        "empty_transcripts": transcript_empty,
        "missing_classes": [i for i in range(num_classes) if counts[i] == 0],
        "class_counts": {str(i): counts[i] for i in range(num_classes)},
        "boxes_per_image": {
            "min": min(boxes_per_image) if boxes_per_image else None,
            "median": float(np.median(boxes_per_image)) if boxes_per_image else None,
            "max": max(boxes_per_image) if boxes_per_image else None,
        },
        "box_width_px": {
            "min": float(wh[:, 0].min()) if len(wh) else None,
            "median": float(np.median(wh[:, 0])) if len(wh) else None,
            "max": float(wh[:, 0].max()) if len(wh) else None,
        },
        "box_height_px": {
            "min": float(wh[:, 1].min()) if len(wh) else None,
            "median": float(np.median(wh[:, 1])) if len(wh) else None,
            "max": float(wh[:, 1].max()) if len(wh) else None,
        },
        "hashes": hashes,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Gate review for the synthetic-data stage before detector training.")
    p.add_argument("--root", default="data/synthetic")
    p.add_argument("--classes", default="config/classes.json")
    p.add_argument("--model-config", default="config/model.json")
    p.add_argument("--require-train", type=int, default=5000)
    p.add_argument("--require-val", type=int, default=500)
    p.add_argument("--require-test", type=int, default=100)
    p.add_argument("--output", default="outputs/synthetic_audit/stage_review.json")
    args = p.parse_args()

    root = Path(args.root)
    classes = load_classes(args.classes)
    model = load_json(args.model_config)
    reports = {
        s: audit_split(root, s, classes["num_classes"], int(model["image_size"]))
        for s in ("train", "val", "test")
    }

    leakages = []
    splits = ("train", "val", "test")
    for i, a in enumerate(splits):
        rev_a = {}
        for stem, digest in reports[a]["hashes"].items():
            rev_a.setdefault(digest, []).append(stem)
        for b in splits[i + 1:]:
            rev_b = {}
            for stem, digest in reports[b]["hashes"].items():
                rev_b.setdefault(digest, []).append(stem)
            for digest in sorted(set(rev_a) & set(rev_b)):
                leakages.append({"hash": digest, a: rev_a[digest], b: rev_b[digest]})

    requirements = {"train": args.require_train, "val": args.require_val, "test": args.require_test}
    reasons = []
    for split in splits:
        r = reports[split]
        if r["images"] < requirements[split]:
            reasons.append(f"{split} has {r['images']} images; requires {requirements[split]}")
        if not r["valid_labels"]:
            reasons.append(f"{split} label validation failed")
        if r["missing_transcripts"] or r["empty_transcripts"]:
            reasons.append(f"{split} transcript validation failed")
        if r["missing_classes"]:
            reasons.append(f"{split} is missing classes {r['missing_classes']}")
    if leakages:
        reasons.append(f"cross-split duplicate images detected: {len(leakages)}")

    compact = {k: {kk: vv for kk, vv in v.items() if kk != "hashes"} for k, v in reports.items()}
    report = {
        "root": str(root),
        "requirements": requirements,
        "splits": compact,
        "cross_split_duplicate_count": len(leakages),
        "cross_split_duplicates_preview": leakages[:20],
        "ready_for_detector_training": not reasons,
        "blocking_reasons": reasons,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if reasons:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
