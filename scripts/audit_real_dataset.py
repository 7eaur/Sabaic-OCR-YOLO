#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sabaic_ocr.config import load_classes
from sabaic_ocr.data.dataset import list_images
from sabaic_ocr.data.labels import validate_dataset


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Audit real train/val/test data before fine-tuning.")
    p.add_argument("--root", default="data/real")
    p.add_argument("--classes", default="config/classes.json")
    p.add_argument("--min-train", type=int, default=200)
    p.add_argument("--min-val", type=int, default=20)
    p.add_argument("--min-test", type=int, default=20)
    p.add_argument("--output", default="outputs/real_audit/report.json")
    args = p.parse_args()

    cfg = load_classes(args.classes)
    root = Path(args.root)
    minimums = {"train": args.min_train, "val": args.min_val, "test": args.min_test}
    reports = {}
    hashes = {}
    blockers = []

    for split in ("train", "val", "test"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        report = validate_dataset(
            images_dir, labels_dir, cfg["num_classes"], require_nonempty=True, check_images=True
        )
        reports[split] = report
        imgs = list_images(images_dir)
        if len(imgs) < minimums[split]:
            blockers.append(f"{split}: {len(imgs)} images, required >= {minimums[split]}")
        if not report["valid"]:
            blockers.append(f"{split}: labels/images failed validation")
        hashes[split] = {file_hash(p): p.name for p in imgs}

    duplicates = []
    splits = ("train", "val", "test")
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlap = set(hashes[a]) & set(hashes[b])
            for digest in overlap:
                duplicates.append({"sha256": digest, a: hashes[a][digest], b: hashes[b][digest]})
    if duplicates:
        blockers.append(f"data leakage: {len(duplicates)} exact image duplicate(s) across splits")

    transcript_dir = root / "transcripts" / "test"
    missing_test_transcripts = []
    for image in list_images(root / "images" / "test"):
        tp = transcript_dir / f"{image.stem}.txt"
        if not tp.exists() or not tp.read_text(encoding="utf-8").strip():
            missing_test_transcripts.append(str(tp))
    if missing_test_transcripts:
        blockers.append(f"test: {len(missing_test_transcripts)} missing/empty OCR transcript(s)")

    result = {
        "minimums": minimums,
        "splits": reports,
        "cross_split_duplicates": duplicates[:50],
        "missing_test_transcripts": missing_test_transcripts[:50],
        "ready_for_finetuning": not blockers,
        "blocking_reasons": blockers,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
