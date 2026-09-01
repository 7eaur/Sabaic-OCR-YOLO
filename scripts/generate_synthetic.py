#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sabaic_ocr.config import load_classes, load_json
from sabaic_ocr.data.synthetic import SyntheticSabaicGenerator, load_corpus, save_sample


def _stems(path: Path, suffix: str) -> set[str]:
    if not path.exists():
        return set()
    return {p.stem for p in path.glob(f"*{suffix}") if p.is_file()}


def existing_split_count(root: Path, split: str) -> int:
    images = _stems(root / "images" / split, ".jpg")
    labels = _stems(root / "labels" / split, ".txt")
    transcripts = _stems(root / "transcripts" / split, ".txt")
    if not (images == labels == transcripts):
        missing_labels = sorted(images - labels)[:10]
        missing_transcripts = sorted(images - transcripts)[:10]
        orphan_labels = sorted(labels - images)[:10]
        raise RuntimeError(
            f"Split {split!r} is inconsistent before generation. "
            f"missing_labels={missing_labels}, missing_transcripts={missing_transcripts}, "
            f"orphan_labels={orphan_labels}"
        )
    return len(images)


def generate_split(generator, root: Path, split: str, target_count: int, start_index: int = 0):
    count = max(0, target_count - start_index)
    for i in range(count):
        sample = generator.make_sample()
        index = start_index + i
        name = f"sabaic_{split}_{index:06d}"
        save_sample(
            sample,
            root / "images" / split / f"{name}.jpg",
            root / "labels" / split / f"{name}.txt",
            root / "transcripts" / split / f"{name}.txt",
        )
        if (i + 1) % 250 == 0 or i + 1 == count:
            print(f"{split}: {index+1}/{target_count}")


def main():
    p = argparse.ArgumentParser(description="Generate custom synthetic Sabaic YOLO dataset.")
    p.add_argument("--output", default="data/synthetic")
    p.add_argument("--font", default="assets/fonts/NotoSansOldSouthArabian-Regular.ttf")
    p.add_argument("--classes", default="config/classes.json")
    p.add_argument("--model-config", default="config/model.json")
    p.add_argument("--corpus", default="")
    p.add_argument("--train", type=int, default=5000, help="Target train image count")
    p.add_argument("--val", type=int, default=500, help="Target validation image count")
    p.add_argument("--test", type=int, default=100, help="Target synthetic test image count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue incomplete splits up to the requested target counts without overwriting existing samples.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output dataset first. Mutually exclusive with --resume.",
    )
    args = p.parse_args()

    if args.resume and args.overwrite:
        raise SystemExit("Use only one of --resume or --overwrite.")
    for name in ("train", "val", "test"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name} must be >= 0")

    classes_cfg = load_classes(args.classes)
    model_cfg = load_json(args.model_config)
    corpus = load_corpus(args.corpus if args.corpus else None)
    root = Path(args.output)

    if args.overwrite and root.exists():
        shutil.rmtree(root)

    existing = {}
    for split in ("train", "val", "test"):
        existing[split] = existing_split_count(root, split)
        target = int(getattr(args, split))
        if existing[split] > target:
            raise SystemExit(
                f"Split {split} already has {existing[split]} images, greater than requested target {target}."
            )
        if existing[split] and not args.resume:
            raise SystemExit(
                f"Refusing to overwrite existing {split} data ({existing[split]} images). "
                "Use --resume to continue or --overwrite to regenerate from scratch."
            )

    split_seed_offsets = {"train": 0, "val": 1_000_003, "test": 2_000_003}
    targets = {"train": args.train, "val": args.val, "test": args.test}
    run_details = {}

    for split in ("train", "val", "test"):
        target = int(targets[split])
        start = int(existing[split])
        if target == start:
            run_details[split] = {"existing": start, "generated": 0, "target": target}
            continue
        split_seed = int(args.seed + split_seed_offsets[split] + start * 7919)
        generator = SyntheticSabaicGenerator(
            classes=classes_cfg["classes"],
            font_path=args.font,
            image_size=int(model_cfg["image_size"]),
            seed=split_seed,
            corpus_lines=corpus,
        )
        generate_split(generator, root, split, target, start)
        run_details[split] = {
            "existing": start,
            "generated": target - start,
            "target": target,
            "seed": split_seed,
        }

    actual = {split: existing_split_count(root, split) for split in ("train", "val", "test")}
    manifest_path = root / "manifest.json"
    previous_runs = []
    if manifest_path.exists():
        try:
            previous_runs = json.loads(manifest_path.read_text(encoding="utf-8")).get("generation_runs", [])
        except Exception:
            previous_runs = []

    manifest = {
        "generator": "custom_internal_generator",
        "target_counts": {"train": args.train, "val": args.val, "test": args.test},
        "actual_counts": actual,
        "image_size": model_cfg["image_size"],
        "base_seed": args.seed,
        "font_path": args.font,
        "corpus_path": args.corpus or None,
        "generation_runs": previous_runs + [{
            "utc": datetime.now(timezone.utc).isoformat(),
            "resume": bool(args.resume),
            "overwrite": bool(args.overwrite),
            "details": run_details,
        }],
        "note": (
            "Random sequences are detector-training data and are not claimed to be "
            "historically valid Sabaic unless an externally verified corpus is supplied."
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
