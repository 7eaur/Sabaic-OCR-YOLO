#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sabaic_ocr.config import load_json
from sabaic_ocr.data.dataset import list_images
from sabaic_ocr.data.labels import validate_dataset
from sabaic_ocr.training.engine import train_detector


def main():
    p = argparse.ArgumentParser(
        description=(
            "Corrective synthetic training after classification-collapse diagnosis. "
            "By default, resumes automatically from checkpoint_dir/last.pt when it exists."
        )
    )
    p.add_argument("--config", default="config/train_synthetic_v2.json")
    p.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable automatic resume from checkpoint_dir/last.pt.",
    )
    args = p.parse_args()

    cfg = load_json(args.config)
    model_cfg = load_json(cfg["model_config"])

    min_train = int(cfg.get("min_train_images", 5000))
    min_val = int(cfg.get("min_val_images", 500))
    train_count = len(list_images(cfg["images_dir"]))
    val_count = len(list_images(cfg["val_images_dir"]))
    if train_count < min_train or val_count < min_val:
        raise SystemExit(
            f"REFUSED: synthetic v2 requires at least train={min_train}, val={min_val}; "
            f"found train={train_count}, val={val_count}."
        )

    train_audit = validate_dataset(
        cfg["images_dir"], cfg["labels_dir"], model_cfg["num_classes"]
    )
    val_audit = validate_dataset(
        cfg["val_images_dir"], cfg["val_labels_dir"], model_cfg["num_classes"]
    )
    if not train_audit["valid"] or not val_audit["valid"]:
        print(json.dumps({"train": train_audit, "val": val_audit}, indent=2))
        raise SystemExit("REFUSED: synthetic labels are invalid/incomplete.")

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    auto_resume = checkpoint_dir / "last.pt"
    if not args.no_auto_resume and not (cfg.get("resume") or "") and auto_resume.exists():
        cfg["resume"] = str(auto_resume)
        # Resume restores the entire model/optimizer/scheduler state. The
        # classification reset must only happen on the first v2 start.
        cfg["reset_classification_head"] = False
        print(f"AUTO-RESUME: {auto_resume}")

    if cfg.get("resume"):
        resume = Path(cfg["resume"])
        if not resume.exists():
            raise SystemExit(f"REFUSED: resume checkpoint missing: {resume}")
        init_checkpoint = None
    else:
        ckpt = Path(cfg["pretrained_checkpoint"])
        if not ckpt.exists():
            raise SystemExit(f"REFUSED: baseline synthetic checkpoint missing: {ckpt}")
        init_checkpoint = str(ckpt)
        print(f"START V2 FROM BASELINE: {ckpt}")

    result = train_detector(
        cfg,
        model_cfg,
        init_checkpoint=init_checkpoint,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
