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
        description="Corrective synthetic training after classification-collapse diagnosis."
    )
    p.add_argument("--config", default="config/train_synthetic_v2.json")
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

    ckpt = Path(cfg["pretrained_checkpoint"])
    if not ckpt.exists():
        raise SystemExit(f"REFUSED: baseline synthetic checkpoint missing: {ckpt}")

    result = train_detector(
        cfg,
        model_cfg,
        init_checkpoint=str(ckpt),
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
