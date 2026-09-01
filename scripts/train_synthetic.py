#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sabaic_ocr.config import load_json
from sabaic_ocr.data.dataset import list_images
from sabaic_ocr.data.labels import validate_dataset
from sabaic_ocr.training.engine import train_detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/train_synthetic.json")
    args = p.parse_args()

    train_cfg = load_json(args.config)
    model_cfg = load_json(train_cfg["model_config"])

    min_train = int(train_cfg.get("min_train_images", 1))
    min_val = int(train_cfg.get("min_val_images", 1))
    train_count = len(list_images(train_cfg["images_dir"]))
    val_count = len(list_images(train_cfg["val_images_dir"]))
    if train_count < min_train or val_count < min_val:
        raise SystemExit(
            f"REFUSED: synthetic dataset incomplete: train={train_count}/{min_train}, "
            f"val={val_count}/{min_val}. Finish and audit data generation first."
        )

    for split_name, images_dir, labels_dir in (
        ("train", train_cfg["images_dir"], train_cfg["labels_dir"]),
        ("val", train_cfg["val_images_dir"], train_cfg["val_labels_dir"]),
    ):
        report = validate_dataset(
            images_dir,
            labels_dir,
            model_cfg["num_classes"],
            require_nonempty=True,
            check_images=True,
        )
        missing_classes = sorted(
            set(range(model_cfg["num_classes"])) - set(int(k) for k in report["class_counts"].keys())
        )
        if not report["valid"] or missing_classes:
            raise SystemExit(
                f"REFUSED: synthetic {split_name} audit failed. "
                f"missing_classes={missing_classes}, errors={report['errors'][:5]}"
            )

    result = train_detector(train_cfg, model_cfg)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
