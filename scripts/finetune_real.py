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
        description="Fine-tune the synthetic-pretrained model on real labeled images."
    )
    p.add_argument("--config", default="config/train_real.json")
    args = p.parse_args()

    cfg = load_json(args.config)
    model_cfg = load_json(cfg["model_config"])
    minimum = int(cfg.get("min_real_train_images", 200))

    image_count = len(list_images(cfg["images_dir"]))
    if image_count < minimum:
        raise SystemExit(
            f"REFUSED: real fine-tuning train split has {image_count} images; "
            f"project requirement is at least {minimum}."
        )

    validation = validate_dataset(
        cfg["images_dir"],
        cfg["labels_dir"],
        model_cfg["num_classes"],
        require_nonempty=True,
        check_images=True,
    )
    if not validation["valid"]:
        print(json.dumps(validation, indent=2))
        raise SystemExit("REFUSED: real training labels are invalid/incomplete.")

    ckpt = Path(cfg["pretrained_checkpoint"])
    if not ckpt.exists():
        raise SystemExit(
            f"REFUSED: synthetic pretrained checkpoint does not exist: {ckpt}"
        )

    result = train_detector(
        cfg,
        model_cfg,
        init_checkpoint=str(ckpt),
        require_min_train_images=minimum,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
