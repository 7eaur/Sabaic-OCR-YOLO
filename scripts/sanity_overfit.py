#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from sabaic_ocr.config import load_json
from sabaic_ocr.data.dataset import YoloCharacterDataset
from sabaic_ocr.model.loss import YoloLoss
from sabaic_ocr.model.yolo import SabaicYOLO


def main() -> None:
    p = argparse.ArgumentParser(description="Cheap preflight: verify model/loss can overfit one labeled image.")
    p.add_argument("--images", default="data/synthetic/images/train")
    p.add_argument("--labels", default="data/synthetic/labels/train")
    p.add_argument("--model-config", default="config/model.json")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--output", default="outputs/training_preflight/overfit.json")
    args = p.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    base = load_json(args.model_config)
    scale = args.image_size / float(base["image_size"])
    anchors = [
        [[float(w) * scale, float(h) * scale] for w, h in group]
        for group in base["anchors"]
    ]

    ds = YoloCharacterDataset(
        args.images, args.labels, int(base["num_classes"]), image_size=args.image_size, augment=None
    )
    image, target = ds[0]
    image = image.unsqueeze(0)
    if target.shape[0] == 0:
        raise SystemExit("First training image has no labels; cannot run overfit preflight.")

    model = SabaicYOLO(
        num_classes=int(base["num_classes"]), width_mult=0.25, depth_mult=0.25
    )
    criterion = YoloLoss(
        anchors=anchors, num_classes=int(base["num_classes"]), image_size=args.image_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    checkpoints = {0, 9, 19, 39, args.steps - 1}
    trace = []
    first_loss = None
    final_loss = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        preds = model(image)
        loss = criterion(preds, [target])
        loss.total.backward()
        optimizer.step()
        current = float(loss.total.detach().item())
        if first_loss is None:
            first_loss = current
        final_loss = current
        if step in checkpoints:
            trace.append({
                "step": step + 1,
                "total": current,
                "box": float(loss.box.item()),
                "obj": float(loss.obj.item()),
                "cls": float(loss.cls.item()),
                "positives": int(loss.positives),
            })

    ratio = final_loss / max(first_loss, 1e-9)
    passed = bool(torch.isfinite(torch.tensor(final_loss)).item() and ratio <= 0.55)
    report = {
        "purpose": "training pipeline sanity check only; not a model accuracy result",
        "image": str(ds.images[0]),
        "ground_truth_boxes": int(target.shape[0]),
        "steps": args.steps,
        "initial_loss": first_loss,
        "final_loss": final_loss,
        "final_over_initial": ratio,
        "required_ratio_max": 0.55,
        "passed": passed,
        "trace": trace,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
