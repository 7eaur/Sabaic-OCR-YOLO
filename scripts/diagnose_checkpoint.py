#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sabaic_ocr.config import load_classes
from sabaic_ocr.data.dataset import YoloCharacterDataset, yolo_collate
from sabaic_ocr.model.box_ops import box_iou_xyxy, xywh_to_xyxy
from sabaic_ocr.model.decode import postprocess_batch
from sabaic_ocr.training.engine import build_model


def gt_to_xyxy(target: torch.Tensor) -> torch.Tensor:
    if target.numel() == 0:
        return torch.empty((0, 5), dtype=target.dtype)
    return torch.cat([xywh_to_xyxy(target[:, 1:5]), target[:, :1]], dim=1)


def greedy_class_agnostic_match(pred: torch.Tensor, gt: torch.Tensor, iou_thr: float = 0.5):
    """Greedily match predictions to GT by IoU, ignoring class during localization matching."""
    if pred.numel() == 0 or gt.numel() == 0:
        return [], int(pred.shape[0]), int(gt.shape[0])

    order = pred[:, 4].argsort(descending=True)
    used_gt = torch.zeros(gt.shape[0], dtype=torch.bool)
    matches = []
    unmatched_pred = 0

    for pi in order.tolist():
        ious = box_iou_xyxy(pred[pi, :4].unsqueeze(0), gt[:, :4]).squeeze(0)
        ious = ious.clone()
        ious[used_gt] = -1.0
        best_iou, gi = ious.max(dim=0)
        if float(best_iou) >= iou_thr:
            used_gt[gi] = True
            matches.append((pi, int(gi), float(best_iou)))
        else:
            unmatched_pred += 1

    unmatched_gt = int((~used_gt).sum().item())
    return matches, unmatched_pred, unmatched_gt


def run_threshold(model, loader, cfg, device, conf: float, iou: float):
    total_pred = total_gt = 0
    loc_matches = same_class = wrong_class = 0
    unmatched_pred = unmatched_gt = 0
    pred_hist = Counter()
    gt_hist = Counter()
    confusion = Counter()
    matched_ious = []

    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            detections = postprocess_batch(
                outputs,
                cfg["anchors"],
                cfg["num_classes"],
                cfg["image_size"],
                conf_threshold=conf,
                iou_threshold=iou,
                max_detections=500,
                pre_nms_topk=4000,
            )
            for det, target in zip(detections, targets):
                det = det.detach().cpu()
                gt = gt_to_xyxy(target.detach().cpu())
                total_pred += int(det.shape[0])
                total_gt += int(gt.shape[0])
                pred_hist.update(int(x) for x in det[:, 5].long().tolist())
                gt_hist.update(int(x) for x in gt[:, 4].long().tolist())

                matches, up, ug = greedy_class_agnostic_match(det, gt, 0.5)
                unmatched_pred += up
                unmatched_gt += ug
                loc_matches += len(matches)
                for pi, gi, miou in matches:
                    pc = int(det[pi, 5].item())
                    gc = int(gt[gi, 4].item())
                    matched_ious.append(miou)
                    confusion[(gc, pc)] += 1
                    if pc == gc:
                        same_class += 1
                    else:
                        wrong_class += 1

    loc_precision = loc_matches / max(1, total_pred)
    loc_recall = loc_matches / max(1, total_gt)
    cls_acc_on_localized = same_class / max(1, loc_matches)
    end_to_end_recall = same_class / max(1, total_gt)

    top_confusions = [
        {"gt": int(gc), "pred": int(pc), "count": int(n)}
        for (gc, pc), n in confusion.most_common(30)
        if gc != pc
    ][:20]

    return {
        "confidence_threshold": conf,
        "nms_iou_threshold": iou,
        "predictions": total_pred,
        "ground_truth_boxes": total_gt,
        "localization_matches_iou50_ignoring_class": loc_matches,
        "localization_precision_iou50": loc_precision,
        "localization_recall_iou50": loc_recall,
        "same_class_among_localized": same_class,
        "wrong_class_among_localized": wrong_class,
        "classification_accuracy_on_localized_matches": cls_acc_on_localized,
        "end_to_end_same_class_recall_iou50": end_to_end_recall,
        "unmatched_predictions": unmatched_pred,
        "unmatched_ground_truth": unmatched_gt,
        "mean_iou_of_localized_matches": sum(matched_ious) / max(1, len(matched_ious)),
        "predicted_class_histogram": dict(sorted(pred_hist.items())),
        "ground_truth_class_histogram": dict(sorted(gt_hist.items())),
        "top_wrong_class_pairs": top_confusions,
    }


def main():
    p = argparse.ArgumentParser(description="Diagnose whether a YOLO checkpoint is failing localization, classification, or both.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--classes", default="config/classes.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--conf", type=float, nargs="+", default=[0.05, 0.10, 0.25])
    p.add_argument("--output", default="outputs/evaluation/checkpoint_diagnostic.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = payload["model_config"]
    classes = load_classes(args.classes)
    model = build_model(cfg).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    ds = YoloCharacterDataset(args.images, args.labels, cfg["num_classes"], cfg["image_size"], None, False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=yolo_collate)

    results = [run_threshold(model, loader, cfg, device, float(conf), args.iou) for conf in args.conf]
    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch_index": int(payload.get("epoch", -1)),
        "best_metric": float(payload.get("best_metric", float("nan"))),
        "test_images": len(ds),
        "class_names": {str(c["id"]): c["name"] for c in classes["classes"]},
        "threshold_runs": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
