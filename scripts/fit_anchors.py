#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sabaic_ocr.data.dataset import read_yolo_labels, list_images


def wh_iou_np(wh, centers):
    wh = wh[:, None, :]
    centers = centers[None, :, :]
    inter = np.minimum(wh, centers).prod(axis=2)
    union = wh.prod(axis=2) + centers.prod(axis=2) - inter
    return inter / np.maximum(union, 1e-9)


def kmeans_iou(box_wh: np.ndarray, k: int, seed: int, iterations: int = 100):
    rng = np.random.default_rng(seed)
    if len(box_wh) < k:
        raise ValueError(f"Need at least {k} boxes; found {len(box_wh)}")
    centers = box_wh[rng.choice(len(box_wh), size=k, replace=False)].copy()

    for _ in range(iterations):
        ious = wh_iou_np(box_wh, centers)
        assignment = np.argmax(ious, axis=1)
        new_centers = centers.copy()
        for j in range(k):
            pts = box_wh[assignment == j]
            if len(pts):
                new_centers[j] = np.median(pts, axis=0)
        if np.allclose(new_centers, centers, atol=0.1):
            break
        centers = new_centers
    return centers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--num-classes", type=int, default=30)
    p.add_argument("--k", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    wh = []
    for img in list_images(args.images):
        labels = read_yolo_labels(Path(args.labels) / f"{img.stem}.txt", args.num_classes)
        if labels.numel():
            wh.extend((labels[:, 3:5].numpy() * args.image_size).tolist())

    arr = np.asarray(wh, dtype=np.float32)
    centers = kmeans_iou(arr, args.k, args.seed)
    centers = centers[np.argsort(centers[:, 0] * centers[:, 1])]
    best_iou = wh_iou_np(arr, centers).max(axis=1)
    result = centers.round(2).tolist()
    grouped = [result[0:3], result[3:6], result[6:9]]
    print(json.dumps({
        "anchors": grouped,
        "boxes": int(arr.shape[0]),
        "mean_best_iou": float(best_iou.mean()),
        "anchor_recall_iou50": float((best_iou >= 0.50).mean()),
        "anchor_recall_iou70": float((best_iou >= 0.70).mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
