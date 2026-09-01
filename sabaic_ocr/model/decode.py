from __future__ import annotations

from typing import List, Sequence

import torch

from .box_ops import box_iou_xyxy, xywh_to_xyxy
from .yolo import reshape_prediction


def decode_predictions(
    predictions: List[torch.Tensor],
    anchors: Sequence[Sequence[Sequence[float]]] | torch.Tensor,
    num_classes: int,
    image_size: int = 640,
    conf_threshold: float = 0.25,
    pre_nms_topk: int = 2000,
) -> List[torch.Tensor]:
    """
    Decode three YOLO scales.

    Returns one tensor per batch item with columns:
      x1, y1, x2, y2, score, class_id
    Coordinates are normalized to [0, 1] before NMS.
    """
    if isinstance(anchors, torch.Tensor):
        anchors_t = anchors.to(predictions[0].device, dtype=predictions[0].dtype)
    else:
        anchors_t = torch.tensor(anchors, device=predictions[0].device, dtype=predictions[0].dtype)

    batch = predictions[0].shape[0]
    decoded_per_image = [[] for _ in range(batch)]

    for scale_idx, raw in enumerate(predictions):
        p = reshape_prediction(raw, num_classes)
        b, a, h, w, _ = p.shape
        device = p.device
        dtype = p.dtype

        gy, gx = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij",
        )
        gx = gx.view(1, 1, h, w)
        gy = gy.view(1, 1, h, w)

        cx = (p[..., 0].sigmoid() + gx) / float(w)
        cy = (p[..., 1].sigmoid() + gy) / float(h)

        aw = anchors_t[scale_idx, :, 0].view(1, a, 1, 1)
        ah = anchors_t[scale_idx, :, 1].view(1, a, 1, 1)
        bw = torch.exp(p[..., 2].clamp(max=8.0)) * aw / float(image_size)
        bh = torch.exp(p[..., 3].clamp(max=8.0)) * ah / float(image_size)

        obj = p[..., 4].sigmoid()
        cls_probs = p[..., 5:].sigmoid()
        cls_conf, cls_id = cls_probs.max(dim=-1)
        score = obj * cls_conf

        boxes = torch.stack([cx, cy, bw, bh], dim=-1)
        boxes = xywh_to_xyxy(boxes).clamp(0.0, 1.0)

        for bi in range(batch):
            mask = score[bi] >= conf_threshold
            if not mask.any():
                continue
            bxs = boxes[bi][mask]
            scs = score[bi][mask]
            ids = cls_id[bi][mask].to(dtype)
            det = torch.cat([bxs, scs[:, None], ids[:, None]], dim=1)
            decoded_per_image[bi].append(det)

    out = []
    for chunks in decoded_per_image:
        if chunks:
            det = torch.cat(chunks, dim=0)
            # Pure-PyTorch NMS is intentionally implemented in this project,
            # but letting tens of thousands of low-confidence candidates reach
            # it would be unnecessarily quadratic. Keep only the strongest
            # candidates before class-aware NMS, as production YOLO pipelines do.
            if pre_nms_topk > 0 and det.shape[0] > pre_nms_topk:
                idx = det[:, 4].topk(pre_nms_topk, largest=True, sorted=False).indices
                det = det[idx]
            out.append(det)
        else:
            out.append(torch.empty((0, 6), device=predictions[0].device))
    return out


def nms_single_class(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Pure PyTorch NMS. Returns indices to keep."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        order = rest[ious <= iou_threshold]
    return torch.stack(keep)


def class_aware_nms(
    detections: torch.Tensor,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
) -> torch.Tensor:
    """
    detections columns: x1,y1,x2,y2,score,class_id.
    """
    if detections.numel() == 0:
        return detections

    kept = []
    class_ids = detections[:, 5].long().unique()
    for cls_id in class_ids:
        mask = detections[:, 5].long() == cls_id
        subset = detections[mask]
        idx = nms_single_class(subset[:, :4], subset[:, 4], iou_threshold)
        kept.append(subset[idx])

    out = torch.cat(kept, dim=0)
    order = out[:, 4].argsort(descending=True)
    return out[order[:max_detections]]


def postprocess_batch(
    predictions: List[torch.Tensor],
    anchors: Sequence[Sequence[Sequence[float]]] | torch.Tensor,
    num_classes: int,
    image_size: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 300,
    pre_nms_topk: int = 2000,
) -> List[torch.Tensor]:
    decoded = decode_predictions(
        predictions, anchors, num_classes, image_size, conf_threshold, pre_nms_topk
    )
    return [
        class_aware_nms(d, iou_threshold=iou_threshold, max_detections=max_detections)
        for d in decoded
    ]
