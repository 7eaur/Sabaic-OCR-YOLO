from __future__ import annotations

import math
import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(boxes)
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return out


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(boxes)
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + eps)


def wh_iou(wh: torch.Tensor, anchors: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    wh = wh[:, None, :]
    anchors = anchors[None, :, :]
    inter = torch.minimum(wh, anchors).prod(dim=-1)
    union = wh.prod(dim=-1) + anchors.prod(dim=-1) - inter
    return inter / (union + eps)


def bbox_ciou(box1_xywh: torch.Tensor, box2_xywh: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    b1 = xywh_to_xyxy(box1_xywh)
    b2 = xywh_to_xyxy(box2_xywh)
    inter_lt = torch.maximum(b1[:, :2], b2[:, :2])
    inter_rb = torch.minimum(b1[:, 2:], b2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]
    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter
    iou = inter / (union + eps)
    enc_lt = torch.minimum(b1[:, :2], b2[:, :2])
    enc_rb = torch.maximum(b1[:, 2:], b2[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    c2 = enc_wh[:, 0].pow(2) + enc_wh[:, 1].pow(2) + eps
    rho2 = (box1_xywh[:, 0] - box2_xywh[:, 0]).pow(2) + (box1_xywh[:, 1] - box2_xywh[:, 1]).pow(2)
    v = (4 / math.pi**2) * torch.pow(
        torch.atan(box2_xywh[:, 2] / (box2_xywh[:, 3] + eps)) - torch.atan(box1_xywh[:, 2] / (box1_xywh[:, 3] + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return iou - (rho2 / c2 + alpha * v)
