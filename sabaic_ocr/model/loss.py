from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .box_ops import bbox_ciou, wh_iou
from .yolo import reshape_prediction


@dataclass
class LossBreakdown:
    total: torch.Tensor
    box: torch.Tensor
    obj: torch.Tensor
    cls: torch.Tensor
    positives: int


class YoloLoss(nn.Module):
    """
    Anchor-based YOLO loss implemented inside this repository.

    Targets are a list of tensors, one per image:
      [N, 5] = class_id, cx, cy, w, h
    with normalized coordinates in [0, 1].

    Important: each detected glyph belongs to exactly one of the 30 classes.
    Therefore classification uses categorical cross entropy on positive matches,
    not independent one-vs-all BCE. The earlier BCE formulation averaged one
    positive class together with 29 negatives and allowed frequent classes to
    dominate while the localization branch still learned well.
    """

    def __init__(
        self,
        anchors: Sequence[Sequence[Sequence[float]]],
        num_classes: int,
        image_size: int = 640,
        box_weight: float = 5.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        noobj_weight: float = 0.25,
        cls_label_smoothing: float = 0.02,
        class_weights: Sequence[float] | torch.Tensor | None = None,
    ):
        super().__init__()
        anchors_t = torch.tensor(anchors, dtype=torch.float32)
        if anchors_t.shape != (3, 3, 2):
            raise ValueError("anchors must have shape [3 scales, 3 anchors, 2].")
        self.register_buffer("anchors", anchors_t)
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.box_weight = float(box_weight)
        self.obj_weight = float(obj_weight)
        self.cls_weight = float(cls_weight)
        self.noobj_weight = float(noobj_weight)
        self.cls_label_smoothing = float(cls_label_smoothing)
        if not 0.0 <= self.cls_label_smoothing < 1.0:
            raise ValueError("cls_label_smoothing must be in [0,1).")

        if class_weights is None:
            weights_t = torch.empty(0, dtype=torch.float32)
        else:
            weights_t = torch.as_tensor(class_weights, dtype=torch.float32)
            if weights_t.numel() != self.num_classes:
                raise ValueError(
                    f"class_weights must contain {self.num_classes} values; "
                    f"got {weights_t.numel()}"
                )
            if not torch.isfinite(weights_t).all() or (weights_t <= 0).any():
                raise ValueError("class_weights must be finite and > 0.")
        self.register_buffer("class_weights", weights_t)

        self.obj_bce = nn.BCEWithLogitsLoss(reduction="none")

    def _decode_positive_boxes(
        self,
        raw: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        h: int,
        w: int,
        anchor_wh: torch.Tensor,
    ) -> torch.Tensor:
        px = (raw[:, 0].sigmoid() + grid_x.float()) / float(w)
        py = (raw[:, 1].sigmoid() + grid_y.float()) / float(h)
        pw = torch.exp(raw[:, 2].clamp(max=8.0)) * anchor_wh[:, 0] / float(self.image_size)
        ph = torch.exp(raw[:, 3].clamp(max=8.0)) * anchor_wh[:, 1] / float(self.image_size)
        return torch.stack([px, py, pw, ph], dim=1)

    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> LossBreakdown:
        if len(predictions) != 3:
            raise ValueError("Expected three prediction scales.")

        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        if len(targets) != batch_size:
            raise ValueError("targets list length must match batch size.")

        total_box = torch.zeros((), device=device)
        total_obj = torch.zeros((), device=device)
        total_cls = torch.zeros((), device=device)
        positive_count = 0

        flat_anchors = self.anchors.reshape(-1, 2).to(device)
        reshaped = [reshape_prediction(p, self.num_classes) for p in predictions]
        obj_targets = [torch.zeros_like(p[..., 4], device=device) for p in reshaped]
        assignments = {0: [], 1: [], 2: []}

        for b, t in enumerate(targets):
            if t.numel() == 0:
                continue
            t = t.to(device)
            if t.ndim != 2 or t.shape[1] != 5:
                raise ValueError("Each target tensor must have shape [N, 5].")

            gt_wh_px = t[:, 3:5] * float(self.image_size)
            match = wh_iou(gt_wh_px, flat_anchors)
            candidate_anchors = match.argsort(dim=1, descending=True)

            occupied = {
                scale_idx: {
                    (item[0], item[1], item[2], item[3])
                    for item in assignments[scale_idx]
                }
                for scale_idx in assignments
            }

            for j in range(t.shape[0]):
                cls_id = int(t[j, 0].item())
                if cls_id < 0 or cls_id >= self.num_classes:
                    raise ValueError(f"Invalid class id {cls_id}")

                chosen = None
                for candidate in candidate_anchors[j].tolist():
                    global_a = int(candidate)
                    scale_idx = global_a // 3
                    anchor_idx = global_a % 3
                    pred = reshaped[scale_idx]
                    _, _, h, w, _ = pred.shape
                    gx = float(t[j, 1].item()) * w
                    gy = float(t[j, 2].item()) * h
                    gi = min(w - 1, max(0, int(gx)))
                    gj = min(h - 1, max(0, int(gy)))
                    key = (b, anchor_idx, gj, gi)
                    if key not in occupied[scale_idx]:
                        chosen = (scale_idx, anchor_idx, gj, gi, key)
                        break

                if chosen is None:
                    continue

                scale_idx, anchor_idx, gj, gi, key = chosen
                occupied[scale_idx].add(key)
                assignments[scale_idx].append(
                    (b, anchor_idx, gj, gi, cls_id, t[j, 1:5])
                )
                obj_targets[scale_idx][b, anchor_idx, gj, gi] = 1.0

        for scale_idx, pred in enumerate(reshaped):
            obj_logits = pred[..., 4]
            target_obj = obj_targets[scale_idx]
            obj_raw = self.obj_bce(obj_logits, target_obj)

            pos_mask = target_obj > 0.5
            neg_mask = ~pos_mask
            pos_loss = obj_raw[pos_mask].mean() if pos_mask.any() else torch.zeros((), device=device)
            neg_loss = obj_raw[neg_mask].mean() if neg_mask.any() else torch.zeros((), device=device)
            total_obj = total_obj + pos_loss + self.noobj_weight * neg_loss

        ce_weight = self.class_weights if self.class_weights.numel() else None
        for scale_idx, items in assignments.items():
            if not items:
                continue

            pred = reshaped[scale_idx]
            b_idx = torch.tensor([x[0] for x in items], dtype=torch.long, device=device)
            a_idx = torch.tensor([x[1] for x in items], dtype=torch.long, device=device)
            gj = torch.tensor([x[2] for x in items], dtype=torch.long, device=device)
            gi = torch.tensor([x[3] for x in items], dtype=torch.long, device=device)
            cls_ids = torch.tensor([x[4] for x in items], dtype=torch.long, device=device)
            gt_boxes = torch.stack([x[5].to(device) for x in items], dim=0)

            raw_pos = pred[b_idx, a_idx, gj, gi]
            anchor_wh = self.anchors[scale_idx, a_idx].to(device)

            decoded = self._decode_positive_boxes(
                raw_pos[:, :4], gi, gj, pred.shape[2], pred.shape[3], anchor_wh
            )
            ciou = bbox_ciou(decoded, gt_boxes)
            total_box = total_box + (1.0 - ciou).mean()

            if self.num_classes > 1:
                total_cls = total_cls + F.cross_entropy(
                    raw_pos[:, 5:],
                    cls_ids,
                    weight=ce_weight,
                    label_smoothing=self.cls_label_smoothing,
                    reduction="mean",
                )

            positive_count += len(items)

        total_obj = total_obj / 3.0
        scales_with_pos = sum(1 for v in assignments.values() if v)
        if scales_with_pos:
            total_box = total_box / scales_with_pos
            total_cls = total_cls / scales_with_pos

        total = (
            self.box_weight * total_box
            + self.obj_weight * total_obj
            + self.cls_weight * total_cls
        )
        return LossBreakdown(
            total=total,
            box=total_box.detach(),
            obj=total_obj.detach(),
            cls=total_cls.detach(),
            positives=positive_count,
        )
