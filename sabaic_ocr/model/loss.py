from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn

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
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def _decode_positive_boxes(
        self,
        raw: torch.Tensor,
        grid_x: torch.Tensor,
        grid_y: torch.Tensor,
        h: int,
        w: int,
        anchor_wh: torch.Tensor,
    ) -> torch.Tensor:
        # raw: [N, 4]
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

        # Flatten all anchors so each GT can choose the best anchor globally.
        flat_anchors = self.anchors.reshape(-1, 2).to(device)

        reshaped = [reshape_prediction(p, self.num_classes) for p in predictions]

        # Objectness targets are created for every scale.
        obj_targets = [
            torch.zeros_like(p[..., 4], device=device)
            for p in reshaped
        ]

        # Accumulate positive assignments: each entry maps scale -> list of tuples.
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

            # Dense text can put two character centers in the same grid cell.
            # Assign each GT to the best still-free anchor/cell rather than
            # forcing contradictory box/class targets onto one prediction.
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

                # There are nine anchor slots across the three scales for a
                # given location, so exhaustion is extremely unlikely. If it
                # does happen, skipping is safer than creating conflicting GTs.
                if chosen is None:
                    continue

                scale_idx, anchor_idx, gj, gi, key = chosen
                occupied[scale_idx].add(key)
                assignments[scale_idx].append(
                    (b, anchor_idx, gj, gi, cls_id, t[j, 1:5])
                )
                obj_targets[scale_idx][b, anchor_idx, gj, gi] = 1.0

        # Objectness loss on all cells, weighted to avoid negatives dominating.
        for scale_idx, pred in enumerate(reshaped):
            obj_logits = pred[..., 4]
            target_obj = obj_targets[scale_idx]
            obj_raw = self.bce(obj_logits, target_obj)

            pos_mask = target_obj > 0.5
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_loss = obj_raw[pos_mask].mean()
            else:
                pos_loss = torch.zeros((), device=device)
            if neg_mask.any():
                neg_loss = obj_raw[neg_mask].mean()
            else:
                neg_loss = torch.zeros((), device=device)

            total_obj = total_obj + pos_loss + self.noobj_weight * neg_loss

        # Box and class loss only at positive assignments.
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
                cls_target = torch.zeros(
                    (len(items), self.num_classes), dtype=raw_pos.dtype, device=device
                )
                cls_target.scatter_(1, cls_ids[:, None], 1.0)
                total_cls = total_cls + self.bce(raw_pos[:, 5:], cls_target).mean()

            positive_count += len(items)

        # Average across scales for stable hyperparameters.
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
