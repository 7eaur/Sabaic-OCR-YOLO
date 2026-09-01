from __future__ import annotations

from typing import List
import torch
from torch import nn
from .blocks import ConvBNAct, CSPBlock, SPPF, Up, make_divisible


class SabaicYOLO(nn.Module):
    """Custom three-scale anchor-based YOLO detector."""
    def __init__(self, num_classes: int = 30, width_mult: float = 0.5, depth_mult: float = 0.5):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_anchors = 3
        out_ch = self.num_anchors * (5 + self.num_classes)
        c32 = make_divisible(32 * width_mult)
        c64 = make_divisible(64 * width_mult)
        c128 = make_divisible(128 * width_mult)
        c256 = make_divisible(256 * width_mult)
        c512 = make_divisible(512 * width_mult)
        n1 = max(1, round(1 * depth_mult))
        n2 = max(1, round(2 * depth_mult))
        n3 = max(1, round(3 * depth_mult))
        n4 = max(1, round(2 * depth_mult))
        self.stem = ConvBNAct(3, c32, 3, 2)
        self.down1 = ConvBNAct(c32, c64, 3, 2)
        self.stage1 = CSPBlock(c64, c64, n1)
        self.down2 = ConvBNAct(c64, c128, 3, 2)
        self.stage2 = CSPBlock(c128, c128, n2)
        self.down3 = ConvBNAct(c128, c256, 3, 2)
        self.stage3 = CSPBlock(c256, c256, n3)
        self.down4 = ConvBNAct(c256, c512, 3, 2)
        self.stage4 = CSPBlock(c512, c512, n4)
        self.sppf = SPPF(c512, c512)
        self.reduce5 = ConvBNAct(c512, c256, 1, 1)
        self.up = Up()
        self.fuse4 = CSPBlock(c256 + c256, c256, n2)
        self.reduce4 = ConvBNAct(c256, c128, 1, 1)
        self.fuse3 = CSPBlock(c128 + c128, c128, n2)
        self.down3_neck = ConvBNAct(c128, c256, 3, 2)
        self.out4_block = CSPBlock(c256 + c256, c256, n2)
        self.down4_neck = ConvBNAct(c256, c512, 3, 2)
        self.out5_block = CSPBlock(c512 + c512, c512, n2)
        self.head_s8 = nn.Conv2d(c128, out_ch, 1)
        self.head_s16 = nn.Conv2d(c256, out_ch, 1)
        self.head_s32 = nn.Conv2d(c512, out_ch, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        for head in (self.head_s8, self.head_s16, self.head_s32):
            if head.bias is not None:
                with torch.no_grad():
                    b = head.bias.view(self.num_anchors, -1)
                    b[:, 4] = -4.5
                    head.bias.copy_(b.reshape(-1))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(self.down1(x))
        p3 = self.stage2(self.down2(x))
        p4 = self.stage3(self.down3(p3))
        p5 = self.sppf(self.stage4(self.down4(p4)))
        n4 = self.fuse4(torch.cat([self.up(self.reduce5(p5)), p4], dim=1))
        n3 = self.fuse3(torch.cat([self.up(self.reduce4(n4)), p3], dim=1))
        o4 = self.out4_block(torch.cat([self.down3_neck(n3), n4], dim=1))
        o5 = self.out5_block(torch.cat([self.down4_neck(o4), p5], dim=1))
        return [self.head_s8(n3), self.head_s16(o4), self.head_s32(o5)]


def reshape_prediction(pred: torch.Tensor, num_classes: int, num_anchors: int = 3) -> torch.Tensor:
    b, _, h, w = pred.shape
    return pred.view(b, num_anchors, 5 + num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
