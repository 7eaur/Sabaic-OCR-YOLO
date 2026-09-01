from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int | None = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True):
        super().__init__()
        hidden = max(8, channels // 2)
        self.cv1 = ConvBNAct(channels, hidden, 1, 1)
        self.cv2 = ConvBNAct(hidden, channels, 3, 1)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class CSPBlock(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        hidden = max(8, c2 // 2)
        self.left = ConvBNAct(c1, hidden, 1, 1)
        self.right = ConvBNAct(c1, hidden, 1, 1)
        self.blocks = nn.Sequential(*[Bottleneck(hidden) for _ in range(n)])
        self.merge = ConvBNAct(hidden * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.blocks(self.left(x))
        b = self.right(x)
        return self.merge(torch.cat([a, b], dim=1))


class SPPF(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        hidden = max(8, c1 // 2)
        self.cv1 = ConvBNAct(c1, hidden, 1, 1)
        self.cv2 = ConvBNAct(hidden * 4, c2, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Up(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="nearest")
