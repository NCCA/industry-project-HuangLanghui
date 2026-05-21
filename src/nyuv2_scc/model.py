from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SEBlock3D(nn.Module):
    """Squeeze-and-excitation attention for 3D feature maps."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.pool(x))


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.att = SEBlock3D(out_channels) if use_attention else nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.att(out)
        return F.relu(out + residual, inplace=True)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ResidualBlock3D(in_channels, out_channels, use_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels, use_attention)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # In case a non-power-of-two grid is used, center-crop/pad via interpolation.
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class ResidualAttentionUNet3D(nn.Module):
    """Lightweight SOTA-inspired 3D U-Net for binary occupancy completion."""

    def __init__(self, in_channels: int = 1, base_channels: int = 8, use_attention: bool = True):
        super().__init__()
        b = base_channels
        self.enc1 = ResidualBlock3D(in_channels, b, use_attention)
        self.enc2 = DownBlock(b, b * 2, use_attention)
        self.enc3 = DownBlock(b * 2, b * 4, use_attention)
        self.enc4 = DownBlock(b * 4, b * 8, use_attention)
        self.bottleneck = DownBlock(b * 8, b * 16, use_attention)
        self.up4 = UpBlock(b * 16, b * 8, b * 8, use_attention)
        self.up3 = UpBlock(b * 8, b * 4, b * 4, use_attention)
        self.up2 = UpBlock(b * 4, b * 2, b * 2, use_attention)
        self.up1 = UpBlock(b * 2, b, b, use_attention)
        self.out = nn.Conv3d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z = self.bottleneck(e4)
        d4 = self.up4(z, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
