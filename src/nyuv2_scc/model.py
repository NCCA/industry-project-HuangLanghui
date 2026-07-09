"""3D residual-attention U-Net for occupancy completion.

The network is a volumetric encoder-decoder (U-Net) that takes an incomplete
binary occupancy grid ``(B, 1, D, H, W)`` and predicts a per-voxel occupancy
logit of the same spatial size. It is "SOTA-inspired" in that it combines three
ingredients common in strong 3D completion models, kept deliberately lightweight
so the whole thing trains on a single GPU (~1.4M parameters at ``base_channels=8``):

* **Residual blocks** (:class:`ResidualBlock3D`) -- ease optimisation of a deep
  3D stack via skip connections.
* **Squeeze-and-excitation attention** (:class:`SEBlock3D`) -- cheap channel
  attention that lets the network re-weight feature channels; toggled by
  ``use_attention`` for the attention ablation.
* **U-Net encoder/decoder with skip connections** -- the four down/up stages
  recover fine geometry by fusing high-resolution encoder features into the
  decoder.

The single output channel is a raw logit (no sigmoid); the sigmoid + threshold
are applied by the loss and metric code.
"""
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
    """Two 3x3x3 conv layers with a residual (skip) connection and optional SE attention.

    The 1x1x1 ``skip`` convolution is only used when the channel count changes,
    so the residual add always has matching shapes.
    """

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
    """Encoder stage: halve the spatial resolution (max-pool), then a residual block."""

    def __init__(self, in_channels: int, out_channels: int, use_attention: bool = True):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ResidualBlock3D(in_channels, out_channels, use_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class UpBlock(nn.Module):
    """Decoder stage: upsample, concatenate the matching encoder skip, then a residual block.

    ``in_channels`` are the deeper (lower-resolution) features being upsampled and
    ``skip_channels`` are the encoder features fused in via the U-Net skip.
    """

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
        """Map an incomplete occupancy grid to per-voxel occupancy logits.

        ``x`` has shape ``(B, 1, D, H, W)``; the return value has the same
        spatial shape ``(B, 1, D, H, W)`` and contains raw logits.
        """
        # Encoder: progressively downsample, keeping each stage's features (e1..e4)
        # so the decoder can fuse them back in through the U-Net skip connections.
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z = self.bottleneck(e4)
        # Decoder: upsample and fuse the matching encoder skip at each stage.
        d4 = self.up4(z, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.out(d1)


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters (used for the model summary line)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
