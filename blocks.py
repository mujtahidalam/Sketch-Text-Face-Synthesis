from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def num_groups(channels: int, requested: int = 32) -> int:
    # largest group count <= requested that divides the channel count
    groups = min(requested, channels)
    while channels % groups:
        groups -= 1
    return groups


class NoiseInjection(nn.Module):
    """x + w * N(0, I), with a learned per-channel weight (StyleGAN2 style)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            b, _, h, w = x.shape
            noise = torch.randn(b, 1, h, w, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class AdaGN(nn.Module):
    """AdaGN32: gamma(s) * GroupNorm_32(x) + beta(s)."""

    def __init__(self, channels: int, style_dim: int, groups: int = 32) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups(channels, groups), channels, affine=False)
        self.affine = nn.Linear(style_dim, channels * 2)
        # start as an identity modulation
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(style).chunk(2, dim=1)
        return self.norm(x) * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class Norm(nn.Module):
    """AdaGN when `adaptive`, otherwise a plain GroupNorm (used in the ablation)."""

    def __init__(self, channels: int, style_dim: int, adaptive: bool = True) -> None:
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.norm = AdaGN(channels, style_dim)
        else:
            self.norm = nn.GroupNorm(num_groups(channels), channels)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.norm(x, style) if self.adaptive else self.norm(x)


class StyledResBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        style_dim: int,
        *,
        adaptive: bool = True,
        noise: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = Norm(in_ch, style_dim, adaptive)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = Norm(out_ch, style_dim, adaptive)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.noise1 = NoiseInjection(out_ch) if noise else nn.Identity()
        self.noise2 = NoiseInjection(out_ch) if noise else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        x = self.noise1(self.conv1(F.silu(self.norm1(x, style))))
        x = self.noise2(self.conv2(F.silu(self.norm2(x, style))))
        return (x + res) / math.sqrt(2.0)


class MinibatchStdDev(nn.Module):
    """Appends the average feature std over a group of samples as one extra channel."""

    def __init__(self, group_size: int = 4) -> None:
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = min(self.group_size, b)
        while b % g:
            g -= 1
        y = x.float().reshape(g, -1, c, h, w)
        y = y - y.mean(dim=0, keepdim=True)
        y = (y.square().mean(dim=0) + 1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, h, w)
        return torch.cat([x, y.to(x.dtype)], dim=1)


class DiscResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
            res = F.avg_pool2d(res, 2)
        return (x + res) / math.sqrt(2.0)


class TextCrossAttention(nn.Module):
    """Spatial features (queries) attend over CLIP token embeddings (keys/values).

    The result is added back to the feature map as a residual update.
    Set `keep_attention = True` to store the head-averaged attention map of the
    last call in `last_attention` with shape (B, H, W, T); used for analysis only.
    """

    def __init__(self, channels: int, token_dim: int = 512, heads: int = 8) -> None:
        super().__init__()
        heads = min(heads, channels)
        while channels % heads:
            heads -= 1
        self.visual_norm = nn.GroupNorm(num_groups(channels), channels)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_projection = nn.Linear(token_dim, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.output = nn.Linear(channels, channels)
        self.keep_attention = False
        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        visual: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, c, h, w = visual.shape
        q = self.visual_norm(visual).flatten(2).transpose(1, 2)  # (B, HW, C)
        kv = self.token_projection(self.token_norm(tokens))       # (B, T, C)
        pad = None if mask is None else ~mask.bool()
        out, weights = self.attention(
            q, kv, kv, key_padding_mask=pad, need_weights=self.keep_attention
        )
        if self.keep_attention:
            self.last_attention = weights.detach().reshape(b, h, w, -1)
        out = self.output(out).transpose(1, 2).reshape(b, c, h, w)
        return visual + out
