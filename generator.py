from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import StyledResBlock, TextCrossAttention, num_groups


class MappingNetwork(nn.Module):
    """Maps the pooled CLIP embedding to the style vector used by AdaGN."""

    def __init__(self, in_dim: int = 512, style_dim: int = 512, layers: int = 3) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, style_dim), nn.LeakyReLU(0.2, inplace=True)]
            d = style_dim
        self.network = nn.Sequential(*mods)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.network(e)


class _BottleneckBlock(nn.Module):
    def __init__(self, ch: int, style_dim: int, adaptive: bool, noise: bool) -> None:
        super().__init__()
        self.block = StyledResBlock(ch, ch, style_dim, adaptive=adaptive, noise=noise)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.block(x, style)


class MultimodalGenerator(nn.Module):
    """U-Net over 4x32x32 VAE latents.

    The sketch latent goes through the encoder/decoder path. Text enters in two ways:
      * global: pooled CLIP embedding -> mapping network -> AdaGN32 in every block
      * local (optional): token-level cross-attention at the bottleneck and the
        first `cross_attention_levels` decoder stages
    """

    def __init__(
        self,
        latent_channels: int = 4,
        text_dim: int = 512,
        style_dim: int = 512,
        base_channels: int = 64,
        max_channels: int = 512,
        levels: int = 4,
        mapping_layers: int = 3,
        use_adagn: bool = True,
        use_skip_connections: bool = True,
        noise_injection: bool = True,
        use_mapping_network: bool = True,
        use_cross_attention: bool = False,
        cross_attention_levels: int = 2,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.use_skip_connections = use_skip_connections
        self.use_cross_attention = use_cross_attention

        if use_mapping_network:
            self.mapping = MappingNetwork(text_dim, style_dim, mapping_layers)
        else:
            self.mapping = nn.Linear(text_dim, style_dim)

        ch = [min(base_channels * 2**i, max_channels) for i in range(levels)]
        self.input = nn.Conv2d(latent_channels, ch[0], 3, padding=1)

        # encoder
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i in range(levels - 1):
            self.down_blocks.append(
                StyledResBlock(ch[i], ch[i], style_dim, adaptive=use_adagn, noise=noise_injection)
            )
            self.downsamples.append(nn.Conv2d(ch[i], ch[i + 1], 4, stride=2, padding=1))

        # bottleneck (4x4 for a 32x32 input)
        self.bottleneck = nn.Sequential(
            _BottleneckBlock(ch[-1], style_dim, use_adagn, noise_injection),
            _BottleneckBlock(ch[-1], style_dim, use_adagn, noise_injection),
        )
        self.bottleneck_attention = (
            TextCrossAttention(ch[-1], text_dim) if use_cross_attention else None
        )

        # decoder
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        for i in range(levels - 2, -1, -1):
            self.upsamples.append(nn.Conv2d(ch[i + 1], ch[i], 3, padding=1))
            in_ch = ch[i] * 2 if use_skip_connections else ch[i]
            self.up_blocks.append(
                StyledResBlock(in_ch, ch[i], style_dim, adaptive=use_adagn, noise=noise_injection)
            )
            stage = levels - 2 - i
            if use_cross_attention and stage < cross_attention_levels:
                self.up_attentions.append(TextCrossAttention(ch[i], text_dim))
            else:
                self.up_attentions.append(nn.Identity())

        self.output_norm = nn.GroupNorm(num_groups(ch[0]), ch[0])
        self.output = nn.Conv2d(ch[0], latent_channels, 3, padding=1)

    def forward(
        self,
        sketch_latent: torch.Tensor,
        text_embedding: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_cross_attention and text_tokens is None:
            raise ValueError("this generator uses cross-attention, pass text_tokens")

        style = self.mapping(text_embedding)
        x = self.input(sketch_latent)

        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            x = block(x, style)
            skips.append(x)
            x = down(x)

        for block in self.bottleneck:
            x = block(x, style)
        if self.bottleneck_attention is not None:
            x = self.bottleneck_attention(x, text_tokens, text_mask)

        for up, block, attn, skip in zip(
            self.upsamples, self.up_blocks, self.up_attentions, reversed(skips)
        ):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = up(x)
            if self.use_skip_connections:
                x = torch.cat([x, skip], dim=1)
            x = block(x, style)
            if isinstance(attn, TextCrossAttention):
                x = attn(x, text_tokens, text_mask)

        return self.output(F.silu(self.output_norm(x)))


def build_generator(cfg: dict) -> MultimodalGenerator:
    m = cfg["models"]
    return MultimodalGenerator(
        text_dim=512,
        style_dim=m["style_dim"],
        base_channels=m["base_channels"],
        max_channels=m["max_channels"],
        levels=m["levels"],
        mapping_layers=m["mapping_layers"],
        use_adagn=m["use_adagn"],
        use_skip_connections=m["use_skip_connections"],
        noise_injection=m["noise_injection"],
        use_mapping_network=m["use_mapping_network"],
        use_cross_attention=m.get("use_cross_attention", False),
        cross_attention_levels=m.get("cross_attention_levels", 2),
    )
