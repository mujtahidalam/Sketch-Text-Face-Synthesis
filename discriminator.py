from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import DiscResBlock, MinibatchStdDev


class LatentDiscriminator(nn.Module):
    """Sketch-conditioned discriminator on 32x32 latents.

    Input is [image latent, sketch latent] stacked on the channel axis; output is
    one realism logit per sample.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        max_channels: int = 512,
        levels: int = 4,
        minibatch_std_group_size: int = 4,
    ) -> None:
        super().__init__()
        ch = [min(base_channels * 2**i, max_channels) for i in range(levels)]
        self.input = nn.Conv2d(latent_channels * 2, ch[0], 3, padding=1)
        blocks = [DiscResBlock(ch[i], ch[i + 1]) for i in range(levels - 1)]
        blocks.append(DiscResBlock(ch[-1], ch[-1], downsample=False))
        self.blocks = nn.Sequential(*blocks)
        self.minibatch_std = (
            MinibatchStdDev(minibatch_std_group_size) if minibatch_std_group_size > 0 else None
        )
        self.output = nn.Linear(ch[-1] + (1 if self.minibatch_std else 0), 1)

    def forward(self, image_latent: torch.Tensor, sketch_latent: torch.Tensor) -> torch.Tensor:
        x = self.input(torch.cat([image_latent, sketch_latent], dim=1))
        x = F.leaky_relu(self.blocks(x), 0.2)
        if self.minibatch_std is not None:
            x = self.minibatch_std(x)
        return self.output(x.mean(dim=(2, 3))).squeeze(1)


def build_discriminator(cfg: dict) -> LatentDiscriminator:
    m = cfg["models"]
    return LatentDiscriminator(
        base_channels=m["base_channels"],
        max_channels=m["max_channels"],
        levels=m["levels"],
        minibatch_std_group_size=m.get("minibatch_std_group_size", 0),
    )
