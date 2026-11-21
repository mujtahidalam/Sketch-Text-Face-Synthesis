from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG16_Weights, vgg16

from .models.encoders import FrozenFaceNet, freeze


class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG16 features (relu1_2, relu2_2, relu3_3, relu4_3)."""

    def __init__(self) -> None:
        super().__init__()
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        self.blocks = nn.ModuleList([feats[:4], feats[4:9], feats[9:16], feats[16:23]])
        freeze(self)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake = ((fake + 1) / 2 - self.mean) / self.std
        real = ((real + 1) / 2 - self.mean) / self.std
        loss = fake.new_zeros(())
        for block in self.blocks:
            fake, real = block(fake), block(real)
            loss = loss + F.l1_loss(fake, real.detach())
        return loss


class IdentityLoss(nn.Module):
    """1 - cos(FaceNet(fake), FaceNet(real))."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = FrozenFaceNet()

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        e_fake = self.encoder(fake.float())
        with torch.no_grad():
            e_real = self.encoder(real.float())
        return (1 - F.cosine_similarity(e_fake, e_real, dim=1)).mean()
