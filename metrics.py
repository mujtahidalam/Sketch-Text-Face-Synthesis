"""Shared metric code: FID, LPIPS, SSIM, CLIP score and identity cosine."""
from __future__ import annotations

import pandas as pd
import torch
from torch.nn import functional as F

from .models.encoders import FrozenCLIPScorer, FrozenFaceNet


def ssim_per_image(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim

    fake = (fake.float() + 1) / 2
    real = (real.float() + 1) / 2
    return torch.stack([ssim(fake[i : i + 1], real[i : i + 1], data_range=1.0) for i in range(len(fake))])


class MetricSuite:
    """Accumulates per-image scores and FID statistics over a test set.

    Images are expected in [-1, 1], shape (B, 3, 256, 256).
    """

    def __init__(self, device, clip_name: str = "openai/clip-vit-base-patch32") -> None:
        import lpips
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.clip = FrozenCLIPScorer(clip_name).to(device)
        self.facenet = FrozenFaceNet().to(device)
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.rows: list[dict] = []

    @torch.inference_mode()
    def update(self, ids, fake: torch.Tensor, real: torch.Tensor, texts) -> None:
        clip = self.clip(fake, texts)
        ident = F.cosine_similarity(self.facenet(fake), self.facenet(real), dim=1)
        lp = self.lpips(fake.float(), real.float()).flatten()
        ss = ssim_per_image(fake, real)
        self.fid.update(((real + 1) / 2).clamp(0, 1), real=True)
        self.fid.update(((fake + 1) / 2).clamp(0, 1), real=False)
        for i, sid in enumerate(ids):
            self.rows.append({
                "id": sid,
                "lpips": float(lp[i]),
                "ssim": float(ss[i]),
                "clip": float(clip[i]),
                "identity_cosine": float(ident[i]),
            })

    def summary(self) -> tuple[pd.DataFrame, dict]:
        df = pd.DataFrame(self.rows)
        agg = {"samples": len(df), "fid": float(self.fid.compute())}
        for col in ("lpips", "ssim", "clip", "identity_cosine"):
            agg[f"{col}_mean"] = float(df[col].mean())
            agg[f"{col}_std"] = float(df[col].std(ddof=1))
        return df, agg
