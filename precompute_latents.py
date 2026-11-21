"""Encode every image and sketch once with the frozen VAE.

Writes float16 tensors to <latents_dir>/<split>/<id>_image.pt and <id>_sketch.pt.
Training picks them up automatically when `data.latents_dir` is set in the config.
Safe to re-run: files that already exist are skipped.

    python -m mmface.precompute_latents --config configs/base.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .data import MultimodalFaceDataset
from .models.encoders import FrozenVAE


def precompute(config_path: str, splits=None, batch_size: int = 32, num_workers: int = 0) -> None:
    cfg = load_config(config_path)
    latents_dir = cfg["data"].get("latents_dir")
    if not latents_dir:
        raise ValueError("set data.latents_dir in the config first")

    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = FrozenVAE(cfg["models"]["vae"], cfg["models"].get("vae_checkpoint")).to(device)
    root = Path(latents_dir)

    for split in splits or ["train", "validation", "test"]:
        out = root / split
        out.mkdir(parents=True, exist_ok=True)
        ds = MultimodalFaceDataset(
            cfg["data"]["manifest"],
            root=cfg["data"]["root"],
            split=split,
            image_size=cfg["data"]["image_size"],
            deterministic_caption=True,  # no flips, no caption sampling
        )
        done = len(list(out.glob("*_image.pt")))
        if done >= len(ds):
            print(f"{split}: all {len(ds)} latents already there")
            continue

        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"encoding {split}"):
                ids = batch["id"]
                todo = [i for i, s in enumerate(ids) if not (out / f"{s}_image.pt").exists()]
                if not todo:
                    continue
                try:
                    z_img = vae.encode(batch["image"].to(device)).half().cpu()
                    z_sk = vae.encode(batch["sketch"].to(device)).half().cpu()
                except Exception as e:  # e.g. one broken file in the batch
                    print(f"skipping batch starting at id {ids[0]}: {e}")
                    continue
                for i in todo:
                    torch.save(z_img[i], out / f"{ids[i]}_image.pt")
                    torch.save(z_sk[i], out / f"{ids[i]}_sketch.pt")
        print(f"{split}: {len(list(out.glob('*_image.pt')))}/{len(ds)} latents in {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    precompute(args.config, args.splits, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
