"""How much of the final quality is just the VAE?

Two reference points on the test split, scored with the same metrics as the model:
  * target_reconstruction : decode(encode(target image))  -> upper bound for any latent generator
  * sketch_latent_decode   : decode(encode(sketch))       -> what you get with no generator at all
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import load_config
from .data import MultimodalFaceDataset
from .evaluate_images import evaluate_images
from .models.encoders import FrozenVAE


def run(config_path: str, output: str, limit: int | None = None) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = cfg["data"]
    ds = MultimodalFaceDataset(d["manifest"], root=d["root"], split="test",
                               image_size=d["image_size"], deterministic_caption=True)
    if limit:
        ds.records = ds.records.iloc[:limit].reset_index(drop=True)
    text_by_id = dict(zip(ds.records["id"].astype(str), ds.records["text"]))
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    vae = FrozenVAE(cfg["models"]["vae"], cfg["models"].get("vae_checkpoint")).to(device)

    out = Path(output).resolve()
    dirs = {"target_reconstruction": out / "target_reconstruction",
            "sketch_latent_decode": out / "sketch_latent_decode"}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    rows = {k: [] for k in dirs}

    for b in tqdm(loader, desc="VAE ceiling"):
        with torch.inference_mode():
            recon = vae.decode(vae.encode(b["image"].to(device)))
            sk = vae.decode(vae.encode(b["sketch"].to(device)))
        for i, sid in enumerate(b["id"]):
            base = {"id": sid, "image": b["image_path"][i], "text": text_by_id[str(sid)]}
            for key, imgs in (("target_reconstruction", recon), ("sketch_latent_decode", sk)):
                path = dirs[key] / f"{sid}.png"
                save_image((imgs[i] + 1) / 2, path)
                rows[key].append({**base, "generated": path.as_posix()})

    for key, r in rows.items():
        manifest = out / f"{key}.csv"
        pd.DataFrame(r).to_csv(manifest, index=False)
        evaluate_images(manifest, out / f"metrics_{key}", root=d["root"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--output", default="outputs/vae_ceiling")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.config, a.output, a.limit)


if __name__ == "__main__":
    main()
