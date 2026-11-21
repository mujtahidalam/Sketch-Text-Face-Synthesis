"""Score a folder of already generated images (baselines, VAE reconstructions, ...).

The manifest needs columns: id, image (ground truth), generated, text.
An optional generation_ms column is averaged into the report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm

from .data import read_captions
from .metrics import MetricSuite
from .train import set_seed


class GeneratedPairs(Dataset):
    def __init__(self, manifest, root=None, size: int = 256) -> None:
        self.manifest = Path(manifest).resolve()
        self.root = Path(root).resolve() if root else self.manifest.parent
        self.records = pd.read_csv(self.manifest)
        self.size = size
        missing = {"id", "image", "generated", "text"} - set(self.records.columns)
        if missing:
            raise ValueError(f"manifest is missing columns {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.records)

    def _path(self, v) -> Path:
        p = Path(str(v))
        return p if p.is_absolute() else self.root / p

    def _load(self, v) -> torch.Tensor:
        img = Image.open(self._path(v)).convert("RGB")
        return TF.to_tensor(TF.resize(img, [self.size, self.size], antialias=True)) * 2 - 1

    def __getitem__(self, i: int) -> dict:
        r = self.records.iloc[i]
        return {
            "id": str(r.id),
            "target": self._load(r.image),
            "generated": self._load(r.generated),
            "text": read_captions(self._path(r.text))[0],
        }


def evaluate_images(manifest, output, *, root=None, batch_size=16, workers=2, seed=42) -> dict:
    set_seed(seed)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(GeneratedPairs(manifest, root), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=True)
    metrics = MetricSuite(device)
    for b in tqdm(loader, desc="scoring"):
        metrics.update(b["id"], b["generated"].to(device), b["target"].to(device), b["text"])

    per_image, agg = metrics.summary()
    per_image.to_csv(out / "per_image_metrics.csv", index=False)
    gen_ms = pd.read_csv(manifest).get("generation_ms", pd.Series(dtype=float)).dropna()
    agg = {
        "manifest": str(Path(manifest).resolve()),
        "seed": seed,
        **agg,
        "generation_ms_per_image": float(gen_ms.mean()) if len(gen_ms) else None,
    }
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--root")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    evaluate_images(a.manifest, a.output, root=a.root, batch_size=a.batch_size, workers=a.workers, seed=a.seed)


if __name__ == "__main__":
    main()
