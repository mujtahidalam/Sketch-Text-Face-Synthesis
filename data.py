from __future__ import annotations

import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from tqdm import tqdm


def read_captions(path: Path) -> list[str]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError(f"no captions in {path}")
    return lines


class MultimodalFaceDataset(Dataset):
    """Image / sketch / caption triplets listed in a manifest CSV.

    Columns: id, image, sketch, text, split. Paths can be relative to `root`
    (defaults to the manifest folder). Each text file holds several captions;
    one is sampled per call unless `deterministic_caption` is set.

    If `latents_dir` is given and contains precomputed latents for every record
    (see precompute_latents.py), the latents are loaded into memory and returned
    instead of pixels. This makes training several times faster.
    """

    REQUIRED = {"id", "image", "sketch", "text", "split"}

    def __init__(
        self,
        manifest: str | Path,
        *,
        root: str | Path | None = None,
        split: str = "train",
        image_size: int = 256,
        horizontal_flip: float = 0.0,
        deterministic_caption: bool = False,
        latents_dir: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.root = Path(root).resolve() if root else self.manifest_path.parent
        df = pd.read_csv(self.manifest_path)
        missing = self.REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"manifest is missing columns {sorted(missing)}")
        self.records = df.loc[df["split"] == split].reset_index(drop=True)
        if self.records.empty:
            raise ValueError(f"no records for split {split!r}")

        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        self.deterministic_caption = deterministic_caption
        self.cache: dict[str, tuple[torch.Tensor, torch.Tensor, list[str]]] | None = None

        if latents_dir is not None:
            self._load_latents(Path(latents_dir) / split)

    def _load_latents(self, folder: Path) -> None:
        n = len(self.records)
        found = len(list(folder.glob("*_image.pt"))) if folder.exists() else 0
        if found < n:
            print(f"[data] only {found}/{n} latents in {folder}, falling back to pixels + live VAE")
            return

        def load(row):
            sid = str(row.id)
            try:
                img = torch.load(folder / f"{sid}_image.pt", weights_only=True).float()
                sk = torch.load(folder / f"{sid}_sketch.pt", weights_only=True).float()
                return sid, (img, sk, read_captions(self.resolve(row.text)))
            except Exception:
                return sid, None

        # lots of tiny files -> threads help a lot with disk latency
        cache, bad = {}, []
        rows = list(self.records.itertuples())
        with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 4) * 4)) as pool:
            for sid, item in tqdm(pool.map(load, rows), total=n, desc="loading latents"):
                if item is None:
                    bad.append(sid)
                else:
                    cache[sid] = item
        if bad:
            print(f"[data] skipping {len(bad)} unreadable latent files, e.g. {bad[:5]}")
            self.records = self.records[~self.records["id"].astype(str).isin(bad)].reset_index(drop=True)
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def resolve(self, value) -> Path:
        p = Path(str(value))
        return p if p.is_absolute() else self.root / p

    def _pick(self, captions: list[str]) -> str:
        return captions[0] if self.deterministic_caption else random.choice(captions)

    def __getitem__(self, index: int) -> dict:
        rec = self.records.iloc[index]
        sid = str(rec.id)

        if self.cache is not None:
            img_z, sk_z, caps = self.cache[sid]
            return {"id": sid, "text": self._pick(caps), "image_latent": img_z, "sketch_latent": sk_z}

        caption = self._pick(read_captions(self.resolve(rec.text)))
        image_path = self.resolve(rec.image)
        sketch_path = self.resolve(rec.sketch)
        size = [self.image_size, self.image_size]
        image = TF.resize(Image.open(image_path).convert("RGB"), size, antialias=True)
        sketch = TF.resize(Image.open(sketch_path).convert("RGB"), size, antialias=True)
        if self.horizontal_flip and random.random() < self.horizontal_flip:
            image, sketch = TF.hflip(image), TF.hflip(sketch)
        return {
            "id": sid,
            "image": TF.to_tensor(image) * 2 - 1,   # [-1, 1]
            "sketch": TF.to_tensor(sketch) * 2 - 1,
            "text": caption,
            "image_path": str(image_path),
            "sketch_path": str(sketch_path),
        }


def seed_worker(worker_id: int) -> None:
    random.seed(torch.initial_seed() % 2**32)
