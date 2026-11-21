"""Build the train/validation/test manifest for MM-CelebA-HQ.

Expects:
    <source-root>/images/*.jpg
    <source-root>/sketch/*.jpg|png
    <source-root>/text/*.txt

Only ids that have all three files are kept (in our copy ids 63, 6400, 9733 and
14147 have no sketch). The split is a seeded 70/15/15 shuffle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def index_files(folder: Path, exts: set[str]) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)
    return {p.stem: p.resolve() for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts}


def relpath(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(source_root, output_root, seed: int = 2026) -> Path:
    src = Path(source_root).resolve()
    out = Path(output_root).resolve()
    images = index_files(src / "images", IMAGE_EXT)
    sketches = index_files(src / "sketch", IMAGE_EXT)
    texts = index_files(src / "text", {".txt"})

    ids = sorted(set(images) & set(sketches) & set(texts), key=int)
    if not ids:
        raise RuntimeError(f"no complete image/sketch/text triplets under {src}")

    order = ids.copy()
    random.Random(seed).shuffle(order)
    n_train = int(len(order) * 0.70)
    n_val = int(len(order) * 0.15)
    split = {}
    for i, sid in enumerate(order):
        split[sid] = "train" if i < n_train else "validation" if i < n_train + n_val else "test"

    rows = [
        {
            "id": sid,
            "image": relpath(images[sid], out),
            "sketch": relpath(sketches[sid], out),
            "text": relpath(texts[sid], out),
            "split": split[sid],
            "sketch_style": "dataset_original",
        }
        for sid in ids
    ]
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    df = pd.DataFrame(rows)
    df.to_csv(manifest, index=False)

    meta = {
        "source_root": str(src),
        "split_seed": seed,
        "complete_triplets": len(ids),
        "split_counts": df["split"].value_counts().sort_index().to_dict(),
        "missing_ids": {
            "image": sorted((set(sketches) | set(texts)) - set(images), key=int),
            "sketch": sorted((set(images) | set(texts)) - set(sketches), key=int),
            "text": sorted((set(images) | set(sketches)) - set(texts), key=int),
        },
        "manifest_sha256": sha256(manifest),
    }
    (out / "manifest_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--output-root", default="data/mm_celeba_hq")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    build_manifest(args.source_root, args.output_root, seed=args.seed)


if __name__ == "__main__":
    main()
