"""Make perturbed sketches of the test split for the robustness experiment.

  sparse : thinned strokes with ~28% of 8x8 patches dropped
  dense  : Canny edges (45/120) from the photo instead of the hand-drawn sketch
  noisy  : original strokes + random speckle

Outputs a manifest with a `sketch_style` column, so evaluate.py can run on each
style with --filter-column sketch_style --filter-value <style>.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def _strokes(sketch: np.ndarray) -> np.ndarray:
    # white background / dark lines -> line mask
    return 255 - cv2.cvtColor(sketch, cv2.COLOR_BGR2GRAY)


def sparse(sketch: np.ndarray, seed: int) -> np.ndarray:
    m = cv2.erode(_strokes(sketch), np.ones((2, 2), np.uint8), iterations=1)
    keep = (np.random.default_rng(seed).random((32, 32)) > 0.28).astype(np.uint8)
    keep = cv2.resize(keep, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_NEAREST)
    return 255 - m * keep


def dense(photo: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), 45, 120)
    return 255 - cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)


def noisy(sketch: np.ndarray, seed: int) -> np.ndarray:
    m = _strokes(sketch)
    speckle = (np.random.default_rng(seed).random(m.shape) > 0.992).astype(np.uint8) * 255
    speckle = cv2.dilate(speckle, np.ones((2, 2), np.uint8), iterations=1)
    return 255 - cv2.max(m, speckle)


def create_variants(manifest, root, output) -> Path:
    root, output = Path(root).resolve(), Path(output).resolve()
    df = pd.read_csv(manifest)
    test = df[df["split"] == "test"]
    rows = []
    for rec in tqdm(test.to_dict("records"), desc="sketch variants"):
        sk_path = Path(rec["sketch"]) if Path(rec["sketch"]).is_absolute() else root / rec["sketch"]
        im_path = Path(rec["image"]) if Path(rec["image"]).is_absolute() else root / rec["image"]
        sk = cv2.imread(str(sk_path), cv2.IMREAD_COLOR)
        im = cv2.imread(str(im_path), cv2.IMREAD_COLOR)
        if sk is None or im is None:
            raise FileNotFoundError(f"could not read {sk_path} / {im_path}")
        sid = int(rec["id"])
        rows.append(rec)  # keep the original sketch as its own style
        for style, img in (("sparse", sparse(sk, sid)), ("dense", dense(im)), ("noisy", noisy(sk, sid))):
            path = output / style / f"{rec['id']}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img)
            rows.append({**rec, "sketch": path.as_posix(), "sketch_style": style})
    out_manifest = output / "robustness_manifest.csv"
    pd.DataFrame(rows).to_csv(out_manifest, index=False)
    return out_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="data/mm_celeba_hq/manifest.csv")
    ap.add_argument("--root", default="data/mm_celeba_hq")
    ap.add_argument("--output", default="data/mm_celeba_hq/sketch_variants")
    a = ap.parse_args()
    print(create_variants(a.manifest, a.root, a.output))


if __name__ == "__main__":
    main()
