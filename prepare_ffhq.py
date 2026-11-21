"""Build a small pseudo-paired FFHQ test set for cross-dataset evaluation.

FFHQ has no sketches or captions, so we derive them:
  sketch  = inverted Canny edges (55/140) + 2x2 dilation
  caption = BLIP-base caption of the photo

Keep in mind this tests domain transfer, it is not a native sketch/text benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

BLIP = "Salesforce/blip-image-captioning-base"
FFHQ = "merkol/ffhq-256"


def edge_sketch(img: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 55, 140), np.ones((2, 2), np.uint8), iterations=1)
    return Image.fromarray(255 - edges).convert("RGB")


def prepare(output, samples: int = 1000) -> Path:
    from datasets import load_dataset
    from transformers import BlipForConditionalGeneration, BlipProcessor

    out = Path(output).resolve()
    dirs = {k: out / k for k in ("images", "sketches", "text")}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = BlipProcessor.from_pretrained(BLIP)
    blip = BlipForConditionalGeneration.from_pretrained(
        BLIP, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device).eval()

    rows = []
    stream = load_dataset(FFHQ, split="train", streaming=True)
    for i, item in enumerate(tqdm(stream.take(samples), total=samples, desc="FFHQ")):
        img = item["image"].convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        sid = f"ffhq_{i:05d}"
        img_p = dirs["images"] / f"{sid}.jpg"
        sk_p = dirs["sketches"] / f"{sid}.png"
        txt_p = dirs["text"] / f"{sid}.txt"
        img.save(img_p, quality=95)
        edge_sketch(img).save(sk_p)
        with torch.inference_mode():
            ids = blip.generate(**proc(images=img, return_tensors="pt").to(device), max_new_tokens=35)
        txt_p.write_text(proc.decode(ids[0], skip_special_tokens=True).strip() + "\n", encoding="utf-8")
        rows.append({
            "id": sid, "image": img_p.as_posix(), "sketch": sk_p.as_posix(), "text": txt_p.as_posix(),
            "split": "test", "sketch_style": "canny_external", "caption_source": "BLIP-base",
        })

    manifest = out / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    (out / "metadata.json").write_text(json.dumps({
        "source": FFHQ,
        "samples": samples,
        "resolution": 256,
        "sketch": "Canny 55/140 + 2x2 dilation, inverted",
        "captions": BLIP,
        "note": "derived pseudo-pairs, evaluation only",
    }, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="data/external_ffhq")
    ap.add_argument("--samples", type=int, default=1000)
    a = ap.parse_args()
    print(prepare(a.output, a.samples))


if __name__ == "__main__":
    main()
