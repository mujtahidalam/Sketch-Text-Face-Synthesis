"""Zero-shot diffusion baseline: SD 1.5 + scribble ControlNet on the test split.

No fine-tuning on MM-CelebA-HQ. The first caption is the prompt and the sketch is
the control image. Score the result with evaluate_images.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .data import read_captions

SD = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET = "lllyasviel/sd-controlnet-scribble"


def generate(manifest, root, output, *, limit=None, seed=42, steps=30, guidance=7.5) -> Path:
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    root, output = Path(root).resolve(), Path(output).resolve()
    img_dir = output / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    df = df[df["split"] == "test"].reset_index(drop=True)
    if limit is not None:
        df = df.iloc[:limit].copy()

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD,
        controlnet=ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=torch.float16),
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    resolve = lambda p: Path(p) if Path(p).is_absolute() else root / p  # noqa: E731
    paths, times = [], []
    for i, rec in enumerate(tqdm(df.to_dict("records"), desc="ControlNet")):
        dst = img_dir / f"{rec['id']}.png"
        ms = None
        if not dst.exists():
            prompt = read_captions(resolve(rec["text"]))[0]
            control = Image.open(resolve(rec["sketch"])).convert("RGB").resize((256, 256))
            t0 = time.perf_counter()
            img = pipe(
                prompt=prompt,
                image=control,
                height=256,
                width=256,
                num_inference_steps=steps,
                guidance_scale=guidance,
                controlnet_conditioning_scale=1.0,
                generator=torch.Generator("cpu").manual_seed(seed + i),
            ).images[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            img.save(dst)
        paths.append(dst.as_posix())
        times.append(ms)

    df["generated"] = paths
    df["generation_ms"] = times
    out_manifest = output / "generated_manifest.csv"
    df.to_csv(out_manifest, index=False)
    (output / "settings.json").write_text(json.dumps({
        "base_model": SD, "controlnet": CONTROLNET, "steps": steps,
        "guidance_scale": guidance, "seed": seed, "resolution": 256,
    }, indent=2), encoding="utf-8")
    return out_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="data/mm_celeba_hq/manifest.csv")
    ap.add_argument("--root", default="data/mm_celeba_hq")
    ap.add_argument("--output", default="outputs/baselines/controlnet")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    print(generate(a.manifest, a.root, a.output, limit=a.limit, seed=a.seed, steps=a.steps))


if __name__ == "__main__":
    main()
