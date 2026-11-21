"""Evaluate a trained checkpoint on a split of the manifest.

    python -m mmface.evaluate --checkpoint outputs/full_model/checkpoints/step_00030000.pt

Writes per_image_metrics.csv (LPIPS, SSIM, CLIP, identity cosine per test pair)
and aggregate.json (FID plus means/stds, latency, peak memory).
The EMA generator is used when the checkpoint has one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import MultimodalFaceDataset
from .metrics import MetricSuite
from .models import build_generator
from .models.encoders import FrozenCLIPTextEncoder, FrozenVAE
from .train import encode_text, set_seed


def load_generator(checkpoint: str | dict, device, weights: str = "auto"):
    state = checkpoint
    if not isinstance(state, dict):
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    key = weights
    if key == "auto":
        key = "generator_ema" if "generator_ema" in state else "generator"
    if key not in state:
        raise ValueError(f"checkpoint has no {key!r} weights")
    G = build_generator(cfg).to(device).eval()
    G.load_state_dict(state[key])
    return G, cfg, key


def evaluate(
    checkpoint: str,
    split: str = "test",
    limit: int | None = None,
    output_dir: str | None = None,
    manifest: str | None = None,
    data_root: str | None = None,
    seed: int | None = None,
    filter_column: str | None = None,
    filter_value: str | None = None,
    weights: str = "auto",
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    seed = int(state["config"]["experiment"]["seed"] if seed is None else seed)
    # seed before building anything: the noise injection layers draw from the global RNG
    set_seed(seed)
    G, cfg, weights_key = load_generator(state, device, weights)

    name = Path(checkpoint).stem
    out = Path(output_dir) if output_dir else Path(cfg["experiment"]["output_dir"]) / "evaluation" / split / name
    out.mkdir(parents=True, exist_ok=True)

    dcfg = cfg["data"]
    manifest = manifest or dcfg["manifest"]
    data_root = data_root or dcfg["root"]
    ds = MultimodalFaceDataset(
        manifest, root=data_root, split=split,
        image_size=dcfg["image_size"], deterministic_caption=True,
    )
    if filter_column:
        if filter_value is None:
            raise ValueError("--filter-value is required with --filter-column")
        ds.records = ds.records[ds.records[filter_column].astype(str) == filter_value].reset_index(drop=True)
        if ds.records.empty:
            raise ValueError(f"nothing matches {filter_column}={filter_value!r}")
    if limit is not None:
        ds.records = ds.records.iloc[:limit].reset_index(drop=True)
    loader = DataLoader(
        ds, batch_size=cfg["evaluation"]["batch_size"], shuffle=False,
        num_workers=dcfg["num_workers"], pin_memory=True,
    )

    vae = FrozenVAE(cfg["models"]["vae"], cfg["models"].get("vae_checkpoint")).to(device)
    text_encoder = FrozenCLIPTextEncoder(cfg["models"]["clip"]).to(device)
    metrics = MetricSuite(device, cfg["models"]["clip"])
    use_xattn = cfg["models"].get("use_cross_attention", False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen_seconds = 0.0
    for batch in tqdm(loader, desc="evaluate"):
        real = batch["image"].to(device, non_blocking=True)
        sketch = batch["sketch"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            z_sketch = vae.encode(sketch)
            e, tok, mask = encode_text(text_encoder, batch["text"], use_xattn)
            fake = vae.decode(G(z_sketch, e, tok, mask))
            # timing covers sketch encode -> generated image, not the metrics
            if device.type == "cuda":
                torch.cuda.synchronize()
            gen_seconds += time.perf_counter() - t0
            metrics.update(batch["id"], fake, real, batch["text"])

    per_image, agg = metrics.summary()
    per_image.to_csv(out / "per_image_metrics.csv", index=False)
    agg = {
        "checkpoint": name,
        "weights": weights_key,
        "split": split,
        "seed": seed,
        "manifest": str(Path(manifest).resolve()),
        "filter": {"column": filter_column, "value": filter_value} if filter_column else None,
        **agg,
        "ms_per_image": gen_seconds * 1000 / max(len(per_image), 1),
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0,
    }
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(json.dumps(agg, indent=2))
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--output-dir")
    ap.add_argument("--manifest", help="use a different manifest (e.g. sketch variants, FFHQ)")
    ap.add_argument("--data-root")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--filter-column")
    ap.add_argument("--filter-value")
    ap.add_argument("--weights", choices=("auto", "generator", "generator_ema"), default="auto")
    a = ap.parse_args()
    evaluate(a.checkpoint, a.split, a.limit, a.output_dir, a.manifest, a.data_root,
             a.seed, a.filter_column, a.filter_value, a.weights)


if __name__ == "__main__":
    main()
