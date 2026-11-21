"""Parameter counts and per-component latency (batch size 1, CUDA)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .config import load_config
from .models import build_discriminator, build_generator
from .models.encoders import FrozenCLIPTextEncoder, FrozenVAE


def count(m: torch.nn.Module) -> dict:
    return {
        "total": sum(p.numel() for p in m.parameters()),
        "trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
    }


def timeit(fn, warmup: int = 20, iters: int = 100) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000)
    t = torch.tensor(ms)
    return {"mean_ms": float(t.mean()), "std_ms": float(t.std()), "p95_ms": float(t.quantile(0.95))}


def profile(config_path: str, output: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("profiling needs CUDA")
    cfg = load_config(config_path)
    dev = torch.device("cuda")
    G = build_generator(cfg).to(dev).eval()
    D = build_discriminator(cfg).to(dev).eval()
    vae = FrozenVAE(cfg["models"]["vae"], cfg["models"].get("vae_checkpoint")).to(dev)
    clip = FrozenCLIPTextEncoder(cfg["models"]["clip"]).to(dev)

    img = torch.randn(1, 3, 256, 256, device=dev).clamp(-1, 1)
    z = torch.randn(1, 4, 32, 32, device=dev)
    e = torch.randn(1, 512, device=dev)
    tok = torch.randn(1, 77, 512, device=dev) if cfg["models"].get("use_cross_attention") else None

    with torch.inference_mode():
        report = {
            "gpu": torch.cuda.get_device_name(0),
            "parameters": {
                "generator": count(G),
                "mapping_network": count(G.mapping),
                "discriminator": count(D),
                "vae": count(vae.vae),
                "clip_text_encoder": count(clip.encoder),
            },
            "latency_batch_1": {
                "vae_encode": timeit(lambda: vae.encode(img)),
                "clip_text_encode": timeit(lambda: clip(["a young woman with glasses"]), iters=30),
                "generator": timeit(lambda: G(z, e, tok)),
                "vae_decode": timeit(lambda: vae.decode(z)),
                "discriminator": timeit(lambda: D(z, z)),
            },
        }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full_model.yaml")
    ap.add_argument("--output", default="outputs/profile/full_model.json")
    a = ap.parse_args()
    profile(a.config, a.output)


if __name__ == "__main__":
    main()
