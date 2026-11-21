"""Train the sketch+text latent GAN.

    python -m mmface.train --config configs/full_model.yaml

Per step:
  1. sketch / target latents (precomputed or encoded on the fly with the frozen VAE)
  2. CLIP text: pooled embedding (+ token states if cross-attention is on)
  3. D update: non-saturating logistic loss, lazy R1 on real latents
  4. G update: L_adv + l1 * L_recon (latent MSE) + l2 * L_perc (VGG) + l3 * L_id (FaceNet)
     The perceptual / identity terms need pixels, so a sub-batch of generated
     latents is decoded through the frozen VAE decoder.
  5. EMA of generator weights (used for evaluation)
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from .config import load_config
from .data import MultimodalFaceDataset, seed_worker
from .losses import IdentityLoss, VGGPerceptualLoss
from .models import build_discriminator, build_generator
from .models.encoders import FrozenCLIPTextEncoder, FrozenVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema.parameters(), model.parameters(), strict=True):
        p_ema.lerp_(p.detach(), 1.0 - decay)
    for b_ema, b in zip(ema.buffers(), model.buffers(), strict=True):
        b_ema.copy_(b)


def save_checkpoint(path, epoch, G, D, opt_g, opt_d, cfg, G_ema=None) -> None:
    state = {
        "epoch": epoch,
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "generator_optimizer": opt_g.state_dict(),
        "discriminator_optimizer": opt_d.state_dict(),
        "config": cfg,
    }
    if G_ema is not None:
        state["generator_ema"] = G_ema.state_dict()
    torch.save(state, path)


def encode_text(text_encoder, captions, use_tokens: bool):
    if use_tokens:
        return text_encoder.encode_with_tokens(captions)
    return text_encoder(captions), None, None


def train(cfg: dict, resume: str | None = None) -> None:
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a CUDA GPU")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False

    out = Path(cfg["experiment"]["output_dir"])
    ckpt_dir, sample_dir = out / "checkpoints", out / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    data, tr, mcfg = cfg["data"], cfg["training"], cfg["models"]
    use_xattn = mcfg.get("use_cross_attention", False)

    dataset = MultimodalFaceDataset(
        data["manifest"],
        root=data["root"],
        split="train",
        image_size=data["image_size"],
        horizontal_flip=data["horizontal_flip"],
        latents_dir=data.get("latents_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=tr["batch_size"],
        shuffle=True,
        num_workers=data["num_workers"],
        pin_memory=data.get("pin_memory", False),
        persistent_workers=data["num_workers"] > 0,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
    )

    G = build_generator(cfg).to(device)
    D = build_discriminator(cfg).to(device)
    vae = FrozenVAE(mcfg["vae"], mcfg.get("vae_checkpoint")).to(device)
    text_encoder = FrozenCLIPTextEncoder(mcfg["clip"]).to(device)
    perceptual = VGGPerceptualLoss().to(device)
    identity = IdentityLoss().to(device) if tr["lambda_identity"] > 0 else None

    betas = tuple(float(b) for b in tr["betas"])
    opt_g = torch.optim.AdamW(G.parameters(), lr=tr["generator_lr"], betas=betas, weight_decay=tr["weight_decay"])
    opt_d = torch.optim.AdamW(D.parameters(), lr=tr["discriminator_lr"], betas=betas, weight_decay=tr["weight_decay"])

    start_epoch, state = 0, None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        G.load_state_dict(state["generator"])
        D.load_state_dict(state["discriminator"])
        opt_g.load_state_dict(state["generator_optimizer"])
        opt_d.load_state_dict(state["discriminator_optimizer"])
        start_epoch = int(state["epoch"]) + 1

    ema_decay = float(tr.get("ema_decay", 0.0))
    G_ema = None
    if ema_decay > 0:
        G_ema = copy.deepcopy(G).eval()
        requires_grad(G_ema, False)
        if state is not None and "generator_ema" in state:
            G_ema.load_state_dict(state["generator_ema"])

    use_amp = bool(tr["mixed_precision"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)  # noqa: E731

    r1_gamma = float(tr.get("r1_gamma", 0.0))
    r1_every = int(tr.get("r1_interval", 16))
    r1_n = int(tr.get("r1_batch_size", 16))
    perc_n_cfg = int(tr.get("perceptual_batch_size", 0))
    max_steps = tr.get("max_steps")

    log_path = out / "training.jsonl"
    step = start_epoch * len(loader)
    done = False

    for epoch in range(start_epoch, int(tr["epochs"])):
        G.train()
        D.train()
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{tr['epochs']}")
        for batch in pbar:
            t0 = time.perf_counter()

            with torch.no_grad():
                if "image_latent" in batch:
                    z_real = batch["image_latent"].to(device, non_blocking=True)
                    z_sketch = batch["sketch_latent"].to(device, non_blocking=True)
                else:
                    z_real = vae.encode(batch["image"].to(device, non_blocking=True))
                    z_sketch = vae.encode(batch["sketch"].to(device, non_blocking=True))
                e_text, tokens, mask = encode_text(text_encoder, batch["text"], use_xattn)

            # ---- discriminator ----
            requires_grad(D, True)
            opt_d.zero_grad(set_to_none=True)
            with autocast():
                z_fake = G(z_sketch, e_text, tokens, mask)
                d_real = D(z_real, z_sketch)
                d_fake = D(z_fake.detach(), z_sketch)
                d_loss = F.softplus(-d_real).mean() + F.softplus(d_fake).mean()

            # lazy R1, done in fp32 on a small sub-batch (double backward is slow in bf16)
            r1 = z_real.new_zeros(())
            if r1_gamma > 0 and step % r1_every == 0:
                n = min(z_real.shape[0], r1_n)
                x = z_real[:n].detach().requires_grad_(True)
                (grad,) = torch.autograd.grad(D(x, z_sketch[:n]).sum(), x, create_graph=True)
                r1 = grad.float().square().flatten(1).sum(1).mean()
                d_loss = d_loss + 0.5 * r1_gamma * r1_every * r1

            scaler.scale(d_loss).backward()
            scaler.unscale_(opt_d)
            nn.utils.clip_grad_norm_(D.parameters(), tr["max_grad_norm"])
            scaler.step(opt_d)

            # ---- generator ----
            requires_grad(D, False)
            opt_g.zero_grad(set_to_none=True)
            with autocast():
                z_fake = G(z_sketch, e_text, tokens, mask)
                adv = F.softplus(-D(z_fake, z_sketch)).mean()
                recon = F.mse_loss(z_fake, z_real)

                # decode only part of the batch to keep memory reasonable
                n = min(z_fake.shape[0], perc_n_cfg) if perc_n_cfg > 0 else z_fake.shape[0]
                x_fake = torch.utils.checkpoint.checkpoint(vae.decode, z_fake[:n], use_reentrant=False)
                if "image" in batch:
                    x_real = batch["image"][:n].to(device, non_blocking=True)
                else:
                    with torch.no_grad():
                        x_real = vae.decode(z_real[:n])

                perc = perceptual(x_fake.float(), x_real.float())
                ident = identity(x_fake, x_real) if identity is not None else x_fake.new_zeros(())
                g_loss = (
                    adv
                    + tr["lambda_reconstruction"] * recon
                    + tr["lambda_perceptual"] * perc
                    + tr["lambda_identity"] * ident
                )

            scaler.scale(g_loss).backward()
            scaler.unscale_(opt_g)
            nn.utils.clip_grad_norm_(G.parameters(), tr["max_grad_norm"])
            scaler.step(opt_g)
            scaler.update()
            if G_ema is not None:
                update_ema(G_ema, G, ema_decay)

            rec = {
                "epoch": epoch,
                "step": step,
                "generator": float(g_loss.detach()),
                "discriminator": float(d_loss.detach()),
                "adversarial": float(adv.detach()),
                "reconstruction": float(recon.detach()),
                "perceptual": float(perc.detach()),
                "identity": float(ident.detach()),
                "r1": float(r1.detach()),
                "seconds": time.perf_counter() - t0,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            pbar.set_postfix(g=f"{rec['generator']:.3f}", d=f"{rec['discriminator']:.3f}")

            if step % 500 == 0:
                # rows: sketch / generated / target
                with torch.no_grad():
                    sk = batch["sketch"][:4].to(device) if "sketch" in batch else vae.decode(z_sketch[:4])
                    tg = batch["image"][:4].to(device) if "image" in batch else vae.decode(z_real[:4])
                grid = torch.cat([sk, x_fake[:4].detach().float(), tg], dim=0)
                save_image((grid.clamp(-1, 1) + 1) / 2, sample_dir / f"step_{step:08d}.jpg", nrow=4)

            step += 1
            if max_steps is not None and step >= int(max_steps):
                done = True
                break

        if (epoch + 1) % int(tr["save_every_epochs"]) == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch + 1:04d}.pt", epoch, G, D, opt_g, opt_d, cfg, G_ema)
        if done:
            save_checkpoint(ckpt_dir / f"step_{step:08d}.pt", epoch, G, D, opt_g, opt_d, cfg, G_ema)
            break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full_model.yaml")
    ap.add_argument("--resume", help="checkpoint to continue from")
    ap.add_argument("--seed", type=int, help="override experiment.seed")
    ap.add_argument("--output-dir", help="override experiment.output_dir")
    ap.add_argument("--max-steps", type=int, help="override training.max_steps")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.max_steps is not None:
        cfg["training"]["max_steps"] = args.max_steps
    train(cfg, args.resume)


if __name__ == "__main__":
    main()
