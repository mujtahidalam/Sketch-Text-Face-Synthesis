"""Fixed-sketch local attribute editing.

For each selected test sample the sketch latent and the injected noise are kept
fixed and only one attribute phrase in the caption is changed (hair colour,
lipstick, eyeglasses). We then measure

  * where the image changed, relative to the face-parsing region of that attribute
    (enrichment = share of change inside region / region area; 1.0 = no localisation)
  * whether the edit was realised (CLIP margin between two attribute prompts)
  * whether identity was kept (FaceNet cosine between base and edited output)
  * for cross-attention models: how much of the edited token's attention falls
    inside the region, compared with the other caption tokens

Cross-attention models are also run with only one pathway receiving the edited
caption ("tokens_only" / "global_only") to separate the two conditioning routes.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from .data import MultimodalFaceDataset, read_captions
from .evaluate import load_generator
from .models.blocks import TextCrossAttention
from .models.encoders import FrozenCLIPTextEncoder, FrozenFaceNet, FrozenVAE, _pooled

# CelebAMask-HQ labels as used by jonathandinu/face-parsing
HAIR, LIPS, EYES = [13], [10, 11, 12], [3, 4, 5, 6, 7]


def _female(c: str) -> bool:
    return re.search(r"\b(She|she|her|woman|lady)\b", c) is not None


EDITS = {
    "hair_blond": dict(
        select=lambda c: re.search(r"\b(black|brown) hair", c) is not None,
        edit=lambda c: re.sub(r"\b(black|brown) hair", "blond hair", c),
        word="blond", region=HAIR,
        clip=("a photo of a person with blond hair", "a photo of a person with dark hair"),
    ),
    "lipstick": dict(
        select=lambda c: _female(c) and "lipstick" not in c and "makeup" not in c,
        edit=lambda c: c.rstrip() + " She is wearing lipstick.",
        word="lipstick", region=LIPS,
        clip=("a photo of a woman wearing bright lipstick", "a photo of a woman with natural lips"),
    ),
    "eyeglasses": dict(
        select=lambda c: "eyeglasses" not in c and "sunglasses" not in c,
        edit=lambda c: c.rstrip() + " This person has eyeglasses.",
        word="eyeglasses", region=EYES,
        clip=("a photo of a person wearing eyeglasses", "a photo of a person without glasses"),
    ),
}


class FaceParser:
    def __init__(self, device) -> None:
        from transformers import SegformerForSemanticSegmentation

        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = ((images + 1) / 2 - self.mean) / self.std
        x = F.interpolate(x, size=512, mode="bilinear", align_corners=False)
        logits = self.model(pixel_values=x).logits
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return logits.argmax(1)


class ClipMargin:
    """CLIP similarity to prompt[0] minus similarity to prompt[1] (x100)."""

    def __init__(self, name: str, device) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.tokenizer = CLIPProcessor.from_pretrained(name).tokenizer
        self.device = device
        self.mean = torch.tensor([0.4815, 0.4578, 0.4082], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2686, 0.2613, 0.2758], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, prompts) -> torch.Tensor:
        x = F.interpolate((images + 1) / 2, size=224, mode="bicubic", align_corners=False)
        x = (x.clamp(0, 1) - self.mean) / self.std
        img = F.normalize(_pooled(self.model.get_image_features(pixel_values=x)), dim=-1)
        tok = self.tokenizer(list(prompts), padding=True, return_tensors="pt").to(self.device)
        txt = F.normalize(_pooled(self.model.get_text_features(**tok)), dim=-1)
        sims = 100 * img @ txt.T
        return sims[:, 0] - sims[:, 1]


def token_positions(tokenizer, caption: str, word: str) -> list[int]:
    pieces = tokenizer.convert_ids_to_tokens(tokenizer(caption, truncation=True, max_length=77)["input_ids"])
    target = tokenizer.tokenize(word)
    n = len(target)
    for start in range(len(pieces) - n, -1, -1):  # last occurrence = the edited phrase
        if pieces[start : start + n] == target:
            return list(range(start, start + n))
    raise ValueError(f"{word!r} not in {caption!r}")


def region_mask(labels: torch.Tensor, classes) -> torch.Tensor:
    m = torch.zeros_like(labels, dtype=torch.float32)
    for c in classes:
        m += (labels == c).float()
    return m.clamp(max=1)


def attention_modules(G) -> list[TextCrossAttention]:
    if not G.use_cross_attention:
        return []
    mods = [G.bottleneck_attention] + [a for a in G.up_attentions if isinstance(a, TextCrossAttention)]
    for m in mods:
        m.keep_attention = True
    return mods


def _u8(img: torch.Tensor) -> np.ndarray:
    return ((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)


def save_examples(folder: Path, img_b, img_e, diff, mask, attn_maps, captions, word, tokenizer) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(len(captions)):
        Image.fromarray(_u8(img_b[k])).save(folder / f"{k}_base.png")
        Image.fromarray(_u8(img_e[k])).save(folder / f"{k}_edit.png")
        d = (diff[k] / (diff[k].max() + 1e-8)).cpu().numpy()
        Image.fromarray((d * 255).astype(np.uint8)).save(folder / f"{k}_diff.png")
        Image.fromarray((mask[k].cpu().numpy() * 255).astype(np.uint8)).save(folder / f"{k}_mask.png")
        if attn_maps:
            tok = token_positions(tokenizer, captions[k], word)
            for w in attn_maps:
                a = w[k, :, :, tok].sum(-1)
                a = (a / a.max()).cpu().numpy()
                Image.fromarray((a * 255).astype(np.uint8)).resize((256, 256), Image.BILINEAR).save(
                    folder / f"{k}_attn{w.shape[1]}.png")
        (folder / f"{k}_caption.txt").write_text(captions[k], encoding="utf-8")


def run(models: dict[str, str], n_per_edit: int, out_dir: Path, seed: int, n_examples: int,
        example_models=("full", "global")) -> None:
    device = torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)

    generators, attn = {}, {}
    cfg = None
    for name, path in models.items():
        G, c, _ = load_generator(path, device, "generator_ema")
        generators[name], attn[name] = G, attention_modules(G)
        cfg = cfg or c

    d = cfg["data"]
    ds = MultimodalFaceDataset(d["manifest"], root=d["root"], split="test",
                               image_size=d["image_size"], deterministic_caption=True)
    captions = [read_captions(ds.resolve(r.text))[0] for r in ds.records.itertuples()]

    vae = FrozenVAE(cfg["models"]["vae"], cfg["models"].get("vae_checkpoint")).to(device)
    text_encoder = FrozenCLIPTextEncoder(cfg["models"]["clip"]).to(device)
    tokenizer = text_encoder.tokenizer
    facenet = FrozenFaceNet().to(device)
    parser = FaceParser(device)
    clip = ClipMargin(cfg["models"]["clip"], device)

    def gen(name, z, e, t, m, batch_seed):
        torch.manual_seed(batch_seed)  # identical noise for base and edited runs
        G = generators[name]
        if not G.use_cross_attention:
            t = m = None
        img = vae.decode(G(z, e, t, m)).float().clamp(-1, 1)
        return img, [a.last_attention for a in attn[name]]

    rng = random.Random(seed)
    rows, attn_rows = [], []
    for edit_name, spec in EDITS.items():
        pool = [i for i, c in enumerate(captions) if spec["select"](c)]
        chosen = sorted(rng.sample(pool, min(n_per_edit, len(pool))))
        print(f"{edit_name}: {len(pool)} candidates, using {len(chosen)}")

        for start in range(0, len(chosen), 16):
            idx = chosen[start : start + 16]
            items = [ds[i] for i in idx]
            base_caps = [captions[i] for i in idx]
            edit_caps = [spec["edit"](c) for c in base_caps]
            batch_seed = seed + start

            with torch.no_grad():
                z = vae.encode(torch.stack([it["sketch"] for it in items]).to(device))
                e_b, t_b, m_b = text_encoder.encode_with_tokens(base_caps)
                e_e, t_e, m_e = text_encoder.encode_with_tokens(edit_caps)

                runs = []
                for name, G in generators.items():
                    if G.use_cross_attention:
                        runs += [(name, "both", e_e, t_e, m_e),
                                 (name, "tokens_only", e_b, t_e, m_e),
                                 (name, "global_only", e_e, t_b, m_b)]
                    else:
                        runs.append((name, "both", e_e, None, None))

                base = {}
                for name, mode, e, t, m in runs:
                    if name not in base:
                        img_b, _ = gen(name, z, e_b, t_b, m_b, batch_seed)
                        base[name] = (img_b, parser(img_b), clip(img_b, spec["clip"]), facenet(img_b))
                    img_b, lab_b, margin_b, id_b = base[name]

                    img_e, maps = gen(name, z, e, t, m, batch_seed)
                    lab_e = parser(img_e)
                    mask = torch.maximum(region_mask(lab_b, spec["region"]), region_mask(lab_e, spec["region"]))
                    diff = (img_e - img_b).abs().mean(1) / 2
                    margin_e = clip(img_e, spec["clip"])
                    id_keep = F.cosine_similarity(facenet(img_e), id_b, dim=1)

                    for k in range(len(idx)):
                        area = float(mask[k].mean())
                        share = float((diff[k] * mask[k]).sum()) / (float(diff[k].sum()) + 1e-8)
                        rows.append(dict(
                            edit=edit_name, model=name, mode=mode, id=items[k]["id"],
                            region_area=area,
                            change_share_in_region=share,
                            enrichment=share / area if area > 0 else np.nan,
                            mean_abs_change=float(diff[k].mean()),
                            clip_margin_gain=float(margin_e[k] - margin_b[k]),
                            edit_success=float(margin_e[k] > margin_b[k]),
                            identity_retained=float(id_keep[k]),
                        ))

                    if mode == "both" and maps:
                        for w in maps:
                            hh, ww = w.shape[1:3]
                            cover = F.adaptive_avg_pool2d(mask[:, None], (hh, ww))[:, 0]
                            for k in range(len(idx)):
                                area = float(cover[k].mean())
                                if area <= 0:
                                    continue
                                tok = token_positions(tokenizer, edit_caps[k], spec["word"])
                                n_valid = int(m[k].sum())
                                a = w[k, :, :, tok].sum(-1)
                                a = a / a.sum()
                                others = [j for j in range(1, n_valid - 1) if j not in tok]
                                o = w[k, :, :, others].mean(-1)
                                o = o / o.sum()
                                attn_rows.append(dict(
                                    edit=edit_name, model=name, level=f"{hh}x{ww}", id=items[k]["id"],
                                    region_area=area,
                                    token_mass_in_region=float((a * cover[k]).sum()),
                                    token_enrichment=float((a * cover[k]).sum()) / area,
                                    other_tokens_enrichment=float((o * cover[k]).sum()) / area,
                                ))

                    if start == 0 and mode == "both" and name in example_models:
                        save_examples(out_dir / "examples" / edit_name / name,
                                      img_b[:n_examples], img_e[:n_examples], diff, mask,
                                      maps, edit_caps[:n_examples], spec["word"], tokenizer)
            print(f"  {edit_name}: {start + len(idx)}/{len(chosen)}")

    edits = pd.DataFrame(rows)
    edits.to_csv(out_dir / "edit_metrics.csv", index=False)
    summary = edits.groupby(["edit", "model", "mode"]).agg(
        n=("id", "count"),
        region_area=("region_area", "mean"),
        change_share_in_region=("change_share_in_region", "mean"),
        enrichment=("enrichment", "median"),
        mean_abs_change=("mean_abs_change", "mean"),
        edit_success=("edit_success", "mean"),
        clip_margin_gain=("clip_margin_gain", "mean"),
        identity_retained=("identity_retained", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "summary_edits.csv", index=False)
    print(summary.round(4).to_string())

    if attn_rows:
        att = pd.DataFrame(attn_rows)
        att["token_beats_others"] = (att["token_enrichment"] > att["other_tokens_enrichment"]).astype(float)
        att.to_csv(out_dir / "attention_metrics.csv", index=False)
        att_sum = att.groupby(["edit", "model", "level"]).agg(
            n=("id", "count"),
            region_area=("region_area", "mean"),
            token_mass_in_region=("token_mass_in_region", "mean"),
            token_enrichment=("token_enrichment", "median"),
            other_tokens_enrichment=("other_tokens_enrichment", "median"),
            frac_token_above_other=("token_beats_others", "mean"),
        ).reset_index()
        att_sum.to_csv(out_dir / "summary_attention.csv", index=False)
        print(att_sum.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--global", dest="global_", default="outputs/base/checkpoints/epoch_0400.pt")
    ap.add_argument("--identity-only", default="outputs/identity_only/checkpoints/step_00030000.pt")
    ap.add_argument("--crossattn-only", default="outputs/crossattn_only/checkpoints/step_00030000.pt")
    ap.add_argument("--full", default="outputs/full_model/checkpoints/step_00030000.pt")
    ap.add_argument("--n", type=int, default=256, help="samples per edit")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--out", default="outputs/local_edit")
    a = ap.parse_args()
    models = {
        "global": a.global_,
        "identity_only": a.identity_only,
        "crossattn_only": a.crossattn_only,
        "full": a.full,
    }
    run(models, a.n, Path(a.out), a.seed, a.examples)


if __name__ == "__main__":
    main()
