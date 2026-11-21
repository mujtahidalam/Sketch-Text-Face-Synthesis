"""Frozen pretrained networks: VAE, CLIP text encoder, CLIP scorer and FaceNet."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _pooled(out):
    # newer transformers versions return a ModelOutput here instead of a tensor
    return out.pooler_output if hasattr(out, "pooler_output") else out


class FrozenVAE(nn.Module):
    """KL autoencoder (sd-vae-ft-mse): 3x256x256 <-> 4x32x32."""

    def __init__(self, model_name: str, checkpoint: str | None = None) -> None:
        super().__init__()
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(model_name)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.vae.load_state_dict(state.get("vae", state))
        self.vae = freeze(self.vae)
        self.scaling_factor = float(self.vae.config.scaling_factor)

    @torch.no_grad()
    def encode(self, image: torch.Tensor, sample: bool = False) -> torch.Tensor:
        dist = self.vae.encode(image).latent_dist
        z = dist.sample() if sample else dist.mode()
        return z * self.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # not wrapped in no_grad: the generator loss backprops through the decoder
        return self.vae.decode(latent / self.scaling_factor).sample.clamp(-1, 1)


class FrozenCLIPTextEncoder(nn.Module):
    """Returns the L2-normalised pooled (EOT) embedding and, optionally, per-token states."""

    def __init__(self, model_name: str, max_length: int = 77) -> None:
        super().__init__()
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.encoder = freeze(CLIPTextModelWithProjection.from_pretrained(model_name))
        self.max_length = max_length

    @torch.no_grad()
    def forward(self, texts) -> torch.Tensor:
        return self.encode_with_tokens(texts)[0]

    @torch.no_grad()
    def encode_with_tokens(self, texts):
        tok = self.tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.encoder.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.encoder(**tok)
        pooled = F.normalize(out.text_embeds.float(), dim=1)
        return pooled, out.last_hidden_state.float(), tok["attention_mask"]


class FrozenCLIPScorer(nn.Module):
    """Image-text cosine similarity with CLIP ViT-B/32 (evaluation only)."""

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name: str, max_length: int = 77) -> None:
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = freeze(CLIPModel.from_pretrained(model_name))
        self.max_length = max_length
        self.register_buffer("mean", torch.tensor(self.MEAN)[None, :, None, None])
        self.register_buffer("std", torch.tensor(self.STD)[None, :, None, None])

    @torch.no_grad()
    def forward(self, images: torch.Tensor, texts) -> torch.Tensor:
        x = F.interpolate((images.float() + 1) / 2, (224, 224), mode="bicubic", align_corners=False)
        x = (x - self.mean) / self.std
        tok = self.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}
        img = F.normalize(_pooled(self.model.get_image_features(pixel_values=x)), dim=1)
        txt = F.normalize(_pooled(self.model.get_text_features(**tok)), dim=1)
        return (img * txt).sum(dim=1)


class FrozenFaceNet(nn.Module):
    """InceptionResnetV1 pretrained on VGGFace2. Returns unit-norm 512-d embeddings."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1

            net = InceptionResnetV1(pretrained="vggface2")
        except ImportError:
            # facenet-pytorch pins old torch versions on pip; loading it through
            # torch.hub avoids having it downgrade torch.
            net = torch.hub.load(
                "timesler/facenet-pytorch:master",
                "inception_resnet_v1",
                pretrained="vggface2",
                trust_repo=True,
            )
        self.encoder = freeze(net)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images, size=(160, 160), mode="bilinear", align_corners=False)
        return F.normalize(self.encoder(x), dim=1)
