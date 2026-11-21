# Modified U-Net Generator for Facial Image Synthesis from Sketch and Text Description

Code for our paper on generating face images from a sketch plus a text description.

The model works in the latent space of a frozen VAE. The sketch gives the structure.
The caption is used in two ways:

- **global**: the pooled CLIP text embedding goes through a 3-layer mapping network,
  and the resulting style vector modulates every residual block through AdaGN32
  (adaptive GroupNorm with 32 groups)
- **local**: token-level cross-attention between the U-Net features and the CLIP token
  embeddings, at the 4x4 bottleneck and the 8x8 and 16x16 decoder stages

Training uses a sketch-conditioned latent discriminator with an adversarial loss,
latent reconstruction loss, VGG perceptual loss and a FaceNet identity-preservation loss.

```
sketch (256x256) --VAE enc--> z_sketch (4x32x32) --> U-Net generator --> z_hat --VAE dec--> face (256x256)
                                                       ^         ^
caption --CLIP--> pooled emb --mapping net--> AdaGN32  |         | cross-attention
              \--> token embs -------------------------+---------+
```

## Repository layout

```
configs/                 YAML configs (base = global conditioning only, full_model = everything)
  ablations/             component ablations
scripts/                 shell scripts that run the full set of experiments
src/mmface/
  models/                generator, discriminator, building blocks, frozen encoders
  data.py                manifest-based dataset (pixels or precomputed latents)
  losses.py              perceptual and identity losses
  train.py               training loop
  evaluate.py            FID / LPIPS / SSIM / CLIP score / identity cosine on a split
  prepare_data.py        builds the 70/15/15 split manifest
  precompute_latents.py  caches VAE latents (much faster training)
  stats.py               seed-level and paired statistics between two models
  vae_ceiling.py         VAE reconstruction upper bound
  sketch_variants.py     sparse / dense / noisy sketches for the robustness test
  prepare_ffhq.py        pseudo-paired FFHQ test set (Canny sketches + BLIP captions)
  controlnet_baseline.py zero-shot SD1.5 + ControlNet baseline
  evaluate_images.py     scores any folder of generated images
  profile_model.py       parameter counts and per-component latency
  local_edit.py          fixed-sketch attribute editing analysis
tests/
```

## Setup

Python 3.10+ and a CUDA GPU. All our runs used a single 16 GB card.

```bash
git clone <this repo>
cd <repo>
pip install -r requirements.txt
pip install -e .
```

If `pip install facenet-pytorch` tries to downgrade your PyTorch, install it with
`pip install --no-deps facenet-pytorch==2.6.0`. If it is missing completely, the
code falls back to loading FaceNet through `torch.hub`.

Pretrained weights (`stabilityai/sd-vae-ft-mse`, `openai/clip-vit-base-patch32`,
VGGFace2 FaceNet, VGG16, AlexNet for LPIPS) are downloaded automatically on first use.

## Data

We use [Multi-Modal CelebA-HQ](https://github.com/IIGROUP/MM-CelebA-HQ-Dataset)
(30,000 images, each with a sketch and 10 captions). Arrange it like this:

```
data/raw/MM-CelebA-HQ/
  images/   0.jpg ... 29999.jpg
  sketch/   0.jpg ...
  text/     0.txt ...
```

Then build the split and cache the latents:

```bash
python -m mmface.prepare_data --source-root data/raw/MM-CelebA-HQ --output-root data/mm_celeba_hq --seed 2026
python -m mmface.precompute_latents --config configs/base.yaml
```

In our copy four ids (63, 6400, 9733, 14147) had no sketch, so they are dropped. With
seed 2026 this gives 20,997 train / 4,499 validation / 4,500 test pairs.
`manifest_metadata.json` stores the manifest checksum so you can check you have the same split.

## Training

```bash
# global conditioning only (AdaGN32), 400 epochs
python -m mmface.train --config configs/base.yaml

# full model: + cross-attention + identity loss
python -m mmface.train --config configs/full_model.yaml

# the two remaining cells of the 2x2 study
python -m mmface.train --config configs/crossattn_only.yaml
python -m mmface.train --config configs/identity_only.yaml
```

Main hyperparameters (see `configs/base.yaml`):

| | |
|---|---|
| optimizer | AdamW, betas (0.0, 0.99) |
| learning rate | G 1e-4, D 4e-4 |
| batch size | 64 |
| loss weights | lambda_rec 50, lambda_perc 1, lambda_id 1 (full model) |
| channels | 64 -> 512, 4 levels |
| regularisation | R1 (gamma 1, every 16 steps), minibatch std (group 4), EMA 0.9999 |
| precision | bf16 autocast |

Checkpoints, sample grids and a JSONL loss log are written to `outputs/<experiment>/`.
`--seed`, `--max-steps` and `--output-dir` override the config from the command line.

## Evaluation

```bash
python -m mmface.evaluate --checkpoint outputs/full_model/checkpoints/step_00030000.pt
```

This reports FID (2048-d Inception features), LPIPS (AlexNet), SSIM, CLIP score and
FaceNet identity cosine on the 4,500 test pairs, using the first caption of each
sample. Per-image scores are saved as well, which is what `stats.py` uses for the
paired tests.

The other experiments in the paper:

```bash
bash scripts/run_ablations.sh          # component ablations
bash scripts/run_seeds.sh              # 5 seeds x 2 models + significance tests
bash scripts/run_extra_experiments.sh  # VAE ceiling, robustness, FFHQ, ControlNet, latency, local edits
```

A few notes on these:

- The FFHQ set is pseudo-paired (Canny sketches, BLIP captions). It measures transfer
  to a different photo distribution, not performance on hand-drawn sketches.
- The ControlNet comparison is zero-shot; it is not fine-tuned on MM-CelebA-HQ.
- FID is a set-level metric, so across seeds we compare it at run level (Welch t-test)
  and only use paired per-image tests for LPIPS, SSIM, CLIP and identity cosine.

## Results

Test split of MM-CelebA-HQ (full model):

| FID | Identity cosine | SSIM | CLIP score |
|---|---|---|---|
| 26.90 | 0.522 | 0.527 | 0.257 |

Please see the paper for the full comparison, ablations and statistical analysis.

## Tests

```bash
pytest
```

## Citation

If you find this code useful, please cite:

```bibtex
@article{elahi_modified_unet_face,
  title   = {Modified U-Net Generator for Facial Image Synthesis from Sketch and Text Description},
  author  = {Elahi, Towha and Aonty, Shuhena Salam and Chowdhury, Nayamat Ullah and Alam, Mujtahid},
  journal = {Multimedia Tools and Applications},
  year    = {2026}
}
```

## Acknowledgements

We build on the pretrained VAE from Stable Diffusion (`diffusers`), CLIP (`transformers`),
[facenet-pytorch](https://github.com/timesler/facenet-pytorch), [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
and `torchmetrics`. The dataset is from the TediGAN authors.
