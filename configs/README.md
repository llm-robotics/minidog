# Configs

All configs train the same model: a 12-layer, 384-wide LightningDiT (6 heads, patch size 1) on
16x16x32 latents from a frozen VAE, conditioned on Qwen3-0.6B caption embeddings, with
velocity-prediction flow matching and 4 timestep tokens. They differ only in the knobs below.
Every FID is a single run of 200 epochs (~20k steps), measured against the 26k pretraining images.

## The walkthrough

| Config | VAE | Alignment | Caption tokens | Epochs | EMA | LR | Notes |
|---|---|---|---|---|---|---|---|
| `pretrain_irepa_eupe_mse.yaml` | E2E-INVAE | iREPA, EUPE, squared error | 128 | 200 | 0.9995 | 1e-4, 100 warmup | **The final recipe.** FID 8.32 |
| `pretrain_repa_dinov2_mse.yaml` | E2E-INVAE | REPA, DINOv2, squared error | 128 | 200 | 0.9995 | 1e-4, 100 warmup | The baseline the lessons build up to. FID 8.80 |
| `sft_eupe_norepa.yaml` | E2E-INVAE | none | 128 | 100 | 0.995 | 5e-5, no warmup | **Fine-tune the 8.32 checkpoint on 2k synthetic dogs, alignment off.** HPSv2 0.1625 -> 0.2427, PickScore 18.42 -> 19.61. Launch with `--ckpt <irepa-eupe ckpt> --init-weights-only` |

## Target representations (Lesson 4)

Sixteen configs named `pretrain_{repa,irepa}_{target}_{mse,cosine}.yaml`, identical apart from the
alignment. REPA uses a linear projector; iREPA uses a 3x3 convolution with spatial normalisation
(gamma 0.7). All use E2E-INVAE, 128 caption tokens and lambda 0.5. FID at 20k:

| Target representation | REPA, squared error | REPA, cosine | iREPA, squared error | iREPA, cosine |
|---|---|---|---|---|
| `dinov2` (ViT-B/14) | **8.80** | 9.14 | 8.68 | 8.89 |
| `pe_spatial` (B/16) | 9.56 | 9.89 | 9.65 | 9.99 |
| `eupe` (ViT-B/16) | 9.71 | 10.20 | **8.32** | 8.37 |
| `dinov3` (ViT-B/16) | 10.12 | 10.32 | 8.46 | 9.01 |

Bold: best overall in each method, across both distances. Each target needs its own latents folder,
cached with the matching `minidog.precompute_latents_{dinov3,pe_spatial,eupe}` script.

## Tokenizer and caption length (Lessons 1-2)

| Config | VAE | REPA | Caption tokens | FID |
|---|---|---|---|---|
| `e2e-invae-norepa-128tok.yaml` | E2E-INVAE | no | 128 | 11.47 |
| `e2e-invae-norepa-64tok.yaml` | E2E-INVAE | no | 64 | 12.70 |
| `e2e-vavae-norepa-128tok.yaml` | E2E-VAVAE | no | 128 | 14.40 |
| `e2e-vavae-norepa-64tok.yaml` | E2E-VAVAE | no | 64 | 16.72 |
| `e2e-invae-repa-64tok.yaml` | E2E-INVAE | yes | 64 | not reported |
| `e2e-vavae-repa-{128,64}tok.yaml` | E2E-VAVAE | yes | 128 / 64 | not reported |

The four `norepa` rows are the tokenizer x caption-length sweep of Lessons 1-2; the three `repa`
rows complete the grid but are not reported in the paper. `norepa` configs reuse their `repa`
sibling's latents.

## Data paths

Every config expects the dataset under `data/dog-t2i-diffusion-data/`, which is where
`hf download reyhanehesi/dog-t2i-diffusion-data --local-dir data/dog-t2i-diffusion-data` puts it.
The three tarballs extract into flat folders of `shard-*.tar`; the latents folders and the FID
reference stats are produced locally:

| Path (under `data/dog-t2i-diffusion-data/`) | What | Produced by |
|---|---|---|
| `dogs_recaptioned_wds/` | 26k real dog photos + captions | `tar -xzf dogs_recaptioned_wds.tar.gz -C dogs_recaptioned_wds/` |
| `dogs_synthetic_2k_wds/` | 2k synthetic SFT images + captions | `tar -xzf dogs_synthetic_2k_wds.tar.gz -C dogs_synthetic_2k_wds/` |
| `dogs_recaptioned_64tok_wds/` | the same 26k photos, captions of about 30-40 words (fit 64 tokens) | `tar -xzf dogs_recaptioned_64tok_wds.tar.gz -C dogs_recaptioned_64tok_wds/` |
| `dogs_recaptioned_stats.npz` | InceptionV3 mu/sigma for FID | `python -m minidog.fid_stats --data-dir .../dogs_recaptioned_wds --output .../dogs_recaptioned_stats.npz` |
| `dogs_recaptioned_latents_e2e-invae_eupe/` | pretraining latents for the final recipe (INVAE + EUPE features) | `python -m minidog.precompute_latents_eupe --config configs/pretrain_irepa_eupe_mse.yaml --input-dir .../dogs_recaptioned_wds --output-dir .../dogs_recaptioned_latents_e2e-invae_eupe` |
| `dogs_recaptioned_latents_e2e-invae/` | pretraining latents for the DINOv2 baseline | `python -m minidog.precompute_latents --config configs/pretrain_repa_dinov2_mse.yaml --input-dir .../dogs_recaptioned_wds --output-dir .../dogs_recaptioned_latents_e2e-invae` |
| `dogs_recaptioned_latents_e2e-invae_{dinov3,pe_spatial}/` | latents for the other target representations | same, with the matching `precompute_latents_*` script and config |
| `dogs_synthetic_2k_latents_e2e-invae_eupe/` | SFT latents (INVAE + EUPE features) | `python -m minidog.precompute_latents_eupe --config configs/sft_eupe_norepa.yaml` and the synthetic shards |
| `dogs_recaptioned_latents_e2e-vavae/` | latents for the VAVAE ablations | same, with a `vavae-*-128tok` config |
| `dogs_recaptioned_64tok_latents_e2e-{invae,vavae}/` | latents for the `*-64tok` ablations | same, with a `*-64tok` config and `--input-dir .../dogs_recaptioned_64tok_wds` |
| `captions_500.json` | 500 held-out eval captions for `minidog.generate` | shipped |
