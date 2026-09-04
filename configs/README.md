# Configs

All configs train the same model: a 12-layer, 384-wide LightningDiT (6 heads, patch size 1) on
16x16x32 latents from a frozen VAE, conditioned on Qwen3-0.6B caption embeddings, with
velocity-prediction flow matching and 4 timestep tokens. They differ only in the knobs below.

| Config | VAE | REPA | Caption tokens | Epochs | EMA | LR | Notes |
|---|---|---|---|---|---|---|---|
| `pretrain.yaml` | E2E-INVAE | yes | 128 | 200 | 0.9995 | 1e-4, 100 warmup | **The pretraining run.** FID 9.4 |
| `sft.yaml` | E2E-INVAE | yes | 128 | 100 | 0.995 | 5e-5, no warmup | **Fine-tune from `pretrain` on 2k synthetic dogs.** Launch with `--ckpt <pretrain ckpt> --init-weights-only` |
| `ablations/e2e-invae-repa-64tok.yaml` | E2E-INVAE | yes | 64 | 200 | 0.9995 | 1e-4 | FID 9.8 |
| `ablations/e2e-invae-norepa-128tok.yaml` | E2E-INVAE | no | 128 | 200 | 0.9995 | 1e-4 | FID 11.5 |
| `ablations/e2e-invae-norepa-64tok.yaml` | E2E-INVAE | no | 64 | 200 | 0.9995 | 1e-4 | FID 12.7 |
| `ablations/e2e-vavae-repa-64tok.yaml` | E2E-VAVAE | yes | 64 | 200 | 0.9995 | 1e-4 | FID 13.4 |
| `ablations/e2e-vavae-repa-128tok.yaml` | E2E-VAVAE | yes | 128 | 200 | 0.9995 | 1e-4 | FID 13.9 |
| `ablations/e2e-vavae-norepa-128tok.yaml` | E2E-VAVAE | no | 128 | 200 | 0.9995 | 1e-4 | FID 14.4 |
| `ablations/e2e-vavae-norepa-64tok.yaml` | E2E-VAVAE | no | 64 | 200 | 0.9995 | 1e-4 | FID 16.7 |

`pretrain.yaml` is the INVAE / REPA / 128-token corner of the ablation grid; the eight rows
together are the tokenizer x REPA x caption-length sweep reported in the tutorial.

## Data paths

Every config expects the dataset under `data/dog-t2i-diffusion-data/`, which is where
`hf download reyhanehesi/dog-t2i-diffusion-data --local-dir data/dog-t2i-diffusion-data` puts it.
The three tarballs extract into flat folders of `shard-*.tar`; the latents folders and the FID
reference stats are produced locally:

| Path (under `data/dog-t2i-diffusion-data/`) | What | Produced by |
|---|---|---|
| `dogs_recaptioned_wds/` | 26k real dog photos + captions | `tar -xzf dogs_recaptioned_wds.tar.gz -C dogs_recaptioned_wds/` |
| `dogs_synthetic_2k_wds/` | 2k synthetic SFT images + captions | `tar -xzf dogs_synthetic_2k_wds.tar.gz -C dogs_synthetic_2k_wds/` |
| `dogs_recaptioned_64tok_wds/` | the same 26k photos, captions of 25-40 words (fit 64 tokens) | `tar -xzf dogs_recaptioned_64tok_wds.tar.gz -C dogs_recaptioned_64tok_wds/` |
| `dogs_recaptioned_stats.npz` | InceptionV3 mu/sigma for FID | `python -m minidog.fid_stats --data-dir .../dogs_recaptioned_wds --output .../dogs_recaptioned_stats.npz` |
| `dogs_recaptioned_latents_e2e-invae/` | pretraining latents (INVAE) | `python -m minidog.precompute_latents --config configs/pretrain.yaml --input-dir .../dogs_recaptioned_wds --output-dir .../dogs_recaptioned_latents_e2e-invae` |
| `dogs_synthetic_2k_latents_e2e-invae/` | SFT latents (INVAE) | same, with `--config configs/sft.yaml` and the synthetic shards |
| `dogs_recaptioned_latents_e2e-vavae/` | latents for the VAVAE ablations | same, with a `vavae-*-128tok` config |
| `dogs_recaptioned_64tok_latents_e2e-{invae,vavae}/` | latents for the `*-64tok` ablations | same, with a `*-64tok` config and `--input-dir .../dogs_recaptioned_64tok_wds` |
| `captions_500.json` | 500 eval captions for `minidog.generate` | shipped |

