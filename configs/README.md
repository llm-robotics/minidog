# Configs

All configs train the same model: a 12-layer, 384-wide LightningDiT (6 heads, patch size 1) on
16x16x32 latents from a frozen VAE, conditioned on 128-token Qwen3-0.6B caption embeddings, with
velocity-prediction flow matching and 4 timestep tokens. They differ only in the knobs below.

| Config | VAE | REPA | Epochs | EMA | LR | Notes |
|---|---|---|---|---|---|---|
| `pretrain.yaml` | E2E-INVAE | yes | 200 | 0.9995 | 1e-4, 100 warmup | **The pretraining run.** FID 9.4 |
| `sft.yaml` | E2E-INVAE | yes | 100 | 0.995 | 5e-5, no warmup | **Fine-tune from `pretrain` on 2k synthetic dogs.** Launch with `--ckpt <pretrain ckpt> --init-weights-only` |
| `ablations/e2e-invae-norepa.yaml` | E2E-INVAE | no | 200 | 0.9995 | 1e-4 | FID 11.5 |
| `ablations/e2e-vavae-repa.yaml` | E2E-VAVAE | yes | 200 | 0.9995 | 1e-4 | FID 13.9 |
| `ablations/e2e-vavae-norepa.yaml` | E2E-VAVAE | no | 200 | 0.9995 | 1e-4 | FID 14.4 |

`pretrain.yaml` is the E2E-INVAE / REPA corner of the ablation grid; the four rows together are the
tokenizer x REPA sweep. All captions use a 128-token budget.

## Data paths

Every config expects the dataset under `data/dog-t2i-diffusion-data/`, which is where
`hf download reyhanehesi/dog-t2i-diffusion-data --local-dir data/dog-t2i-diffusion-data` puts it.
The two tarballs extract into flat folders of `shard-*.tar`; the latents folders and the FID
reference stats are produced locally:

| Path (under `data/dog-t2i-diffusion-data/`) | What | Produced by |
|---|---|---|
| `dogs_recaptioned_wds/` | 26k real dog photos + captions | `tar -xzf dogs_recaptioned_wds.tar.gz -C dogs_recaptioned_wds/` |
| `dogs_synthetic_2k_wds/` | 2k synthetic SFT images + captions | `tar -xzf dogs_synthetic_2k_wds.tar.gz -C dogs_synthetic_2k_wds/` |
| `dogs_recaptioned_stats.npz` | InceptionV3 mu/sigma for FID | `python -m minidog.fid_stats --data-dir .../dogs_recaptioned_wds --output .../dogs_recaptioned_stats.npz` |
| `dogs_recaptioned_latents_e2e-invae/` | pretraining latents (INVAE) | `python -m minidog.precompute_latents --config configs/pretrain.yaml --input-dir .../dogs_recaptioned_wds --output-dir .../dogs_recaptioned_latents_e2e-invae` |
| `dogs_synthetic_2k_latents_e2e-invae/` | SFT latents (INVAE) | same, with `--config configs/sft.yaml` and the synthetic shards |
| `dogs_recaptioned_latents_e2e-vavae/` | latents for the VAVAE ablations | same, with a `vavae-*` config |
| `captions_500.json` | 500 eval captions for `minidog.generate` | shipped |

