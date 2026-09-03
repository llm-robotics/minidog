# Configs

All configs train the same model: a 12-layer, 384-wide LightningDiT (6 heads, patch size 1) on
16x16x32 latents from a frozen VAE, conditioned on Qwen3-0.6B caption embeddings, with
velocity-prediction flow matching and 4 timestep tokens. They differ only in the knobs below.

| Config | VAE | REPA | Caption tokens | Epochs | EMA | LR | Notes |
|---|---|---|---|---|---|---|---|
| `pretrain.yaml` | E2E-INVAE | yes | 128 | 200 | 0.9995 | 1e-4, 100 warmup | **The pretraining run.** FID 9.4 |
| `sft.yaml` | E2E-INVAE | yes | 128 | 100 | 0.995 | 5e-5, no warmup | **Fine-tune from `pretrain` on 2k synthetic dogs.** Launch with `--ckpt <pretrain ckpt> --init-weights-only` |
| `ablations/invae-repa-64tok.yaml` | E2E-INVAE | yes | 64 | 200 | 0.9995 | 1e-4 | FID 9.8 |
| `ablations/invae-norepa-128tok.yaml` | E2E-INVAE | no | 128 | 200 | 0.9995 | 1e-4 | FID 11.5 |
| `ablations/invae-norepa-64tok.yaml` | E2E-INVAE | no | 64 | 200 | 0.9995 | 1e-4 | FID 12.7 |
| `ablations/vavae-repa-64tok.yaml` | E2E-VAVAE | yes | 64 | 200 | 0.9995 | 1e-4 | FID 13.4 |
| `ablations/vavae-repa-128tok.yaml` | E2E-VAVAE | yes | 128 | 200 | 0.9995 | 1e-4 | FID 13.9 |
| `ablations/vavae-norepa-128tok.yaml` | E2E-VAVAE | no | 128 | 200 | 0.9995 | 1e-4 | FID 14.4 |
| `ablations/vavae-norepa-64tok.yaml` | E2E-VAVAE | no | 64 | 200 | 0.9995 | 1e-4 | FID 16.7 |

`pretrain.yaml` is the INVAE / REPA / 128-token corner of the ablation grid; the eight rows
together are the tokenizer x REPA x caption-length sweep reported in the tutorial.

The 64-token configs need latents precomputed with `max_length: 64`, so they point at a separate
latents directory. All data paths (`dataset.data_dir`, `eval.datasets.dogs.data_dir`,
`eval.datasets.dogs.reference_npz`) currently need to be edited to your own locations.
