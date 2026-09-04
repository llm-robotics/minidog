# MiniDog-T2I

Dog-breed text-to-image generation on 4 GPUs: a 12-layer LightningDiT trained with flow matching on
frozen-VAE latents, conditioned on Qwen3-0.6B captions, with REPA alignment to DINOv2 features.
All commands run from the repo root.

## Setup

```bash
# Install environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Prepare data
export DATA=data/dog-t2i-diffusion-data
uv run hf download reyhanehesi/dog-t2i-diffusion-data --local-dir $DATA --repo-type dataset
mkdir -p $DATA/dogs_recaptioned_wds $DATA/dogs_synthetic_2k_wds
tar -xzf $DATA/dogs_recaptioned_wds.tar.gz  -C $DATA/dogs_recaptioned_wds && rm $DATA/dogs_recaptioned_wds.tar.gz
tar -xzf $DATA/dogs_synthetic_2k_wds.tar.gz -C $DATA/dogs_synthetic_2k_wds && rm $DATA/dogs_synthetic_2k_wds.tar.gz
```

## Preprocess (once)

```bash
# FID reference statistics
uv run python -m minidog.fid_stats \
    --data-dir $DATA/dogs_recaptioned_wds \
    --output $DATA/dogs_recaptioned_stats.npz

# cache VAE latents, text embeddings and DINOv2 features for both datasets
uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents \
    --config configs/pretrain.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-invae
uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents \
    --config configs/sft.yaml \
    --input-dir $DATA/dogs_synthetic_2k_wds \
    --output-dir $DATA/dogs_synthetic_2k_latents_e2e-invae
```

## Train

```bash
# pretrain
export EXPERIMENT_NAME=pretrain
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \
    --config configs/pretrain.yaml \
    --compile

# fine-tune from the pretraining checkpoint
export EXPERIMENT_NAME=sft
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \
    --config configs/sft.yaml \
    --compile \
    --ckpt ckpts/pretrain/checkpoints/ep-0000200.pt \
    --init-weights-only
```

Checkpoints and sample grids land in `ckpts/$EXPERIMENT_NAME/`, FID/IS in `results/evals/`.
FID/IS are logged every `eval.eval_interval` steps; to score a specific checkpoint (or sweep the
CFG scale) run `torchrun --standalone --nproc_per_node=4 -m minidog.offline_eval --config <cfg>
--checkpoint <ckpt> [--cfg-scale 1.5 2.0 6.0]`.
`--compile` wraps the loss in `torch.compile`: ~25% higher throughput after a one-time warm-up
of a minute or two, identical training curves. Drop it if compilation fails on your setup.
Add `--wandb` with `ENTITY`, `PROJECT` and `WANDB_KEY` set to log to Weights & Biases.
Re-running with the same `EXPERIMENT_NAME` resumes from the latest checkpoint.

## Configs

`configs/` holds one yaml per experiment: `pretrain.yaml`, `sft.yaml`, and the eight
`ablations/e2e-{invae,vavae}-{repa,norepa}-{128,64}tok.yaml` from the tutorial's ablation table
(hyperparameters and reported FID in [`configs/README.md`](../configs/README.md)).
To launch any of them, point `--config` at it and name the run after it:

```bash
CONFIG=configs/ablations/e2e-invae-norepa-128tok.yaml
export EXPERIMENT_NAME=$(basename $CONFIG .yaml)
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train --config $CONFIG --compile
```

The `e2e-vavae-*` configs read latents from `$DATA/dogs_recaptioned_latents_e2e-vavae`, so run
`minidog.precompute_latents` once with one of them first. The `*-64tok` configs need 64-token
recaptions that are not in the HF dataset.

## Generate and score

Sample the 500 evaluation captions from two checkpoints, then compare the two folders with
two learned human-preference models, PickScore and HPSv2:

```bash
for RUN in pretrain sft; do
  uv run python -m minidog.generate \
      --config configs/$RUN.yaml \
      --checkpoint $(ls ckpts/$RUN/checkpoints/*.pt | tail -1) \
      --captions-json $DATA/captions_500.json \
      --output-dir results/samples/$RUN --group-by-breed
done
uv run python -m minidog.score --a results/samples/pretrain --b results/samples/sft
```

`score` prints per-breed and overall PickScore preference and win rate of B over A, and HPSv2
mean scores. FID and Inception Score are already logged during training.

## Layout

Where to look, by what you want to understand:

**Core: the model and the objective**

- [`dit.py`](dit.py): the LightningDiT denoiser (patch embedding, RoPE, attention blocks, time and text tokens, REPA read-out).
- [`transport.py`](transport.py): flow matching (forward process, loss, x/v conversion), the Euler sampler, and classifier-free guidance.
- [`vae.py`](vae.py): the frozen VAE tokenizer that maps images to 16x16x32 latents and back.
- [`text_encoder.py`](text_encoder.py): Qwen3-0.6B caption embeddings.
- [`dinov2.py`](dinov2.py): the frozen DINOv2 encoder used as the REPA target.

**Training**

- [`engine.py`](engine.py): one training epoch: batches, loss, EMA update, sample grids, periodic FID eval.
- [`data.py`](data.py): WebDataset loaders for raw image shards and precomputed latent shards.
- [`eval.py`](eval.py): FID and Inception Score computed across GPUs during training.
- [`config.py`](config.py): the dataclasses every yaml in [`configs/`](../configs/) is parsed into.

**Entry points**, run as `python -m minidog.<name>`

- [`precompute_latents.py`](precompute_latents.py): cache VAE latents, text embeddings and DINOv2 features once.
- [`fid_stats.py`](fid_stats.py): InceptionV3 reference statistics for FID.
- [`train.py`](train.py): pretraining and fine-tuning.
- [`offline_eval.py`](offline_eval.py): FID/IS of a saved checkpoint, optionally sweeping the CFG scale.
- [`generate.py`](generate.py): sample images for a caption set from a checkpoint.
- [`score.py`](score.py): PickScore and HPSv2 comparison of two generated sets.

**Utilities** ([`utils/`](utils/)): checkpoint save/load, distributed setup, optimizer and LR schedule, experiment directories and resume, W&B logging.
