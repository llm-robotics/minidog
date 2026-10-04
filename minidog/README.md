# MiniDog: the text-to-image pipeline

Dog-breed text-to-image generation on 4 GPUs: a 12-layer LightningDiT trained with flow matching on
frozen-VAE latents, conditioned on Qwen3-0.6B captions, with representation alignment to a frozen
vision encoder. The final recipe aligns to EUPE with iREPA and reaches FID 8.32; the lessons build up
to it from REPA with DINOv2 (FID 8.80). All commands run from the repo root.

## Setup

```bash
# Install environment
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Prepare data
export DATA=data/dog-t2i-diffusion-data
uv run hf download reyhanehesi/dog-t2i-diffusion-data --local-dir $DATA --repo-type dataset
for NAME in dogs_recaptioned_wds dogs_synthetic_2k_wds dogs_recaptioned_64tok_wds; do
  mkdir -p $DATA/$NAME && tar -xzf $DATA/$NAME.tar.gz -C $DATA/$NAME && rm $DATA/$NAME.tar.gz
done
```

## Preprocess (once)

```bash
# FID reference statistics of the real photos
uv run python -m minidog.fid_stats \
    --data-dir $DATA/dogs_recaptioned_wds \
    --output $DATA/dogs_recaptioned_stats.npz

# cache VAE latents, text embeddings and the alignment target's features. The config picks the
# tokenizer and the caption length; each (tokenizer, caption length, target) triple gets its own
# latents folder. Swap in precompute_latents_{dinov3,pe_spatial,eupe} for the other targets.
PRE="uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents"

# final recipe: EUPE features on the 26k pretraining set
uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents_eupe \
    --config configs/pretrain_irepa_eupe_mse.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-invae_eupe

# DINOv2 baseline + e2e-invae-*-128tok ablations
$PRE --config configs/pretrain_repa_dinov2_mse.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-invae

# sft (EUPE features: the final recipe fine-tunes the EUPE/iREPA checkpoint)
uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents_eupe \
    --config configs/sft_eupe_norepa.yaml \
    --input-dir $DATA/dogs_synthetic_2k_wds \
    --output-dir $DATA/dogs_synthetic_2k_latents_e2e-invae_eupe

# e2e-invae-*-64tok
$PRE --config configs/e2e-invae-repa-64tok.yaml \
    --input-dir $DATA/dogs_recaptioned_64tok_wds \
    --output-dir $DATA/dogs_recaptioned_64tok_latents_e2e-invae

# e2e-vavae-*-128tok
$PRE --config configs/e2e-vavae-repa-128tok.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-vavae

# e2e-vavae-*-64tok
$PRE --config configs/e2e-vavae-repa-64tok.yaml \
    --input-dir $DATA/dogs_recaptioned_64tok_wds \
    --output-dir $DATA/dogs_recaptioned_64tok_latents_e2e-vavae
```

The first two blocks cover the pretrain and SFT walkthrough; the rest are for the ablations only.
Each takes a few minutes on 4 GPUs. Re-running into an existing folder replaces its shards, and any
GPU count works.

## Train

```bash
# pretrain, final recipe: EUPE with iREPA (FID 8.32)
export EXPERIMENT_NAME=pretrain
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \
    --config configs/pretrain_irepa_eupe_mse.yaml \
    --compile

# fine-tune that checkpoint, with no alignment during SFT
export EXPERIMENT_NAME=sft
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \
    --config configs/sft_eupe_norepa.yaml \
    --compile \
    --ckpt ckpts/pretrain/checkpoints/ep-0000200.pt \
    --init-weights-only
```

- Outputs: checkpoints and sample grids in `ckpts/$EXPERIMENT_NAME/`, FID/IS in `results/evals/`.
- FID/IS are logged every `eval.eval_interval` steps. To score one checkpoint, or sweep the CFG scale:

  ```bash
  uv run torchrun --standalone --nproc_per_node=4 -m minidog.offline_eval --config <cfg> --checkpoint <ckpt> --cfg-scale 1.5 2.0 6.0
  ```

- `--wandb` logs to Weights & Biases; set `ENTITY`, `PROJECT` and `WANDB_KEY` environment variables first.
- Re-running with the same `EXPERIMENT_NAME` resumes from the latest checkpoint.

## Configs

One yaml per experiment; hyperparameters and reported FID in [`configs/README.md`](../configs/README.md).

- `pretrain_irepa_eupe_mse.yaml`, `sft_eupe_norepa.yaml`: the final recipe.
- `pretrain_repa_dinov2_mse.yaml`: the REPA/DINOv2 baseline the lessons build up to.
- `e2e-{invae,vavae}-{repa,norepa}-{128,64}tok.yaml`: the tokenizer x REPA x caption-length grid.
- Each config reads the latents for its tokenizer and caption length (Preprocess step). `norepa` configs reuse their `repa` sibling's latents.

To train any config, point `--config` at it and name the run after it:

```bash
CONFIG=configs/e2e-invae-norepa-128tok.yaml
export EXPERIMENT_NAME=$(basename $CONFIG .yaml)
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train --config $CONFIG --compile
```

## Generate and score

Sample the 500 evaluation captions from two checkpoints, then compare the two folders with PickScore and HPSv2:

```bash
for RUN in "pretrain configs/pretrain_irepa_eupe_mse.yaml" "sft configs/sft_eupe_norepa.yaml"; do
  set -- $RUN
  uv run python -m minidog.generate \
      --config $2 \
      --checkpoint $(ls ckpts/$1/checkpoints/*.pt | tail -1) \
      --captions-json $DATA/captions_500.json \
      --output-dir results/samples/$1 \
      --noise-file results/samples/shared_noise_500.pt --group-by-breed
done
uv run python -m minidog.score_absolute --dirs results/samples/pretrain results/samples/sft
```

`--noise-file` draws the 500 noise tensors once and reuses them, so each pair of images differs only
by the checkpoint. `score_absolute` prints the per-folder HPSv2 and PickScore means reported in the
paper (0.1625 -> 0.2427 and 18.42 -> 19.61); `minidog.score` instead compares two folders pairwise
and prints PickScore win rate and HPSv2 means, per breed and overall.

## Layout

| | Files | What |
|---|---|---|
| Model and objective | [`dit.py`](dit.py), [`transport.py`](transport.py) | LightningDiT; flow matching, Euler sampler, CFG |
| Frozen components | [`vae.py`](vae.py), [`text_encoder.py`](text_encoder.py), [`dinov2.py`](dinov2.py) | tokenizer, Qwen3 captions, REPA target |
| Training | [`engine.py`](engine.py), [`data.py`](data.py), [`eval.py`](eval.py), [`config.py`](config.py) | epoch loop, loaders, FID/IS, config dataclasses |
| Entry points | [`precompute_latents`](precompute_latents.py), [`fid_stats`](fid_stats.py), [`train`](train.py), [`offline_eval`](offline_eval.py), [`generate`](generate.py), [`score`](score.py) | `python -m minidog.<name>`, in pipeline order |
| Utilities | [`utils/`](utils/) | checkpoints, distributed, optimizer, resume, W&B |
