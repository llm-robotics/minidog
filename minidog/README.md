# MiniDog: Task 2, text-to-image

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

# cache VAE latents, text embeddings and DINOv2 features. The config picks the tokenizer and the
# caption length; each (tokenizer, caption length) pair gets its own latents folder.
PRE="uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents"
$PRE --config configs/pretrain.yaml                        --input-dir $DATA/dogs_recaptioned_wds       --output-dir $DATA/dogs_recaptioned_latents_e2e-invae         # pretrain + e2e-invae-*-128tok
$PRE --config configs/sft.yaml                             --input-dir $DATA/dogs_synthetic_2k_wds      --output-dir $DATA/dogs_synthetic_2k_latents_e2e-invae        # sft
$PRE --config configs/ablations/e2e-invae-repa-64tok.yaml  --input-dir $DATA/dogs_recaptioned_64tok_wds --output-dir $DATA/dogs_recaptioned_64tok_latents_e2e-invae   # e2e-invae-*-64tok
$PRE --config configs/ablations/e2e-vavae-repa-128tok.yaml --input-dir $DATA/dogs_recaptioned_wds       --output-dir $DATA/dogs_recaptioned_latents_e2e-vavae         # e2e-vavae-*-128tok
$PRE --config configs/ablations/e2e-vavae-repa-64tok.yaml  --input-dir $DATA/dogs_recaptioned_64tok_wds --output-dir $DATA/dogs_recaptioned_64tok_latents_e2e-vavae   # e2e-vavae-*-64tok
```

The first two lines cover the pretrain and SFT walkthrough. The last three are for the ablations only.
Each takes a few minutes on 4 GPUs.

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

- Outputs: checkpoints and sample grids in `ckpts/$EXPERIMENT_NAME/`, FID/IS in `results/evals/`.
- FID/IS are logged every `eval.eval_interval` steps. To score one checkpoint, or sweep the CFG scale:

  ```bash
  uv run torchrun --standalone --nproc_per_node=4 -m minidog.offline_eval --config <cfg> --checkpoint <ckpt> --cfg-scale 1.5 2.0 6.0
  ```

- `--wandb` logs to Weights & Biases; set `ENTITY`, `PROJECT` and `WANDB_KEY`.
- Re-running with the same `EXPERIMENT_NAME` resumes from the latest checkpoint.

## Configs

One yaml per experiment; hyperparameters and reported FID in [`configs/README.md`](../configs/README.md).

- `pretrain.yaml`, `sft.yaml`: the main walkthrough.
- `ablations/e2e-{invae,vavae}-{repa,norepa}-{128,64}tok.yaml`: the tokenizer x REPA x caption-length grid.
- Each config reads the latents for its tokenizer and caption length (Preprocess step). `norepa` configs reuse their `repa` sibling's latents.

To train any config, point `--config` at it and name the run after it:

```bash
CONFIG=configs/ablations/e2e-invae-norepa-128tok.yaml
export EXPERIMENT_NAME=$(basename $CONFIG .yaml)
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train --config $CONFIG --compile
```

## Generate and score

Sample the 500 evaluation captions from two checkpoints, then compare the two folders with PickScore and HPSv2:

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

`score` prints PickScore preference and win rate of B over A, and HPSv2 means, per breed and overall.

## Layout

| | Files | What |
|---|---|---|
| Model and objective | [`dit.py`](dit.py), [`transport.py`](transport.py) | LightningDiT; flow matching, Euler sampler, CFG |
| Frozen components | [`vae.py`](vae.py), [`text_encoder.py`](text_encoder.py), [`dinov2.py`](dinov2.py) | tokenizer, Qwen3 captions, REPA target |
| Training | [`engine.py`](engine.py), [`data.py`](data.py), [`eval.py`](eval.py), [`config.py`](config.py) | epoch loop, loaders, FID/IS, config dataclasses |
| Entry points | [`precompute_latents`](precompute_latents.py), [`fid_stats`](fid_stats.py), [`train`](train.py), [`offline_eval`](offline_eval.py), [`generate`](generate.py), [`score`](score.py) | `python -m minidog.<name>`, in pipeline order |
| Utilities | [`utils/`](utils/) | checkpoints, distributed, optimizer, resume, W&B |
