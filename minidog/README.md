# MiniDog: the text-to-image pipeline

Dog-breed text-to-image generation on 4 GPUs: a 12-layer, 22M-parameter LightningDiT trained with flow
matching on frozen-VAE latents, conditioned on Qwen3-0.6B captions, with representation alignment to a
frozen vision encoder. The final recipe (E2E-INVAE tokenizer, 128-token captions, iREPA with EUPE,
squared error) reaches FID 8.32; each comparison in the paper changes one setting of it. All commands
run from the repo root.

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

# Cache VAE latents, text embeddings and the alignment target's features. The config picks the
# tokenizer, the caption length and the target; each combination gets its own latents folder,
# which is the config's dataset.data_dir.
EUPE="uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents_eupe"

# final recipe: E2E-INVAE latents + EUPE features of the 26k pretraining images
$EUPE --config configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-invae_eupe

# SFT: latents and text embeddings of the 2k synthetic images (alignment is off, so no features)
$EUPE --config configs/sft_e2e-invae_128tok_mse_norepa.yaml \
    --input-dir $DATA/dogs_synthetic_2k_wds \
    --output-dir $DATA/dogs_synthetic_2k_latents_e2e-invae_eupe
```

The two commands above cover the pretrain and SFT walkthrough. The lessons need these as well:

```bash
# Lesson 1, tokenizer: E2E-VAVAE latents + EUPE features
$EUPE --config configs/pretrain_e2e-vavae_128tok_mse_irepa_eupe.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-vavae_eupe

# Lesson 2, captions: the 64-token captions
$EUPE --config configs/pretrain_e2e-invae_64tok_mse_irepa_eupe.yaml \
    --input-dir $DATA/dogs_recaptioned_64tok_wds \
    --output-dir $DATA/dogs_recaptioned_64tok_latents_e2e-invae_eupe

# Lessons 3-4, alignment and target: DINOv2 features (the no-alignment run reads this folder too),
# then DINOv3 and PE-Spatial. The EUPE runs reuse the final recipe's folder.
uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents \
    --config configs/pretrain_e2e-invae_128tok_mse_repa_dinov2.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-invae
uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents_dinov3 \
    --config configs/pretrain_e2e-invae_128tok_mse_repa_dinov3.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-invae_dinov3
uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents_pe_spatial \
    --config configs/pretrain_e2e-invae_128tok_mse_repa_pe-spatial.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-invae_pe_spatial

# Extra configs not reported in the paper (DINOv2 features)
PRE="uv run torchrun --standalone --nproc_per_node=4 -m minidog.precompute_latents"
$PRE --config configs/pretrain_e2e-invae_64tok_mse_repa_dinov2.yaml \
    --input-dir $DATA/dogs_recaptioned_64tok_wds \
    --output-dir $DATA/dogs_recaptioned_64tok_latents_e2e-invae
$PRE --config configs/pretrain_e2e-vavae_128tok_mse_repa_dinov2.yaml \
    --input-dir $DATA/dogs_recaptioned_wds \
    --output-dir $DATA/dogs_recaptioned_latents_e2e-vavae
$PRE --config configs/pretrain_e2e-vavae_64tok_mse_repa_dinov2.yaml \
    --input-dir $DATA/dogs_recaptioned_64tok_wds \
    --output-dir $DATA/dogs_recaptioned_64tok_latents_e2e-vavae
```

Each takes a few minutes on 4 GPUs. Re-running into an existing folder replaces its shards, and any
GPU count works. `norepa` configs reuse the folder of a sibling config with the same tokenizer and
caption length.

## Train

```bash
# pretrain, final recipe: iREPA with EUPE (FID 8.32)
export EXPERIMENT_NAME=pretrain
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \
    --config configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml \
    --compile

# fine-tune that checkpoint, with no alignment during SFT
export EXPERIMENT_NAME=sft
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train \
    --config configs/sft_e2e-invae_128tok_mse_norepa.yaml \
    --compile \
    --ckpt ckpts/pretrain/checkpoints/ep-0000200.pt \
    --init-weights-only
```

- On 4 RTX 3090s, pretraining takes 55 minutes and SFT 15 minutes, not counting the FID evaluation
  during training. That evaluation runs every `eval.eval_interval` steps (5,000 for pretraining, 150
  for SFT), doubles the pretraining time and adds about an hour to SFT; set `eval.eval_interval: 0` to
  turn it off.
- Outputs: checkpoints and sample grids in `ckpts/$EXPERIMENT_NAME/`, FID/IS in `results/evals/`.
- To score one checkpoint, or sweep the CFG scale:

  ```bash
  uv run torchrun --standalone --nproc_per_node=4 -m minidog.offline_eval --config <cfg> --checkpoint <ckpt> --cfg-scale 1.5 2.0 6.0
  ```

- `--wandb` logs to Weights & Biases; set `ENTITY`, `PROJECT` and `WANDB_KEY` environment variables first.
- Re-running with the same `EXPERIMENT_NAME` resumes from the latest checkpoint.

## Configs

One yaml per experiment, named `{stage}_{tokenizer}_{caption length}_{loss}_{alignment}_{target}.yaml`.
Hyperparameters and reported FIDs are in [`configs/README.md`](../configs/README.md).

| Field | Values |
|---|---|
| stage | `pretrain`, `sft` |
| tokenizer | `e2e-invae`, `e2e-vavae` |
| caption length | `128tok`, `64tok` |
| loss | `mse` (squared error) or `cosine`: the alignment distance. `norepa` runs use `mse`, the flow-matching loss |
| alignment | `irepa`, `repa`, `norepa` |
| target | `dinov2`, `dinov3`, `pe-spatial`, `eupe`; left out for `norepa` |

The final recipe is `pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml` followed by
`sft_e2e-invae_128tok_mse_norepa.yaml`. To train any config, point `--config` at it and name the run
after it:

```bash
CONFIG=configs/pretrain_e2e-invae_128tok_mse_norepa.yaml
export EXPERIMENT_NAME=$(basename $CONFIG .yaml)
uv run torchrun --standalone --nproc_per_node=4 -m minidog.train --config $CONFIG --compile
```

## Generate and score

Sample the 500 evaluation captions from two checkpoints, then compare the two folders with PickScore and HPSv2:

```bash
for RUN in "pretrain configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml" "sft configs/sft_e2e-invae_128tok_mse_norepa.yaml"; do
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
| Frozen components | [`vae.py`](vae.py), [`text_encoder.py`](text_encoder.py), [`eupe.py`](eupe.py), [`dinov2.py`](dinov2.py), [`dinov3.py`](dinov3.py), [`pe_spatial.py`](pe_spatial.py) | tokenizer, Qwen3 captions, the four target representations |
| Training | [`engine.py`](engine.py), [`data.py`](data.py), [`eval.py`](eval.py), [`config.py`](config.py) | epoch loop, loaders, FID/IS, config dataclasses |
| Entry points | [`precompute_latents`](precompute_latents.py) (and `_eupe`, `_dinov3`, `_pe_spatial`), [`fid_stats`](fid_stats.py), [`train`](train.py), [`offline_eval`](offline_eval.py), [`generate`](generate.py), [`score_absolute`](score_absolute.py), [`score`](score.py) | `python -m minidog.<name>`, in pipeline order |
| Utilities | [`utils/`](utils/) | checkpoints, distributed, optimizer, resume, W&B |
