<p align="center"><img src="assets/minidog-cute.jpg" alt="MiniDog" width="320"></p>

# MiniDog

A teaching resource for flow-matching generative models, in two tasks:

1. **Flow-matching basics on 2D toy data, on CPU** — `toy_flow_matching.ipynb`, runs in minutes.
2. **Pretraining and fine-tuning a text-to-image diffusion transformer on dog photos, on 4 consumer GPUs** — the `minidog/` package, ~3 hours end to end.

The two tasks share one idea: a network learns the velocity that carries noise to data along a straight
line, and is sampled by integrating that velocity. Task 1 shows it on points you can plot; Task 2 shows the
same objective driving a 12-layer LightningDiT on VAE latents with text conditioning, REPA alignment and
classifier-free guidance, evaluated with FID and human-preference models.

**Prerequisites**: linear algebra, probability, and PyTorch at the level of writing an `nn.Module` and a training loop.

## Where each concept lives

Read the notebook cell and the code side by side; the toy version is the minimal form of what the package does.

| Concept | Task 1: `toy_flow_matching.ipynb` | Task 2: `minidog/` |
|---|---|---|
| Forward process, `x_t = (1-t) x + t eps` | *Flow matching* cell, `FlowMatching.training_losses` | `transport.py`, `Transport.sample` and the first lines of `training_losses` |
| Training target and loss, `v = eps - x`, MSE | same cell | `transport.py`, `Transport.compute_loss`; per-step timestep sampling in `get_time_sampler` |
| x- vs v-prediction and converting between them | same cell, the `pred_type` branches | `transport.py`, `Transport.convert_model_pred`; chosen by `transport.prediction` in the config |
| Sampling: Euler steps from noise to data | `FlowMatching.sample` | `transport.py`, `Sampler.sample_ode` |
| Time conditioning | *Model* cell, `SinusoidalEmbedding` | `dit.py`, `GaussianFourierEmbedding`: Fourier features of `t` turned into 4 learned tokens that join the sequence |
| The denoiser | `MLPDenoiser`, an MLP on `[x_t, t_emb]` | `dit.py`, `LightningDiT`: patch embedding, RoPE, RMSNorm attention blocks, SwiGLU MLP, zero-initialised final layer |
| Conditioning on text | — | `text_encoder.py` (Qwen3-0.6B hidden states) and `dit.py`, `ConditionEmbedder`: caption tokens appended to the sequence, masked in `_build_attn_mask` |
| Classifier-free guidance | — | `transport.py`: `apply_cfg_dropout` during training, `forward_with_cfg` at sampling; scale in `guidance.cfg` |
| Working in a latent space | the orthogonal projection to `D` dims | `vae.py`: frozen E2E-INVAE encodes 256x256 images to 16x16x32 latents; `precompute_latents.py` caches them once |
| Representation alignment (REPA) | — | `dinov2.py` provides the target; `dit.py` `repa_projector` reads out layer `repa_layer_depth`; the loss term is in `Transport.training_losses` |
| Training loop, EMA weights | *Training loop* cell | `engine.py`, `train_one_epoch`; EMA in `utils/train_utils.update_ema` |
| Evaluation | look at the samples | `eval.py` (FID, Inception Score), `score.py` (PickScore, HPSv2 on two sets of samples) |
| Experiments | Experiments 1 and 2 in the notebook | `configs/`: `pretrain.yaml`, `sft.yaml`, the tokenizer x REPA ablation grid ([`configs/README.md`](configs/README.md)) |

## Getting started

**Task 1.** `uv sync`, then open `toy_flow_matching.ipynb` and run it top to bottom. Every cell is
explained in the markdown above it. Change `DIMS`, `TRAIN_STEPS` or `HIDDEN_DIMS` in the experiment cells
and re-run.

**Task 2.** Follow [`minidog/README.md`](minidog/README.md): download the dataset, precompute latents,
pretrain, fine-tune, generate, score. The hyperparameters and the FID each config reached are in
[`configs/README.md`](configs/README.md).

## Going further

Things to change once both tasks run:

- Switch `transport.prediction` between `velocity` and `x` in a config and compare the FID curves: the
  Task 1 question at full scale.
- Turn `repa.use_repa` off, or move `repa.repa_layer_depth`, and watch what REPA buys.
- Swap the tokenizer between `e2e-invae` and `e2e-vavae` (`stage_1.params.vae_type`), the strongest
  effect in the ablation table.
- Sweep `guidance.cfg.scale` on a trained checkpoint with `minidog.offline_eval` and see how guidance
  trades FID against sample quality.
- Fine-tune on your own images: pack them as `jpg` + `txt` WebDataset shards, precompute latents with
  `sft.yaml`, and train from the pretraining checkpoint.

## Layout

```
toy_flow_matching.ipynb   Task 1
minidog/                  Task 2 package and its README
configs/                  experiment configs and results table
LICENSE                   MIT
```
