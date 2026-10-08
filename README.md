<p align="center"><img src="assets/minidog-full.png" alt="MiniDog" width="70%"></p>

<p align="center">
  <a href="https://huggingface.co/datasets/reyhanehesi/dog-t2i-diffusion-data" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"></a>
  <a href="https://huggingface.co/reyhanehesi/minidog-checkpoints" target="_blank"><img src="https://img.shields.io/badge/Checkpoints-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="Checkpoints"></a>
  <a href="about:blank" target="_blank"><img src="https://img.shields.io/badge/Paper-PDF-EC1C24?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper"></a>
</p>

<p align="center">
  <a href="assets/minidog-video.mp4"><img src="assets/minidog-teaser.gif" alt="MiniDog in 44 seconds" width="88%"></a>
</p>
<p align="center"><sub><a href="assets/minidog-video.mp4">Full 44-second video</a></sub></p>

MiniDog is a minimal teaching and research resource for flow-matching generative models, in two parts.

- **[Warm-up](#warm-up-flow-matching-basics)**: learn flow-matching basics on 2D toy data. Runs on CPU in [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb), ~15 minutes.
- **[The five lessons](#the-five-lessons)**: build a text-to-image diffusion transformer and test one design decision at a time, each backed by a controlled experiment you can rerun. Runs on 4 consumer GPUs (RTX 3090) from the [`minidog/`](minidog/) package. Pretraining reaches FID 8.32 in 55 minutes, fine-tuning takes 15 more.

Both tasks train the same objective:

- A network learns the velocity that moves noise to data along a straight line.
- Sampling integrates that velocity.
- The warm-up shows this on 2D points you can plot.
- The lessons run it at full scale: a [LightningDiT](minidog/dit.py) on [VAE latents](minidog/vae.py), with [text conditioning](minidog/text_encoder.py), representation alignment ([REPA](minidog/dinov2.py) and [iREPA](minidog/eupe.py)), [classifier-free guidance](minidog/transport.py), and [FID](minidog/eval.py) and [preference scores](minidog/score_absolute.py).

**Prerequisites**: linear algebra, probability, and basic PyTorch. You should be able to write an `nn.Module`
and a training loop.

## Where each concept lives

The same flow-matching pieces appear in the warm-up and in the full pipeline. Read them side by side; every pipeline entry links to the line in the code.

| Flow-matching concept | Warm-up: [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb) | Pipeline: [`minidog/`](minidog/) |
|---|---|---|
| Forward process `x_t = (1-t) x + t eps` | *Flow matching* cell, `training_losses` | [`Transport.sample`](minidog/transport.py#L103) |
| Target `v = eps - x`, MSE loss | same cell | [`compute_loss`](minidog/transport.py#L135) |
| x- vs v-prediction | `pred_type` branches | [`convert_model_pred`](minidog/transport.py#L128), config [`transport.prediction`](configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml#L28) |
| Timestep sampling during training | uniform `t` | [`get_time_sampler`](minidog/transport.py#L79), logit-normal |
| Euler sampling from noise to data | `sample` | [`Sampler.sample_ode`](minidog/transport.py#L155) |
| Time conditioning of the network | *Model* cell, `SinusoidalEmbedding` | [`GaussianFourierEmbedding`](minidog/dit.py#L86), 4 time tokens |
| The denoiser | `MLPDenoiser` | [`LightningDiT`](minidog/dit.py#L140) |
| Conditioning on text | — | [`TextEncoder`](minidog/text_encoder.py#L8), [`ConditionEmbedder`](minidog/dit.py#L104) |
| Classifier-free guidance | — | [`apply_cfg_dropout`](minidog/transport.py#L15) (train), [`forward_with_cfg`](minidog/transport.py#L179) (sample) |
| Flow matching in a latent space | — | [`VAE.encode`](minidog/vae.py#L146), [`precompute_latents`](minidog/precompute_latents.py#L180) |
| Representation alignment (REPA) | — | [`DINOv2Encoder`](minidog/dinov2.py#L9), [`repa_projector`](minidog/dit.py#L178), [`loss_repa`](minidog/transport.py#L125) |
| Target representation and iREPA | — | [`EUPEEncoder`](minidog/eupe.py), [`DINOv3Encoder`](minidog/dinov3.py), [`PESpatialEncoder`](minidog/pe_spatial.py); conv projector and spatial norm in [`dit.py`](minidog/dit.py#L178), [`transport.py`](minidog/transport.py#L131) |

## Getting started

### Warm-up: flow-matching basics

CPU, ~15 minutes.

```bash
uv sync
uv run jupyter lab toy_flow_matching.ipynb
```

Run [the notebook](toy_flow_matching.ipynb) top to bottom. Each cell is explained in the markdown above it. You will learn:

- how a flow-matching model is trained and sampled;
- why predicting the clean data beats predicting the velocity when the data lives in a high-dimensional space, and why MiniDog can still use v-prediction in its compressed latent space.

### The five lessons

4x3090 GPUs, 55 minutes pretraining, 15 minutes fine-tuning, not counting the FID evaluation during
training. The configs evaluate FID every 5,000 steps; set `eval.eval_interval: 0` to turn it off.
Leaving it on doubles the pretraining time and adds about an hour to fine-tuning.

Each lesson asks one design question and answers it with a controlled experiment that changes one
setting of the final recipe (E2E-INVAE tokenizer, 128-token captions, iREPA with EUPE, squared error)
and keeps everything else fixed. Every FID is at 200 epochs (~20k steps).

| Lesson | Question | Answer | FID |
|---|---|---|---|
| 1. Tokenizer | Which tokenizer? | E2E-INVAE: better reconstruction on all four metrics, and latents that are easier to model | 8.32 vs 12.23 (E2E-VAVAE) |
| 2. Captions | How detailed should captions be? | The 128-token captions | 8.32 vs 8.54 (64 tokens) |
| 3. Alignment | Does representation alignment help? | Yes; with EUPE, iREPA helps more than REPA | 8.32 (iREPA) vs 9.71 (REPA) vs 11.47 (none) |
| 4. Target | Which target representation, and which distance? | EUPE for iREPA, DINOv2 for REPA (8.80); squared error beats cosine in every case | **8.32** (iREPA, EUPE) |
| 5. SFT | Does fine-tuning on high-quality images help? | Yes: HPSv2 0.1625 -> 0.2427, PickScore 18.42 -> 19.61 | — |

Follow [`minidog/README.md`](minidog/README.md) to run them: download the data, precompute latents,
pretrain, fine-tune, generate, score. The configs behind every number are listed in
[`configs/README.md`](configs/README.md).

## Going further

Ideas to explore once both tasks run:

- Set [`transport.prediction`](configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml#L28) to `x` and compare FID curves: the warm-up question at full scale.
- Turn [`repa.use_repa`](configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml#L86) off, change `repa.repa_layer_depth`, or swap the target representation: the four encoders compared in the paper have one config per method and distance, `configs/pretrain_e2e-invae_128tok_{mse,cosine}_{repa,irepa}_{dinov2,dinov3,pe-spatial,eupe}.yaml`.
- Try another tokenizer: add a class with `encode`/`decode` to [`minidog/vae.py`](minidog/vae.py) and point [`stage_1.target`](configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml#L2) at it, e.g. the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) VAE or [RAEv2](https://github.com/nanovisionx/RAEv2).
- Sweep [`guidance.cfg.scale`](configs/pretrain_e2e-invae_128tok_mse_irepa_eupe.yaml#L36) on a checkpoint with [`minidog.offline_eval`](minidog/offline_eval.py).
- Fine-tune on your own images: pack `jpg` + `txt` WebDataset shards, build FID stats with [`minidog.fid_stats`](minidog/fid_stats.py), precompute with [`sft_e2e-invae_128tok_mse_norepa.yaml`](configs/sft_e2e-invae_128tok_mse_norepa.yaml), train from the pretrain checkpoint.

## Layout

- [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb): the warm-up
- [`minidog/`](minidog/): the text-to-image package, with its own [README](minidog/README.md)
- [`configs/`](configs/): experiment configs, with the results table in [`configs/README.md`](configs/README.md)
