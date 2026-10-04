<p align="center"><img src="assets/minidog-full.png" alt="MiniDog" width="70%"></p>

<p align="center">
  <a href="https://huggingface.co/datasets/reyhanehesi/dog-t2i-diffusion-data" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace"></a>
  <a href="https://huggingface.co/reyhanehesi/minidog-checkpoints" target="_blank"><img src="https://img.shields.io/badge/Checkpoints-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="Checkpoints"></a>
  <a href="about:blank" target="_blank"><img src="https://img.shields.io/badge/Paper-PDF-EC1C24?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper"></a>
</p>

<p align="center">
  <a href="assets/minidog-60s.mp4"><img src="assets/minidog-teaser.gif" alt="MiniDog in 60 seconds" width="88%"></a>
</p>
<p align="center"><sub><a href="assets/minidog-60s.mp4">Full 60-second version</a></sub></p>

MiniDog is a minimal teaching and research resource for flow-matching generative models, in two parts.

- **[Warm-up](#warm-up-flow-matching-basics)**: learn flow-matching basics on 2D toy data. Runs on CPU in [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb), ~15 minutes.
- **[The five lessons](#the-five-lessons)**: build a text-to-image diffusion transformer one design decision at a time, each backed by a controlled experiment you can rerun. Runs on 4 consumer GPUs (RTX 3090) from the [`minidog/`](minidog/) package. Pretraining reaches FID 8.32 in 75 minutes of training, fine-tuning takes 20 more.

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
| x- vs v-prediction | `pred_type` branches | [`convert_model_pred`](minidog/transport.py#L128), config [`transport.prediction`](configs/pretrain_repa_dinov2_mse.yaml#L28) |
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

4x3090 GPUs, 75 minutes pretraining, 20 minutes fine-tuning (about 2.5 hours with the in-training FID evaluations left on).

Each lesson asks one design question and answers it with a controlled experiment, in the order the
pipeline is built. Together they take FID from 16.72 to 8.32 in the same training budget.

| Lesson | Question | Answer | FID |
|---|---|---|---|
| — | Starting point: E2E-VAVAE, 64-token captions, no alignment | | 16.72 |
| 1. Tokenizer | Which tokenizer? | E2E-INVAE. Good reconstruction is necessary but not sufficient; the latents also have to be easy to model | 12.70 |
| 2. Captions | How detailed should captions be? | The 128-token captions | 11.47 |
| 3. Alignment | Does representation alignment help? | Yes, most on a small training budget | 8.80 |
| 4. Target | Which target representation? | EUPE with iREPA. Which target is best depends on how it is aligned | **8.32** |
| 5. SFT | Does fine-tuning on high-quality images help? | Yes: HPSv2 0.1625 -> 0.2427, PickScore 18.42 -> 19.61 | — |

Follow [`minidog/README.md`](minidog/README.md) to run them: download the data, precompute latents,
pretrain, fine-tune, generate, score. The configs behind every number are listed in
[`configs/README.md`](configs/README.md).

## Going further

Ideas to explore once both tasks run:

- Set [`transport.prediction`](configs/pretrain_repa_dinov2_mse.yaml#L28) to `x` and compare FID curves: the warm-up question at full scale.
- Turn [`repa.use_repa`](configs/pretrain_repa_dinov2_mse.yaml#L86) off, change `repa.repa_layer_depth`, or swap the target representation: the four encoders compared in the paper each have a config pair, `configs/pretrain_{repa,irepa}_{dinov2,dinov3,pe_spatial,eupe}_{mse,cosine}.yaml`.
- Try another tokenizer: add a class with `encode`/`decode` to [`minidog/vae.py`](minidog/vae.py) and point [`stage_1.target`](configs/pretrain_repa_dinov2_mse.yaml#L2) at it, e.g. the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) VAE or [RAEv2](https://github.com/nanovisionx/RAEv2).
- Sweep [`guidance.cfg.scale`](configs/pretrain_repa_dinov2_mse.yaml#L36) on a checkpoint with [`minidog.offline_eval`](minidog/offline_eval.py).
- Fine-tune on your own images: pack `jpg` + `txt` WebDataset shards, build FID stats with [`minidog.fid_stats`](minidog/fid_stats.py), precompute with [`sft_eupe_norepa.yaml`](configs/sft_eupe_norepa.yaml), train from the pretrain checkpoint.

## Layout

- [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb): the warm-up
- [`minidog/`](minidog/): the text-to-image package, with its own [README](minidog/README.md)
- [`configs/`](configs/): experiment configs, with the results table in [`configs/README.md`](configs/README.md)
