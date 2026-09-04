<p align="center"><img src="assets/minidog-full.png" alt="MiniDog" width="70%"></p>

<p align="center">
  <a href="https://huggingface.co/datasets/reyhanehesi/dog-t2i-diffusion-data" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace">
  </a>
  <a href="about:blank" target="_blank">
    <img src="https://img.shields.io/badge/Paper-PDF-EC1C24?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Paper">
  </a>
</p>

MiniDog is a minimal teaching and research resource for flow-matching generative models. It has two tasks.

- **[Task 1](#task-1-flow-matching-basics)**: learn flow-matching basics on 2D toy data. Runs on CPU in [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb), ~15 minutes.
- **[Task 2](#task-2-minidog-text-to-image)**: pretrain and fine-tune a text-to-image diffusion transformer on dog photos. Runs on 4 consumer GPUs (RTX 3090) from the [`minidog/`](minidog/) package. Pre-training takes 2.5 hours, fine-tuning takes 20 minutes.

Both tasks train the same objective:

- A network learns the velocity that moves noise to data along a straight line.
- Sampling integrates that velocity.
- Task 1 shows this on 2D points you can plot.
- Task 2 runs it at full scale: a [LightningDiT](minidog/dit.py) on [VAE latents](minidog/vae.py), with [text conditioning](minidog/text_encoder.py), [REPA](minidog/dinov2.py), [classifier-free guidance](minidog/transport.py), and [FID](minidog/eval.py) and [preference scores](minidog/score.py).

**Prerequisites**: linear algebra, probability, and basic PyTorch. You should be able to write an `nn.Module`
and a training loop.

## Where each concept lives

The same flow-matching pieces appear in both tasks. Read them side by side; every Task 2 entry links to the line in the code.

| Flow-matching concept | Task 1: [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb) | Task 2: [`minidog/`](minidog/) |
|---|---|---|
| Forward process `x_t = (1-t) x + t eps` | *Flow matching* cell, `training_losses` | [`Transport.sample`](minidog/transport.py#L103) |
| Target `v = eps - x`, MSE loss | same cell | [`compute_loss`](minidog/transport.py#L135) |
| x- vs v-prediction | `pred_type` branches | [`convert_model_pred`](minidog/transport.py#L128), config [`transport.prediction`](configs/pretrain.yaml#L28) |
| Timestep sampling during training | uniform `t` | [`get_time_sampler`](minidog/transport.py#L79), logit-normal |
| Euler sampling from noise to data | `sample` | [`Sampler.sample_ode`](minidog/transport.py#L155) |
| Time conditioning of the network | *Model* cell, `SinusoidalEmbedding` | [`GaussianFourierEmbedding`](minidog/dit.py#L86), 4 time tokens |
| The denoiser | `MLPDenoiser` | [`LightningDiT`](minidog/dit.py#L140) |
| Conditioning on text | — | [`TextEncoder`](minidog/text_encoder.py#L8), [`ConditionEmbedder`](minidog/dit.py#L104) |
| Classifier-free guidance | — | [`apply_cfg_dropout`](minidog/transport.py#L15) (train), [`forward_with_cfg`](minidog/transport.py#L179) (sample) |
| Flow matching in a latent space | — | [`VAE.encode`](minidog/vae.py#L146), [`precompute_latents`](minidog/precompute_latents.py#L180) |
| Representation alignment (REPA) | — | [`DINOv2Encoder`](minidog/dinov2.py#L9), [`repa_projector`](minidog/dit.py#L178), [`loss_repa`](minidog/transport.py#L125) |

## Getting started

### Task 1: flow-matching basics

CPU, ~15 minutes.

```bash
uv sync
uv run jupyter lab toy_flow_matching.ipynb
```

Run [the notebook](toy_flow_matching.ipynb) top to bottom. Each cell is explained in the markdown above it. You will learn:

- how a flow-matching model is trained and sampled;
- why predicting the clean data beats predicting the velocity when the data lives in a high-dimensional space, and why MiniDog can still use v-prediction in its compressed latent space.

### Task 2: MiniDog text-to-image

4x3090 GPUs, 2.5 hours pretraining, 20 minutes fine-tuning.

Follow [`minidog/README.md`](minidog/README.md). It walks through six steps: download the data,
precompute latents, pretrain, fine-tune, generate, score. The configs are listed in [`configs/README.md`](configs/README.md). You will learn:

- how a text-to-image diffusion transformer is built and trained with the same objective;
- how the tokenizer, REPA and guidance change the result;
- why FID and human-preference scores disagree after fine-tuning.

## Going further

Ideas to explore once both tasks run:

- Set [`transport.prediction`](configs/pretrain.yaml#L28) to `x` and compare FID curves: the Task 1 question at T2I scale.
- Turn [`repa.use_repa`](configs/pretrain.yaml#L86) off, change `repa.repa_layer_depth`, or swap the REPA target encoder in [`minidog/dinov2.py`](minidog/dinov2.py).
- Try another tokenizer: add a class with `encode`/`decode` to [`minidog/vae.py`](minidog/vae.py) and point [`stage_1.target`](configs/pretrain.yaml#L2) at it, e.g. the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) VAE or [RAEv2](https://github.com/nanovisionx/RAEv2).
- Sweep [`guidance.cfg.scale`](configs/pretrain.yaml#L36) on a checkpoint with [`minidog.offline_eval`](minidog/offline_eval.py).
- Fine-tune on your own images: pack `jpg` + `txt` WebDataset shards, build FID stats with [`minidog.fid_stats`](minidog/fid_stats.py), precompute with [`sft.yaml`](configs/sft.yaml), train from the pretrain checkpoint.

## Layout

- [`toy_flow_matching.ipynb`](toy_flow_matching.ipynb): Task 1
- [`minidog/`](minidog/): Task 2 package, with its own [README](minidog/README.md)
- [`configs/`](configs/): experiment configs, with the results table in [`configs/README.md`](configs/README.md)
