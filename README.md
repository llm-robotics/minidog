<h1>
  <img src="assets/minidog-cute.jpg" alt="DiffusionBench logo" width="40" align="top">
  MiniDog
</h1>

MiniDog is a teaching resource for flow-matching generative models. It has two tasks.

- **Task 1**: learn flow-matching basics on 2D toy data. Runs on CPU in `toy_flow_matching.ipynb`, ~15 minutes.
- **Task 2**: pretrain and fine-tune a text-to-image diffusion transformer on dog photos. Runs on 4 consumer GPUs from the `minidog/` package, ~3 hours.

Both tasks train the same objective:

- A network learns the velocity that moves noise to data along a straight line.
- Sampling integrates that velocity.
- Task 1 shows this on 2D points you can plot.
- Task 2 runs it at full scale: a LightningDiT on VAE latents, with text conditioning, REPA, classifier-free guidance, and FID and preference scores.

**Prerequisites**: linear algebra, probability, and basic PyTorch. You should be able to write an `nn.Module`
and a training loop.

## Where each concept lives

The same flow-matching pieces appear in both tasks. Read them side by side.

| Flow-matching concept | Task 1: `toy_flow_matching.ipynb` | Task 2: `minidog/` |
|---|---|---|
| Forward process `x_t = (1-t) x + t eps` | *Flow matching* cell, `training_losses` | [`transport.py`](minidog/transport.py) `Transport.sample` |
| Target `v = eps - x`, MSE loss | same cell | [`transport.py`](minidog/transport.py) `compute_loss` |
| x- vs v-prediction | `pred_type` branches | [`transport.py`](minidog/transport.py) `convert_model_pred`, config `transport.prediction` |
| Timestep sampling during training | uniform `t` | [`transport.py`](minidog/transport.py) `get_time_sampler`, logit-normal |
| Euler sampling from noise to data | `sample` | [`transport.py`](minidog/transport.py) `Sampler.sample_ode` |
| Time conditioning of the network | *Model* cell, `SinusoidalEmbedding` | [`dit.py`](minidog/dit.py) `GaussianFourierEmbedding`, 4 time tokens |
| The denoiser | `MLPDenoiser` | [`dit.py`](minidog/dit.py) `LightningDiT` |
| Conditioning on text | — | [`text_encoder.py`](minidog/text_encoder.py), [`dit.py`](minidog/dit.py) `ConditionEmbedder` |
| Classifier-free guidance | — | [`transport.py`](minidog/transport.py) `apply_cfg_dropout`, `forward_with_cfg` |
| Flow matching in a latent space | orthogonal projection to `D` dims | [`vae.py`](minidog/vae.py), [`precompute_latents.py`](minidog/precompute_latents.py) |
| Representation alignment (REPA) | — | [`dinov2.py`](minidog/dinov2.py), `repa_projector` in [`dit.py`](minidog/dit.py), loss in `training_losses` |

## Getting started

**Task 1: flow-matching basics** (CPU, ~15 minutes)

```bash
uv sync
uv run jupyter lab toy_flow_matching.ipynb
```

Run the notebook top to bottom. Each cell is explained in the markdown above it. You will learn:

- how a flow-matching model is trained and sampled;
- why predicting the clean data beats predicting the velocity when the data lives in a high-dimensional space.

**Task 2: MiniDog text-to-image** (4 GPUs, ~3 hours)

Follow [`minidog/README.md`](minidog/README.md). It walks through six steps: download the data,
precompute latents, pretrain, fine-tune, generate, score. You will learn:

- how a text-to-image diffusion transformer is built and trained with the same objective;
- how the tokenizer, REPA and guidance change the result;
- why FID and human-preference scores disagree after fine-tuning.

## Going further

Ideas to explore once both tasks run:

- Set `transport.prediction` to `x` and compare FID curves: the Task 1 question at T2I scale.
- Turn `repa.use_repa` off, change `repa.repa_layer_depth`, or swap the REPA target encoder in `minidog/dinov2.py`.
- Try another tokenizer: add a class with `encode`/`decode` to `minidog/vae.py` and point `stage_1.target` at it, e.g. the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) VAE or [RAEv2](https://github.com/nanovisionx/RAEv2).
- Sweep `guidance.cfg.scale` on a checkpoint with `minidog.offline_eval`.
- Fine-tune on your own images: pack `jpg` + `txt` WebDataset shards, build FID stats with `minidog.fid_stats`, precompute with `sft.yaml`, train from the pretrain checkpoint.

## Layout

```
toy_flow_matching.ipynb   Task 1
minidog/                  Task 2 package and README
configs/                  experiment configs and results
```
