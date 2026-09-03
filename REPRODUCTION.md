# Reproduction log

Reference numbers are from the MiniDog tutorial (section 8 ablation table, section 9 SFT curve,
section 11 preference scores). Reproduced on the cleaned `minidog` package, commit `69d3ac2`+,
4x H100 80GB, bf16, `--compile`, one seed (42) per run, all data from the HF dataset as
described in `minidog/README.md`. Run logs: `results/logs/`, in-training FID/IS CSVs:
`results/evals/<run>/`.

## Pretraining (`configs/pretrain.yaml`, 200 epochs = 20,200 steps)

| Step | FID (ref) | FID (ours) | IS (ref) | IS (ours) |
|---|---|---|---|---|
| 5,000 | | 89.8 | | 11.2 |
| 10,000 | | 16.5 | | 17.3 |
| 15,000 | | 10.5 | | 17.7 |
| 20,000 | **9.4** | **8.6** | **17.9** | **17.8** |

Wall time 59 min including four evals (tutorial: 2h15m on 4x RTX 3090). FID is evaluated at
`eval.eval_interval` multiples, so the last measured point is step 20,000; the shipped checkpoint
is step 20,200 (`ep-0000200.pt`). See "CFG sweep" below for the step-20,200 number.

## SFT (`configs/sft.yaml`, from `pretrain/ep-0000200.pt`, 100 epochs = 700 steps)

| Step | FID (ref) | FID (ours) | IS (ours) |
|---|---|---|---|
| 150 | 13.6 | 13.9 | 16.5 |
| 300 | 20.6 | 21.8 | 15.4 |
| 450 | 25.7 | 27.1 | 14.7 |
| 600 | 28.8 | 30.2 | 14.3 |

Monotonically rising FID, as the tutorial reports. Wall time 34 min, almost all of it the four evals.

## Human-preference scores (`minidog.score`, 500 captions, pretrain `ep-0000200` vs SFT `ep-0000100`)

| Metric | pretrain | SFT | Tutorial claim |
|---|---|---|---|
| PickScore win rate | 1.8% | **98.2%** | SFT preferred |
| PickScore mean preference | 0.24 | 0.76 | |
| HPSv2 mean | 0.155 | **0.239** | SFT preferred |

Per-pair scores: `results/scores/pretrain_vs_sft.json`.

## REPA ablation (`configs/ablations/e2e-invae-norepa-128tok.yaml`, same schedule, REPA off)

| Step | FID no-REPA (ours) | FID REPA (ours) | IS no-REPA (ours) |
|---|---|---|---|
| 5,000 | 110.4 | 89.8 | 8.9 |
| 10,000 | 33.1 | 16.5 | 15.7 |
| 15,000 | 16.0 | 10.5 | 17.2 |
| 20,000 | **11.54** | **8.56** | 17.65 |

Reference: 11.5 without REPA vs 9.4 with (gain 2.1). Ours: 11.54 vs 8.56 (gain 3.0). The
flow-matching loss is the same in both runs at every epoch (0.88 at epoch 100); REPA changes what
is learned, not how well the denoising objective is fit.

## CFG sweep on the epoch-200 checkpoints (`minidog.offline_eval`, step 20,200, EMA weights)

| CFG scale | pretrain FID | pretrain IS | no-REPA FID | no-REPA IS |
|---|---|---|---|---|
| 1.5 | 17.36 | 16.25 | 24.42 | 15.57 |
| 2.0 | 12.73 | 17.09 | 17.39 | 16.39 |
| 3.0 | 9.48 | 17.61 | 12.94 | 17.27 |
| 4.0 | **8.44** | 17.77 | 11.66 | 17.59 |
| 6.0 (training-time eval setting) | 8.48 | 17.76 | **11.43** | 17.62 |

The shipped `pretrain/ep-0000200.pt` scores 8.48 at the config's CFG 6.0, vs 8.56 measured at
step 20,000 during training: the 200-step gap between last eval and final checkpoint is immaterial.
FID falls monotonically from CFG 1.5 to 4.0 and is flat between 4.0 and 6.0, so the config's 6.0
is not costing FID; the usual "low CFG minimizes FID" pattern does not hold for this small model.
The REPA gap grows at low guidance (7.1 FID at CFG 1.5 vs 3.0 at CFG 6.0): the alignment loss
helps most where the model has to rely on its own conditioning.

All runs finished 2026-09-04 04:55. Nothing pending except the 64-token data question below.

## Not reproducible from the shipped data

The four `*-64tok` ablation rows (ref. FID 9.8, 12.7, 13.4, 16.7) need the 26k photos
recaptioned to 25-40 words (`dogs_recaptioned_64tok_wds`), which is not on the HF dataset.
Waiting on the original caption files; otherwise drop the rows and configs.

## Bugs fixed while reproducing

- raw-shard sample count was a hardcoded 26,000 (SFT set reported 26k instead of 2k)
- concurrent torch.hub downloads of DINOv2 corrupted the hub cache
- eval-dataset setup decoded all 26k images to read captions (~2 min per launch -> 5 s)
- loader rebuilt every epoch, refilling the shuffle buffer (~5 s/epoch, ~17 min per pretrain)
- hpsv2 wheel ships without its BPE vocab
- config passed constructor kwargs the trimmed DiT no longer accepts
