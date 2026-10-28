# CIFAR-10: Riesz training progress

## Riesz U-Net training progress

Snapshot: **2026-10-27 23:51:34 EDT**. This is a fixed report; training continues. Run states below come from each trainer's status file; scheduler state was not independently verified for this snapshot.

### Current result

The original Riesz U-Net has reached **81,700 / 100,000 updates**, with best FID **20.49 at 80,000**. Every completed 5,000-update FID evaluation has improved so far: 84.65 at 5,000 to 20.49 at 80,000, a **75.8% reduction**. Improvements continue, although gains are smaller later in training.

The fresh 250-step trial has reached **7,800 / 100,000 updates**. Its first FID is **86.72 at 5,000**, versus **84.65** for the original run at the same update (2.07 points higher). This single evaluation does not establish an advantage for the smaller reference interval.

| Run | Job | Trainer state | Progress | Best FID ↓ | GPU |
| --- | --- | --- | --- | --- | --- |
| Riesz U-Net · 100 Euler steps | 49846683 | running | 81,700 / 100,000 (81.7%) | 20.49 at 80,000 | NVIDIA H200 MIG 7g.141gb |
| Riesz U-Net · 250 Euler steps | 49894698 | running | 7,800 / 100,000 (7.8%) | 86.72 at 5,000 | NVIDIA A100-PCIE-40GB |

### Training setup

Both runs train `CIFAR10UNetParticleGenerator` from random weights (seed 0): **61,804,419 parameters**, base channels 128, four residual blocks per level, and time conditioning. Neither uses diffusion/FM weights, reconstruction priming, a pretrained feature encoder in the loss, or an auxiliary velocity-MSE objective.

- **Objective:** symmetric paired pixel-space Riesz, row/column affinities, temperatures 0.2 / 0.05 / 0.02, per-temperature normalization, and self negatives included.
- **Reference:** affine Gaussian-to-data path; sample a shared continuous reference time per interaction batch. Each update uses one network call at that time, rather than a full sampling rollout.
- **Optimization:** Adam, batch 256, peak learning rate 2e-4, 500-update warmup, cosine decay with a 10% floor over 100,000 updates; gradient norm clipped to 1. EMA decay is 0.99 through update 1,000, then 0.999. Random horizontal flips are enabled.
- **Difference:** original reference interval `h=0.01`, 100 Euler sampling steps; new trial `h=0.004`, 250 Euler sampling steps. Both the training interval and inference cost change, so this comparison does not isolate either effect. Hardware also differs.
- **Evaluation:** EMA FID every 5,000 updates, 10,000 generated images against all 50,000 CIFAR-10 training images, Inception 2048 features. These are training-reference FIDs, not test-set scores. Fixed-noise 32-image previews run every 1,000 updates.

### FID learning curves

![FID learning curves](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/figures/fid_curves.png)

Markers are completed evaluations; lines only connect measurements. The 250-step trial has one FID point. The right panel compares both at 5,000 updates. This is a single-seed comparison with no confidence intervals.

| Update | Original: 100 steps | New trial: 250 steps |
| ---: | ---: | ---: |
| 5,000 | 84.65 | 86.72 |
| 10,000 | 62.55 | — |
| 15,000 | 50.62 | — |
| 20,000 | 44.26 | — |
| 25,000 | 36.67 | — |
| 30,000 | 33.83 | — |
| 35,000 | 30.41 | — |
| 40,000 | 27.84 | — |
| 45,000 | 26.47 | — |
| 50,000 | 24.83 | — |
| 55,000 | 23.39 | — |
| 60,000 | 22.82 | — |
| 65,000 | 21.74 | — |
| 70,000 | 21.23 | — |
| 75,000 | 20.74 | — |
| 80,000 | 20.49 | — |

### Loss and gradient curves

![Training loss and gradient norms](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/figures/training_diagnostics.png)

Curves use deduplicated status observations from the persistent monitor plus the latest status files: Riesz U-Net · 100 Euler steps: 523 observations, updates 100–81,700; Riesz U-Net · 250 Euler steps: 77 observations, updates 100–7,800. These are **sampled observations**, not a complete per-update training log. Faint lines show raw observations; bold lines show a trailing median over 11 available observations. Vertical axes use a symmetric log scale. The two runs have independent horizontal axes.

The gradient norm is recorded **before clipping**; values above 1 do not mean clipping failed. Loss depends on randomly sampled reference times and the reference interval, so its fluctuations and absolute scale should not be interpreted as FID or directly ranked across these runs. Latest recorded losses and gradients are finite.

| Run | Latest loss | Pre-clip gradient norm | Recorded training elapsed | Peak allocated GPU memory |
| --- | ---: | ---: | ---: | ---: |
| Riesz U-Net · 100 Euler steps | 398.164 | 0.420 | 5.58 h | 26.43 GB |
| Riesz U-Net · 250 Euler steps | 1420.795 | 1.278 | 1.43 h | 24.01 GB |

Elapsed time includes evaluation after the training timer starts and excludes queue time and initial setup.

### Preview diagnostics and visual review

![Preview statistics](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/figures/preview_diagnostics.png)

These statistics come from 32 fixed-noise samples. Pixel variance and out-of-range rates are diagnostics, not guarantees of diversity or quality.

#### Original run: update 81,000

![Original Riesz U-Net samples](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/figures/preview_0_0081000.png)

#### 250-step trial: update 7,000

![250-step Riesz U-Net samples](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/figures/preview_1_0007000.png)

Visual review of these two grids shows varied vehicles, animals, and outdoor scenes, with blurred or distorted details still present. Neither grid shows an obvious all-samples-identical collapse. The grids are from different training updates and are a limited visual check, not a matched-update quality comparison.

### Next milestones and artifacts

Continue the original run toward 100,000 updates; its FID is still improving. The next FID checkpoints after this snapshot are 85,000 for the original and 10,000 for the 250-step trial. Compare the latter against the original at matched updates, and use matched-sampler evaluations before attributing a difference to reference discretization. Both runs have `latest.pt` and `best.pt` recovery artifacts.

- **Riesz U-Net · 100 Euler steps**: [run directory](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_b256_seed0_20261027), [configuration](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_b256_seed0_20261027/training/config.json), [best FID](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_b256_seed0_20261027/training/best.json), [frozen trainer](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_b256_seed0_20261027/source/scripts/lsf/run_small_image_flow.py).
- **Riesz U-Net · 250 Euler steps**: [run directory](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_ref250_b256_seed0_20261027), [configuration](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_ref250_b256_seed0_20261027/training/config.json), [best FID](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_ref250_b256_seed0_20261027/training/best.json), [frozen trainer](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_ddpm_unet_ref250_b256_seed0_20261027/source/scripts/lsf/run_small_image_flow.py).

Report data: [snapshot JSON](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/snapshot.json), [FID CSV](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/fid_history.csv), [training observations CSV](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/training_history.csv), [preview CSV](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/preview_history.csv). Figures are available as PNG and SVG in `figures/`. Copied preview images are unchanged. [Report generation script](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/riesz_unet_progress_20261027_235134/build_report.py).

---

## DiT experiment: original launch record

The following records the DiT submission on 2026-10-27; its queue states are historical, not a current progress check. This transfers the successful MNIST/FashionMNIST
symmetric paired pixel-Riesz recipe to CIFAR-10. It is a new, randomly
initialized experiment; CIFAR performance is not yet known.

### Jobs and results

| Job | ID | Purpose |
| --- | --- | --- |
| A100 preflight | 49832328 | Actual DiT forward/backward, finite gradients, memory/runtime and sampler checks |
| A100 training | 49832366 | Starts only after preflight exits successfully |
| CPU monitor | 49832404 | Records scheduler state, metrics, previews and numerical warnings every 30 seconds |

At submission, the A100s on `arpa-gpu01` were occupied. Both GPU jobs are
queued in `longgpu02`; submission does not mean training has started.
The H100 queues inspected were restricted to other groups.

- [Training study and frozen source](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_dit_s2_b256_seed0_20261027)
- [Training manifest](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_dit_s2_b256_seed0_20261027/manifest.json)
- [Preflight study](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_symmetric_dit_s2_b256_smoke_20261027)
- [Campaign and monitor](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_symmetric_dit_campaign_20261027)

Each GPU job requests one A100 with 32 GB of GPU memory, four CPUs and
16 GB of host memory. Training has a 48-hour scheduler limit. The monitor
runs for up to 72 hours, including queue time. It records numerical evidence
and image paths; it does not visually inspect images or send chat updates.

### Forward process and loss

The full derivation is in [the implementation walkthrough](PIXEL_RIESZ_IMPLEMENTATION.md).
For centered RGB images `y` and independent Gaussian noise `z`:

```text
t ~ Uniform(0, 0.999), shared within the interaction batch
x_t = (1-t) z + t y
h = min(0.01, 1-t)
r = (1-t-h) z + (t+h) y

D = DiT clean-image prediction
v = (D - x_t) / max(1-t, 0.001)
q = x_t + h v
```

The Riesz field attracts `q` toward the paired reference population `r` and
repels it from the detached predicted population. Each particle retains its
own paired positive. Its zero-displacement self negative also participates
in affinity normalization, making the field cancel when the two empirical
populations coincide.

The loss uses row/column softmax affinities at temperatures `0.2, 0.05, 0.02`,
per-temperature field RMS normalization, and the same detached-target
regression with time weighting as the successful MNIST run. There is no
distance smoothing, multiscale pooling, learned feature encoder, reconstruction
priming, FM initialization, or added velocity-MSE term. Each training update
uses one network call at a sampled reference time.

### Model and training

| Setting | Value |
| --- | --- |
| Dataset | CIFAR-10, native RGB 32×32 |
| Backbone | Time-conditioned DiT, width 384, depth 12, 6 heads |
| Patches | 2×2 pixels; 256 tokens per image |
| Output | Clean RGB prediction via `2*sigmoid(logits)-1` |
| Initialization | Random weights, seed 0 |
| Batch | 256 |
| Augmentation | Random horizontal flips |
| Optimizer | Adam, peak learning rate 0.0002 |
| Schedule | 500-update warmup, cosine decay over 100,000 updates |
| Gradient clipping | Norm 1 |
| EMA | Decay 0.99 through update 1,000, then 0.999 |
| Sampling | 100 Euler steps using EMA weights |
| Preview cadence | Update 0 and every 1,000 updates, fixed 32-noise sample |
| FID cadence | Every 5,000 updates |
| FID protocol | 10,000 generated images versus all 50,000 training images |
| FID features | Native RGB, pytorch-fid Inception 2048 |

The existing DiT adapter bounds clean predictions to `[-1, 1]`; the MNIST
U-Net clean predictor is unbounded. This output mapping, the RGB dataset,
architecture, and horizontal flips are changes from the MNIST recipe.
The paired reference process and symmetric Riesz loss remain the same.

The first FID will be at update 5,000. Previews are sampling diagnostics,
not FID estimates. Earlier CIFAR FM runs use different sampling and training
settings, so their scores require protocol checks before direct comparison.

### Validation and recovery

The existing small-flow and reference-DiT suites passed: **41 tests**.
GPU compatibility, full-batch memory and finite gradients are checked by
the queued preflight before training is allowed to run.

Training saves `latest.pt` every 1,000 updates, and updates `best.pt` at
FID milestones. These are recovery artifacts; this runner does not implement
automatic exact-state resume. Any continuation after the scheduler limit
needs an explicit recovery implementation or a documented Riesz-only fork.
The preflight is a memory/gradient check without optimizer updates; actual
training additionally allocates Adam state and EMA weights.
