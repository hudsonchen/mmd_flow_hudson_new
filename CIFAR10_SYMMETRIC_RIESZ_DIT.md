# CIFAR-10: symmetric paired Riesz with a DiT

Submitted on 2026-10-27. This transfers the successful MNIST/FashionMNIST
symmetric paired pixel-Riesz recipe to CIFAR-10. It is a new, randomly
initialized experiment; CIFAR performance is not yet known.

## Jobs and results

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

## Forward process and loss

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

## Model and training

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

## Validation and recovery

The existing small-flow and reference-DiT suites passed: **41 tests**.
GPU compatibility, full-batch memory and finite gradients are checked by
the queued preflight before training is allowed to run.

Training saves `latest.pt` every 1,000 updates, and updates `best.pt` at
FID milestones. These are recovery artifacts; this runner does not implement
automatic exact-state resume. Any continuation after the scheduler limit
needs an explicit recovery implementation or a documented Riesz-only fork.
The preflight is a memory/gradient check without optimizer updates; actual
training additionally allocates Adam state and EMA weights.
