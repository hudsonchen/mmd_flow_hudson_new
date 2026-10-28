# CIFAR-10: direct-velocity Riesz setup

This document describes the four-GPU velocity Riesz experiment launched on
2026-10-28. It replaces the earlier CIFAR-10 DiT/denoiser progress report.
The setup below is taken from this run's saved configuration and frozen source.

## Run and model

| Setting | Value |
| --- | --- |
| Run | `pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028` |
| Training job | `49976959` |
| Hardware | 4 × RTX 2080 Ti on `longgpu01`, queue `longgpu` |
| Dataset | CIFAR-10 training set, 50,000 RGB images at 32 × 32 |
| Data preprocessing | Convert uint8 pixels to `[-1, 1]`; random horizontal flip with probability 0.5 |
| Data sampling | Random image indices with replacement on each rank |
| Backbone | `CIFAR10UNetParticleGenerator`, base channels 128, four residual blocks per level, time conditioning |
| Parameters | 61,804,419 |
| Output | Unrestricted RGB velocity, using the `DDPMFlow` wrapper |
| Time embedding | `999 * (1 - t)` |
| Initialization | Random weights, seed 0 |
| Batch | 64 images per GPU; 256 interacting particles globally |
| Training budget | 100,000 successful optimizer updates |

The wrapper and skewed-time helper are shared with the flow-matching code.
Training uses the Riesz objective throughout, with no flow-matching or diffusion
weights, reconstruction priming, pretrained feature encoder, or auxiliary
velocity-MSE objective.

## Reference process and velocity prediction

Let `y` be a centered training image and `z ~ N(0, I)` independent Gaussian
noise. Draw one scalar time for the entire global interaction batch:

```text
a ~ N(0, 1)
t = clamp(sigmoid(1.2 - 1.2*a), 0.0001, 0.999)
h = min(0.01, 1 - t)

x_t = (1 - t)*z + t*y
r   = (1 - t - h)*z + (t + h)*y
q   = x_t + h*v_theta(x_t, t)
```

Rank 0 draws `t` and broadcasts it to all ranks. The source `x_t` and reference
`r` use the same image/noise pair. All 256 predicted particles interact with
all 256 reference particles. Each update evaluates the network once on each
rank's local batch; it does not backpropagate through a sampling rollout.

The network predicts velocity directly. The retained time weight is `(1-t)/h`
on the Riesz surrogate; there is no conversion from a clean-image prediction
through division by `1-t`.

## Pixel-space Riesz objective

Flatten each RGB image to 3,072 coordinates. The support population concatenates
256 detached predictions (negatives) and 256 reference particles (positives).
The paired positive and the zero-displacement self negative are both included
in affinity normalization. The self negative contributes zero displacement.

| Setting | Value |
| --- | --- |
| Distance powers `p` | 0.8, 1.0, 1.5 |
| Temperatures `tau` | 0.2, 0.05, 0.02 |
| Affinity | Geometric mean of row and column softmax affinities |
| Repulsion weight | 1 |
| Field aggregation | Sum all nine power/temperature fields |
| Field RMS normalization | None |
| Distance smoothing | None |
| Query-tail reweighting | None |

For query-to-support distances `d_ij`, the detached batch distance scale is
`s = max(mean(d_ij), 0.001)`. For each power and temperature:

```text
logits_ij = -(d_ij / s)^p / tau
A_ij = sqrt(softmax_over_support(logits)_ij
            * softmax_over_queries(logits)_ij)
```

Split `A` into negative and positive columns. Each positive affinity is
multiplied by that query's total negative affinity, and each negative affinity
by its total positive affinity. These cross-mass factors balance attraction
and repulsion. Each displacement contributes its unit direction multiplied
by `p * (d/s)^(p-1) / tau` and the corresponding balanced affinity; attraction
is added and repulsion subtracted. Distances in force denominators and power
factors are floored at `1e-6`, so exact zero displacements remain zero.

Let `F` be the sum of the nine fields and
`c = max(s / sqrt(3072), 0.001)`. The training surrogate is:

```text
goal = stop_gradient(q / c + F)
L = ((1 - t) / h) * mean((q / c - goal)^2)
```

The scale, support, affinities, and field are detached. Gradients flow only
through the prediction in the first `q/c` term. `normalization='none'` disables
field RMS normalization; coordinate scaling and softmax normalization still
apply. Logged loss magnitudes depend on sampled time and field strength and
are not quality metrics.

## Distributed training and optimization

Every rank gathers detached predictions and targets from all four GPUs and
constructs the same 256-particle field. Only that rank's 64 predictions are
reattached to its autograd graph. The backward loss is multiplied by four to
compensate for DDP's gradient averaging, preserving the global-batch gradient.

| Setting | Value |
| --- | --- |
| Optimizer | Adam, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay 0 |
| Learning rate | Peak `2e-4`; 500-update warmup; cosine decay to 10% of peak |
| Gradient clipping | Global parameter norm 1, after AMP unscaling |
| Precision | FP16 autocast for network calls; predictions and particle-field calculations in FP32 |
| Gradient scaler | Dynamic AMP scaler, initial scale 1,024 |
| EMA | Decay 0.99 through update 1,000, then 0.999 |
| Distributed backend | NCCL, one process per GPU |
| CPU threads | Two per process |

For zero-based successful-update count `u`, the learning rate is:

```text
lr(u) = 2e-4 * min((u + 1)/500, 1)
              * (0.1 + 0.45*(1 + cos(pi*u/100000)))
```

An AMP overflow skips the optimizer update and does not advance the successful
update count or EMA. More than 20 consecutive overflows raises an error.
The logged gradient norm is measured before clipping.

## Sampling, evaluation, and checkpoints

Sampling uses EMA weights and 100 uniform Euler intervals, starting from
Gaussian noise at `t=0` and ending at `t=1`:

```text
x_0 ~ N(0, I)
x_(k+1) = x_k + v_EMA(x_k, k/100)/100,  k = 0,...,99
```

This costs 100 network evaluations per generated image. Sampling uses FP16
network autocast with FP32 state updates. Raw states are not clipped during
integration; final images are mapped to `[0,1]` and clamped for FID and display.

| Artifact / evaluation | Schedule and protocol |
| --- | --- |
| Training status | Every 100 successful updates; loss, pre-clip gradient, learning rate, sampled time, memory, AMP scale, and overflow count |
| Preview | Update 0 and every 1,000 updates; 32 fixed-noise images, seed 8765432 |
| FID | Every 5,000 updates and at completion; 10,000 generated images against all 50,000 training images |
| FID features | Inception 2048; unquantized float pixels in `[0,1]` |
| Evaluation batching | Sampling batch 32 and Inception batch 32 per GPU |
| Evaluation noise | Reused CPU RNG seeds `100000 + seed + rank` across milestones |
| Recovery | Atomic `latest.pt` every 1,000 updates and at completion |
| Best checkpoint | `best.pt` and `best.json` updated when a completed FID improves |

Checkpoints contain model and EMA weights, optimizer, AMP scaler, successful
update and attempt counts, configuration, and per-rank CPU/CUDA RNG states.
The saved trainer supports `--resume` from a compatible checkpoint, checks that
the configuration matches, rejects flow-matching ancestry, and restores each
rank's RNG state after initialization.

These FIDs use the training reference set. The run does not automatically
compute a separate test-set FID. Comparisons with flow matching also involve
different solvers and time grids, even when both use 100 network evaluations.

## Progress snapshot

Snapshot read **2026-10-28 19:19 EDT**; this is a fixed observation, not a live
status display. The trainer reported **43,600 / 100,000 updates (43.6%)**.
Best completed FID: **13.47 at 40,000 updates**. The scheduler was independently
checked during the preceding progress review and reported job `49976959` running.

| Update | FID ↓ |
| ---: | ---: |
| 5,000 | 58.30 |
| 10,000 | 33.69 |
| 15,000 | 25.15 |
| 20,000 | 20.07 |
| 25,000 | 18.32 |
| 30,000 | 15.38 |
| 35,000 | 13.66 |
| 40,000 | 13.47 |

At this snapshot, loss and gradient norm were finite, rank-0 peak allocated
GPU memory was 6.83 GB, and 11 AMP overflow attempts had been skipped.
Visual review of the 43,000-update preview showed varied recognizable objects
with some blurred or distorted details and no obvious collapse in that grid.
The next scheduled FID is at 45,000 updates.

## Configuration and source artifacts

- [Run directory](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028)
- [Saved configuration](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/training/config.json)
- [Manifest and frozen-source hashes](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/manifest.json)
- [Frozen training runner](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/source/scripts/lsf/run_cifar_riesz_ddp.py)
- [Frozen velocity loss and sampler](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/source/cifar_riesz_velocity.py)
- [Frozen power-distance Riesz implementation](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/source/small_flow.py)
- [Latest trainer status](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/training/status.json)
- [Best completed FID](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/training/best.json)
- [43,000-update preview](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/pixel_riesz_cifar10_velocity_power08_10_15_skewed_4x2080_seed0_20261028/training/preview_0043000.png)
