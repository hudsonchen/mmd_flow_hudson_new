# CIFAR-10 velocity Riesz flow

This checkout contains the direct-velocity, pixel-space Riesz training path for
CIFAR-10. The model uses the time-conditioned `CIFAR10UNetParticleGenerator`
(base channels 128, four residual blocks per level, **61,804,419 parameters**).
Fresh training starts from random weights; continuation accepts Riesz-only
checkpoints and restores optimizer, EMA, AMP scaler, and per-rank RNG state.

The paired reference path is `x_t = (1-t)z + ty`. Training compares
`x_t + h*v_theta(x_t,t)` against `x_(t+h)`, using one shared sampled time and
one global interacting particle population across GPUs. The loss sums the
power-distance Riesz fields for powers **0.8, 1.0, 1.5** and temperatures
**0.2, 0.05, 0.02**, with row-column affinities, included self negatives, and
no field RMS normalization. The time weight is `(1-t)^gamma / h`.

Default evaluation uses **50 uniform Heun intervals (100 network evaluations)**,
FID with 10,000 generated images versus 50,000 training images every **5,000
updates**, and fixed-noise previews/recovery checkpoints every **1,000 updates**.
See [the study record](CIFAR10_SYMMETRIC_RIESZ_DIT.md) for the full objective and
historical experiments; its frozen source links refer to the original modules.

## Run

Use the existing `.venv` (PyTorch with compatible CUDA, torchvision, NumPy,
SciPy, and Pillow). FID requires the checksum-verified Inception weights in
`assets/pt_inception-2015-12-05-6726825d.pth`. Data is downloaded or validated
under `/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/datasets`.

Freeze the source and validate it, then submit a GPU preflight followed by
training on four RTX 2080 Ti GPUs:

```bash
.venv/bin/python scripts/lsf/launch_cifar_riesz_ddp.py \
  --name velocity_riesz_my_run \
  --reference-time-power 1 \
  --submit
```

The run name must be new. Omitting `--submit` still creates the run directory,
frozen source, test log, and manifest, but submits no jobs. New runs use
`run_storage.new_run_path` under
`/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/`.

For a single H200 with global batch 256:

```bash
.venv/bin/python scripts/lsf/launch_cifar_riesz_ddp.py \
  --name velocity_riesz_h200_my_run \
  --gpus 1 --batch-per-rank 256 \
  --queue dbeigpu --host dbei-ai1 --required-gpu H200 \
  --gpu-memory 48 --gpu-mig 7/7 \
  --reference-time-power 0.5 --submit
```

Resume a velocity Riesz checkpoint by adding `--resume /absolute/path/latest.pt`
and retaining its training configuration and GPU layout. The launcher preserves
a hard-linked copy as `resume.pt`; source and destination must share a filesystem.
`--allow-evaluation-change` permits a new output directory and changed
solver/sample-step settings while retaining training settings. Changing the
sampler starts a new best-FID record. Earlier Euler checkpoints can continue
with `--solver euler --sample-steps 100`, or switch to Heun with the evaluation
change flag. Existing jobs continue using their own frozen source snapshots.

The trainer can also be launched directly with `torch.distributed.run`; inspect
`scripts/lsf/run_cifar_riesz_ddp.py --help` for its arguments. The launcher checks
the expected GPU model and runs a three-update GPU preflight, including endpoint
times, distributed synchronization, sampling, FID, and checkpoint round-trip.

## Files

| File | Purpose |
| --- | --- |
| `velocity_model.py`, `backbone.py` | Velocity adapter, skewed times, and the U-Net |
| `riesz_loss.py`, `cifar_riesz_velocity.py` | Pixel Riesz field, distributed loss, Euler/Heun sampling |
| `data.py` | Checksum-verified CIFAR-10 loading |
| `fid.py`, `_fid_inception.py`, `assets/` | Inception features and distributed FID statistics |
| `run_storage.py` | Project-storage routing |
| `scripts/lsf/launch_cifar_riesz_ddp.py` | Frozen-source validation and LSF submission |
| `scripts/lsf/run_cifar_riesz_ddp.py` | Training, continuation, previews, evaluation, checkpoints |
| `scripts/lsf/monitor_pixel_flow.py` | Numerical campaign status logging |
| `scripts/lsf/audit_cifar_velocity_samplers.py` | Fixed-checkpoint sampler comparisons |
| `tests/` | Field/gradient, integration, model, resume-config, and storage checks |

The monitor reads a campaign JSON containing `runs: [{"path": "/path/to/run"}]`:

```bash
.venv/bin/python scripts/lsf/monitor_pixel_flow.py \
  --campaign /path/to/campaign.json --output cifar/my_progress --interval 30
```

It records numerical checks and preview paths; images still require visual
review. The sampler audit supports both current snapshots and archived velocity
runs whose shared helpers were named `cifar_flow_matching.py`.

## Validate

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -v
```

Obsolete MMD, Sinkhorn, reconstruction, diffusion, flow-matching, DiT, and
small-dataset experiment entry points have been removed from the working code.
Historical results and frozen run snapshots remain in their recorded locations.
