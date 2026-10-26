# CIFAR-10 priming on LSF

## DDPM architecture sanity check

`ddpm.py` trains standard discrete epsilon-prediction DDPM on the existing
CIFAR DDPM++ spatial backbone. It adds sinusoidal timestep conditioning to
every residual block and replaces the sigmoid image head with unrestricted
noise prediction. Channel widths, residual block counts, attention, resampling,
and skip topology match the one-pass model. Existing one-pass behavior and
checkpoint keys are preserved. This tests the shared backbone under diffusion;
it does not isolate the one-pass sigmoid head or MMD objective.

The extended run uses 250,000 updates, batch 128, seed 42, Adam at 2e-4 with 5,000-step
warmup, gradient clipping at 1, EMA 0.9999, and random horizontal flips.
It follows the [original DDPM CIFAR recipe](https://github.com/hojonathanho/diffusion/blob/master/scripts/run_cifar.py)
for 1,000 linear-beta steps (1e-4 to 0.02), epsilon MSE, clipped reconstructed
images, and fixed-large ancestral sampling variance. This is an adaptation to
our backbone and an architecture sanity check, not a reproduction of a published FID.

```bash
.venv/bin/python scripts/lsf/submit_cifar_ddpm.py \
  --study results/cifar/ddpm_sanity_seed42_YYYYMMDD --submit
```

Without `--submit`, only the source snapshot and deployment record are created.
Each run requests one 80-GB-or-larger GPU, four CPUs, 32 GB host memory, and
120 hours. Results are under `STUDY/training/`: training/held-out noise MSE,
checkpoints every 1,000 updates, 64-image previews every 5,000 updates, and final
FID-10K against all 50,000 CIFAR training images using the same Inception as MMD.
FID uses all 1,000 reverse steps per image; it is much slower than one-pass FID.
The worker automatically resumes an existing checkpoint with matching arguments.
The total update budget may increase on resume; shorter budgets are rejected.
Final FID is evaluated again at the extended endpoint, preserving earlier scores.
It never initializes from the priming run. Do not submit a duplicate live run.

`smoke_ddpm.py DATA_ROOT RESULT_JSON` checks full-width updates and a short
reverse chain on an allocated GPU; its samples are not a quality measurement.
`tests/test_ddpm.py` covers diffusion math, conditioning, sampling RNG isolation,
fixed-batch learning, resume equivalence, and original-backbone compatibility.

`evaluate_ddpm_fid.py --checkpoint PRESERVED_CHECKPOINT --data-root DATA_ROOT
--output EVALUATION_DIR --notebook NOTEBOOK` evaluates a fixed DDPM EMA while
training continues. It uses 10,000 generated images, 1,000 ancestral reverse
steps, seed 12345, batches of 64, and all 50,000 training reference images.
Preserve the checkpoint before launching; training atomically replaces its
recovery checkpoint. The evaluator writes progress to `job_status.json`, a
complete score with checkpoint provenance to `fid_metrics.json`, and refreshes
the notebook on completion. The comparison notebook merges completed results
under `training/fid_evaluations/*/` and displays evaluation-job progress.

For comparable Riesz diagnostics, `evaluate_ddpm_riesz.py` reads a saved DDPM
EMA checkpoint and the MMD run's `data_indices.json`. It evaluates 64 generated
images against the same 512-image train/test subsets using unquantized pixels
and the existing biased empirical Riesz MMD². This diagnostic is distinct from
the multiscale softmax-Riesz training surrogate. It writes `riesz_metrics.json`
without changing the checkpoint or training metrics. The notebook
`notebooks/cifar_priming_ddpm_progress.ipynb` reads these diagnostics alongside
FID and the two MMD runs' matching diagnostics.

## Reconstruction priming

For a fresh unprimed MMD-flow baseline with the same 250,000 distributional
updates, seed 42, multiscale objective, optimizer, EMA, and evaluation settings:

```bash
.venv/bin/python scripts/lsf/submit_cifar_priming.py \
  --study results/cifar/mmd_baseline_seed42_YYYYMMDD --baseline-only --submit
```

This submits only the baseline stage, from random initialization, in a new
output directory. It neither resumes the cancelled baseline nor launches
reconstruction. FID-10K runs every 2,500 updates and checkpoints every 1,000.

This checkout is on the Penn LSF cluster. Its `dbeigpu` queue provides H200s;
the older `scripts/cs_cluster/` launchers target a separate SGE cluster.

Prepare the local environment and assets before submitting:

```bash
python -m venv --system-site-packages .venv
# The installed system torch is 2.7.1+cu126.
.venv/bin/python -m pip install --no-deps torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python main.py --dataset cifar10 --data-root data --download-only
```

FID also requires `assets/pt_inception-2015-12-05-6726825d.pth`, verified against
`assets/inception.sha256`. Its public download URL is in `_fid_inception.py`.

```bash
.venv/bin/python scripts/lsf/submit_cifar_priming.py \
  --study results/cifar/priming_seed42_YYYYMMDD --submit
```

Without `--submit`, this creates a source snapshot and deployment record only.
Use a fresh study path for each invocation. Each job requests one GPU with
80 GB GPU memory, four CPUs, 32 GB host memory, and a 48-hour limit.

| Stage | Budget | Batch | EMA | Objective |
| --- | --- | --- | --- | --- |
| Reconstruction | 80 full epochs (62,560 updates) | 64 paired images | 0.9999 | MSE at corruption 0.8 |
| Post-training | 250,000 updates | 64 generated / 256 real | 0.999 | Existing multiscale softmax-Riesz |

All stages use the 56.35M-parameter DDPM++ CIFAR U-Net, seed 42, Adam
`lr=2e-4`, betas `(0.9,0.999)`, no weight decay, and no warmup.
Post-training waits for successful reconstruction via LSF `done(job_id)`.
Only these two stages are submitted. This is an initial single-training-seed
experiment.

FID-10K uses all 50,000 training images as the reference. Reconstruction saves
FID and image grids at epochs 0/20/40/60/80; distributional stages save at step
0 and every 2,500 steps. Checkpoints are saved every reconstruction epoch and
every 1,000 post-training steps, plus final checkpoints. The next stage loads
online reconstruction weights and inherits its EMA; it resets the optimizer.

`deployment.json` records job IDs and exact submission commands. Each stage
has `scheduler.log`, live `run.log`, `job_status.json`, `config.json`, metrics,
images, and `checkpoint_latest.pt`. Source files are snapshotted with SHA-256
hashes; dataset and FID assets are reused and checksum-verified by the trainer.

```bash
bjobs JOB_ID
bpeek JOB_ID
tail -f results/cifar/STUDY/reconstruction/run.log
```

The current comparison extends DDPM, primed post-training, and unprimed
training from 25,000 to 250,000 updates. Reconstruction remains at 80 epochs.
Continuation records and preserved pilot checkpoints are under each study’s
`extensions/updates_250000/`; original source snapshots remain unchanged.

To resume, resubmit the recorded stage command (with the same source snapshot
and output). The worker automatically adds `--resume` when a checkpoint exists.
If reconstruction failed, its held dependent post-training job will not run:
resubmit reconstruction, then update the held job's dependency to the new ID
using `bmod -w 'done(NEW_JOB_ID)' POST_JOB_ID` before allowing post-training.
Do not submit a duplicate stage while the original job is still running.

Checks:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 DRIFT_COMPILE=0 \
  .venv/bin/python -m unittest discover -s tests -v
bash -n scripts/lsf/run_cifar_priming.sh
```

`smoke_priming.py DATA_ROOT RESULT_JSON` exercises full-width reconstruction,
distributional updates, EMA, and the FID encoder on an allocated GPU. It writes
timings and peak GPU memory; it does not produce a model-quality result.

`finalize_cifar_comparison.py --run RUN --expected-update 250000 --notebook NOTEBOOK`
archives the completed checkpoint and metrics under `RUN/checkpoints/step_250000/`
and executes the comparison notebook. Archives retain model, EMA, optimizer, and
random state. Omitting `--run` and `--expected-update` only refreshes the notebook.
CPU finalizers wait for the pilot DDPM FID and each extended training job; their
commands and IDs are in `results/cifar/comparison_finalization_20261025/deployment.json`.
The notebook displays diffusion FID sampling progress until a complete FID-10K
measurement is available, then includes that measurement in its tables and plot.

The tuned MMD continuations in `results/cifar/tuned_mmd_seed42_20261025/`
start from preserved 36,000-update primed/unprimed checkpoints. Both use 128
generated training images, 256 real images, horizontal flips on training images,
gradient clipping at norm 1, EMA 0.9999, four crops per patch size, and global /
coarse / patch weights 2:1:1. Learning rate warms up for 1,000 additional updates
to 1e-4 and follows cosine decay to 5e-5 at total update 250,000. Reconstruction
remains at 80 epochs. Diagnostics retain 64 fixed generated images and FID-10K.
This is a combined tuning experiment, not an individual-setting ablation.

`run_tuned_cifar.py` reconstructs the original checkpoint arguments and records
the explicit overrides. `--fork-checkpoint` retains model, EMA, Adam moments,
random states, and prior metrics in a new output directory; ordinary `--resume`
still rejects unrecorded recipe changes. Each job performs a full-width GPU smoke
check, retains the initial checkpoint and subsequent numbered checkpoints every
25,000 updates, updates recovery checkpoints every 1,000 updates, and archives
the final checkpoint and refreshes the notebook inline. The original GPU jobs
were stopped after checkpoint preservation, and their obsolete CPU finalizers
were cancelled. Diffusion jobs are unchanged.

The primed run was subsequently restarted at post-training step zero in
`results/cifar/priming_lr1e3_seed42_20261025/`. It loads the completed 80-epoch
reconstruction weights and EMA with fresh Adam state and random streams. The
tuned settings apply from the first update, with 1,000-step warmup to `1e-3`
and cosine decay to `5e-4` over 250,000 updates. Its deployment record identifies
the replacement job and preserved prior run. The baseline was subsequently
stopped by user request with its checkpoint preserved at update 77,000; the
comparison differs in learning rate and training history.
`run_tuned_cifar.py --restart-from-priming CHECKPOINT --learning-rate 0.001
--min-learning-rate 0.0005` selects this behavior; the study's `paused_runs.json`
supplies the original argument recipe only, not model or optimizer state.

After the `1e-3` run deteriorated, it was stopped with its update-8,000 checkpoint
preserved. The active primed restart is now
`results/cifar/priming_lr2e4_seed42_20261025/`: fresh post-training from the same
completed reconstruction, 1,000-step warmup to `2e-4`, cosine decay to `1e-4`,
and unchanged EMA decay `0.9999`. The comparison notebook selects this run.
