# CIFAR-10 priming on LSF

## Direct-velocity mixed-kernel Riesz on four RTX 2080 Ti GPUs

`launch_cifar_riesz_ddp.py --name NAME --submit` freezes and tests a fresh
61,804,419-parameter DDPM U-Net run, then submits a four-GPU preflight and
dependent training on `longgpu01`. The network directly predicts unrestricted
velocity, without the denoiser division by `1-t`. Initialization is random;
there is no FM, diffusion, reconstruction, or encoder weight transfer.

The existing paired pixel Riesz force sums powers 0.8, 1.0, and 1.5 over
temperatures 0.2, 0.05, and 0.02, without force RMS normalization. Each rank
handles 64 images. Predictions and targets are gathered into the same global
256-particle population on every rank before calculating affinities and forces.
Only local queries retain gradients; multiplying the loss by four compensates
for DDP's gradient averaging. Independent local 64-particle losses would change
the objective and are not used. CPU tests compare partitioned gradients with
the original full-batch coupled loss, including endpoint intervals.

One time is drawn for the entire global batch from Meta's skewed distribution
`sigmoid(1.2 - 1.2 * N(0,1))`, clipped to `[0.0001, 0.999]`. Training uses the
affine reference path and interval `min(0.01, 1-t)`. The existing `(1-t)` loss
weight is retained as an explicit weighting choice, although direct velocity
no longer needs it to cancel a denoiser Jacobian. The existing Riesz Adam
schedule (500-update warmup to 2e-4, cosine floor 2e-5), clipping at 1, and
EMA schedule are retained; FP16 autocast and gradient scaling fit the GPUs.

The budget is 100,000 optimizer updates. Sampling remains 100 uniform Euler
steps, previews and full recovery checkpoints occur every 1,000 updates, and
distributed FID-10K against all 50,000 training images occurs every 5,000.
Latest/best checkpoints include optimizer, GradScaler, and all four RNG states;
resume requires matching configuration and Riesz-only provenance. The GPU
preflight tests times 0.001, 0.77, and 0.999, actual sampling/Inception, identical
parameters across ranks, memory use, and checkpoint serialization. The preflight
checkpoint is not used to initialize training. All artifacts use project storage.

## Distributed flow matching with the diffusion U-Net

`launch_cifar_fm_ddp.py --name NAME --submit` creates a frozen experiment in
project CIFAR storage, pins the Meta `facebookresearch/flow_matching` recipe
references to a Git commit, runs the mathematical tests, and submits a four-GPU
preflight followed by training on `longgpu01` / `longgpu`. The model is the exact
61,804,419-parameter time-conditioned DDPM U-Net, predicting unrestricted velocity
from random weights. All four RTX 2080 Ti GPUs train one synchronized DDP model.
The default global batch is 256: two FP16 microbatches of 32 on each GPU, with
FP32 optimizer states and loss, gradient scaling, and norm clipping at 1.

The adapted Meta CIFAR recipe uses independent Gaussian/data coupling,
`t = sigmoid(1.2 - 1.2 * Normal(0,1))`, AdamW at 1e-4 with betas (0.9, 0.95),
weight decay 0.01, and warm-started EMA capped at 0.999. The DDPM backbone retains
dropout 0.1 and maps time to `999*(1-t)`. It is not Meta's published architecture
or a claim to reproduce its 2.07 FID. The initial budget is 250,000 updates.

Sampling uses 50 second-order Heun intervals (100 actual network calls) on an
EDM-derived rho=7 time grid with exact noise/data endpoints. FID-10K against all
50,000 training images runs every 5,000 updates and previews every 1,000.
At update 5,000, a separate 100-interval audit checks integration sensitivity;
final evaluation also audits 100 and 250 intervals. FID keeps the existing local
unquantized-pixel protocol. Training and FID generation are distributed over all
four GPUs; Inception moments are combined in float64.

The preflight exercises full-sized updates, verifies parameter equality between
ranks, tests the actual sampler and Inception, records per-GPU peak memory, and
checks checkpoint serialization. Atomic `latest.pt` checkpoints include model,
EMA, optimizer, GradScaler, update count, and each rank's CPU/CUDA RNG. Resume
the frozen recorded command with `--resume TRAINING/latest.pt` and otherwise
identical settings. Checkpoints are written before evaluation and best models
are retained. The preflight checkpoint is never used to initialize training.

## Multiscale VGG DiT continuation

Add `--restart-from-priming` and provide a completed reconstruction checkpoint
to start fresh post-training at update zero. This retains reconstruction online
weights and EMA but resets Adam, random streams, and the 1,000-update warmup;
no distributional training history or optimizer state is inherited. The GPU
smoke check uses the same initialization choice.

`launch_vgg_multiscale.py --source-study STUDY --checkpoint NUMBERED_CHECKPOINT
--name NAME --submit` forks a preserved distributional checkpoint into project
storage using `run_storage.new_run_path`. It keeps online weights, EMA, Adam,
random states, and the original learning-rate schedule. The combined experiment
defaults to the original 128 generated and 256 real images per interaction batch;
`--train-particles` and `--target-batch-size` explicitly override these sizes. It uses FID-10K
every 5,000 updates, recovery checkpoints every 1,000, and numbered checkpoints
every 5,000. Prior metrics through the fork point are inherited history.

`--vgg-multiscale --perceptual-pool-size 8` computes 15 equally weighted fields:
for each of five channel-normalized VGG layers, spatial features pooled to at
most 8x8, local 2x2 mean/std maps pooled to at most 4x4, and global channel
mean/std. Every branch has independent batch-distance scaling and neighbor
weights. Temperatures, repulsion, and the absence of field RMS normalization
remain unchanged. Per-branch force RMS and effective neighbor counts are logged.

The launcher freezes source and tests, copies the initialization and VGG assets,
runs the test suite, and submits a full-size GPU smoke check followed by training.
The smoke check loads the actual source model, EMA, and Adam state, keeps the
dataset and FID network resident, and checks gradients, memory, and FID features
at the requested batch sizes. It must pass before training begins. Queue and GPU
memory request can be set explicitly; defaults target an exclusive A100 on
`longgpu02` with a 32-GB minimum-memory request.

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
reconstruction. New jobs run FID-10K every 5,000 updates and checkpoints every 1,000.

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
0 and every 5,000 steps for new jobs. Checkpoints are saved every reconstruction epoch and
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


## Two Riesz experiments per H200

`run_cifar_pair.py --manifest PAIR/manifest.json` runs two independent training
processes on the same LSF-allocated GPU, with two CPU threads per process.
Request four CPU slots and `num=1:mode=shared:j_exclusive=yes:gmem=80G`:
shared compute mode permits both CUDA contexts, while job exclusivity keeps the
GPU reserved to the pair. The existing per-run full-width smoke checks execute
sequentially before both trainers start. Model, optimizer, RNG, and output files
remain independent. Existing checkpoints automatically resume; each completed
run is archived and refreshes the notebook. A failed trainer is recorded while
its companion continues. `--dry-run` prints the two effective commands.

The deployment under `results/cifar/shared_h200_20261026/` pairs normalized and
unnormalized softmax at LR 2e-4, and softmax LR 5e-4 with standard Riesz LR 2e-4.
The original run source snapshots are retained. Only CPU threads change from
four to two. Diffusion was stopped by user request with a durable update-192,000
checkpoint preserved in its `training/checkpoints/paused_20261026T000911Z/`
directory; its dependent finalizer was cancelled. GPU sharing changes throughput,
so wall-clock comparisons should account for this scheduling change.

Standard Riesz was subsequently paused at update 9,000. Its place alongside
softmax LR 5e-4 is used by the primed Sinkhorn OT experiment in
`results/cifar/priming_sinkhorn_ot_lr2e4_seed42_20261026/`. The other pair continues.

## Direct Sinkhorn OT comparison

`run_tuned_cifar.py --objective sinkhorn --sinkhorn-mode ot --sinkhorn-iters 10
--sinkhorn-tolerance 0 --learning-rate 0.0002 --min-learning-rate 0.0001
--restart-from-priming CHECKPOINT` selects direct OT post-training from the same
reconstruction checkpoint. The existing default `--sinkhorn-mode wflow` retains
the normalized barycentric-target regression objective.

The new mode uses detached balanced transport plans: attraction to real samples
minus repulsion from an independent generated batch. It combines regularization
strengths 0.001 and 0.01 with weights 0.75 and 0.25, inspired by the Priming paper
(https://arxiv.org/abs/2610.20331). This is a pixel-space adaptation: it retains
our batch-distance scaling and multiscale pixel views, with dimension-averaged
quadratic costs, rather than the paper's calibrated frozen encoder features.
There is no field RMS normalization. The scalar is a gradient surrogate, not
the complete entropy-regularized Sinkhorn divergence. The original run switched to exactly
ten inner iterations with convergence checks disabled
(`--sinkhorn-tolerance 0`) after update 1,000. The original first 1,000 updates
used convergence checks every ten iterations, capped at 1,000. The original
checkpoint and source are preserved; the continuation retains model, EMA,
optimizer, and RNG state. A final marginal-error diagnostic is still recorded.
Metrics record attraction, repulsion, raw velocity RMS, marginal error, and solver
iteration counts.

The comparison retains seed 42, generated/real batches 128/256, view weights
2:1:1, four crops per size, Adam, clipping at norm 1, EMA 0.9999, and 1,000-step
warmup followed by cosine decay from 2e-4 to 1e-4 over 250,000 updates. FID and
held-out Riesz MMD use the existing evaluation protocol. Each run has a frozen
source snapshot and a deployment manifest containing its exact command.

## Smaller-batch GPU benchmark

`benchmark_cifar_batch.py --source SOURCE --data-root DATA --checkpoint PRIMING
--output RESULTS` tests generated batches of 64 and 96 for normalized softmax
and Sinkhorn OT, plus batch 64 for softmax without field RMS normalization.
If batch 64 runs out of memory, it tries 32 instead of 96. Each case uses a fresh
process, the reconstruction weights, 256 real images, the production multiscale
views, Adam and EMA. The full real dataset and FID network remain on the GPU.
Two warmup updates precede eight timed updates, followed by an evaluation batch
of 64 with optimizer state and gradients still resident. Results include timing,
peak allocated/reserved memory, finite-output checks, and any out-of-memory error.
These are short capacity/speed checks, not FID or training-quality comparisons.
Reducing the generated batch changes the empirical particle interaction loss;
ordinary gradient accumulation does not reproduce the original larger-batch loss.

## Conservative Sinkhorn pair

`results/cifar/sinkhorn_pair_stable_20261026/manifest.json` deploys two independent
trainers on one H200: `priming_sinkhorn_ot_stable_seed42_20261026` and
`priming_sinkhorn_wflow_stable_seed42_20261026`. Both restart from the same
80-epoch reconstruction checkpoint, with fresh Adam and random streams, peak
LR 5e-5, cosine floor 2.5e-5, 5,000-step warmup, seed 42, EMA 0.9999, and the
existing 128/256 generated/real batches and multiscale 2:1:1 view weights.
Both use exactly ten inner iterations with convergence checks disabled.

Direct OT selects `--sinkhorn-ot-epsilon 0.05`; omitting that option preserves
the historical 0.001/0.01 mixture. W-Flow uses the existing barycentric,
RMS-normalized fixed-point target loss with `--sinkhorn-epsilon 0.05`.
The no-RMS softmax Riesz run remains on its separate H200. The preceding
Sinkhorn run was stopped with a resumable update-12,000 checkpoint preserved.

Both new runs log 100-update averages of pre-clip gradient norms, clipping
frequency, relative parameter-update norms, and transport diagnostics. FID-10K
and recovery checkpoints occur every 1,000 updates; numbered checkpoints remain
every 25,000 updates. The pair worker retains the checkpoint corresponding to
the best completed FID under `checkpoints/best_fid/` and refreshes the comparison
notebook when new FID scores arrive. Separate outputs, checkpoints, and RNG
states are maintained for each trainer. The pair manifest records both exact
commands, and each study retains a frozen source snapshot with hashes.
# Small reference-process CIFAR experiments

`launch_small_flow.py --dataset cifar10 --name NAME` freezes the RGB reference-flow
implementation and submits to `longgpu` on `longgpu01`. New artifacts go to
`/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar/NAME`; the launcher
uses `run_storage.new_run_path` and never falls back to home storage. CIFAR uses
native 3×32×32 images, per-channel local statistics, and the same configurable
small U-Net as the grayscale studies. `--deep --width 32` has 1,372,035 parameters.
`--random-flip` augments both reconstruction priming and transport training.

FID defaults to 10,000 generated images against all 50,000 training images every
5,000 transport optimizer updates. `--fid-reference-samples` explicitly overrides
the reference count. Final training evaluation also reports test FID; use
`audit_small_image_flow.py` with an independent noise seed for confirmation after
checkpoint selection. Sampling solver, step count and network evaluations must
be included in comparisons. Reconstruction priming updates are additional to
the reported transport updates.

Riesz permits reconstruction-only priming and Riesz-only continuation. Parent
dataset and ancestry checks reject incompatible checkpoints or FM weight
transfer. A `path-endpoint` run combines endpoint and next-reference-marginal
Riesz losses; `reference-hybrid` additionally uses a velocity-MSE loss and must
be labeled as a hybrid. The separate FM baseline does not supply any weights.

Before a new configuration is trained, `--smoke` tests full backward passes on
the allocated GPU. The October 27 RGB preflight passed on a 2080 Ti with PyTorch
2.7.1+cu126: batch 128, 20 checkpointed rollout steps, and reconstruction/spatial
features used 1.95 GB peak allocated memory. This excludes the FID network and
optimizer states and is a capacity check, not a quality measurement.

`review_cifar_perceptual.py --output cifar/REVIEW_NAME` snapshots the recorded
October 26 VGG and LPIPS experiments without modifying them. It verifies the
actual reconstruction checkpoint hashes, records scheduler status, and writes
JSON/CSV evidence plus FID and sample comparison figures. LPIPS reconstruction
priming, unpaired VGG distribution matching, and the additive VGG-field pilot
are reported separately.

The small RGB trainer accepts `--vgg-endpoint-weight 1 --vgg-pool-size 8
--vgg-cache CACHE` to average its existing endpoint objective with 15 independent
VGG feature Riesz fields. The next-reference-marginal loss remains in pixel
space. VGG inputs use the unclipped affine conversion from centered RGB states;
the encoder is frozen, targets are detached, and LPIPS pairwise heads are unused.
The launcher copies the cached VGG16 weights into each new study and records their
hash. A completed `reconstruction.pt` can be supplied through `--fork`; the
generator and frozen reconstruction encoder are both restored and verified
against that reconstruction-only checkpoint. This supports feature comparisons
without repeating priming or inheriting FM weights. A fork starts new optimizer
and random streams; comparisons with uninterrupted training are recipe comparisons.

`launch_small_flow.py --dataset cifar10 --gradient-diagnostic --name NAME
--checkpoint CHECKPOINT --recipe CONFIG --device cuda --batch 128 --repeats 16`
freezes a checkpoint and loss recipe, then measures endpoint and local-reference
parameter gradients without optimizer updates. It reports their norms and
alignment across time, including the candidate `(1-t)` reference weighting.
Use an explicit `--vgg-cache` for a VGG recipe. These diagnostics guide ablations;
they do not establish an improvement in FID. The launcher's default storage and
2080 Ti placement also apply to this mode.

For a small sampling check while GPUs train, `audit_small_image_flow.py
--preview-only --device cpu --preview-count 16` supports CPU previews. It records
the device, count, solver and noise seed and rejects nonfinite samples. CPU and
CUDA random streams differ even for the same seed. FID audits still require CUDA;
small previews cannot establish image quality or diversity.

`audit_cifar_perceptual.py --output cifar/AUDIT_NAME` copies the two saved
multiscale-VGG update-25,000 checkpoints and one atomic snapshot of the running
FM checkpoint, freezes their compatible archived implementations, and submits
an independent FID-10K evaluation against all 10,000 CIFAR test images. It records
the exact FM update, checkpoint hashes, common fresh noise seed, sampling cost,
and GPU memory. This evaluation never updates the models or their original runs.

The reference trainer also supports `--architecture dit --dit-width 384
--dit-depth 12 --dit-heads 6 --dit-patch-size 2` (32,253,324 parameters).
For reconstruction initialization, use `--parameterization denoiser
--dit-prime-checkpoint CHECKPOINT --solver euler`. The checkpoint must be a
from-scratch reconstruction stage; its SHA256 and exact EMA tensor transfer are
verified. New time modulation output gates start at zero, preserving the primed
clean prediction at initialization. No FM weights or optimizer state transfer.
The DiT currently uses pixel and optional VGG endpoint features. Fixed-corruption
priming can produce saturated multi-step samples before transport training;
`--preview-every 1000` records 32-image EMA diagnostics from update zero without
changing the FID schedule. These previews are not quality estimates.

For the H200 host, use launcher options `--queue dbeigpu --host dbei-ai1
--gpu-mig 7/7 --gpu-memory 128 --cpus 4 --memory-gb 16`. Explicit GPU and compute
instance sizes matter: the October 27 request for 32 GB without a MIG size was
assigned a `2g.35gb` partition. The full-size request was assigned `7/7` by LSF.
Use `--smoke` before training to verify the visible device, finite gradients,
memory and runtime. `--exclusive` is for non-MIG compute mode and cannot be
combined with `--gpu-mig`. Default launcher placement remains the 2080 Ti host.

For the CIFAR Riesz comparison against the diffusion backbone, use
`--architecture ddpm-unet --parameterization denoiser`. This selects the exact
61,804,419-parameter diffusion U-Net (base channels 128, four residual blocks
per level), with the same layers, attention, dropout 0.1, time embedding, and
unrestricted output head. The adapter only flattens its output and maps flow
time `t` to embedding coordinate `999*(1-t)`; the reference path remains linear.
The head predicts centered clean pixels instead of epsilon, and no diffusion
weights are loaded. EMA evaluation disables dropout. The small `unet` option
remains a different architecture and must not be described as the diffusion
backbone. Run a full-batch GPU preflight before this larger training job.

For a cheaper reference-process candidate, `--smoke --local-only
--local-diagnostics 4 --affinity row-column --feature-space pixels` compares
single-transition Riesz objectives with independent, paired, and paired
leave-one-out reference populations. It performs no endpoint rollout backward
pass and no optimizer updates. The diagnostic reports gradient alignment with
the analytic affine-path velocity-MSE gradient, without a teacher model or FM
weight transfer. This alignment alone cannot establish image quality. The
training objective `--objective coupled --reference-coupling paired` reuses the
same sampled image/noise particles across adjacent reference marginals, while
retaining all-target Riesz attraction and generated-particle repulsion. It must
be distinguished from independent-population distribution matching.

`--objective local-hybrid --parameterization denoiser --reference-weight 1
--riesz-weight 0.1` jointly trains paired clean-image reference MSE and the local
Riesz loss from one network call. Clean-image MSE is the affine conditional
velocity-MSE objective weighted by `(1-t)^2`, which avoids the denoiser Jacobian
singularity. `--reference-coupling paired-loo --reference-time-power 1` selects
matched-target exclusion and remaining-time weighting for the Riesz term.
This is explicitly a hybrid supervised/distributional objective, not pure Riesz
and not a separately trained FM initialization. Both loss weights must be
positive. It uses no rollout loss, trained FM teacher, or VGG post-training
encoder. Reconstruction-only LPIPS priming remains supported. The local smoke
test accepts `--local-hybrid` to verify finite backward gradients and measure its
incremental cost against plain local Riesz on the same GPU allocation.


### CIFAR combined-power Riesz follow-up (2026-10-28)

The unchanged 61,804,419-parameter `ddpm-unet` can use
`--distance-powers 0.8 1.0 1.5 --normalization none` with the paired
`--objective coupled --affinity row-column` recipe. These options sum all
three power-kernel fields in **one loss**, across every configured temperature;
they do not launch separate runs and do not average or RMS-normalize the fields.

For distance `r` and the existing detached mean-distance scale `s`, power `p`
uses affinity logits `-(r/s)^p / tau` and radial force
`p * (r/s)^(p-1) * (target-query)/r / tau`, with the existing row/column
cross-mass coefficients. The three powers have equal coefficient 1. This is
a sum of separately constructed kernel forces, not a single softmax of the
sum of powers. Softmax affinity normalization and coordinate scaling remain;
force RMS normalization is disabled. Gradient clipping at norm 1 remains.
The existing `1e-6` force-distance floor regularizes the fractional-power
singularity, and exact coincident points contribute zero displacement.
No additional distance smoothing is enabled in this trial.

`--distance-powers 1` preserves the original loss, including existing
normalization choices. Multiple/nonunit powers currently require the coupled
row-column image objective. The GPU preflight accepts the same powers with
`--local-only`; its output records the actual configuration. The launcher
freezes the power-kernel tests alongside the source.

The combined-power trial starts from random weights, seed 0, using batch 256,
100,000 updates, peak LR 2e-4, reference interval 0.01, 100 Euler sampling
steps, previews every 1,000 updates, and FID-10K every 5,000 updates against
the full 50,000-image training split. No diffusion/FM weights or learned
feature encoder are used. Architecture and optimizer schedule match the
original 100-step Riesz U-Net run.
