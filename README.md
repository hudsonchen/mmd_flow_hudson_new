# Neural Riesz image generation

One shared trainer generates MNIST or CIFAR-10 images from fresh Gaussian
noise using ordinary Riesz MMD, softmax Riesz, Drifting, or Sinkhorn. Training updates only
network parameters. Fixed, independent evaluation inputs track the same sample
identities across epochs.

## Layout

```text
main.py          CLI, training loop, optimizer, experiment configuration
fid.py           CIFAR-10 FID evaluation of generators or saved NumPy samples
assets/          pretrained FID Inception weights, checksum, and license
backbone.py      MLP, compact MNIST U-Net, larger CIFAR RGB attention U-Net
loss.py          Riesz fields/MMD, softmax Riesz, Drifting, and Sinkhorn losses
sinkhorn.py      compatibility imports from loss.py
multiscale.py    coarse-image and aligned-patch training objectives
data.py         verified dataset downloads and both dataset loaders
visualization.py shared image grids, trajectories, and MMD plots
run.sh           shared GPU launcher
scripts/cs_cluster/ CIFAR-10 H100 launcher and result transfer watcher
tests/          mathematical, gradient, backbone, and data/rendering checks
notebooks/       reproducible saved-result comparisons
results/mnist/   MNIST experiments
results/cifar/   CIFAR-10 experiments
figures/         comparison figures and summaries
```

Historical run configurations retain their original launch paths.
`results/reorganization.json` records the old/new paths and migration-time
file checksums. The already-running CIFAR job uses a temporary link at its old
path; a cleanup watcher removes that link once the process exits. Its physical
outputs are stored under `results/cifar/`.

## Run experiments

The 2026-10-27 symmetric paired-reference follow-up tests a finite-population
property of the local row/column Riesz field. With included matched positives
but masked self negatives, identical predicted and target populations can
still have a nonzero field. `--include-self-negative` includes zero-displacement
self negatives in affinity normalization, so equal empirical populations
cancel exactly at unit repulsion. This changes the finite-batch estimator; it
does not establish improved population transport or FID. Direct paired reference
information remains present and must be reported separately from paired-loo.

`--distance-smoothing 0.05` uses the smooth norm
`sqrt(distance**2 + epsilon**2) - epsilon`, with epsilon equal to 0.05 times
the mean batch distance, and the corresponding softened unit directions.
The smooth/no-RMS trial uses `--normalization none` so small forces can shrink
near a match. Both options are opt-in for coupled pixel-only training; existing
recipes and frozen runs retain their behavior. Trials start from random weights,
use no encoder or MSE term, save previews every 1,000 updates, and evaluate
FID every 5,000 updates. Campaign artifacts are under
`/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/mnist/pixel_symmetric_campaign_20261027/`.

New runs save under `/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/`.
The trainer, shared launcher, and LSF submission scripts route new relative
destinations such as `results/cifar/my_run` to that directory, including logs,
checkpoints, metrics, and source snapshots. New absolute paths under the old home
`results/` directory are routed there too. Existing runs retain their recorded
paths; the already-running primed DiT study remains in home storage.
`MMD_RESULTS_ROOT` can override the root; an explicit absolute destination outside
the old home results directory is also honored. Missing LongGroup access fails
normally rather than falling back to home storage.

### Reconstruction priming (two-stage CIFAR-10)

The optional reconstruction stage adapts Eq. (1) of
[Are We Overthinking One-Step Generation?](https://arxiv.org/abs/2610.20331).
It trains the existing generator on paired noisy/clean images at fixed `s=0.8`.
For stored pixels `x` in `[0,1]`, inputs are `(1-s)*(2*x-1) + s*noise`,
with standard Gaussian noise; the sigmoid output predicts `x` using mean MSE.
There is no time input. The centered input convention matches the paper's
noise scale; the output and MSE retain this repo's `[0,1]` convention.

```bash
python main.py --dataset cifar10 --data-root data --target-size 0 \
  --training-stage reconstruction --corruption-level 0.8 \
  --epochs 80 --target-batch-size 64 --particles 64 --seed 42 \
  --ema-decay 0.9999 --save-every 20 --checkpoint-every 1 \
  --device cuda --output results/cifar/priming/reconstruction

python main.py --dataset cifar10 --data-root data --target-size 0 \
  --init-checkpoint results/cifar/priming/reconstruction/checkpoint_latest.pt \
  --objective softmax-riesz --multiscale-loss --steps 250000 \
  --target-batch-size 256 --particles 64 --seed 42 --ema-decay 0.999 \
  --save-every 5000 --checkpoint-every 1000 \
  --device cuda --output results/cifar/priming/posttraining
```

Initialization requires a completed, compatible reconstruction checkpoint.
It transfers **online weights and the pretrained EMA shadow**, starts a fresh
optimizer and post-training random streams, and records the source checkpoint
SHA-256 in `config.json`. `--resume` instead restores the entire interrupted run,
including optimizer, random streams, metrics, and EMA. For a resumed post-training
run, keep its original `--init-checkpoint` argument alongside `--resume`.

The reconstruction CSV adds training MSE and fixed-noise held-out MSE at saved
snapshots. FID always measures generation from **pure Gaussian noise**, including
during reconstruction; it does not evaluate reconstructions. Checkpoints are
written atomically when `--checkpoint-every` is enabled.

This first experiment deliberately retains the existing unconditional DDPM++
network, Adam optimizer, and multiscale pixel-space softmax-Riesz loss. It uses
plain reconstruction MSE, without perceptual losses. It tests the two-stage idea;
it does not reproduce the paper's conditional ImageNet feature-space OT recipe.
See [the LSF launcher](scripts/lsf/README.md) for the current cluster's two-stage
submission workflow.

```bash
bash run.sh                                  # CIFAR-10, RGB U-Net, softmax Riesz
DATASET=mnist bash run.sh                     # MNIST, compact U-Net, softmax Riesz
DATASET=mnist OBJECTIVE=riesz bash run.sh      # standard Riesz control
DATASET=mnist ARCHITECTURE=mlp bash run.sh     # MLP control
```

The launcher uses the `mmd_flow_hudson` environment and requires CUDA.
`DATASET`, `ARCHITECTURE`, `OBJECTIVE`, `PARTICLES`, `EPOCHS`, `BATCH_SIZE`,
`SAVE_EVERY`, `BASE_CHANNELS`, `NUM_RES_BLOCKS`, `LATENT_DIM`, `LEARNING_RATE`, `SEED`,
`DATA_ROOT`, `PYTHON`, and `OUTPUT` are configurable shell variables.
The local data root defaults to `/home/zongchen/data`. CIFAR-10 is loaded from
`/home/zongchen/data/CIFAR10/raw/cifar-10-python.tar.gz`; the loader also accepts
the binary archive in the same directory. Set `DATA_ROOT` to use another root.
`EMA_DECAY` defaults to 0.999 for CIFAR-10 and 0 for MNIST; 0 disables EMA.
After each optimizer update, EMA weights are updated from the live training
weights. FID, MMD diagnostics, sample grids, and trajectories use EMA weights
when enabled. The training objective always uses the live model. The EMA copy
starts from the initial model, without a decay ramp; `config.json` records the
decay, update unit, and evaluation weights. CLI runs accept `--ema-decay`.
The default is 2,000 epochs for CIFAR-10 and 100 for MNIST, with image output
every 20 epochs, 64 generated samples per update, and target batches
of 256. Each epoch visits all eligible training images exactly once, including
its final partial batch. This gives 235 updates per MNIST epoch (60,000 images;
last batch 96) or 196 per CIFAR epoch (50,000 images; last batch 80).
Fresh training noise uses its own seeded RNG, independent of target shuffling.
Existing output configurations are protected against overwriting.

For full CLI control, use the main entry point:

```bash
/home/zongchen/miniconda3/envs/mmd_flow_hudson/bin/python main.py \
  --dataset cifar10 --architecture unet --objective softmax-riesz \
  --target-size 0 --target-batch-size 256 --particles 64 --epochs 2000 \
  --device cuda --save-every 20 --output results/cifar/my_run
```

`--steps` selects individual updates instead of `--epochs`. `--target-size 0`
uses all eligible training images; positive sizes select a fixed subset.
`--class-id` restricts either dataset to one class. MNIST also accepts `--digit`.
Without either duration option, the CLI defaults to 2,000 updates.

## Backbones

The compact CIFAR transformer is selected with `--architecture dit`. Its defaults
are 2x2 patches (256 tokens), width 384, 12 blocks, six attention heads, and a
4x GELU MLP. `--dit-width`, `--dit-depth`, `--dit-heads`, and `--dit-patch-size`
configure it. This unconditional DiT-style generator uses fixed 2D sine/cosine
positions, learned constant adaLN-Zero modulation, no dropout, and a sigmoid
pixel head. It has no time/class input and uses the same distributional objectives
as the U-Net. The small nonzero output head lets residual gates learn immediately.

The primed DiT comparison uses its own 80-epoch reconstruction stage, followed by
250,000 updates of corrected softmax Riesz without field RMS normalization.
Post-training transfers both DiT online and EMA weights with fresh Adam state;
U-Net checkpoints are incompatible. Seed 42, generated/real batches 128/256,
multiscale weights 2:1:1, four crops per size, EMA 0.9999, and the LR schedule
2e-4 to 1e-4 match the corrected U-Net reference. New jobs run FID-10K every 5,000
post-training updates and checkpoints every 20,000. The architecture is smaller
than the 56M U-Net, so this comparison does not match parameter count.

```bash
# Without --submit, only create the frozen source and exact command manifest.
.venv/bin/python scripts/lsf/submit_compact_dit.py \
  --study results/cifar/priming_compact_dit_seed42_20261026 --submit
```

One GPU job runs the full-batch smoke check, reconstruction, and post-training
sequentially. It records stage status and supports restart from each stage's latest
checkpoint via `scripts/lsf/run_compact_dit.py --manifest STUDY/manifest.json`.
Use `--dry-run` on that worker to print the recorded commands.

For perceptual priming, install `lpips==0.1.4` and pass `--lpips-weight 0.1`
to `scripts/lsf/submit_compact_dit.py`, using a fresh study name. Reconstruction
then minimizes `[0,1]` pixel MSE plus `0.1 * LPIPS` with frozen VGG16 and calibrated
LPIPS v0.1 weights. This weight is an initial experimental choice, not a tuned
CIFAR optimum. LPIPS receives native 32x32 RGB images, converted internally to
`[-1,1]`, with no resizing. The 80-epoch priming budget and pixel-only Riesz
post-training recipe are retained. The encoder is used only during priming;
its weights are frozen but gradients pass through it to the generator.
The launcher stores its pretrained-weight cache inside the project study,
records weight hashes and package versions, and tests the perceptual objective
in the GPU smoke check. Reconstruction metrics report MSE, LPIPS, and their
weighted total separately. Direct `main.py` use requires both `--lpips-weight`
and an explicit `--lpips-cache`; LPIPS is rejected outside CIFAR reconstruction.

For a perceptual distance inside the existing MMD flow, use
`--softmax-feature-space vgg` with `--lpips-cache` and distributional softmax
Riesz. The generator's outputs are mapped through frozen VGG16 before the
existing attraction/repulsion rule is evaluated. The five VGG feature slices
use LPIPS channel normalization and adaptive average pooling to at most 4x4
per layer (`--perceptual-pool-size`). Each flattened layer is divided by the
square root of its spatial area and the number of layers, then concatenated
into one embedding. Thus the squared distance is the mean across layers of
spatially averaged squared feature differences. LPIPS's learned pairwise heads
are not used.

The softmax logits and unit directions both use this single feature-space
Euclidean distance, with the existing batch-distance scaling, temperatures,
repulsion coefficient, detached field target, and gradient clipping. Gradients
pass through the frozen VGG encoder into the generator; real features are
fixed. There is no separate perceptual supervision term. Disable
`--multiscale-loss`: VGG's five layers define the representation in this mode.
FID and pixel MMD diagnostics retain their original definitions; velocity
and force diagnostics use the selected feature space. Configurations record
the encoder hash and feature geometry, and resume checks preserve them.

To launch from a completed LPIPS priming study, pass `--source-study STUDY`,
`--name NEW_NAME`, `--repulsion-weight 1.0`, `--softmax-feature-space vgg`, and
`--submit` to `scripts/lsf/launch_dit_repulsion.py`. It starts fresh posttraining
from the final reconstruction checkpoint, validates the frozen source, and runs
a full-size GPU smoke check before training.

For CIFAR-10 H100 submission with `-P aihub_ucl` and result copying back from
the cluster, see [scripts/cs_cluster/README.md](scripts/cs_cluster/README.md).

The ten-step prototype in `train_time_flow.py` uses a time-conditioned DiT and
the affine path `x(t)=(1-t)*z+t*y`, with Gaussian `z` and clean images `y` in
`[-1,1]`. This is a prescribed [flow-matching probability path](https://facebookresearch.github.io/flow_matching/generated/flow_matching.path.AffineProbPath.html),
not a claim that the interpolation itself is an exact Riesz gradient-flow trajectory.
One shared transition index `k=0,...,9` is sampled per minibatch. Source states
come from the path at `k/10`; independent real-image/noise pairs supply targets
at `(k+1)/10`. The prediction is `x_next=x+0.1*v_theta(x,k/10)`.
Riesz-softmax attraction and repulsion match its distribution to that next
marginal, using the same frozen VGG feature embedding as the one-pass VGG flow.
No paired velocity MSE or auxiliary perceptual supervision is used.

The time-conditioned network starts from fresh parameters, has sinusoidal time
embeddings and per-block adaLN modulation, and predicts unrestricted velocity.
Intermediate states and the affine VGG input conversion `(state+1)/2` are not
clipped. Such early noisy states are outside VGG's clean-image training domain;
this geometry is an experimental choice. Sampling starts from Gaussian noise
and executes all ten updates without access to real images. Only final images
are mapped to `[0,1]` and clipped for FID. FID therefore tests the complete
rollout, including any accumulated error from training on prescribed path states.

`scripts/lsf/launch_time_flow.py --name NAME --source-cache EXISTING_VGG_CACHE --submit`
freezes source and cache into project storage, runs recovery and transport tests,
checks every time bin at full batch size on a GPU, then launches training.
Defaults are 250,000 optimizer updates, 128 source and 256 target particles,
repulsion 1.0, FID every 5,000 updates, and recovery checkpoints every 1,000.
`transition_metrics.csv` records coverage and diagnostics for each time bin;
`trajectory.png` shows the ten sampling updates.

For pixel-only continuation with exposure to model-generated states:

```bash
python scripts/lsf/launch_time_flow.py --name NAME \
  --softmax-feature-space pixels --ema-decay 0.999 \
  --rollout-fraction 0.5 --rollout-warmup-steps 5000 \
  --fork-checkpoint EXISTING_CHECKPOINT --queue longgpu02 --gpu-memory 32 --submit
```

This creates a separate stage with an immutable copy of the parent checkpoint,
loads its live weights, and resets Adam, EMA, RNG, and the stage update count.
No VGG model or cache is loaded in pixel mode. Over 5,000 stage updates, the
fraction of source particles obtained from current-model rollouts rises linearly
to 50%; the rest use the prescribed affine path. Rollout prefixes are detached,
and only the selected transition receives gradients. Targets remain independent
samples from the next affine marginal. At time zero both source choices are
Gaussian noise. Training and per-transition CSVs record actual rollout exposure.
The full ten-step EMA sampler is evaluated for FID every 5,000 stage updates.

To supervise the complete sampler directly, add `--endpoint-weight 1.0` to
the pixel Riesz launcher above. Each optimizer update then adds a separate
Riesz field between 128 full noise-to-image rollouts and 256 clean targets.
The endpoint population contains only model-generated samples. Gradients pass
through all ten transitions, with non-reentrant activation checkpointing to
bound memory; no intermediate state is detached or clipped. Local and endpoint
gradients accumulate before one clipping/Adam/EMA update. The weight is an
initial experimental choice. This remains unpaired Riesz training.

The endpoint term is disabled by default for existing recipes, rejected for
VGG/flow-matching configurations, and fixed across checkpoint resumes. Use a
fresh fork to enable it on an existing checkpoint. Logs record local and
endpoint surrogates separately; neither is a calibrated image-quality metric.
FID evaluations also record 2x2 patch-phase RMS against a real-image reference,
to track checkerboard artifacts. Checkpointing adds recomputation time; the
production GPU smoke check exercises the complete combined update at full batch.

For a separate standard conditional flow-matching baseline from fresh weights:

```bash
python scripts/lsf/launch_time_flow.py --name NAME \
  --objective flow-matching --softmax-feature-space pixels --ema-decay 0.999 \
  --queue longgpu02 --gpu-memory 32 --submit
```

This baseline samples an independent continuous `t ~ Uniform[0,1)` for each
image and minimizes `mean((v_theta((1-t)*z+t*y,t) - (y-z))**2)`. Sampling still
uses exactly ten Euler steps and the same FID protocol. It uses no Riesz field,
VGG encoder, or rollout mixing. The 128-example batch, DiT, learning-rate
schedule, 250,000-update budget, and checkpoint handling use the shared trainer.
`velocity_mse` is logged explicitly; per-time-bin counts are numbers of images.
Fresh initialization differs from the pixel-MMD continuation's inherited
weights, so that comparison is between training recipes, not only objectives.

For Riesz training with continuous time and no training grid:

```bash
python scripts/lsf/launch_time_flow.py --name NAME \
  --objective riesz --riesz-time-mode continuous --transport-horizon 0.1 \
  --softmax-feature-space pixels --ema-decay 0.999 --flow-steps 10 \
  --queue longgpu02 --gpu-memory 32 --submit
```

Each update samples one shared `t ~ Uniform[0,1)` for the entire particle
population and directly constructs `x_t=(1-t)*z+t*y`. With
`h=min(transport_horizon,1-t)`, one network call predicts
`x_next=x_t+h*v_theta(x_t,t)`. Its Riesz field matches independent samples of
the affine marginal at `t+h`. Sharing a time keeps different marginals out of
the same particle interaction; it does not round times to bins. This remains
finite-interval distribution matching, not conditional velocity MSE or an
exact infinitesimal flow-matching objective. The interval shortens near the
endpoint; no inverse-interval loss reweighting is applied.

`--flow-steps` controls only numerical sampling in this mode. Changing it
does not change training inputs, targets, or gradients. Coverage histograms
use ten reporting bins only; logs retain the sampled source/target times and
actual interval. Continuous Riesz rejects rollout mixing and endpoint loss,
uses fresh weights unless explicitly forked, and retains FID every 5,000
updates. Its default maximum horizon 0.1 retains the old interval scale while
removing the fixed observation times. Existing jobs keep their frozen source
and recorded discrete-time behavior.

The separate Gaussian-mixture control is `scripts/test_gaussian_cascade.py`.
It trains ten independent residual MLP maps sequentially toward the marginals
`(1-t)*noise + t*data`, freezing each map before training the next. The default
runs staged energy-distance Riesz MMD on actual-prefix inputs across three
seeds. Optional recipes include ideal-marginal inputs, a direct generator,
the existing softmax-Riesz surrogate, and Gaussian-kernel MMD. Held-out metrics
and sample plots assess both mode coverage and concentration around the modes.

```bash
# Default: RTX 2080 Ti GPUs, longgpu queue on longgpu01.
python scripts/test_gaussian_cascade.py --name NAME --submit
# Explicit CPU alternative:
python scripts/test_gaussian_cascade.py --name NAME --device cpu --submit
```

`--updates-per-stage` defaults to 1,000. For the requested longer staged run,
use `--updates-per-stage 50000 --walltime 04:00`. `--final-stage-updates` can
override only the last map's budget; `total_updates_by_recipe` records the
actual total. The optional direct generator receives ten times the base count.
Outputs,
stage checkpoints, frozen source, tests, and manifests go to project
`results/toy/NAME`. Hardware can be overridden with `--queue`, `--host`, and
`--device`. The initial CPU studies keep their recorded locations.

For the staged Riesz-softmax setup with one fixed temperature and 50,000
updates per map:

```bash
python scripts/test_gaussian_cascade.py --name NAME \
  --recipes softmax_actual --softmax-tau 0.1 \
  --updates-per-stage 50000 --walltime 04:00 --submit
```

This uses plain MLP maps. The single temperature stays constant across all
stages and updates, using the repository's batch-mean-distance scaling and
no field RMS normalization. Logs record the number of temperatures, field
magnitude, and effective positive/generated neighbor counts. Omitting
`--softmax-tau` retains the earlier three-temperature mixture.

The `rbf_actual` recipe uses an unbiased MMD estimator with an equal mixture
of Gaussian kernels, with standard deviations 0.1, 0.2, 0.4, 0.8, and 1.6 in
data coordinates. `rbf_fourier_actual` additionally augments each map's input
with sine/cosine features from 64 fixed random frequencies at four scales.
No target centers are supplied to the network. This increases total trainable
parameters from 44,820 to 126,740, so it does not isolate input features from
capacity. Both recipes retain ten separate maps and actual-prefix training.
For the staged-only refinement with extra final-stage training:

```bash
python scripts/test_gaussian_cascade.py --name NAME \
  --recipes rbf_fourier_actual --updates-per-stage 2000 \
  --final-stage-updates 10000 --submit
```

The small reference-process study uses `small_flow.py` and
`scripts/lsf/launch_small_flow.py`. It runs MoG first, then MNIST and
FashionMNIST on `longgpu01` in `longgpu`. New runs freeze their source and
configuration under the project results root. Flow matching is a standalone
control: Riesz jobs reject direct and inherited flow-matching checkpoints.
Image reconstruction priming is allowed, including masked reconstruction;
the reconstruction output head is reset before learning an unrestricted velocity.
`--reprime-features --prime-steps N --feature-space primed --fork PATH` trains
a fresh reconstruction model, freezes it as the feature encoder, and then
restores only the generator EMA from the Riesz parent. The new encoder retains
its reconstruction-only weights. Parent ancestry is checked before priming;
`initialization.json` verifies exact generator restoration and encoder origin.
This optional path tests longer feature pretraining independently of the
generator's initialization. Priming batch losses are saved every 5,000 updates
and at the end; transport FID keeps its separate 5,000-update cadence. This
extra encoder stage preserves the pre-priming transport RNG state so it does
not change the subsequent data/noise sequence relative to a fork-only control.

`--objective path-endpoint` combines independent next-marginal Riesz-softmax
matching along `(1-t)*noise+t*data` with Riesz-softmax matching of a full
differentiable noise-to-data rollout. `--solver euler --train-rollout-steps 10
--sample-steps 10` keeps the trained and sampled discrete transport identical.
The optional image `--feature-space primed` adds frozen reconstruction features
to the endpoint loss; local noisy-state matching stays in pixel space.
`--deep --width 16` uses a compact 28/14/7 U-Net with attention at 7x7.
`--no-rollout-checkpointing` trades memory for faster image updates.
`--affinity row-column` uses joint row/column softmax affinities and
cross-population mass balancing, as in Drifting, while retaining Riesz
unit-direction forces. The default `row` preserves the original separately
normalized attraction/repulsion neighborhoods.

`--parameterization denoiser` instead predicts a clean state `D(x,t)` and uses
velocity `(D(x,t)-x)/(1-t)`. Euler integration directly emits the clean predictor
at its last step. This option requires Euler, avoiding the singular velocity at
time one, and retains the reconstruction head when primed. It changes the model
parameterization without introducing an FM checkpoint or paired velocity loss.

Image audits can use `--decode-primed --reference-split train` to diagnose a
frozen reconstruction decoder applied after transport. This postprocessing is
explicitly labeled in the audit; development choices use the training reference.
In image training, `--decode-primed --feature-space primed` differentiates the
endpoint Riesz loss through that frozen reconstruction decoder and evaluates
the same decoded output. Its weights remain fixed and no FM teacher is used.
`--pixel-statistics-weight 1` adds a separately averaged Riesz loss over aligned
2x2/4x4 pixel means and standard deviations, targeting local texture and
background leakage without paired images or labels.
`--pixel-gradient-weight 1` adds a separately averaged Riesz loss on signed
horizontal/vertical differences at native resolution and pooled to 14x14 and
7x7. It matches the
data's spatial-derivative distributions, rather than penalizing all edges.
`--spatial-feature-weight 1 --spatial-feature-locations 16` adds independent
Riesz kernels at sampled spatial locations of each frozen encoder feature map,
plus global channel means and standard deviations. Locations are shared between
generated and target populations; images remain unpaired. Feature and field
normalization are shared across locations within each map, following the
[Drifting multi-scale feature design](https://arxiv.org/html/2602.04770v2#A5).
The branch retains unit-direction Riesz forces, uses explicit differences for
numerical stability, and requires primed features and row/column affinities.
Its weight defaults to zero, preserving previous recipes and encoder weights.
For image `path-endpoint` training, `--reference-coupling paired` reuses each
noise/data pair at the current and next reference time in the local Riesz loss.
The endpoint population remains independent. This changes the Monte Carlo
coupling of the reference loss without introducing velocity regression or a
learned teacher. Audits label this particle coupling separately from analytic
paired velocity supervision; the default remains `independent`.
With row/column affinities, `--reference-coupling paired-loo` additionally masks
each query's matched positive target, alongside its self-negative. This retains
correlated support populations while excluding the direct self-pair attraction.
The MoG trainer supports the same coupling options. `--objective coupled`
trains only the local reference loss, with `paired` as its default coupling;
`path-endpoint` defaults to independent reference populations.
For encoder-free random-time image trials, `--objective marginal` matches
independent next-time pixel populations; `--objective coupled
--reference-coupling paired-loo --affinity row-column` shares reference particles
while excluding each query's own positive target. Both use one network call per
update and can start with `--prime-steps 0 --feature-space pixels`. The optional
`--local-pixel-scales 28 14 7` averages separate local Riesz losses on native and
average-pooled pixels. This adds fixed spatial resolutions, with no learned
feature encoder or velocity-regression term. Sampling resolution remains a
separate setting. Omit this option to retain the original native-pixel loss.

Image `--reference-time-power 1` multiplies the local marginal Riesz loss by
`1-t`. For the denoiser parameterization this cancels its explicit
`1/(1-t)` output Jacobian in that loss, while preserving the reference path and
endpoint loss. The default power zero preserves existing recipes. Compare this
experimental weighting against a matched continuation with power zero.
`--reference-whitening-floor 0.1` instead preconditions pixel-space Riesz losses
by the reference path's per-pixel standard deviation:
`sqrt((1-t)^2 + t^2 * Var(data))`, with the specified positive floor. Endpoint
and local next-marginal losses use their respective reference times. Learned
reconstruction features and model inputs retain their original coordinates.

`--objective reference-hybrid` jointly trains analytic reference-path velocity
regression and endpoint Riesz-softmax matching from the first update, with
`--reference-weight` and `--riesz-weight`. This is explicitly a hybrid loss,
not pure Riesz training; it uses no pretrained FM weights or teacher. Image
`--endpoint-batch` can reduce the rollout batch independently of the reference
velocity batch. Audits distinguish paired reference-velocity supervision from
inheritance of a trained FM model.

The experimental `--objective reference-centered` subtracts a matched
current-time field from the next-time field. Its positive and negative support
populations are independent and equal-sized, and each population reuses its
own noise/data particles across the two times. Repulsion must equal one.
The baseline field then has zero expectation by positive/negative exchange
symmetry. This is a sampled reference-process control variate, not a learned
teacher or paired velocity-regression loss.
`--paired-reference-supports` instead shares the positive and negative path
particles, cancelling the baseline field exactly; the query remains independent.

MoG audits report the fraction farther than 0.36 from every mode center,
mode balance, radial-distribution KS error, within-mode spread, and sliced
Wasserstein distance on fresh samples. Optional terminal Gaussian noise is
recorded in the configuration and applied in both training and sampling.
For MoG row-column endpoint objectives, experimental `--endpoint-tail-scale s`
multiplies each query's detached force by `min(1 + (d/s)^2, w)`, where `d` is
its nearest positive-particle distance divided by the batch distance scale and
`w` is `--endpoint-tail-max-weight` (default 4). Weighting precedes the existing
force RMS normalization. It uses target particles, not mode labels or centers;
the default scale zero preserves the original loss. The local reference loss
is unchanged. Compare tail geometry and spread alongside the outside count.
Image jobs benchmark FID-10K every 5,000 transport updates, with grayscale
repeated into RGB and resized from 28x28 to 32x32 before the existing Inception
pipeline. Reconstruction priming is recorded as a separate stage. Recovery
checkpoints are saved every 1,000 transport updates; preview-only audits do not
compute off-schedule FID. `audit_small_image_flow.py` evaluates selected
checkpoints on fresh noise and the held-out test split. For a sampling-resolution
diagnostic, pass `--reference-split train --sample-steps 10 20 50`; this reuses
the same checkpoint and fresh noise at each resolution. Pass `--noise-seed N`
to use a separate noise draw for confirmation after sampler selection; the
default FID noise seed remains 7754321. The audit records
transport and decoder evaluations separately, archives checkpoint bytes and
hashes, and checks inherited reconstruction encoders against their original
priming checkpoint. Its default `--fid-batch-size 32` reduces evaluation memory
when small jobs share a GPU; it does not change the FID sample count.

The one-pass architectures are built through `backbone.build_backbone` and described by
`backbone.backbone_metadata`:

- MNIST U-Net: resolutions $28,14,7$, channels $32,64,128$, residual blocks,
  concatenated skips, GroupNorm, SiLU, and sigmoid pixels. It has 1,018,721
  parameters and takes one $1\times28\times28$ Gaussian noise image.
- CIFAR U-Net: the official Score-SDE CIFAR DDPM++ spatial topology,
  with 56,353,923 parameters in our adaptation,
  with resolutions $32,16,8,4$, base width 128, multipliers $(1,2,2,2)$,
  four encoder residual blocks and five decoder residual blocks per resolution.
  Every encoder block and downsample contributes a concatenated decoder skip.
  Single-head attention follows each encoder block at resolution 16, the decoder
  stage at 16, and the bottleneck. BigGAN residual blocks perform nearest-neighbour
  upsampling and average-pool downsampling, with residual scaling by $1/\sqrt{2}$.
  GroupNorm uses epsilon $10^{-6}$, SiLU activations, dropout 0.1, and DDPM
  fan-average initialization with near-zero residual/attention output branches.
  The spatial topology follows [Song et al.'s CIFAR DDPM++ configuration](https://github.com/yang-song/score_sde_pytorch/blob/main/configs/vp/cifar10_ddpmpp.py)
  and [implementation](https://github.com/yang-song/score_sde_pytorch/blob/main/models/ncsnpp.py).
  Our one-pass adaptation omits timestep embedding and applies a sigmoid to
  generate image pixels from $3\times32\times32$ Gaussian noise. It is not a
  diffusion denoiser and cannot load an official diffusion checkpoint directly.
  Source attribution and the Apache-2.0 license are in `backbone.py` and
  `assets/SCORE_SDE_LICENSE`.
- MLP: default latent dimension 32 and hidden widths $128,256$, with SiLU and
  sigmoid output. `--hidden-dims` customizes these widths for either dataset.

U-Net latent dimensions must match the image size: 784 for MNIST, 3072 for
CIFAR. `--base-channels` controls U-Net width; `--num-res-blocks` controls the
CIFAR stage depth. Adam defaults to learning rate $10^{-3}$ for MNIST and
$2\times10^{-4}$ for CIFAR, with betas $(0.9,0.999)$, epsilon $10^{-8}$,
and no weight decay. These are our experiment budgets and training settings,
rather than a reproduction of the full Score-SDE training recipe.

## Objectives

### Optional multiscale pixel losses

Enable the full-image, coarse-image, and local-patch combination with:

```bash
MULTISCALE_LOSS=1 bash run.sh
```

The generator still takes Gaussian noise and emits one full image in one pass.
Training applies the selected objective independently to the full image,
average-pooled views at 8x8 and 16x16, and aligned crops at 7x7 and 14x14.
For each crop size, one uniformly sampled valid location is shared across the
generated and real populations; images are not paired. Locations are resampled
each update using a dedicated RNG with seed `seed + 2`.

The default loss is `(global + mean(coarse views) + mean(patch views)) / 3`.
CLI options `--coarse-sizes`, `--patch-sizes`, `--crops-per-size`,
`--global-loss-weight`, `--coarse-loss-weight`, and `--patch-loss-weight` configure
the views and group weights. A zero weight disables a group. We average within
groups and normalize the active group weights to sum one. Ordinary Riesz uses
Euclidean distance divided by the square root of the view dimension; softmax
Riesz retains its existing batch-distance and per-coordinate normalization.

The configuration records this geometry and RNG convention. Extra CSV columns
`train_global_surrogate`, `train_coarse_surrogate`, `train_patch_surrogate`, and
`train_combined_surrogate` average training surrogates over updates in each
iteration; these values are not MMD energies. Initialization and disabled groups
have blank entries. FID and the existing full-image MMD diagnostics retain their
evaluation definitions. Without the flag, training follows the original objective.

### Base objectives

The ordinary Riesz kernel is $k(x,y)=-\|x-y\|_2$. For generated images
$x_i=G_\theta(z_i)$, the energy is half the biased empirical MMD squared.
`loss.py` computes its exact image gradient via the field
$v_i=\operatorname{mean}_j (x_i-x_j)/\|x_i-x_j\|_2
-\operatorname{mean}_j (x_i-y_j)/\|x_i-y_j\|_2$.
Backpropagation uses $-v_i/N$, preserving the exact energy gradient through
the generator. Exact collisions contribute zero and pairwise operations are
chunked to bound memory.

Softmax Riesz reuses the loss from
[`hudsonchen/riesz_flow_hudson`](https://github.com/hudsonchen/riesz_flow_hudson),
revision `777d696f31340a1370f5b39a2e7d000464b10526`. The upstream source hash
and license reference are retained in `loss.py`; the license text is in
`SOFTMAX_RIESZ_LICENSE.txt`. It applies row-softmax weights to unit directions
at temperatures `[0.2, 0.05, 0.02]`, masking generated self-diagonals. All other
particles participate; there is no hard top-k truncation. Each temperature
field is RMS-normalized before summation, then the generator regresses toward
a detached goal in batch-scaled coordinates. `--tau-list` controls temperatures;
`--no-normalize-per-temperature` selects upstream's combined-field normalization.
`--no-field-rms-normalization` bypasses both field RMS normalizations while
retaining batch-distance coordinate scaling, softmax temperatures, unit
directions, and the `1/tau` factors. This softmax-only ablation records raw
combined/per-temperature field RMS, attraction/repulsion RMS, pre-clipping
parameter gradient norm, and clipping frequency in `training_diagnostics.csv`.
Rows average updates in intervals of at most 100 updates (also flushed at
checkpoints and completion); field magnitudes first average equally over active
views. `clip_fraction` is the fraction of updates exceeding the configured
gradient norm threshold, not a fraction of parameters. Learning rate is the
last rate in each interval. The existing modes retain their behavior.

`--repulsion-weight` (default `1.0`) scales the softmax Riesz repulsive fields
in every global, coarse, and patch view before field normalization. Attraction
and the batch-distance scale remain unchanged. Use `--repulsion-weight 0.5`
with `--no-field-rms-normalization` for attraction minus half the repulsion.
The coefficient is recorded in the configuration and checked when resuming;
the ordinary MMD diagnostic keeps its original definition.
`scripts/lsf/launch_dit_repulsion.py` creates a frozen posttraining run from
a completed priming study, validates its source, and optionally submits it
with `--submit`. New outputs use `run_storage.new_run_path`.

`loss.backpropagate_objective` dispatches the Riesz and Drifting training rules. Softmax Riesz's
regression surrogate is distinct from the ordinary Riesz energy. Both are
used in raw pixel space with no feature encoder or CFG fixed negatives.
Finite Adam/minibatch steps need not decrease MMD monotonically.

### Official Drifting loss without a feature encoder

Select `--objective drifting`, or `OBJECTIVE=drifting bash run.sh`.
This ports the [official JAX loss](https://github.com/lambertae/drifting/blob/8861066430db11d2fb0858d042c7845b99df49f1/drift_loss.py)
to PyTorch in `loss.drifting_loss` (also available as `loss.drift_loss`).
The upstream revision/hash are recorded in the source and run configuration;
its MIT license is retained in `DRIFTING_LICENSE.txt`.

The loss uses the geometric mean of row and column softmax affinities over
joint generated/real support, balances attraction and repulsion using their
opposite population's affinity mass, and uses displacement vectors directly.
It preserves the official distance floor, finite self-distance mask, affinity
floor, batch coordinate scaling, per-temperature RMS normalization, and detached
regression target. Optional weighted positives and fixed negatives are supported
by the loss API. `--tau-list` selects temperatures; per-temperature normalization
is always enabled for Drifting.

The trainer applies this loss directly to pixels, without a feature encoder.
`--multiscale-loss` optionally applies it to the existing pooled images and aligned
patches too. This is the official loss within our generator/training setup, rather
than the full ImageNet training recipe: pretrained MAE features, class conditioning,
and sample queues are not added. The published ImageNet models use pretrained
feature encoders, including for pixel-output generation, so changing the loss
alone does not reproduce those results.

```bash
OBJECTIVE=drifting MULTISCALE_LOSS=1 bash run.sh
```

### Sinkhorn implementation

`loss.py` also owns `sinkhorn_plan`, `sinkhorn_barycenter`, and
`sinkhorn_target_loss`. The two-independent-generated-batch W-Flow rule and its
normalization are unchanged. `--objective sinkhorn` selects it; `sinkhorn.py`
retains compatibility imports for existing callers.

## Data and outputs

MNIST downloads from the CVDF mirror to the data root, with canonical checksums.
CIFAR accepts either official binary or Python archives after verifying the
[official checksums](https://cave.cs.toronto.edu/kriz/cifar.html). Verified existing
archives in the data root or Torch vision cache are reused. Archive members are
read directly without extraction. CIFAR-10 FID requires torchvision and SciPy. Pixels are
scaled from uint8 to $[0,1]$; no augmentation or conditioning is used by default.

Each run saves:

- `config.json`, including architecture, optimizer, objective, seed, data source,
  checksums, update budget, evaluation convention, and snapshot epochs.
- `data_indices.json`, training/test example grids, generated snapshot grids,
  and `trajectory.png` with fixed evaluation identities.
- `metrics.csv` and PNG curves. MMD is recorded each epoch. CIFAR-10 saves
  `fid_vs_iteration.png` with MMD and FID in two horizontal panels; MNIST saves
  `mmd_vs_iterations.png`. No PDFs are produced by the trainer.
- `run.log` when launched through `run.sh`.

Pixel-space diagnostic evaluation uses 64 generated images and fixed subsets of up to 512
training and 512 test images. The CSV records ordinary biased Riesz MMD squared;
run-level plots show its square root, without smoothing. This is a pixel-space
diagnostic. The field RMS column
uses the objective-specific convention recorded in the configuration.
Only fixed evaluation outputs are recomputed at each epoch, avoiding a second
large-model forward pass per training minibatch. Set `--checkpoint-every N` to
save resumable checkpoints; the two-stage launchers enable this automatically.
Passed smoke-test outputs are removed; full experiment artifacts are retained.

### CIFAR-10 FID

CIFAR-10 uses FID as the primary quality metric, evaluated at initialization
and every saved snapshot, including the final model. The default compares
10,000 generated images against all 50,000 training images, independently of
the training subset. `--class-id` also filters the FID reference. Use
`--fid-samples 50000` for FID-50K and `--fid-batch-size 64` to control memory.
Report the sample count and reference split/class when comparing scores.
The separate fixed evaluation seed does not consume training randomness.
Reference features are computed once per run; FID adds substantial computation.
The launcher evaluates FID and saves generated grids and the combined metric
plot at epoch 0 and every 20 epochs through epoch 2,000.

The evaluator uses 2048-dimensional features from the TensorFlow-compatible
[pytorch-fid v0.3.0 Inception](https://github.com/mseitzer/pytorch-fid/tree/v0.3.0).
Weights and their SHA-256 checksum are in `assets/`; `_fid_inception.py` vendors
the architecture, with its Apache license in `assets/PYTORCH_FID_LICENSE`.
Preprocessing uses RGB pixels in $[0,1]$, bilinear resizing to $299\times299$
with `align_corners=False`, and scaling to $[-1,1]$.
The `mmd_flow_hudson` environment includes the required torchvision and SciPy.

Scores are saved in `fid_metrics.json` and the `fid` column of `metrics.csv`
(blank between FID snapshots). The configuration records the reference size,
feature dimension, seed, sample count, preprocessing, and weight hash.

Standalone evaluation accepts a NumPy array of individual generated images:

```bash
/home/zongchen/miniconda3/envs/mmd_flow_hudson/bin/python fid.py \
  --samples generated.npy --data-root /home/zongchen/data --output fid.json
```

Samples must have shape `(N,3,32,32)` or `(N,3072)`, with uint8 pixels or
floating pixels in $[0,1]$. The default reference is the full training split;
`--reference-split test` selects the 10,000 test images instead.
The output JSON records FID, counts, split/class, and input hashes.
Rendered grids are unsuitable for FID; historical runs without saved samples
or checkpoints cannot be evaluated retrospectively. New runs evaluate the
live generator directly.

## Comparisons and checks

Softmax Riesz unit directions use direct pairwise differences, with queries
chunked in groups of 32 while retaining the full softmax support. This avoids
false zero distances and cancellation in weighted coordinate sums when
features nearly coincide. Batch distance scaling also uses direct distances.
The detached fixed-point target, temperature weights, and normalization options
are unchanged. Regression tests cover near-collisions, exact duplicates, and
weighted supports against explicit reference calculations.

The corrected CIFAR no-RMS run is
`results/cifar/priming_softmax_no_rms_directdiff_seed42_20261026/` (current LSF job
49819812; initial job 49819790). It starts from the same 80-epoch reconstruction
checkpoint with
temperatures 0.2/0.05/0.02, peak LR 2e-4, 1,000-update warmup, and EMA 0.9999.
It requests one exclusive full GPU and four CPU cores. All four active CIFAR
runs evaluate FID-10K every 2,500 updates and save/retain
checkpoints every 20,000 updates plus completion. Extra best-FID copies are
disabled. The workers refresh the comparison notebook after new FID scores.
Each run's frozen source and manifest preserve its deployed recipe. The earlier no-RMS
run and shared-GPU Sinkhorn pair continue independently.

`notebooks/compare_unet_riesz.ipynb` selects the two exact MNIST seed-0 U-Net
runs. It verifies matching settings, initial-image/data-index hashes, and
complete update coverage, then saves curves, final grids, and metric summaries
under `figures/unet_riesz_comparison_seed0/`. It compares complete recipes:
softmax weighting, normalization, and regression scaling all differ. A single
seed does not isolate these factors or establish general superiority.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 DRIFT_COMPILE=0 \
  /home/zongchen/miniconda3/envs/mmd_flow_hudson/bin/python \
  -m unittest discover -s tests -v
```

The additional upstream softmax function tests are in `tests/test_softmax.py`
and can be run with pytest. End-to-end checks use temporary output directories,
which are removed after successful validation.
