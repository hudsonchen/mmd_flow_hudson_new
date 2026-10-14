# Neural Riesz image generation

One shared trainer generates MNIST or CIFAR-10 images from fresh Gaussian
noise using either ordinary Riesz MMD or softmax Riesz. Training updates only
network parameters. Fixed, independent evaluation inputs track the same sample
identities across epochs.

## Layout

```text
main.py          CLI, training loop, optimizer, experiment configuration
fid.py           CIFAR-10 FID evaluation of generators or saved NumPy samples
assets/          pretrained FID Inception weights, checksum, and license
backbone.py      MLP, compact MNIST U-Net, larger CIFAR RGB attention U-Net
loss.py          ordinary Riesz fields/MMD and softmax Riesz loss
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

For CIFAR-10 H100 submission with `-P aihub_ucl` and result copying back from
the cluster, see [scripts/cs_cluster/README.md](scripts/cs_cluster/README.md).

All architectures are built through `backbone.build_backbone` and described by
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

`loss.backpropagate_objective` dispatches both training rules. Softmax Riesz's
regression surrogate is distinct from the ordinary Riesz energy. Both are
used in raw pixel space with no feature encoder or CFG fixed negatives.
Finite Adam/minibatch steps need not decrease MMD monotonically.

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
large-model forward pass per training minibatch. No `.pt` checkpoints are saved.
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
