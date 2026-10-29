# ImageNet64 paired-reference Riesz with JiT

This experiment adapts the best documented CIFAR paired-reference velocity
Riesz recipe to ImageNet64. Its loss uses RGB pixels directly. It loads no
pretrained generator, encoder, or latent autoencoder. Inception is used only
for evaluation, outside the training graph.

The backbone is the authors' JiT-B/16 at input resolution 64: 12 layers, width
768, 12 attention heads, bottleneck width 128, 32 class tokens inserted at
block 4, adaLN-Zero, SwiGLU, RMSNorm, RoPE, and QK-Norm. There are 131,136,000
parameters. The 16-pixel patch size is retained, giving 16 image tokens at
64px. The network predicts unrestricted RGB velocity, preserving the Riesz
parameterization; this is not the JiT denoising loss. Official code revision
and the small device-portability changes are in `vendor/jit/provenance.json`.

Training uses Adam, betas (0.9, 0.95), epsilon 1e-8, zero weight decay, peak
LR 0.0002, linear warmup for five dataset-equivalent epochs, then constant LR.
The global interacting batch is exactly 1024, split 342/341/341 across three
H100s. Unequal shards are padded only for communication and trimmed before
the loss. DDP's backward multiplier of three preserves the full-population
gradient. The budget is 600 dataset-equivalent epochs: 750,684 updates, based
on the 1,281,167-image training set. Images are sampled with replacement, so
these are image-count epochs rather than exhaustive shuffled dataset sweeps.

Each update samples one class uniformly and independent images from that
class across the ranks. With probability 0.1 the whole interaction batch is
unconditional and images are instead drawn from all training images. This
batch-level CFG dropout keeps each particle field conditional on a single
label (or the true unconditional mixture); it avoids mixing incompatible
conditional and unconditional particles in one Riesz field.

One time is shared across the interacting population, with
`logit(t) ~ Normal(-0.8, 0.8^2)`, clipped to [0.0001, 0.999]. The source and
reference use paired image/noise draws, `x_t=(1-t)z+t*y`,
`r=(1-t-h)z+(t+h)y`, and `q=x_t+h*v(x_t,t,label)`, with
`h=min(0.01,1-t)`. The unchanged Riesz field sums powers 0.8/1/1.5 and
temperatures 0.2/0.05/0.02 with row-column affinities, self negatives,
no field RMS normalization, and loss time weight `(1-t)/h`.

Network calls use BF16, particle calculations FP32, gradient norm clipping
at 1, and EMA decay 0.9999. RGB images use the official JiT/ADM 64px center
crop, random horizontal flip, and scaling to [-1,1].

Evaluation uses EMA, 50 uniform Heun intervals, CFG 2.9 only on [0.1,1], and
balanced class labels. There are 100 solver velocity evaluations; CFG adds
the null-class network evaluation inside the guidance interval. FID-10K
is measured against all ImageNet training images with the same 64px crop,
every 5000 updates and at the end. It is not a published ImageNet256 FID or
the canonical downsampled-ImageNet64 reference protocol. Reference statistics
are cached at the first scheduled FID. Previews and recovery checkpoints are
saved every 1000 updates; an initial recovery checkpoint is saved at update 3.

The approved benchmark-reference continuation instead uses the saved
`VIRTUAL_imagenet64_labeled.npz` mean/covariance, extracted without numerical
changes into `VIRTUAL_imagenet64_fid_only.npz`. Set `FID_REFERENCE` and
`FID_REFERENCE_SHA256` in the cluster launcher, plus `RESUME_FROM` for an
existing checkpoint. The evaluator validates the hash, dimensions, finite
values, and covariance symmetry before training and never reads reference
images in this mode. `fid_reference.json` records the reference provenance;
metric rows identify the benchmark and its hash. The archive does not encode
its reference sample count, so that field is null rather than inferred from
the 10,000 images bundled in the original archive. Generated samples retain
the existing unquantized float preprocessing and pytorch-fid Inception.
Changing reference requires explicit evaluation-change permission during
resume and resets best-FID selection, while restoring model, EMA, optimizer,
scaler, update, attempt counter, and each rank's RNG state.

Deployment uses the dedicated GitHub branch, an isolated cluster checkout,
and explicit SAN outputs. The three-update preflight uses the full 1024 batch,
endpoint times, parameter synchronization, short CFG sampling, Inception,
and a checkpoint round trip before starting fresh production training.

```bash
qsub -v STUDY=/SAN/intelsys/imagenet_mmd_flow/mmd_flow_hudson_new/results/imagenet64/jit_reference_3h100_20261029 \
  scripts/cs_cluster/run_imagenet64_jit_3h100.sh
```
