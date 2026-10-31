# Tiny ImageNet: matched JiT-S/4 Riesz and flow matching

Requested 2026-10-30: two independent runs, one H100 each. Both start from the
same random initialization; no weight transfer or priming. Custom smaller JiT:
12 blocks, width 384, 6 heads, 4x4 patches, bottleneck intermediate width 128,
32 class tokens introduced at block 4, 200 classes plus the unconditional label.
The network predicts clean RGB; velocity conversion retains denominator floor .05.

The original Riesz loss and global particle wrapper were copied from the
ImageNet64 B/4 x-prediction experiment. The Tiny ImageNet Riesz continuation
uses an algebraically equivalent vectorized field; the frozen original remains
in `tests/riesz_loss_reference.py`. Paired references, h=.01 capped by 1-t,
time weight (1-t)/h, powers [.8,1,1.5], temperatures [.2,.05,.02], joint row-column
affinities, no field RMS normalization, and all 1024 interacting particles remain.
Flow matching uses mean squared error against clean minus noise. It uses the
same class-matched replacement batches, shared random t, CFG dropout, flips,
seed, optimizer and model. A shared t gives an unbiased FM objective with a
different gradient variance from the usual per-image independent-time recipe.

Official Tiny ImageNet: 100000 train images, 200 classes with 500 images each;
10000 held-out validation images, 50 per class. Preserve native 64px RGB and
training-only random horizontal flips. Archive checksum, complete decoding,
class counts, split indices and packed RGB arrays are verified before GPU launch.
Sampling with replacement is intentional: a 1024-particle conditional batch can
contain repeated training images, each with independently sampled noise.

Both runs: batch 1024, Adam (.9,.95), LR .0002, 5 equivalent warmup epochs,
600 equivalent epochs / 58594 updates; EMA .9999; gradient clip 1; BF16 network,
FP32 velocity/loss. Time sigmoid(N(-.8,.8^2)), clipped [.0001,.999]. Two workers.
Both sample with 50 Heun intervals (last Euler), CFG 2.9 on [.1,1].
Current and EMA previews every 1000 updates; FID-10K every 5000 and final against
all 10000 validation images. These scores are not comparable to ImageNet FID.
The first three preflight updates force t=.001,.77,.999; preflight also verifies
finite sampling/Inception, full optimizer checkpoint roundtrip and GPU identity.
Production starts afresh with the same seed after preflight, never from its weights.

Cluster outputs and frozen source use explicit SAN paths under
/SAN/intelsys/imagenet_mmd_flow/mmd_flow_hudson_new/results/tinyimagenet/.
Other running experiments are left intact. latest.pt every 1000 updates and
after update 3; best.pt retained at improving FID. Resumption restores optimizer
and RNG and rejects objective/configuration changes. Training-only seconds are
logged separately from evaluation overhead. See deployment.json and submission.json.

The vectorized loss reuses direct FP32 pairwise distances, combines weights over
all powers/temperatures, and evaluates forces with two centered matrix products.
It retains direct paired-positive displacements and removes zero-displacement
weights to avoid cancellation of singular self coefficients. Distances are not
replaced with squared-norm matrix products. Use strict FP32 matrix multiplication
(`NVIDIA_TF32_OVERRIDE=0` in the continuation launcher).

`tests/test_riesz_vectorized.py` compares loss, detached field and query gradient
against the frozen original, including endpoints, duplicate particles, unequal
populations and common translations. `scripts/benchmark_riesz.py` additionally
checks all 1024 particles on H100 with a real checkpoint and data batch, compares
model gradients through BF16 training, and times full optimizer steps. This
benchmark never saves its updated model into the training run. Deployment requires
a passed benchmark and exact source hashes; `scripts/cs_cluster/resume_vectorized.py`
restores the production checkpoint, optimizer, EMA, scaler and RNG through the
unchanged runner. The original shared source, flow matching and ImageNet64 remain
untouched. Separate continuation manifests retain the implementation boundary.
