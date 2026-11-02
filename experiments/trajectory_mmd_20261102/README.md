# Direct trajectory MMD: MoG followed by MNIST

This is a fresh implementation of Approach A. A time-conditioned velocity
network defines an ODE starting from standard Gaussian noise. Each update draws
one independent uniform time $t$, integrates fresh particles from zero to $t$,
and compares their law with $p_t=\operatorname{Law}((1-t)Z+tX)$ using Gaussian
MMD. Backpropagation follows every Heun step; no generated states are detached
or replaced by reference states. There are no velocity labels, Riesz kernels,
softmax affinities, teacher weights, or encoder features in the objective.

## What the objective establishes

For a characteristic kernel, zero population MMD identifies each marginal
([Gretton et al., 2012](https://www.jmlr.org/papers/v13/gretton12a.html)).
Zero integrated loss implies equality almost everywhere in time. With weakly
continuous marginal paths and a bounded continuous kernel, equality extends
to every time, including the endpoint. This does not identify the velocity:
adding a field $w_t$ with $\nabla\cdot(p_t w_t)=0$ preserves the continuity
equation. Finite loss and a finite numerical grid give neither exact equality
nor a guarantee of visual quality. A deterministic regular ODE also cannot
exactly map a continuous source to the finite empirical MNIST distribution;
the intended target is an approximation to the underlying image distribution.

MMD-trained generators are established work, e.g.
[Li et al., 2015](https://proceedings.mlr.press/v37/li15.html).
Here the generator is an ODE constrained across its time marginals. No novelty
claim is made from this pilot.

## Estimator and numerical controls

MoG uses an exact mixture of Gaussian kernels with bandwidths
0.2, 0.4, 0.8, and 1.6. The target has eight equally weighted radius-2 centers
with isotropic component standard deviation 0.15. Source and reference
populations use separate RNG streams. The loss is the full unbiased
U-statistic, including both repulsion and reference-reference terms; its
finite-sample value can be negative.

MNIST uses all 60,000 training images, normalized to [-1,1], with labels
discarded for training. Kernels act on full pixels and fixed nonoverlapping
2x2 and 4x4 average-pooled images, equally weighted. The full-pixel component
retains a characteristic kernel. For view dimension $d$ and pooling scale
$s$, Gaussian squared length is $d[(tb)^2+(1-t)^2/s^2]$, for
$b\in\{.05,.1,.2,.4\}$. Reference Gaussian noise is integrated analytically
in cross and reference-reference terms. This is variance reduction for the
same marginal objective, not matching a velocity or a denoised proxy.

Time sampling estimates the integral without an extra endpoint term. Numerical
time steps are at most 1/16 in the cluster pilot. Models are ordinary globally
connected SiLU velocity networks, width 64 for MoG and 512 for MNIST, with
zero-initialized output layers. The MNIST model is deliberately a simple
baseline; failure of this architecture/optimizer is not failure of the
population objective. There is no state clipping or terminal projection in
the dynamics. Display grids and classifier inputs alone are clipped to [0,1];
pixel MMD and out-of-range diagnostics use the actual generated values.

## Frozen initial pilot

Run three MoG seeds (0,1,2), 4,000 updates each, then MNIST seed 0 for 6,000
updates. Generated/reference batches are 256/512. Adam uses betas (0.9,0.95),
gradient clipping at 1, and cosine learning-rate decay to 10% of its initial
value. Initial learning rates are 0.001 (MoG) and 0.0002 (MNIST). A separate
CPU development run uses 1,200 updates, batches 128/256, and 12 training steps;
its results must not be pooled with the cluster recipe.

Both datasets save independent held-out reference samples, initial-generator
and real-versus-real baselines, final marginal checks at five times, and
16/64/128-step endpoint evaluations using identical noise. MoG additionally
reports mode fractions, within-three-sigma mass, and 128-direction sliced
Wasserstein distance. MNIST reports raw-pixel MMD, out-of-range mass,
unfiltered grids, and optional evaluation-only classifier Frechet/confidence
and digit frequencies. This Frechet score is not Inception FID.

Training completion and numerical audits are distinct from quality acceptance.
Pilot results require inspection; no automatic quality claim is made. Fresh
MNIST seeds should follow only after the pilot is understood. Neither previous
experiments nor their checkpoints are modified.

## Run

From the repository root:

```bash
python -m unittest discover -s experiments/trajectory_mmd_20261102 -p test_core.py -v
python experiments/trajectory_mmd_20261102/run.py --dataset mog --device cpu
qsub -v STUDY_ROOT=/path/to/fresh/results,DATA_ROOT=/path/to/data,CLASSIFIER=/path/to/classifier.pt experiments/trajectory_mmd_20261102/run_cluster.sh
```

Existing run directories are rejected. Each run stores its exact source and
configuration, RNG states, model/optimizer checkpoint, held-out samples,
metrics, solver refinements, and hash manifest. `ready.json` means the run
finished; it explicitly does not mean quality has been accepted.

## Local execution and architecture diagnostic

The cluster launcher was prepared but **not deployed or submitted**. Automatic
approval review rejected the GitHub push because ownership/privacy of the
destination and upload authorization were not verified. Experiments continued
locally on CPU. `results/trajectory_mmd_20261102/study_plan.json` records the
local plan: MoG development seed 0 and replication seeds 1/2 at 12,000 updates,
and an MNIST seed-0 pilot at 3,000 updates (generated/reference batches 128/256,
1,024 held-out samples). The plotting notebook is
`notebooks/trajectory_mmd.ipynb`.

The first MNIST architecture has a structural limitation independent of the
loss. Its 784-dimensional velocity lies in the span of a fixed 784-by-512
output matrix and its bias. At fixed trained parameters, projections onto
the orthogonal complement of this span cannot change during sampling, so at
least 271 directions retain their original Gaussian noise. This is an
architectural limitation of the pilot, not of trajectory MMD.

The matched `cpu_mnist_skip_pilot` adds `--linear-skip`, giving
$u_\theta(x,t)=a_\theta(t)x+b_\theta(x,t)$ with a learned scalar time network
$a_\theta$ and the original MLP $b_\theta$. Both output heads start at zero;
both are trained only by marginal MMD. This allows contraction in all pixel
directions. The baseline and skip pilot retain identical base-network
initialization, independent-population RNG streams, times, optimizer budget,
and evaluation noise. No analytic target velocity, teacher, or reconstruction
loss is introduced. The baseline remains available for comparison.

The unconstrained skip pilot subsequently developed escaping raw-pixel tails,
despite low pixel MMD; those tails persisted on refined solver grids. The final
`cpu_mnist_stable_pilot` uses `--stable-field` and a bounded 2,000-update budget.
Its velocity is $u_\theta(x,t)=-\operatorname{softplus}(a_\theta(t))x+
4\tanh(b_\theta(x,t)/4)$. Both heads still start at zero, so the initial
velocity is $-(\log 2)x$. This changes parameterization and initialization,
not the marginal loss. The smooth tanh acts on the velocity residual, never
on generated states. Coordinatewise outward speed is at most 4, preventing
finite-time escape of the exact ODE; numerical convergence is still checked.
The coordinatewise tanh also removes the fixed linear output-span restriction.
This diagnostic has a different budget and two architecture changes, so it is
not a single-factor comparison with the earlier pilots.

`diagnose_mnist.py` records unclipped pixel tails and recomputes endpoint MMD
in float64 with exact self diagonals. This guards against cancellation in
float32 distance calculations for very large escaped samples; original
training-monitor metrics and source snapshots are retained. The existing
cluster launcher is an unexecuted reproduction template for the original
baseline, not a validated training recommendation.

This failure is consistent with the population theory. For a Gaussian kernel
bounded by 1, contamination $P_\epsilon=(1-\epsilon)P+\epsilon\delta_R$ has
$\operatorname{MMD}^2(P_\epsilon,P)\leq 2\epsilon^2$, independently of how far
away $R$ lies. Thus small nonzero MMD does not control pixel moments or exclude
extreme outliers. Characteristicness identifies equality at zero; it does not
make every small empirical loss a reliable image-quality certificate. This
motivates checking raw tails, marginal paths, and solver convergence alongside
MMD, and controlling velocity growth through the model parameterization.

## Completed pilot findings

All three MoG runs and all three MNIST pilots completed locally, with source,
checkpoint, sample, and metric hashes replayed by `audit.py`. Six scientific
checks passed. `results/trajectory_mmd_20261102/study_summary.json` is the
machine-readable summary. Audit completion is not quality acceptance.

For MoG, fresh 8,192-sample evaluation at 128 Heun steps gives Gaussian MMD
squared $0.001065\pm0.000250$ and sliced Wasserstein distance
$0.06663\pm0.00588$ (three-seed mean and sample SD). Every mode is represented;
individual mode fractions range from 11.66% to 13.51%, around the true 12.5%.
Within-three-sigma concentration is $88.93\%\pm1.61\%$, versus 98.88% for
real mixture samples. Excess bridge/tail mass remains. The largest MMD-squared
change from 64 to 128 Heun steps is $6.77\times10^{-6}$, and the largest paired
sample RMS change is 0.000476. Seed 0 is development; seeds 1 and 2 replicate
the fixed recipe. This is an approximate proof of concept, not exact recovery.

MNIST is **not successful at these pilot budgets**. Final diagnostics use
1,024 held-out samples and unclipped ODE outputs:

| Velocity parameterization | Updates | Pixel MMD squared, float64 | Classifier Frechet | Maximum absolute pixel |
| --- | ---: | ---: | ---: | ---: |
| Plain MLP | 3,000 | 0.04337 | 1494.4 | 5.19 |
| MLP + unconstrained learned skip | 3,000 | 0.006117 | 1494.5 | 3,922,363.5 |
| Bounded residual + contractive skip | 2,000 | 0.005985 | 1526.4 | 4.61 |
| Real versus real | — | -0.000155 | 26.5 | at most 1 |

The classifier is evaluation-only and verified at 99.14% test accuracy; its
Frechet score is not Inception FID. Displays and classifier inputs are clipped,
but pixel MMD and the raw-tail audit are not. Negative real-real MMD squared is
normal for the unbiased estimator. The unequal stable-field budget and changed
initialization prevent treating this table as a controlled superiority claim.

The plain MLP leaves 271 orthogonal directions unchanged: measured projected
Gaussian standard deviation is 1.001 before and after sampling, versus 0.291
for real MNIST in those directions. Its projected state change is only
$1.32\times10^{-7}$ RMS. The unconstrained skip removes that invariant but
produces escaping tails: 16.3% of images have raw pixel RMS above 2. These tails
persist under refinement, with large absolute solver errors. The bounded-field
pilot eliminates extreme escape and has 64-to-128-step paired RMS change
0.000116, but still has noisy, malformed and imbalanced digits, 42.6% of pixel
values outside [-1,1], and 2.8% of images with RMS above 2. Stability alone has
not solved image quality.

The next useful controlled test is a convolutional velocity with the same
raw-pixel marginal objective and stable growth control. Larger batches and
terminal-time emphasis should be separate follow-ups, with any extra endpoint
loss explicitly distinguished from the original integral. These pilot results
do not show that the population marginal-matching objective fails; they show
that the tested finite-budget image recipes are inadequate.
