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
