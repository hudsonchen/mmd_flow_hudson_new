# Continuing trajectory MMD: bounded prediction in a continuous image flow

This follows the completed `trajectory_mmd_20261102` pilots. The objective
still compares actual evolved particle distributions with independent
reference marginals. There are no paired velocity targets, paired image
reconstruction targets, class labels, Riesz kernels, or softmax affinities.

## Continuous field and solver

A convolutional U-Net produces a bounded array $D_\theta(x,t)\in[-1,1]^{784}$.
It parameterizes $u_\theta(x,t)=(D_\theta(x,t)-x)/(1-t+\epsilon)$, with
$\epsilon=0.001$. This is a model parameterization, not a denoising loss.
The field is regular on the whole interval. Under
$s=\log((1+\epsilon)/(1-t+\epsilon))$, the same ODE becomes
$dy/ds=D_\theta(y,t(s))-y$.

We differentiate through a second-order exponential midpoint discretization.
Both predictor and corrector are convex combinations of the current state
and a bounded prediction. Every numerical grid preserves the coordinatewise
bound $|Y_t|\leq r_t|Z|+(1-r_t)$, where
$r_t=(1-t+\epsilon)/(1+\epsilon)$. Thus the remaining unbounded Gaussian
component has coefficient about 0.001 at the endpoint, rather than the large
uncontrolled residual in the earlier pilots. The generated state is never
clipped or detached. Different step counts approximate the same continuous
field, and their samples and quality metrics are checked explicitly.

The U-Net uses fixed coordinate channels and time conditioning. Hidden
features are not used by the kernel, and there is no pretrained encoder.
Its bounded output is $D_\theta(x,t)=\tanh(x+f_\theta(x,t))$. The input skip
keeps random initial particles distinguishable through the long contraction
path; a nearly zero constant predictor would compress them almost to one
image before training. This skip is not a reconstruction loss.

## Distribution objective

The reference is $R_t=(1-t)Z'+tX$, with independent training-image samples
and reference noise. Source Gaussian particles use a different RNG stream.
The reference noise is integrated analytically in the Gaussian kernel.
Generated-generated pairs use the off-diagonal U-statistic; source-reference
interactions are all-to-all. The reference-reference term is omitted during
training because it has no model gradient. Consequently the logged training
loss is **not** full MMD squared. Held-out evaluation reports full MMD.

The kernel combines full pixels and fixed 2x/4x average pooling (75% weight)
with aligned random 7x7/14x14 raw-pixel crops (25%). There are four Gaussian
bandwidths in each view. Crop coordinates are shared across populations;
image identities are not paired. The positive full-image Gaussian component
retains characteristicness. This adds local stroke sensitivity without an
encoder or a new pointwise loss.

Two initial runs isolate endpoint emphasis:

- `integral`: a uniform time per update, estimating the requested integral.
- `endpoint`: time is 1 with probability 0.5 and uniform otherwise, estimating
  half the original integral plus half an extra endpoint MMD. This is an
  explicitly augmented objective, not an unchanged implementation of Approach A.

Both use width 24, generated/reference populations 256/2,048, 6,000 updates,
an eight-step terminal training solver, Adam at 0.0002 with 100-update warmup
and a fixed 12,000-update cosine schedule, and EMA 0.995. Near earlier times
the solver uses proportionately fewer logarithmic-time steps, with minimum 2.
The GPU launcher uses BF16 network operations and FP32 kernel calculations;
quality evaluation uses FP32 networks and float64 MMD sums. Each run has
separate source, reference, time, crop and evaluation random streams.

## Evidence and reproduction

`test_science.py` checks the exact constant-field solution and parameter
gradient, the state bound, second-order convergence, checkpoint gradient
equivalence, analytic-reference kernel equivalence and finite differences,
and independent population permutations with crops. The launcher runs a
short real-GPU preflight at the actual training population sizes before its
full run. It does not alter other jobs.

```bash
python -m unittest discover -s experiments/trajectory_mmd_xpred_20261102 -p test_science.py -v
qsub experiments/trajectory_mmd_xpred_20261102/cluster.sh
```

The cluster checkout is isolated and obtains source through the existing
public `hudsonchen/mmd_flow_hudson_new` repository. Ownership and visibility
were verified after the earlier deployment rejection. MNIST and the existing
evaluation-only classifier are separate assets, not GitHub uploads.

Each run retains source hashes, data/classifier identities, EMA and raw
weights, optimizer and RNG states, every evaluated checkpoint, unfiltered
samples, full-space marginal checks, and solver refinement. `ready.json`
means artifact completion, not image-quality acceptance. The final model
must also show coherent digits, coverage, controlled raw pixels, fresh-sample
quality, and numerical stability; a small MMD alone is insufficient.
