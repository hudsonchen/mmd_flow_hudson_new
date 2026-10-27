# Pixel-space paired-reference Riesz flow: implementation walkthrough

This document explains the **symmetric paired-reference recipe launched on
2026-10-27**. It describes the reference process, model, training loss, and
generation procedure. Equations use plain-text notation so they are readable
without a Markdown math renderer.

The central idea is:

> Choose a known path from Gaussian noise to real images. Train a network to
> move a batch of particles a small distance along that path, using an
> attractive–repulsive Riesz field. Generate images by chaining the learned
> steps together.

## 1. Notation

| Symbol | Meaning |
| --- | --- |
| `y_i` | A real image, scaled to `[-1, 1]` |
| `z_i` | Independent standard Gaussian noise with the same shape as the image |
| `t` | Reference time: `0` means noise and `1` means a real image |
| `x_i(t)` | The reference state at time `t` |
| `h` | A small time increment |
| `D_theta(x, t)` | The network's clean-image prediction |
| `v_theta(x, t)` | Velocity obtained from that prediction |
| `q_i` | The network's predicted next state |
| `r_i` | The analytically constructed next-time reference state |
| `B` | Batch size: `256` in this recipe |
| `d` | Image dimension: `28 * 28 = 784` for MNIST/FashionMNIST |
| `stop_gradient(...)` | Keep the numerical value, but do not backpropagate through its construction |

## 2. How the reference process is chosen

Draw a real image and independent Gaussian noise:

```text
y ~ data distribution
z ~ Normal(0, I)
```

Choose the linear interpolation:

```text
x(t) = (1 - t) * z + t * y
```

| Time | Reference state |
| --- | --- |
| `t = 0` | `z`: Gaussian noise |
| `t = 0.5` | `0.5 * z + 0.5 * y` |
| `t = 1` | `y`: a real image |

Conditional on a particular image, the distribution is known:

```text
x(t) given y ~ Normal(mean=t*y, covariance=(1-t)^2 * I)
```

This time convention runs **from noise to images**, the reverse of the usual
diffusion noising-time convention.

### Why use this path?

- We can sample a reference state at any time directly.
- We can construct its next-time reference without a trained teacher.
- Training does not require a complete endpoint rollout.
- The path has a simple conditional velocity: `y - z`.

The path is a fixed design choice. The network does not learn this reference
process. The same image/noise pair defines the whole conditional path; fresh
noise is not injected at each time step.

## 3. What one training batch contains

Each optimizer update draws `B = 256` real images and Gaussian noise samples.
It then draws **one common time for the entire batch**:

```text
t ~ Uniform(0, 0.999)
h = min(0.01, 1 - t)
```

All particles share the time so that the loss compares populations belonging
to the same reference marginal.

For each pair `(y_i, z_i)`, construct:

```text
current state:
    x_i = (1 - t) * z_i + t * y_i

next-time reference:
    r_i = (1 - t - h) * z_i + (t + h) * y_i
```

Equivalently:

```text
r_i = x_i + h * (y_i - z_i)
```

The current state and next-time target reuse the **same** real image and
noise. This is the paired-reference coupling.

Pairing gives a clear small-step signal: the difference between a reference
particle's two states is exactly `h * (y_i - z_i)`. With unrelated source and
target populations, differences between individual particles also contain
large differences from unrelated images and noise. Our independent-population
trials performed much worse.

Pairing changes the finite-batch estimator. The successful recipe should be
reported separately from independent-population or paired leave-one-out runs.

## 4. What the network predicts

The model is a time-conditioned U-Net with width `32`, a deeper `28 -> 14 -> 7`
spatial hierarchy, and approximately `1.37 million` parameters. It receives
only the current state and time:

```text
clean_prediction = D_theta(x_i, t)
```

The output is unrestricted. Calling it a clean-image predictor describes the
parameterization; it is not a pretrained denoiser.

Convert the prediction to velocity:

```text
v_theta(x_i, t) = (D_theta(x_i, t) - x_i) / (1 - t)
```

The implementation floors the denominator at `0.001`. The chosen training
time range and Euler sampling grid avoid the singular endpoint.

The predicted next state is:

```text
q_i = x_i + h * v_theta(x_i, t)
```

Define `alpha = h / (1 - t)`. Then:

```text
q_i = (1 - alpha) * x_i + alpha * D_theta(x_i, t)
```

We now have two populations at the next time:

- `Q = {q_i}`: predicted next states.
- `R = {r_i}`: reference next states constructed from the data/noise pairs.

## 5. How the Riesz field is chosen

We want predicted particles to approach the reference population while
retaining distributional spread. The field combines:

- **Attraction** from predictions toward reference particles.
- **Repulsion** between predicted particles.

The Riesz-style geometric building block here is the gradient direction of
Euclidean distance:

```text
u(delta) = delta / max(norm(delta), 1e-6)
```

This is a unit-direction force away from a nonzero displacement's origin.
Its magnitude does not grow linearly with distance as a squared-distance
gradient would.

The successful run computes distances on the full `784` pixels. It uses no
learned feature encoder, multiscale pooling, or whitening in the loss.

### 5.1 Distance scaling and neighbor weights

Use both predicted and reference particles as support:

```text
S = concatenate(Q, R)                # B predictions + B references
distance[i, j] = norm(q_i - S_j)
m = max(mean(distance), 1e-3)
```

For each of three temperatures:

```text
tau in {0.2, 0.05, 0.02}
```

compute:

```text
logit[i, j] = -distance[i, j] / (m * tau)

row_weight = softmax(logit, over support columns j)
column_weight = softmax(logit, over query rows i)

A[i, j] = sqrt(row_weight[i, j] * column_weight[i, j])
```

The row softmax chooses neighbors for each query. The column softmax also
accounts for competition among queries for a support point. Dividing by the
mean distance adapts the temperature scale to the batch's geometry.

The larger temperature gives broader interactions; the smaller temperatures
emphasize more local interactions. These are empirical design choices, not a
claim that this weighting is uniquely optimal.

### 5.2 Balance attraction and repulsion

Split the affinity matrix into two parts:

```text
A_minus[i, j] = affinity to predicted particle q_j
A_plus[i, j]  = affinity to reference particle r_j

negative_mass[i] = sum_j A_minus[i, j]
positive_mass[i] = sum_j A_plus[i, j]
```

For each temperature, construct:

```text
attraction_i = negative_mass[i]
               * sum_j A_plus[i, j] * u(r_j - q_i)

repulsion_i  = positive_mass[i]
               * sum_j A_minus[i, j] * u(q_j - q_i)

F_tau[i] = (attraction_i - repulsion_i) / tau
```

The cross-mass factors make the total positive and negative coefficient mass
equal for each query. Repulsion strength is `1` in this run.

Finally, normalize each temperature's field using the RMS across all particles
and pixel coordinates, and sum the three fields:

```text
rms_tau = sqrt(max(mean(F_tau ** 2), 1e-8))
normalized_F_tau = F_tau / rms_tau

F = sum_tau normalized_F_tau
```

**The implemented objective is a softmax-weighted, normalized Riesz-force
surrogate. It is not the ordinary unbiased Riesz MMD formula differentiated
directly.**

## 6. The self-normalization change that improved the results

The paired recipe retains each particle's own reference target `r_i` among
the positives. Previously, its self-negative `q_i` was masked out of the
negative affinity normalization.

That was asymmetric: the positive self-match contributed to weights and mass
balancing, while the corresponding negative self-match did not. Consequently,
identical predicted and reference populations could still have a nonzero
repulsive field.

The new option is:

```text
include_self_negative = True
```

The zero-distance self-negative now participates in affinity normalization.
Its actual force remains zero:

```text
u(q_i - q_i) = 0
```

When `Q = R`, the positive and negative supports and affinities match, their
forces cancel, and:

```text
F(Q, Q) = 0
```

This changes the finite-batch estimator. Exact cancellation is a useful
consistency property; it is not by itself a proof of convergence or better
generalization. We checked the property with tests and then measured actual
sample quality.

## 7. The scalar loss used for backpropagation

The field calculation above runs under `torch.no_grad()`. Distances, scales,
affinities, support particles, and the resulting field are detached.

Define a per-coordinate scale:

```text
s = max(m / sqrt(d), 1e-3)
```

Construct a frozen target:

```text
T_i = stop_gradient(q_i / s + F_i)
```

The applied scalar loss is:

```text
loss = ((1 - t) / h) * mean((q / s - T) ** 2)
```

The mean is over the batch and pixel coordinates. Backpropagation passes
through the first `q / s`, but not through `T` or `s`.

This uses squared regression to implement a prescribed Riesz update. It does
not add a separate regression loss directly onto clean images or reference
velocities.

### Why the time factor?

The network's clean prediction affects the next state through:

```text
dq / dD = h / (1 - t)
```

Multiplying the loss by `(1 - t) / h` cancels that explicit factor. It prevents
the small transition horizon from automatically shrinking the gradient
through the predicted step.

At the current forward-pass values, the residual `q / s - T` equals `-F`.
Although the numerical loss value is related to the field's magnitude, the
gradient is nonzero because `T` is detached. The logged surrogate loss should
not be interpreted as an ordinary MMD value that must decrease monotonically.

## 8. One optimizer update, in pseudocode

```python
# Fresh data and Gaussian noise; one common time for the batch.
y = sample_real_images(batch_size=256)       # pixels in [-1, 1]
z = randn_like(y)
t = uniform(0.0, 0.999)
h = min(0.01, 1.0 - t)

# Analytic reference process.
x = (1.0 - t) * z + t * y
r = (1.0 - t - h) * z + (t + h) * y

# One neural-network call; no training rollout.
D = network(x, t)
q = x + h * (D - x) / max(1.0 - t, 0.001)

# Frozen particle field, with both positive and negative self terms included.
with no_grad():
    F, s = symmetric_row_column_riesz_field(
        q, r,
        temperatures=[0.2, 0.05, 0.02],
        normalize_each_temperature=True,
        include_self_negative=True,
    )
    target = q / s + F

loss = ((1.0 - t) / h) * mean((q / s - target) ** 2)
loss.backward()
clip_gradient_norm(max_norm=1.0)
optimizer.step()
update_ema()
```

## 9. How generation works

Generation starts with fresh Gaussian noise and integrates the learned velocity
for `100` Euler steps using EMA weights:

```python
x = randn(image_shape)

for k in range(100):
    t = k / 100
    D = ema_network(x, t)
    x = x + 0.01 * (D - x) / (1.0 - t)
```

No real images or Riesz interactions are needed during generation. Each sample
is generated using the network and its initial noise.

At the last step, `t = 0.99`, so the update becomes:

```text
final_image = D_EMA(x_at_0.99, 0.99)
```

The network is never evaluated at the singular time `t = 1`. For rendering and
FID, the final centered output is mapped to `[0, 1]` and clipped to that range.

## 10. Relationship to flow matching and paired supervision

For this reference path, the conditional velocity is:

```text
reference_velocity = y_i - z_i
```

A standard flow-matching baseline regresses the predicted velocity directly
onto this target. Our recipe instead constructs the Riesz field over predicted
and reference next-state populations.

However, pairing contains strong information. Algebraically:

```text
q_i - r_i = (h / (1 - t)) * (D_theta(x_i, t) - y_i)
```

Therefore, attraction toward a particle's own reference target carries direct
denoising information. **This is paired-reference Riesz training.** It should
be distinguished from training using independent source and target marginal
samples, or from paired leave-one-out training that masks the own positive.

The model starts from random weights. There is no pretrained feature encoder,
flow-matching weight initialization, trained teacher, or added reference-MSE
term. The reference path itself uses the same linear interpolation idea that
is useful in flow matching.

## 11. Active settings of the successful recipe

| Setting | Value |
| --- | --- |
| Model | Time-conditioned deep U-Net, width 32 |
| Initialization | Random; no reconstruction priming |
| Image geometry | Full 28-by-28 pixels, scaled to `[-1, 1]` |
| Batch | 256 paired image/noise particles |
| Reference time | One common `Uniform(0, 0.999)` draw per update |
| Local horizon | `min(0.01, 1-t)` |
| Affinities | Geometric mean of row and column softmax weights |
| Temperatures | `0.2`, `0.05`, `0.02` |
| Repulsion strength | 1 |
| Positive self-match | Included |
| Negative self-match | Included in normalization; zero self-displacement force |
| Field normalization | RMS normalization separately for each temperature |
| Distance smoothing | None in the strongest run |
| Optimizer | Adam; peak learning rate `2e-4` |
| Learning-rate schedule | 500-update warmup; cosine decay toward `2e-5` over 100,000 updates |
| Gradient clipping | Global norm 1 |
| EMA | Decay 0.99 for the first 1,000 updates, then 0.999 |
| Sampling | 100 Euler steps using EMA weights |
| Previews | Every 1,000 updates |
| FID | Every 5,000 updates |

The shared configuration also contains options for other objectives. In
particular, `riesz_weight=0.1`, `marginal_weight`, `reference_weight`, and
`train_rollout_steps` are not used by this `coupled` training branch. The loss
in Section 7 is the applied loss.

## 12. Evidence for the change

These are recorded milestones, not a live status report.

For a fuller comparison with flow matching, including FID curves, matched-update
generated-image panels, and the independent test audit, see the
[performance report with generated images](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/mnist/riesz_vs_flow_matching_report_20261027/REPORT.md).

| MNIST recipe | Updates | Training-reference FID-10K |
| --- | ---: | ---: |
| Original paired-positive Riesz | 10,000 | 18.29 |
| Symmetric paired Riesz | 10,000 | **4.75** |
| Symmetric, smoothed distance, no field RMS normalization | 10,000 | 7.11 |

The symmetric-only comparison preserves the original paired-positive model,
batch size, optimizer/schedule, initialization, and sampling settings. Its
initial preview matched the control exactly. The smoothed/no-RMS variant
changes additional settings and is a separate experiment.

An independent evaluation of the **preserved 5,000-update symmetric checkpoint**
gave held-out MNIST test FID **6.30**, using fresh noise. The FM baseline gave
**11.13** under the same audit at its 20,000-update checkpoint. FM uses a smaller
network and batch, so that is a recipe comparison, not an isolated objective
or equal-compute comparison. The 6.30 held-out score does not belong to the
10,000-update checkpoint.

## 13. Where to find the implementation and records

- [Reference path, model parameterization, and Riesz field](small_flow.py)
- [Training loop, optimizer, previews, and evaluation](scripts/lsf/run_small_image_flow.py)
- [Model and loss tests](tests/test_small_flow.py)
- [Persistent numerical and metric review](scripts/lsf/supervise_pixel_flow.py)
- [10,000-update milestone report](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/mnist/pixel_symmetric_campaign_20261027/milestone_review_10000.json)
- [Independent held-out verification](/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/mnist/pixel_symmetric_campaign_20261027/independent_test_review.json)

All experiment artifacts remain in project storage. This file is repository
documentation describing the implementation.
