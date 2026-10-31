# Lorenz–96 hybrid residual ensemble comparison

This improves the exploratory single-scale Lorenz–96 pilot. It models forecast
uncertainty around the known deterministic numerical forecast. It is a hybrid
physics/generative task, not a reproduction of the two-scale Lorenz–96 protocol
in Pacchiardi et al. (JMLR 2024), and not a real-weather benchmark.

## Shared architecture and data

All four methods use exactly `ForecastNet(width=48, depth=3)`: the same circular
convolutions, inputs, output shape, parameter count, and paired initialization.
Inputs are the residual/noise state, initial context, deterministic forecast,
lead time and time embedding. No method has an extra learned network. Each
network has a receptive field of 15 cyclic grid sites. The energy-score network
is used as a residual one-pass generator at t=0. FM and Riesz use the same
network as a direct velocity field. These different output/sampling semantics
are intrinsic to the objectives and do not imply identical inference cost.

Use the byte-identical data from `lorenz96_ensemble_20261030`: 40 variables,
F=8, independent initial uncertainty sigma=0.5, leads 0.2/0.5/1.0. Split identity
and numerical reference validation are retained. Context/control normalization
and per-lead residual RMS use training data only. Both context and deterministic
forecast are available at inference. No future observation enters conditioning.
The earlier pilot's test set is reused for continuity, so this is a follow-up
exploratory experiment, not a pristine confirmatory test.

## Objectives and validation protocol

- Conditional flow matching, unchanged paired-reference Riesz field with direct
  velocity parameterization, direct energy-score minimization, and the existing
  no-repulsion mechanism ablation. Energy uses the unbiased within-ensemble
  U-statistic; no diagonal self-pair bias. Interactions never cross conditions.
- All methods share batches of 8 contexts x 32 members, paired random draws,
  Adam betas (0.9, 0.95), 200-step warmup, gradient clipping at 1 and EMA 0.99.
- Every method gets the same LR grid (3e-4, 1e-3, 3e-3), 3,000 updates per
  candidate, and tuning seed 100. Validation is performed every 1,000 updates
  with fixed shared noise, 32 contexts, 32 members and 64 reference outcomes.
  Selection minimizes mean per-lead CRPS divided by training residual RMS.
- Freeze chosen learning rates before test scoring. A validation-only gate
  requires short-lead FM to beat deterministic-control CRPS and attain empirical
  90% coverage between 70% and 98%. Failure stops before the larger comparison.
- Run 12,000 updates for each method at seeds 0,1,2. Report validation-selected
  checkpoints, final equal-update checkpoints and latest checkpoints fitting
  the minimum training-time budget across methods per seed. Timing excludes
  validation/checkpoint I/O, reports kernel warmup separately, and includes
  batches/optimization/EMA. Tuning compute is reported separately.
- Heun 20 steps (40 evaluations) and 40-step refinement use the same noise.
  Energy-score generation needs one network evaluation. Report neural sampling
  cost plus the shared CPU deterministic-forecast cost separately.

## Evaluation, audit and limitations

128-member test ensembles are evaluated against the first 256 independent
reference futures. The next 128 futures provide the simulator baseline.
Also retain deterministic and independent Gaussian-residual baselines. Report
fair CRPS/energy, interval coverage and width, mean RMSE, covariance/correlation,
extreme-event Brier score, rank histograms, p=0.5 variogram score, and a
coordinatewise member-shuffle control preserving every empirical marginal.
Variogram scores use the standard finite-ensemble estimator; ensemble size is
matched. More spread alone does not establish useful diversity.

The source and data are hashed. The final audit checks every run and evaluation,
paired initialization, identical architecture/parameter counts, finite coverage,
checkpoint hashes, and score replay from saved draws. `ready.json` follows audit.
Three-seed paired bootstrap intervals are exploratory. Ensembles at separate
leads are conditional marginals, not coherent multilead trajectories. The
no-repulsion ablation need not preserve consistency. The reference-path theorem
does not automatically guarantee the learned sampler's distribution.

## References

- Pacchiardi et al., JMLR 2024 (direct energy/kernel scoring-rule generators):
  https://jmlr.org/papers/v25/23-0038.html
- DiffCast, CVPR 2024 (deterministic prediction plus stochastic residuals):
  https://arxiv.org/abs/2312.06734
- SEEDS, Science Advances 2024 (physics-ensemble augmentation/correction):
  https://research.google/pubs/seeds-emulation-of-weather-forecast-ensembles-with-diffusion-models/
- Howard and Barnes, August 2026 preprint (joint-structure/shuffle diagnostics):
  https://arxiv.org/abs/2608.08954

The scoring-rule baseline implements the energy-score objective, not the full
published GRU architecture or its two-scale benchmark. All methods deliberately
share the architecture required by this experiment.
