# Lorenz–96 conditional ensemble experiment

This is an exploratory scientific test of the existing paired-reference Riesz
training recipe, not a weather-forecasting claim or an assumption of superior
diversity. The primary question is whether it improves conditional distribution
accuracy and calibration at a matched budget relative to flow matching.

## Frozen protocol

- Forty cyclic variables, forcing 8. Independent context trajectories are spun
  up for 20 model-time units. Given a context, the initial state has independent
  Gaussian uncertainty with standard deviation 0.5 at each coordinate.
- RK4 with step 0.005, lead times 0.2, 0.5, 1.0 model-time units. Data creation
  checks step-halving and DOP853 agreement. These horizons are not labelled as
  actual meteorological hours or days.
- 1,024 training, 64 validation and 128 held-out test contexts. Each has 64,
  256 and 512 independently perturbed reference members respectively.
  Test scores use the first 256 reference members; an independent set of 128
  members forms an exact-simulator ensemble baseline. An unperturbed numerical
  trajectory is a deterministic baseline. Neither baseline is a learned model.
- Three methods: conditional velocity MSE (FM), the paired Riesz surrogate,
  and Riesz with only the negative displacement force removed. The latter
  retains affinities/scales and is a mechanism ablation, not a consistent
  alternative objective or a fully tuned competitor.
- Seeds 0, 1, 2 share initialization, context/reference selection, noise, times,
  cyclic shifts, and evaluation noise. Each batch has four separate contexts
  with 64 particles each. Forces never cross conditioning groups.
- Identical residual MLP: width 256, three blocks, x-prediction; velocity is
  `(x_hat-x_t)/max(1-t,0.05)`. State normalization uses only training data.
  All models use Adam (0.9,0.95), LR 2e-4, 500-update warmup, gradient norm
  clipping at 1, EMA 0.999, and 10,000 updates. No classifier-free guidance.
- The Riesz horizon, distance powers, temperatures, shared logit-normal time,
  time weighting, self-negatives, and detached-field convention are copied
  from the current Tiny ImageNet recipe. `original_riesz.py` is a byte-for-byte
  copy of that implementation. `test_science.py` checks batched loss, force,
  and gradient agreement, including times close to the endpoint.
- FP32 throughout. The architecture and EMA are adaptations to this small
  scientific benchmark; this does not reproduce the JiT image architecture.
- Evaluate EMA at 10,000 updates and at the last saved checkpoint fitting the
  fastest method's training-time budget within each seed. Checkpoints are
  spaced 1,000 updates apart. Actual used times are retained; budgets are
  rounded down, not represented as exactly equal. An initialization checkpoint
  at zero training time handles budgets below the first saved update; such a
  comparison is explicitly identifiable by selected_step=0 and is not evidence
  about a trained model. Training time includes
  data draws, forward/backward, optimizer and EMA, excluding checkpoint I/O.
  Five backward-only warm-up passes initialize objective-specific GPU kernels;
  they use separate random draws and do not update parameters or optimizer
  state. Their cost is reported separately in `warmup.json`.
- Generate 128-member ensembles with 50-step Heun (final Euler, 99 evaluations).
  Repeat final-checkpoint sampling at 100 steps with the same noise to diagnose
  integration sensitivity. Test results do not drive tuning or stopping.

## Evaluation and interpretation

Primary outcomes are fair CRPS and fair multivariate energy score in physical
state units. Secondary outcomes are central 90% coverage and interval width,
ensemble-mean RMSE, variance ratio, covariance/correlation errors, and a Brier
score for the event that the spatial maximum exceeds a training-only 95th
percentile threshold for that lead. Event thresholds are fixed before test
evaluation. More spread alone is not an improvement.

Per-context scores and generated ensembles are retained. Paired differences
against FM use a two-way bootstrap over training seeds and held-out contexts;
three seeds are only an exploratory estimate of training variability. Report
all leads and methods, including unfavourable results. Compare sampling costs
as well as training costs. Fixed shared hyperparameters are not evidence that
either method is optimally tuned. Larger or weather experiments require a
subsequent decision based on these findings.

Run `python -m unittest -v test_science` for scientific checks. The scheduler
launcher runs these and a complete GPU smoke study before production. It
requires exactly one scheduler-allocated visible GPU and never modifies other
jobs. Production can resume from checkpoints in the same output directory;
configuration/source mismatches fail instead of being silently accepted.

`ready.json` is written only after all nine training runs, validation and test
evaluations, refinement evaluations, paired initialization checks and artifact
hash checks pass. `status.json` reports intermediate progress. Results are not
complete merely because preflight passes or the job leaves the queue.

## Scientific references

- Lorenz–96 implementation and context: https://docs.dart.ucar.edu/en/v10.9.2/models/lorenz_96/readme.html
- Proper forecast scores: https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
- Flow matching: https://arxiv.org/abs/2210.02747
- Weather ensemble emulation as a possible later application:
  https://research.google/pubs/seeds-emulation-of-weather-forecast-ensembles-with-diffusion-models/

This experiment does not establish that a reference-path convergence theorem
controls generated trajectories; that theorem-to-algorithm question is separate.
