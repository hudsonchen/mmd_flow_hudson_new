Reference YAMLs fetched on 2026-10-15:

- W-Flow: https://github.com/hanjq17/W-Flow/blob/main/configs/gen/latent_sota_B_ot_8node.yaml
- Drifting: https://github.com/lambertae/drifting/blob/main/configs/gen/latent_sota_B.yaml

Neither release provides a CIFAR-10 configuration. The launcher
`scripts/cs_cluster/run_cifar_official_b_h100.sh` transfers the following B-model
hyperparameters to this repository's existing unconditional CIFAR pixel model:

| Setting | Both methods |
| --- | --- |
| Generated / positive population per interaction group | 64 / 64 |
| Independent negative generated population (Sinkhorn) | 64 |
| Optimizer | AdamW, betas (0.9, 0.95), epsilon 1e-8, weight decay 0 |
| Peak learning rate | 0.0004 |
| Warmup | 10,000 updates, linear from 1e-6; constant thereafter |
| Gradient norm cap | 2 |
| EMA | 0.999 per update |
| Training budget | 200,000 optimizer updates |
| Seed | 42 |
| Detached normalized-field target step | 1 |

Drifting uses temperatures [0.2, 0.05, 0.02]. Sinkhorn uses quadratic cost,
epsilon 0.05 times view dimension, 10 iterations, independent resampled
generated support, and no diagonal mask. `--field-step 1` records the unit step;
the Sinkhorn and Drifting losses directly construct their normalized targets.

Adaptation boundaries for the original unconditional launcher: the official conditional ImageNet model uses 128
interaction groups per train batch, positive memory banks, 32 unconditional
negative samples, CFG, a latent DiT, VAE, and MAE activations. These CIFAR jobs
use one unconditional interaction group, fresh random real batches from all
50,000 training images, the existing DDPM++ generator, and equally weighted
global/coarse/aligned-patch pixel views. They have no unconditional-negative
population or CFG. The official global batch sizes (4096 W-Flow / 2048 Drifting)
are not reproduced. The CIFAR diagnostics remain FID-10K every 5,000 updates;
only the latest resumable checkpoint is kept, written every 2,000 updates.
These are adaptations of released B hyperparameters, not full paper reproductions.

Sinkhorn audit: `tests/fixtures/sinkhorn_official.json` stores losses, gradients,
and plans evaluated directly with the unmodified released `drift_loss_ot.py`
(SHA256 c987285caf69cb1085e3e50450936368373087d6066bb6fe15020bae8796b58e).
Cases include unequal population sizes and 64/64/64 particles. The tests compare
the local unconditional, quadratic-cost path against these reference values.
The official Drifting fixtures separately check its loss and generator gradient.
The W-Flow and Drifting licenses are retained in the project root.

## Normalization audit and CFG extension

The normalization formulas retain the released conventions:

- Drifting and softmax Riesz divide coordinates by the weighted mean pair
  distance divided by sqrt(view dimension), floored at 1e-3. Each temperature
  field is divided by sqrt(max(mean(field squared), 1e-8)), then summed.
  The summed field is not normalized again: three aligned unit-RMS fields can
  have RMS three. Softmax Riesz's optional legacy combined-field mode is
  separate and remains available through `--no-normalize-per-temperature`.
- Quadratic Sinkhorn uses the RMS pair distance divided by sqrt(view
  dimension), the same coordinate-scale floor, regularization epsilon times
  view dimension, and the same squared-RMS field floor. Its one-temperature
  normalized field has RMS one unless the floor is active. The independent
  generated support participates in its coordinate scale; the new-CFG
  unconditional real support does not, matching W-Flow.
- Every view's squared regression error is averaged over particles and
  coordinates. Multiscale losses average views within each group and take
  the normalized weighted average over global/coarse/patch groups.

The audit found numerical cancellation at nearly identical pixel outputs:
float32 distance calculations based on squared norms, and barycentric
subtractions of large nearly equal coordinates, can corrupt small
displacements. The implementation now computes pair distances directly,
centers softmax-Riesz direction calculations, and centers Drifting force
calculations. It preserves the reference floors and detached gradients.
The existing official loss/gradient fixtures still pass. A CIFAR-dimensional
near-constant direction test compares against independent float64 direct
displacements; scale and duplicated-coordinate invariance tests cover all
three guided losses. This improves numerical fidelity, not evidence of an
improved training FID.

`--cfg` enables a class- and scale-conditioned CIFAR U-Net, with real
unconditional samples drawn from the full training split (including the
selected class). Scale 1 means ordinary class-conditional generation.

- Drifting uses reference unconditional affinity weights
  `(scale-1)*(generated_count-1)/unconditional_count`.
- Softmax Riesz uses the per-temperature field
  `scale*A_positive - A_generated - (scale-1)*A_unconditional`, then normalizes.
  Each A is a separately normalized softmax average of unit directions.
  This is an explicit guided extension of our Riesz field; it does not reuse
  Drifting's population-count coefficient for population averages.
  For coordinate-scale estimation only, unconditional weights additionally
  use `(generated_count-1)/unconditional_count`, so duplicating the
  unconditional Monte Carlo population changes neither its total geometry
  mass nor its averaged field.
- Sinkhorn uses the reference new-CFG field
  `scale*T_positive - T_generated - (scale-1)*T_unconditional`, then normalizes.
  `tests/fixtures/sinkhorn_cfg_official.json` records official loss/gradient
  values at scales 1, 2.5, and 4 from the same pinned W-Flow source above.

Guidance is trained into a single generator evaluation, as in the releases;
there is no interpolation between conditional and unconditional image outputs.
The CFG launcher and artifact contract are described in
[the cluster instructions](../../scripts/cs_cluster/README.md).
Class groups are independently normalized before gradient averaging, rather
than jointly normalized across all groups as in the distributed references.

`metrics.csv` now includes `scale_inputs`, `normalized_field_rms`,
`normalization_floor_active` (fraction of view/temperature fields hitting the
force floor), and the unclipped `gradient_norm`. CFG also logs
`cfg_scale_mean`. The historical fixed-target velocity diagnostic remains an
unguided mixed-population diagnostic; training normalization columns measure
the actual guided class-group fields.

Validation: `DRIFT_COMPILE=0 python -m unittest discover -s tests -v` checks
reference parity, normalization, CFG=1 equivalence, conditional parameter
gradients, balanced-label FID dispatch, and exact CPU checkpoint/resume for all
three objectives. The trainer integration test uses synthetic CIFAR-shaped
data and a lightweight FID stub; it does not report an Inception FID result.
