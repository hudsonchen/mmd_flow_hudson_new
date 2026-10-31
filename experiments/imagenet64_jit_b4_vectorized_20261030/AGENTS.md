# Run storage preference

Save all new experiment outputs under
`/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/` (CIFAR studies under
`cifar/`, MNIST under `mnist/`). This includes checkpoints, metrics, figures,
logs, manifests, and frozen source snapshots. Use `run_storage.new_run_path`
in new launchers. Do not silently fall back to home storage.

Existing runs keep their recorded paths. In particular, the active primed DiT
study launched on 2026-10-26 remains in the home checkout; do not move an active
run or rewrite archived manifests without a separate migration request.

The user can override this preference with an explicit destination.

# FID evaluation preference

For new experiment jobs, benchmark FID every 5,000 optimizer updates by default.
Preserve existing runs' recorded schedules unless the user requests a change.
The ten-step time-conditioned Riesz run was explicitly switched to this cadence.

# CIFAR Riesz U-Net comparison

The user clarified on 2026-10-27 that the CIFAR Riesz U-Net run must use the
same backbone as the diffusion baseline: `CIFAR10UNetParticleGenerator` with
base channels 128, four residual blocks per level, time conditioning, and
61,804,419 parameters. Use `--architecture ddpm-unet`; do not substitute the
small `ImageVelocity` U-Net. Train from random weights with the Riesz objective,
without loading diffusion weights.

# Small experiment hardware

Prefer RTX 2080 Ti GPUs for small experiments such as Gaussian-mixture
studies, as clarified by the user. Use the `longgpu` queue on `longgpu01`.
Verify CUDA compatibility before launching on that hardware.

# Reference-process study initialization

For the MoG → MNIST → FashionMNIST reference-process study, trained flow-matching
models are comparison baselines. Do not initialize, distill, or otherwise transfer
weights from a flow-matching model into the Riesz/MMD models, including through
an intermediate checkpoint. Ideas from flow matching may inform the loss.
Reconstruction priming and continuation from Riesz-only checkpoints are allowed.
Joint reference-velocity/Riesz losses trained from scratch are allowed by the
user's permission to use flow-matching ideas in loss construction; label these
as hybrid objectives and distinguish them from pure Riesz training.
Exclude the earlier FM-initialized pilot runs from claims about this study.

# Pixel-only reference-process follow-up

For the MNIST/FashionMNIST follow-up requested on 2026-10-27, prioritize
pixel-space Riesz/MMD training without a pretrained encoder in the generator
loss. The initial random-time trials start from random weights without
reconstruction priming. Larger training budgets and more than ten sampling
steps are allowed; sample reference times during training rather than requiring
a full endpoint rollout. Keep encoder-assisted results separate.
Track progress and generated images during training, rather than waiting for
completion. Save early previews at least every 1,000 updates, inspect numerical
stability and sample diversity, and use clear early failures to revise trials.
Keep FID on its separate 5,000-update schedule.

# Autonomous experiment follow-through

The user explicitly requested proactive progress checks on 2026-10-27. After
launching improvement trials, follow through to subsequent previews and FID
milestones without waiting for another user prompt. Inspect samples, evaluate
progress against the recorded baselines, and revise clear failures. Maintain
persistent progress review while jobs run; distinguish numerical/metric checks
from actual visual review and do not imply that a logging process performs
model-driven analysis or sends chat updates. Preserve best/latest checkpoints
before early stopping, record the reason, and report meaningful outcomes.
