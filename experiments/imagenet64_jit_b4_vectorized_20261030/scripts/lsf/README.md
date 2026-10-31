# Velocity Riesz cluster scripts

The maintained entry point is `launch_cifar_riesz_ddp.py`. It freezes the velocity
model, loss, data/FID utilities, trainer, monitor, sampler audit, and tests before
submitting a GPU preflight and dependent training job. See the root
[README](../../README.md) for launch and continuation examples.

- `run_cifar_riesz_ddp.py`: one- or four-GPU velocity Riesz training, resume,
  preview/checkpoint cadence, and FID evaluation.
- `monitor_pixel_flow.py`: log numerical status and preview paths from a campaign
  manifest; this does not perform visual review or automatically tune jobs.
- `audit_cifar_velocity_samplers.py`: compare Euler and uniform/EDM Heun using a
  fixed checkpoint and shared noise. Its small-sample FID is exploratory and must
  not be compared directly with training FID-10K.

New experiment outputs go to project storage via `run_storage.new_run_path`.
Existing jobs and archived runs retain their recorded paths and frozen source.
