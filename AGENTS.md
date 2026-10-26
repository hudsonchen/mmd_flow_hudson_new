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
