# CIFAR-10 on cs-cluster

`run_cifar_h100.sh` launches the CIFAR-10 DDPM++ one-pass generator with
softmax Riesz, EMA 0.999, and a 2,000-epoch budget. It requests one H100,
32 GB host memory, 48 hours, and the `aihub_ucl` project.

On `cs-cluster`, change to the project root under `/SAN/`, with the source,
FID weights in `assets/`, and the verified CIFAR-10 archive in `data/`:

```bash
qsub -P aihub_ucl scripts/cs_cluster/run_cifar_h100.sh
```

The dedicated multiscale launcher enables the full-image, 8x8/16x16 coarse-image,
and aligned 7x7/14x14 patch losses with the same DDPM++/EMA recipe:

```bash
qsub -P aihub_ucl scripts/cs_cluster/run_cifar_multiscale_h100.sh
```

Its result directory includes `_multiscale` before the job ID. The source copy
on SAN must include `multiscale.py` and `tests/test_multiscale.py`.

The merged scheduler log is written to the home directory as
`/home/zongchen/cifar_ddpmpp_ema.o<JOB_ID>` for the base launcher, or
`/home/zongchen/cifar_multiscale.o<JOB_ID>` for the multiscale launcher,
using the number returned by `qsub`.

This default applies to new submissions. An explicit `qsub -o` overrides it.

The default Python environment is
`/home/zongchen/miniconda3/envs/mmd_flow/bin/python`. To select another existing
environment, pass `-v PYTHON=/absolute/path/to/python` to `qsub`.

The launcher checks the allocated GPU and runs the unit tests before training.
It saves outputs under `results/cifar/h100_ddpmpp_ema0999_<JOB_ID>/`, including
`job_status.json`, `run.log`, metrics, and images. Successful completion produces
`artifact_manifest.json` for SHA-256 verification.

Run the transfer watcher on the destination machine:

```bash
python scripts/cs_cluster/sync_cifar_h100.py path/to/deployment.json
```

The deployment JSON contains `job_id`, `remote_source`, `remote_output`, and
`local_output`. For new submissions, also set `remote_log` to
the appropriate home-directory scheduler log above. Older deployment records without this
field retain their original `remote_source/scheduler.log` location.
The watcher uses the `cs-cluster` SSH alias, copies available
outputs every 300 seconds, and verifies final artifact hashes after completion.
The interval can be changed with `--interval SECONDS`. While training continues,
copied outputs are marked `partial_synced` in `transfer_status.json`.

Existing submitted jobs keep their original source snapshots and script paths.
