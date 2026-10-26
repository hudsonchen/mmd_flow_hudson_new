# Four-H100 CIFAR-10 priming

This launcher adapts the Penn LSF recipe at `7a93ac4`. It runs fresh
reconstruction priming for 80 epochs (62,560 updates, corruption 0.8, batch 64),
then 1,000,000 softmax-Riesz updates without field RMS normalization. It retains
distance scaling and the three temperatures 0.2/0.05/0.02.

Post-training uses global generated/real batches 128/256, horizontal flips,
multiscale weights 2:1:1, four crops per size, clipping at norm 1, Adam,
EMA 0.9999, and seed 42. Learning rate warms up for 1,000 steps to 2e-4,
then follows cosine decay to 1e-4 over the full million updates.
FID-10K runs every 2,500 steps, recovery checkpoints every 1,000 steps, and
numbered checkpoints every 25,000 steps. Reconstruction retains the original
constant 2e-4 learning rate and EMA 0.9999.

`--data-parallel-gpus 4` splits generator forwards across the four allocated
GPUs. Outputs are gathered before the global particle interactions. Batch sizes
and loss normalization therefore retain their Penn meaning. The optimizer,
EMA, and checkpoint keys belong to the unwrapped model on the first GPU.
Dropout uses each GPU's random stream; this does not promise bitwise agreement
with a one-GPU trajectory.

Submit from the dedicated GitHub deployment checkout, with the FID weights in
`assets/` and the existing CIFAR archive in the shared data directory:

```bash
mkdir -p "$STUDY"
qsub -o "$STUDY/scheduler.log" -v "STUDY=$STUDY" scripts/cs_cluster/run_cifar_priming_4h100.sh
```

The job requests four H100s on one node, 16 GB host memory per slot, and up to
720 hours; it exits when the two stages finish. A compute-node preflight checks
the allocation, FID asset checksum, global-loss gradient parity, and full-width
reconstruction/post-training updates. Results include `preflight.json`, source
hashes and exact commands in `deployment.json`, and stage-specific logs,
configs, status, metrics, and checkpoints. Resubmitting the same study resumes
its latest checkpoint and skips a completed stage; an advisory lock prevents
concurrent writers. No Penn checkpoint is copied: priming runs fresh on CS.
