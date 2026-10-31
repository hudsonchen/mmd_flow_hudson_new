# ImageNet64 B/4 vectorized Riesz continuation

Applies the Tiny ImageNet validated Riesz implementation to the existing
four-H100 ImageNet64 B/4 run. Direct FP32 distances, all 1024 interacting
particles, powers, temperatures, row-column affinities, paired references and
the detached surrogate are preserved. Weighted fields use centered matrix
products; close paired-positive terms are evaluated directly and zero
displacements are removed after affinity construction. Strict FP32 GEMMs are
required. The original implementation remains in tests/riesz_loss_reference.py.

The production runner and global-particle/DDP wrapper are unchanged. Each rank
reattaches its own predictions, and the factor of four before backward retains
the existing global-gradient convention. Existing data, model, EMA, optimizer,
scaler, four rank RNG states and attempt count resume from an immutable copy of
latest.pt. The 10000-update recovery cadence, best/final exceptions, 5000-update
FID cadence, 1000-update previews, benchmark statistics and sampler are unchanged.

Before optimized training, the replacement allocation validates the actual
four-rank B/4 model at batch 1024, including endpoint field/loss/query gradients,
DDP model gradients, synchronized parameters and finite sampling. It benchmarks
both implementations with full forward/backward/Adam steps. Benchmark updates
are discarded. If the benchmark fails or times out, the launcher resumes the
preserved production checkpoint with the original source. Original source and
deployment records are retained, with a separate implementation-change audit.

Both Tiny ImageNet jobs are outside this change's scope. This is a continuation
of the existing SAN run; its recorded storage destination is preserved.
