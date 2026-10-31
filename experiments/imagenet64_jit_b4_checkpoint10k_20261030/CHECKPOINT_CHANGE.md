# ImageNet64 checkpoint cadence change, 2026-10-30

User requested less frequent checkpoint saves: routine recovery checkpoints every
10000 updates instead of 1000. Keep previews every 1000, FID every 5000, and
current best/final recovery checkpoints. An improving FID between periodic saves
must first save the current model/EMA/optimizer/RNG state; never link an older
latest.pt to a newer best.json.

Only checkpoint scheduling and explicit resume validation are changed. All loss,
model, optimizer, data, time-sampling and evaluation settings remain identical.
The existing running interpreter cannot reload its hard-coded loop, so perform
one controlled restart after the next completed recovery checkpoint. Preserve
the original deployment manifest; record the continuation separately, retaining
the same SAN study path and four-H100 allocation. No Tiny ImageNet run changes.
