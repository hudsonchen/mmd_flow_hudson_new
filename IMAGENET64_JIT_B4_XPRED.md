# ImageNet64 JiT-B/4 clean-image prediction with paired-reference Riesz

This is a fresh four-H100 experiment. The network predicts the clean RGB image;
the velocity interface computes `(x_hat-x_t)/max(1-t,0.05)` in FP32, following
the official JiT conversion. The user explicitly selected the existing
MMD/Riesz objective, not the paper's paired velocity MSE.

Training retains `q=x_t+h*v_theta` and paired reference `r=x_(t+h)`, the global
pixel Riesz field, powers 0.8/1/1.5, temperatures 0.2/0.05/0.02, no field RMS
normalization, and time weight `(1-t)/h`. This is not a reproduction of the
paper's MSE objective or its ImageNet64 ablation architecture.

The B backbone has 12 blocks, width 768, 12 heads, patch size 4, 256 image
tokens, 32 class tokens inserted at block 4, and the existing 128-dimensional
patch projection. The released architecture enhancements remain enabled.
Activation checkpointing recomputes block activations without changing the
objective or interacting batch. All weights start randomly; no B/16 checkpoint
is loaded.

Global batch 1024 is split 256 per GPU. Adam uses LR 0.0002, betas (0.9,0.95),
zero weight decay, five dataset-equivalent warmup epochs and constant LR
thereafter. The target is 600 dataset-equivalent epochs (750684 updates).
Shared logit-normal times use mean -0.8 and standard deviation 0.8. These
settings retain the preceding Riesz run's recipe, rather than claim an exact
ImageNet64 paper recipe. EMA is 0.9999, gradient clipping is 1, and the network
uses BF16.

Sampling uses the converted velocity, 50 uniform intervals, Heun with a final
Euler interval (99 velocity evaluations), and CFG 2.9 on the existing guidance
interval [0.1,1]. FID-10K uses the verified saved ImageNet64 benchmark statistics
every 5000 updates. Previews and recovery checkpoints are saved every 1000
updates, with an initial recovery checkpoint at update 3.

The cs-cluster launcher uses the established SAN result root: the LongGroup
path in the inherited instructions belongs to the other cluster and is absent
here. Deployment is via a dedicated GitHub branch and isolated checkout.
The four-GPU preflight checks the full batch, endpoint gradients, synchronization,
sampling, Inception, and checkpoint roundtrip before fresh production training.

Official conversion and sampler: https://github.com/LTH14/JiT/blob/main/denoiser.py
