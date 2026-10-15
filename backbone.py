"""Shared backbones for one-pass MNIST and CIFAR-10 image generation.

CIFAR DDPM++ blocks/topology adapted from Google Research's
yang-song/score_sde_pytorch (Copyright 2020 The Google Research Authors,
Apache-2.0; see assets/SCORE_SDE_LICENSE). Modifications omit time embedding,
use a sigmoid image head, and implement attention with PyTorch SDPA.
"""

import math
import torch


class ParticleGenerator(torch.nn.Module):
    """Small shared MLP mapping Gaussian latents to flattened image pixels."""

    def __init__(self, latent_dim: int = 32, hidden_dims: tuple[int, ...] = (128, 256), output_dim: int = 784):
        super().__init__()
        layers = []
        width = latent_dim
        for hidden in hidden_dims:
            layers.extend([torch.nn.Linear(width, hidden), torch.nn.SiLU()])
            width = hidden
        layers.extend([torch.nn.Linear(width, output_dim), torch.nn.Sigmoid()])
        self.network = torch.nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)



class ResidualConvBlock(torch.nn.Module):
    """Per-image normalization and residual convolutions for spatial features."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.GroupNorm(math.gcd(8, in_channels), in_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
            torch.nn.GroupNorm(math.gcd(8, out_channels), out_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (torch.nn.Identity() if in_channels == out_channels
                     else torch.nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (self.layers(features) + self.skip(features)) / math.sqrt(2)



class UNetParticleGenerator(torch.nn.Module):
    """Compact U-Net adaptation for one-pass generation, without diffusion time."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        width = base_channels
        self.input_conv = torch.nn.Conv2d(1, width, 3, padding=1)
        self.encoder28 = ResidualConvBlock(width, width)
        self.down14 = torch.nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.encoder14 = ResidualConvBlock(2 * width, 2 * width)
        self.down7 = torch.nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.bottleneck = torch.nn.Sequential(
            ResidualConvBlock(4 * width, 4 * width),
            ResidualConvBlock(4 * width, 4 * width),
        )
        self.up14 = torch.nn.Conv2d(4 * width, 2 * width, 3, padding=1)
        self.decoder14 = ResidualConvBlock(4 * width, 2 * width)
        self.up28 = torch.nn.Conv2d(2 * width, width, 3, padding=1)
        self.decoder28 = ResidualConvBlock(2 * width, width)
        self.output = torch.nn.Sequential(
            torch.nn.GroupNorm(math.gcd(8, width), width),
            torch.nn.SiLU(), torch.nn.Conv2d(width, 1, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        noise = latent.reshape(-1, 1, 28, 28)
        skip28 = self.encoder28(self.input_conv(noise))
        skip14 = self.encoder14(self.down14(skip28))
        features = self.bottleneck(self.down7(skip14))
        features = self.up14(torch.nn.functional.interpolate(
            features, size=skip14.shape[-2:], mode="bilinear", align_corners=False))
        features = self.decoder14(torch.cat([features, skip14], dim=1))
        features = self.up28(torch.nn.functional.interpolate(
            features, size=skip28.shape[-2:], mode="bilinear", align_corners=False))
        features = self.decoder28(torch.cat([features, skip28], dim=1))
        return self.output(features).flatten(1)



def ddpm_conv(in_channels, out_channels, kernel_size=3, init_scale=1.0):
    """DDPM fan-average uniform initialization, including its near-zero branches."""
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                           padding=kernel_size // 2)
    scale = 1e-10 if init_scale == 0 else init_scale
    bound = math.sqrt(3 * scale / ((in_channels + out_channels) * kernel_size ** 2 / 2))
    torch.nn.init.uniform_(conv.weight, -bound, bound)
    torch.nn.init.zeros_(conv.bias)
    return conv


def ddpm_norm(channels):
    return torch.nn.GroupNorm(math.gcd(min(channels // 4, 32), channels), channels, eps=1e-6)


class ResidualBlock(torch.nn.Module):
    """DDPM++ BigGAN block with optional residual up/downsampling, without time embedding.

    Adapted from score_sde_pytorch/models/layerspp.py (Apache-2.0).
    """
    def __init__(self, in_channels, out_channels=None, *, up=False, down=False, dropout=0.1,
                 conditioning_dim=0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.up, self.down = up, down
        self.norm1 = ddpm_norm(in_channels)
        self.conv1 = ddpm_conv(in_channels, out_channels)
        self.norm2 = ddpm_norm(out_channels)
        self.conditioning = (torch.nn.Linear(conditioning_dim, 2 * out_channels)
                             if conditioning_dim else None)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ddpm_conv(out_channels, out_channels, init_scale=0)
        self.skip = (ddpm_conv(in_channels, out_channels, 1)
                     if in_channels != out_channels or up or down else torch.nn.Identity())

    def forward(self, x, conditioning=None):
        h = torch.nn.functional.silu(self.norm1(x))
        if self.up:
            h = torch.nn.functional.interpolate(h, scale_factor=2, mode='nearest')
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        elif self.down:
            h = torch.nn.functional.avg_pool2d(h, 2)
            x = torch.nn.functional.avg_pool2d(x, 2)
        h = self.conv1(h)
        h = self.norm2(h)
        if self.conditioning is not None:
            if conditioning is None:
                raise ValueError('Conditional residual block needs class/CFG embedding')
            scale, shift = self.conditioning(torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(torch.nn.functional.silu(h)))
        return (self.skip(x) + h) / math.sqrt(2)


class SpatialAttention(torch.nn.Module):
    """Single-head DDPM++ attention with near-zero output projection."""
    def __init__(self, channels):
        super().__init__()
        self.norm = ddpm_norm(channels)
        self.q = ddpm_conv(channels, channels, 1, init_scale=0.1)
        self.k = ddpm_conv(channels, channels, 1, init_scale=0.1)
        self.v = ddpm_conv(channels, channels, 1, init_scale=0.1)
        self.projection = ddpm_conv(channels, channels, 1, init_scale=0)

    def forward(self, x):
        batch, channels, height, width = x.shape
        normalized = self.norm(x)
        q, k, v = [layer(normalized).flatten(2) for layer in (self.q, self.k, self.v)]
        attended = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).unsqueeze(1), k.transpose(1, 2).unsqueeze(1),
            v.transpose(1, 2).unsqueeze(1))
        attended = attended.squeeze(1).transpose(1, 2).reshape(batch, channels, height, width)
        return (x + self.projection(attended)) / math.sqrt(2)


class CIFAR10UNetParticleGenerator(torch.nn.Module):
    """Official CIFAR DDPM++ spatial topology adapted to one-pass image generation.

    Reference: yang-song/score_sde_pytorch models/ncsnpp.py and
    configs/vp/cifar10_ddpmpp.py. No time embedding; sigmoid image output.
    """
    def __init__(self, base_channels: int = 128, num_res_blocks: int = 4, num_classes: int = 0):
        super().__init__()
        self.num_classes = num_classes
        conditioning_dim = 4 * base_channels if num_classes else 0
        if num_classes:
            self.class_embedding = torch.nn.Embedding(num_classes, conditioning_dim)
            self.cfg_embedding = torch.nn.Sequential(torch.nn.Linear(32, conditioning_dim),
                                                     torch.nn.SiLU(),
                                                     torch.nn.Linear(conditioning_dim, conditioning_dim))
            self.register_buffer('cfg_frequencies', torch.exp(-math.log(10000) * torch.arange(16) / 16))
        widths = [base_channels * multiplier for multiplier in (1, 2, 2, 2)]
        self.input_conv = ddpm_conv(3, widths[0])
        self.encoders = torch.nn.ModuleList()
        self.downsamples = torch.nn.ModuleList()
        skip_channels = [widths[0]]
        channels = widths[0]
        for level, width in enumerate(widths):
            stage = torch.nn.ModuleList()
            for _ in range(num_res_blocks):
                stage.append(torch.nn.Sequential(
                    ResidualBlock(channels, width, conditioning_dim=conditioning_dim),
                    SpatialAttention(width) if level == 1 else torch.nn.Identity()))
                channels = width
                skip_channels.append(channels)
            self.encoders.append(stage)
            if level < 3:
                self.downsamples.append(ResidualBlock(channels, down=True, conditioning_dim=conditioning_dim))
                skip_channels.append(channels)
        self.middle = torch.nn.Sequential(ResidualBlock(channels, conditioning_dim=conditioning_dim),
                                          SpatialAttention(channels),
                                          ResidualBlock(channels, conditioning_dim=conditioning_dim))
        self.decoders = torch.nn.ModuleList()
        self.decoder_attention = torch.nn.ModuleList()
        self.upsamples = torch.nn.ModuleList()
        for level in (3, 2, 1, 0):
            stage = torch.nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                stage.append(ResidualBlock(channels + skip_channels.pop(), widths[level],
                                           conditioning_dim=conditioning_dim))
                channels = widths[level]
            self.decoders.append(stage)
            self.decoder_attention.append(SpatialAttention(channels) if level == 1 else torch.nn.Identity())
            if level > 0:
                self.upsamples.append(ResidualBlock(channels, up=True, conditioning_dim=conditioning_dim))
        assert not skip_channels
        self.output = torch.nn.Sequential(ddpm_norm(channels), torch.nn.SiLU(),
                                          ddpm_conv(channels, 3, init_scale=0), torch.nn.Sigmoid())

    def forward(self, latent, labels=None, cfg_scale=None):
        conditioning = None
        if self.num_classes:
            if labels is None or cfg_scale is None:
                raise ValueError('Conditional generator needs labels and CFG scales')
            labels = torch.as_tensor(labels, device=latent.device, dtype=torch.long)
            cfg_scale = torch.as_tensor(cfg_scale, device=latent.device, dtype=latent.dtype).expand(len(latent))
            if labels.shape != (len(latent),) or (labels < 0).any() or (labels >= self.num_classes).any():
                raise ValueError('Expected one valid class label per latent')
            if not torch.isfinite(cfg_scale).all() or (cfg_scale < 1).any():
                raise ValueError('CFG scales must be finite and at least one')
            angles = cfg_scale[:, None] * self.cfg_frequencies[None] * 1000
            conditioning = self.class_embedding(labels) + self.cfg_embedding(
                torch.cat([angles.cos(), angles.sin()], dim=1))
        elif labels is not None or cfg_scale is not None:
            raise ValueError('Unconditional generator does not accept class/CFG inputs')
        x = self.input_conv(latent.reshape(-1, 3, 32, 32))
        skips = [x]
        for level, stage in enumerate(self.encoders):
            for block in stage:
                x = block[1](block[0](x, conditioning))
                skips.append(x)
            if level < 3:
                x = self.downsamples[level](x, conditioning)
                skips.append(x)
        for block in self.middle:
            x = block(x, conditioning) if isinstance(block, ResidualBlock) else block(x)
        for level, stage in enumerate(self.decoders):
            for block in stage:
                x = block(torch.cat([x, skips.pop()], dim=1), conditioning)
            x = self.decoder_attention[level](x)
            if level < 3:
                x = self.upsamples[level](x, conditioning)
        assert not skips
        return self.output(x).flatten(1)



def build_backbone(dataset, architecture, latent_dim, base_channels, hidden_dims, num_res_blocks,
                   num_classes=0):
    """Construct an architecture without dataset-specific trainer entry points."""
    if architecture == "mlp":
        return ParticleGenerator(latent_dim, tuple(hidden_dims), 784 if dataset == "mnist" else 3072)
    if architecture != "unet":
        raise ValueError(f"Unknown architecture: {architecture}")
    if dataset == "mnist":
        return UNetParticleGenerator(base_channels)
    if dataset == "cifar10":
        return CIFAR10UNetParticleGenerator(base_channels, num_res_blocks, num_classes=num_classes)
    raise ValueError(f"Unknown dataset: {dataset}")


def backbone_metadata(dataset, architecture, latent_dim, base_channels, hidden_dims, num_res_blocks):
    if architecture == "mlp":
        return [latent_dim, *hidden_dims, 784 if dataset == "mnist" else 3072]
    metadata = {
        "name": "compact_residual_unet" if dataset == "mnist" else "cifar10_ddpmpp_onepass_unet",
        "channels": [base_channels, 2 * base_channels, 4 * base_channels] if dataset == "mnist"
                    else [base_channels, 2 * base_channels, 2 * base_channels, 2 * base_channels],
        "resolutions": [28, 14, 7] if dataset == "mnist" else [32, 16, 8, 4],
        "skip_connections": "concatenation", "normalization": "GroupNorm",
        "upsampling": "bilinear_then_conv", "input_shape": [1, 28, 28] if dataset == "mnist" else [3, 32, 32],
        "time_conditioning": False, "sampling_network_evaluations": 1,
    }
    if dataset == "cifar10":
        metadata.update(residual_blocks_per_stage=num_res_blocks,
                        decoder_residual_blocks_per_stage=num_res_blocks + 1,
                        decoder_resolutions=[4, 8, 16, 32],
                        skip_connections="concatenation_after_each_encoder_block_and_downsample",
                        upsampling="biggan_residual_nearest", downsampling="biggan_residual_average_pool",
                        attention_resolutions=[16], bottleneck_attention=True,
                        encoder_attention="after_each_residual_block_at_16",
                        decoder_attention="after_stage_at_16", residual_block_type="biggan",
                        dropout=0.1, groupnorm_eps=1e-6, skip_rescale=True, fir=False,
                        initialization="ddpm_fan_average_uniform_near_zero_output_branches",
                        output_activation="sigmoid",
                        adaptations=["omit_time_embedding", "sigmoid_image_output"],
                        scale_reference="https://github.com/yang-song/score_sde_pytorch/blob/main/configs/vp/cifar10_ddpmpp.py")
    return metadata
