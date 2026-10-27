"""Frozen VGG for paired LPIPS and unpaired feature distribution matching."""
import hashlib
from importlib.metadata import version
from pathlib import Path

import torch


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, cache, device='cpu'):
        super().__init__()
        import lpips

        if cache is None:
            raise ValueError('LPIPS requires an explicit cache directory')
        cache = Path(cache).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        previous_hub = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(cache / 'hub'))
            # Torchvision initializes layers before loading weights. Preserve RNG.
            with torch.random.fork_rng(devices=[]):
                self.metric = lpips.LPIPS(net='vgg', version='0.1', pretrained=True,
                                          pnet_rand=False, pnet_tune=False,
                                          eval_mode=True, verbose=False)
        finally:
            torch.hub.set_dir(previous_hub)
        self.requires_grad_(False).eval()
        digest = hashlib.sha256()
        for name, tensor in sorted(self.metric.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.numpy().tobytes())
        self.metadata = dict(package='lpips', package_version=version('lpips'),
                             torchvision_version=version('torchvision'),
                             network='vgg16', lpips_version='0.1',
                             weights_sha256=digest.hexdigest(), frozen=True,
                             input='native CIFAR RGB 32x32; [0,1] converted to [-1,1]',
                             resize=False, cache=str(cache))
        self.to(device)

    def forward(self, prediction, target):
        if prediction.shape != target.shape or prediction[0].numel() != 3072:
            raise ValueError('Perceptual priming requires matching CIFAR RGB images')
        # Frozen weights preserve the differentiable path to generated images.
        return self.metric(prediction.reshape(-1, 3, 32, 32),
                           target.detach().reshape(-1, 3, 32, 32), normalize=True).mean()

    def distribution_features(self, images, pool_size=4, batch_size=32):
        """LPIPS-normalized VGG activations, spatially pooled per layer.

        Uses the five VGG feature slices, not LPIPS's learned pairwise heads.
        Pooling caps spatial dimensions without upsampling small feature maps.
        """
        import lpips
        from torch.nn import functional as F
        if images.ndim != 2 or images.shape[1] != 3072 or len(images) == 0:
            raise ValueError('VGG distribution features require flattened CIFAR RGB images')
        if pool_size < 1 or batch_size < 1:
            raise ValueError('Feature pool and batch sizes must be positive')
        self.metric.eval()
        layers = None
        for batch in images.split(batch_size):
            centered = 2 * batch.reshape(-1, 3, 32, 32) - 1
            outputs = self.metric.net(self.metric.scaling_layer(centered))
            if layers is None:
                layers = [[] for _ in outputs]
            for values, feature in zip(layers, outputs):
                feature = lpips.normalize_tensor(feature)
                size = tuple(min(pool_size, s) for s in feature.shape[-2:])
                values.append(F.adaptive_avg_pool2d(feature, size).flatten(1))
        return [torch.cat(values) for values in layers]

    def flow_features(self, images, pool_size=4):
        """One embedding: equal layer weights and spatial averaging in squared distance."""
        layers = self.distribution_features(images, pool_size)
        # The native VGG slices have spatial sizes 32,16,8,4,2 for CIFAR.
        spatial_sizes = [min(pool_size, side) for side in (32, 16, 8, 4, 2)]
        if len(layers) != len(spatial_sizes):
            raise ValueError('Expected the five VGG16 LPIPS feature slices')
        return torch.cat([feature / (side * len(layers)**.5)
                          for feature, side in zip(layers, spatial_sizes)], dim=1)

    def multiscale_features(self, images, pool_size=8, batch_size=32):
        """Separate spatial, local-statistic and global-statistic VGG branches.

        Statistics are computed before spatial pooling on channel-normalized
        activations. Each branch receives its own distance scale and neighbors
        in the distributional loss. No learned LPIPS pairwise heads are used.
        """
        import lpips
        from torch.nn import functional as F
        if images.ndim != 2 or images.shape[1] != 3072 or len(images) == 0:
            raise ValueError('Expected flattened CIFAR RGB images')
        if pool_size < 1 or batch_size < 1:
            raise ValueError('Pool and batch sizes must be positive')
        self.metric.eval()
        branches = {}
        for batch in images.split(batch_size):
            centered = 2 * batch.reshape(-1, 3, 32, 32) - 1
            for index, feature in enumerate(self.metric.net(self.metric.scaling_layer(centered))):
                feature = lpips.normalize_tensor(feature)
                side = min(pool_size, *feature.shape[-2:])
                spatial = F.adaptive_avg_pool2d(feature, side).flatten(1)
                local_mean = F.avg_pool2d(feature, 2, 2)
                local_var = (F.avg_pool2d(feature.square(), 2, 2) - local_mean.square()).clamp_min(0)
                local_std = (local_var + 1e-8).sqrt()
                patch_side = min(4, *local_mean.shape[-2:])
                patches = torch.cat([F.adaptive_avg_pool2d(v, patch_side).flatten(1)
                                     for v in (local_mean, local_std)], dim=1)
                global_mean = feature.mean((-2, -1))
                global_std = (feature.var((-2, -1), unbiased=False) + 1e-8).sqrt()
                for name, value in [('spatial', spatial), ('patch_stats', patches),
                                    ('global_stats', torch.cat([global_mean, global_std], dim=1))]:
                    branches.setdefault(f'layer{index + 1}_{name}', []).append(value)
        return {name: torch.cat(values) for name, values in branches.items()}


def multiscale_vgg_loss(generated, real, *, tau_list, normalize_per_temperature,
                        no_field_rms_normalization, repulsion_weight):
    """Equally weight the 15 independently scaled feature distribution losses."""
    from loss import _riesz_topk_softmax_loss_impl, raw_field_diagnostics
    if generated.keys() != real.keys() or not generated:
        raise ValueError('Generated and real feature branches must match')
    losses, diagnostics = [], {}
    for name, features in generated.items():
        value, info = _riesz_topk_softmax_loss_impl(
            features.unsqueeze(0), real[name].detach().unsqueeze(0),
            tau_list=tau_list, normalize_per_temperature=normalize_per_temperature,
            no_field_rms_normalization=no_field_rms_normalization,
            repulsion_weight=repulsion_weight, epsilon=1e-6)
        losses.append(value.mean())
        values = raw_field_diagnostics(info)
        for key, value in values.items():
            diagnostics[key] = diagnostics.get(key, 0) + value / len(generated)
        diagnostics[f'raw_{name}_force_rms'] = values['raw_force_rms']
        diagnostics[f'raw_{name}_positive_ess'] = info['riesz_topk_softmax_positive_ess']
        diagnostics[f'raw_{name}_generated_ess'] = info['riesz_topk_softmax_generated_ess']
    return torch.stack(losses).mean(), diagnostics
