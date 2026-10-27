"""Time-conditioned DiT reference flows with reconstruction-only initialization."""
import hashlib
from pathlib import Path

import torch
from flow_path import TimeConditionedDiT


class ReferenceDiT(TimeConditionedDiT):
    """Raw velocity or bounded centered clean-image prediction, selected explicitly."""
    def __init__(self, width=384, depth=12, heads=6, patch_size=2, centered_output=False):
        super().__init__(width, depth, heads, patch_size)
        self.centered_output = centered_output

    def forward(self, state, time):
        logits = super().forward(state, time)
        return 2 * logits.sigmoid() - 1 if self.centered_output else logits


def image_base_from_config(config):
    architecture = config.get('architecture', 'unet')
    if architecture == 'unet':
        from small_flow import ImageVelocity, image_geometry
        shape = image_geometry(config['dataset'])
        return ImageVelocity(config['width'], config.get('spatial_fourier', False),
                             config.get('deep', False), channels=shape[0], image_size=shape[1])
    if architecture != 'dit' or config['dataset'] != 'cifar10':
        raise ValueError('DiT reference flows require CIFAR-10')
    return ReferenceDiT(config.get('dit_width', 384), config.get('dit_depth', 12),
                        config.get('dit_heads', 6), config.get('dit_patch_size', 2),
                        centered_output=config.get('parameterization') == 'denoiser')


def verified_reconstruction_checkpoint(path, expected_sha256=None):
    """Accept only an auditable, from-scratch reconstruction checkpoint, never FM."""
    path = Path(path)
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('DiT priming checkpoint checksum mismatch')
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    config = checkpoint['config']
    if (checkpoint.get('lineage_has_flow_matching', False)
            or config.get('objective') in ['fm', 'flow-matching']
            or config.get('training_stage') != 'reconstruction'
            or config.get('dataset') != 'cifar10'
            or config.get('training_rule') not in ['paired_denoising_mse', 'paired_denoising_mse_plus_lpips']):
        raise ValueError('DiT initialization requires reconstruction-only weights, never flow matching')
    if any(config.get(key) for key in ['init_checkpoint', 'fork_checkpoint', 'fork', 'resume']):
        raise ValueError('DiT reconstruction source must have verified from-scratch ancestry')
    if checkpoint.get('ema_model') is None:
        raise ValueError('DiT priming source must contain reconstruction EMA weights')
    return checkpoint, digest


def initialize_from_reconstruction(model, path):
    """Retain the exact primed clean predictor; new time-dependent gates start at zero."""
    if not isinstance(model, ReferenceDiT) or not model.centered_output:
        raise ValueError('Reconstruction initialization requires a centered DiT clean predictor')
    checkpoint, digest = verified_reconstruction_checkpoint(path)
    weights = checkpoint['ema_model']
    result = model.load_state_dict(weights, strict=False)
    expected = {key for key in model.state_dict()
                if key.startswith(('time_embedding.', 'time_modulations.'))}
    if set(result.missing_keys) != expected or result.unexpected_keys:
        raise ValueError('Priming source does not match the DiT backbone')
    for module in model.time_modulations:
        if torch.count_nonzero(module[-1].weight) or torch.count_nonzero(module[-1].bias):
            raise ValueError('New time-dependent gates must start at zero')
    if any(not torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in weights.items()):
        raise ValueError('A transferred reconstruction tensor changed')
    return dict(source=str(Path(path).resolve()), sha256=digest,
        source_stage='reconstruction', source_rule=checkpoint['config']['training_rule'],
        source_updates=checkpoint['update_count'], source_iteration=checkpoint['iteration'],
        transferred_weights='reconstruction EMA only', transferred_tensors=len(weights),
        transferred_tensors_exact=True, new_time_modulations_zero=True,
        output_mapping='2*sigmoid(primed logits)-1; velocity=(clean_prediction-state)/(1-t)',
        optimizer_inherited=False, flow_matching_weights=False)
