"""Allocated-device check, global-loss gradient parity, and full-width updates."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import build_backbone
from main import generator_step, update_ema, evaluation_images
from priming import reconstruction_step

torch.set_num_threads(8)
torch.manual_seed(42)
assert torch.cuda.device_count() == 4, 'Expected exactly four allocated GPUs'
names = [torch.cuda.get_device_name(i) for i in range(4)]
assert all('H100' in name for name in names), names
asset = Path(__file__).resolve().parents[2] / 'assets/pt_inception-2015-12-05-6726825d.pth'
assert hashlib.sha256(asset.read_bytes()).hexdigest() == '6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2'

# The global softmax interactions must agree with an unsplit forward/backward.
single = torch.nn.Sequential(torch.nn.Linear(24, 48), torch.nn.Sigmoid()).cuda()
parallel_raw = copy.deepcopy(single)
parallel = torch.nn.DataParallel(parallel_raw, device_ids=list(range(4)))
latent, real = torch.randn(16, 24, device='cuda'), torch.rand(32, 48, device='cuda')
for model in (single, parallel):
    generator_step(model, latent, real, torch.optim.SGD(model.parameters(), lr=0.), 64,
                   objective='softmax-riesz', normalize_per_temperature=False,
                   no_field_rms_normalization=True)
for first, second in zip(single.parameters(), parallel_raw.parameters()):
    torch.testing.assert_close(first.grad, second.grad, rtol=1e-4, atol=1e-7)
del single, parallel_raw, parallel

model = build_backbone('cifar10', 'unet', 3072, 128, [128, 256], 4).cuda()
ema = copy.deepcopy(model).eval().requires_grad_(False)
parallel = torch.nn.DataParallel(model, device_ids=list(range(4)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
clean = torch.rand(64, 3072, device='cuda')
reconstruction = reconstruction_step(parallel, clean, torch.randn_like(clean), optimizer, .8)
update_ema(model, ema, .9999)
options = dict(image_shape=(3, 32, 32), coarse_sizes=[8, 16], patch_sizes=[7, 14],
               crops_per_size=4, patch_rng=torch.Generator().manual_seed(44),
               global_weight=2., coarse_weight=1., patch_weight=1.)
times = []
for _ in range(3):
    torch.cuda.synchronize()
    started = time.monotonic()
    diagnostics = generator_step(parallel, torch.randn(128, 3072, device='cuda'),
        torch.rand(256, 3072, device='cuda'), optimizer, 64, objective='softmax-riesz',
        normalize_per_temperature=False, no_field_rms_normalization=True,
        multiscale_options=options, max_grad_norm=1.)
    update_ema(model, ema, .9999)
    images = evaluation_images(torch.nn.DataParallel(ema, device_ids=list(range(4))),
                               torch.randn(64, 3072, device='cuda'))
    assert torch.isfinite(images).all()
    assert all(torch.isfinite(torch.as_tensor(value)) for value in diagnostics.values())
    torch.cuda.synchronize()
    times.append(time.monotonic() - started)
assert all(torch.isfinite(p).all() for p in model.parameters())
assert not any(key.startswith('module.') for key in model.state_dict())
assert all(torch.cuda.max_memory_allocated(i) > 0 for i in range(4))
result = dict(status='passed', gpus=names, global_gradient_parity=True,
              checkpoint_keys_unwrapped=True, reconstruction=reconstruction,
              training_generated_batch=128, training_real_batch=256,
              field_rms_normalization='none', step_seconds=times,
              peak_gib=[torch.cuda.max_memory_allocated(i) / 2**30 for i in range(4)])
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
