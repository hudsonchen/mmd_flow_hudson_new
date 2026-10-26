"""Check the full-width tuned batch on its allocated GPU before training."""
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from backbone import build_backbone
from main import generator_step, horizontal_flip_batch, update_ema

torch.set_num_threads(4)
torch.manual_seed(42)
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
model = build_backbone('cifar10', 'unet', 3072, 128, [128, 256], 4).cuda()
ema = copy.deepcopy(model).eval().requires_grad_(False)
learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4
objective = sys.argv[3] if len(sys.argv) > 3 else 'softmax-riesz'
no_field_rms_normalization = len(sys.argv) > 4 and sys.argv[4] == 'none'
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
options = dict(image_shape=(3, 32, 32), coarse_sizes=[8, 16], patch_sizes=[7, 14],
               crops_per_size=4, patch_rng=torch.Generator().manual_seed(44),
               global_weight=2., coarse_weight=1., patch_weight=1.)
started = time.monotonic()
for _ in range(2):
    real = horizontal_flip_batch(torch.rand(256, 3072, device='cuda'), (3, 32, 32))
    diagnostics = generator_step(model, torch.randn(128, 3072, device='cuda'), real,
                                 optimizer, 64, objective=objective,
                                 no_field_rms_normalization=no_field_rms_normalization,
                                 multiscale_options=options, max_grad_norm=1.)
    update_ema(model, ema, .9999)
    assert all(torch.isfinite(value) for value in diagnostics.values())
assert all(torch.isfinite(p).all() for p in model.parameters())
torch.cuda.synchronize()
result = dict(status='passed', training_generated_batch=128, real_batch=256,
              learning_rate=learning_rate, objective=objective,
              field_rms_normalization='none' if no_field_rms_normalization else 'per_temperature',
              diagnostics={key: float(value) for key, value in diagnostics.items()},
              crops_per_size=4, gpu=torch.cuda.get_device_name(),
              peak_gib=torch.cuda.max_memory_allocated() / 2**30,
              seconds_per_update=(time.monotonic() - started) / 2)
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
