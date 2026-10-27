"""Measure local pixel-Riesz gradient signal at an archived checkpoint; no updates."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['DRIFT_COMPILE'] = '0'
import torch
from reference_dit import image_base_from_config
from run_storage import new_run_path
from small_flow import DenoiserVelocity, marginal_loss, reference_centered_loss, validate_riesz_parent
from scripts.lsf.run_small_image_flow import load_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--times', type=float, nargs='+', default=[.1, .5, .9, .99])
    parser.add_argument('--horizons', type=float, nargs='+', default=[.01, .1])
    parser.add_argument('--include-centered', action='store_true')
    parser.add_argument('--self-normalization-variants', action='store_true')
    args = parser.parse_args()
    if args.batch < 2 or args.repeats < 1 or any(not 0 <= t < 1 for t in args.times) or any(h <= 0 for h in args.horizons):
        parser.error('Invalid diagnostic batch, repeats, times or horizons')
    output = new_run_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    payload = args.checkpoint.read_bytes()
    (output / 'checkpoint.pt').write_bytes(payload)
    checkpoint = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    validate_riesz_parent(checkpoint)
    cfg = checkpoint['config']
    if checkpoint.get('encoder') is not None or cfg.get('prime_steps') or cfg.get('vgg_endpoint_weight'):
        raise ValueError('This diagnostic requires a pixel-only checkpoint without priming')
    torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    model = image_base_from_config(cfg)
    if cfg.get('parameterization') == 'denoiser':
        model = DenoiserVelocity(model)
    model.load_state_dict(checkpoint['ema'])
    model.eval()
    parameters = tuple(model.parameters())
    train, _ = load_images(cfg['dataset'], new_run_path('datasets'))
    train = train.flatten(1).float() / 127.5 - 1
    options = dict(taus=cfg['tau'], normalization=cfg['normalization'], affinity=cfg['affinity'],
                   repulsion_weight=cfg['repulsion_weight'], time_weight_power=cfg['reference_time_power'])

    def gradient(loss):
        values = torch.autograd.grad(loss, parameters)
        vector = torch.cat([v.flatten() for v in values]).detach()
        if not torch.isfinite(vector).all():
            raise RuntimeError('Nonfinite diagnostic gradient')
        return vector

    def cosine(a, b):
        a,b=a.double(),b.double()
        return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))

    rows = []
    for repeat in range(args.repeats):
        torch.manual_seed(913771 + repeat)
        clean, target = [train[torch.randint(len(train), (args.batch,))] for _ in range(2)]
        noise, target_noise = torch.randn_like(clean), torch.randn_like(target)
        for t in args.times:
            state = (1-t) * noise + t * clean
            # Analytic direction only: no learned teacher, optimizer or MSE training.
            fm_gradient = gradient((model(state, t) - (clean-noise)).square().mean())
            estimates = {}
            for h in args.horizons:
                for coupling in ['independent', 'paired-loo', 'paired']:
                    paired = coupling != 'independent'
                    loss = marginal_loss(model, clean, noise, clean if paired else target,
                                         noise if paired else target_noise, t, h,
                                         exclude_target_diagonal=coupling == 'paired-loo', **options)
                    g = gradient(loss)
                    row = dict(repeat=repeat, time=t, horizon=h, actual_transition=min(h,1-t),
                               coupling=coupling, gradient_norm=float(g.norm()),
                               analytic_velocity_mse_gradient_norm=float(fm_gradient.norm()),
                               cosine_with_analytic_velocity_mse=cosine(g, fm_gradient))
                    estimates[(h, coupling)] = g
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                if args.self_normalization_variants:
                    for name,smoothing,normalization in [('symmetric',0.,cfg['normalization']),
                            ('symmetric-smooth',.05,cfg['normalization']),('symmetric-smooth-no-rms',.05,'none')]:
                        variant=dict(options,include_self_negative=True,distance_smoothing=smoothing,normalization=normalization)
                        value=marginal_loss(model,clean,noise,clean,noise,t,h,**variant)
                        g=gradient(value)
                        row=dict(repeat=repeat,time=t,horizon=h,coupling='paired',variant=name,
                            gradient_norm=float(g.norm()),cosine_with_analytic_velocity_mse=cosine(g,fm_gradient))
                        rows.append(row)
                        print(json.dumps(row),flush=True)
                if args.include_centered:
                    loss = reference_centered_loss(model, clean, noise, target, target_noise,
                                                   target, target_noise, t, h,
                                                   taus=cfg['tau'], normalization=cfg['normalization'])
                    loss = loss * (1-t)**cfg['reference_time_power']
                    g = gradient(loss)
                    row = dict(repeat=repeat, time=t, horizon=h, actual_transition=min(h,1-t),
                               coupling='centered-shared-supports', affinity='row',
                               gradient_norm=float(g.norm()),
                               analytic_velocity_mse_gradient_norm=float(fm_gradient.norm()),
                               cosine_with_analytic_velocity_mse=cosine(g, fm_gradient))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
            for coupling in ['independent', 'paired-loo', 'paired']:
                if len(args.horizons) == 2:
                    rows.append(dict(repeat=repeat, time=t, coupling=coupling,
                                     horizons=args.horizons,
                                     cosine_between_horizons=cosine(estimates[(args.horizons[0], coupling)],
                                                                   estimates[(args.horizons[1], coupling)])))
    files = ['run_storage.py', 'small_flow.py', 'reference_dit.py', 'flow_path.py', 'backbone.py',
             'loss.py', 'sinkhorn.py', 'data.py', 'fid.py', '_fid_inception.py', 'perceptual.py',
             'scripts/lsf/run_small_flow.py', 'scripts/lsf/run_small_image_flow.py',
             'scripts/lsf/diagnose_local_pixel_flow.py']
    hashes = {}
    for name in files:
        dest = output / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    report = dict(checkpoint_source=str(args.checkpoint), checkpoint_update=checkpoint['update'],
                  checkpoint_sha256=hashlib.sha256(payload).hexdigest(), source_hashes=hashes,
                  config=vars(args), rows=rows, optimizer_updates=0, device='cpu',
                  interpretation='Analytic velocity-MSE gradients are a diagnostic reference only. '
                  'Alignment on these finite minibatches is not a proof of transport correctness or image quality. '
                  'No encoder, pretrained teacher, parameter updates, or off-schedule FID.')
    (output / 'diagnostics.json').write_text(json.dumps(report, indent=2, default=str)+'\n')


if __name__ == '__main__':
    main()
