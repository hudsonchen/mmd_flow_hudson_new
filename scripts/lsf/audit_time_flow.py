"""Read-only checkpoint audit of teacher-state versus rollout transport."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['DRIFT_COMPILE'] = '0'
import torch
from data import load_dataset
from flow_path import TimeConditionedDiT, interpolate
from loss import _riesz_topk_softmax_loss_impl
from run_storage import new_run_path


def stats(x):
    im = x.reshape(-1, 3, 32, 32)
    smooth = torch.nn.functional.avg_pool2d(im, 2, 2).repeat_interleave(2, -1).repeat_interleave(2, -2)
    return dict(rms=float(x.square().mean().sqrt()), mean=float(x.mean()),
                highpass_rms=float((im-smooth).square().mean().sqrt()),
                outside_range=float((x.abs()>1).float().mean()),
                diversity_rms=float(x.var(0, unbiased=False).mean().sqrt()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--resume-audit', action='store_true')
    args = parser.parse_args()
    output = new_run_path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume_audit)
    source = output/'source'
    source.mkdir(exist_ok=args.resume_audit)
    for f in ['flow_path.py', 'loss.py', 'backbone.py', 'data.py', 'run_storage.py', 'scripts/lsf/audit_time_flow.py']:
        dest = source/f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/f, dest)
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    _, (raw, _), _ = load_dataset('cifar10', ROOT/'data')
    clean = raw[torch.randperm(len(raw))[:2048]].float().flatten(1)/127.5-1
    n = args.batch_size
    y = clean[:n]
    z = torch.randn_like(y)
    base = Path('/project/LongGroup/zonghaoc/mmd_flow_hudson_new/results/cifar')
    studies = ['time_dit_riesz10_pixels_rollout50_ema999_seed42_20261026',
               'time_dit_fm10_pixels_ema999_fresh_seed42_20261026']
    results = dict(seed=20261026, batch_size=n, real=stats(clean), checkpoints=[], trajectories=[],
                   caveat='Small fixed CPU diagnostic batch; not a FID estimate. Conditional paired next-state MSE is diagnostic, not the Riesz objective.')
    if args.resume_audit:
        results = json.loads((output/'audit.json').read_text())
        assert results['batch_size'] == n
    started = time.monotonic()
    def save():
        (output/'audit.json').write_text(json.dumps(results, indent=2)+'\n')
    with torch.no_grad():
        for study in studies:
            complete = {key for key in ('model','ema_model') if sum(
                r['study']==study and r['weights']==key for r in results['trajectories']) == 10}
            if len(complete)==2:
                continue
            # Atomic rename by trainers makes this open checkpoint a consistent snapshot.
            checkpoint_path = base/study/'training/checkpoint_latest.pt'
            with checkpoint_path.open('rb') as stream:
                checkpoint_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
                stream.seek(0)
                cp = torch.load(stream, map_location='cpu', weights_only=False)
            a = cp['arguments']
            prior = [r for r in results['checkpoints'] if r['study']==study]
            if prior:
                assert prior[0]['sha256'] == checkpoint_hash, 'Checkpoint changed since partial audit'
            else:
                results['checkpoints'].append(dict(study=study, path=str(checkpoint_path),
                    sha256=checkpoint_hash, update=cp['update_count'], arguments=a))
            model = TimeConditionedDiT(a['dit_width'], a['dit_depth'], a['dit_heads'], a['dit_patch_size']).eval()
            weights = {key:cp[key] for key in ('model', 'ema_model')}
            del cp
            for key, state_dict in weights.items():
                if key in complete:
                    continue
                results['trajectories'] = [r for r in results['trajectories']
                    if not (r['study']==study and r['weights']==key)]
                model.load_state_dict(state_dict)
                state = z.clone()
                for k in range(10):
                    teacher = interpolate(y, z, k/10)
                    expected = interpolate(y, z, (k+1)/10)
                    # Batch both passes together for comparable model execution.
                    inputs = torch.cat([teacher, state])
                    v = model(inputs, k/10)
                    predicted = teacher+.1*v[:n]
                    state = state+.1*v[n:]
                    row = dict(study=study, weights=key, transition=k,
                               target=stats(expected), teacher_prediction=stats(predicted), rollout=stats(state),
                               teacher_next_mse=float((predicted-expected).square().mean()),
                               identity_next_mse=float((teacher-expected).square().mean()),
                               teacher_displacement_rms=float((.1*v[:n]).square().mean().sqrt()),
                               rollout_displacement_rms=float((.1*v[n:]).square().mean().sqrt()))
                    results['trajectories'].append(row)
                    save()
                print(study, key, 'final', row, 'elapsed',time.monotonic()-started, flush=True)
            del model, weights

        # Same-marginal control measures the stochastic field floor at exact distribution match.
        # Compare to a no-movement predictor at t_k; keep production populations 128/256.
        results['field_controls'] = []
        for k in (0, 4, 8, 9):
            for mode in ('identity', 'exact_next_marginal'):
                forces = []
                for repeat in range(4):
                    # Pair the random draws between the two control modes and use
                    # the complete training population, as the production job does.
                    torch.manual_seed(20261026 + 100*k + repeat)
                    src = raw[torch.randint(len(raw), (128,))].float().flatten(1)/127.5-1
                    sn = torch.randn_like(src)
                    t = k/10 if mode=='identity' else (k+1)/10
                    gen = interpolate(src,sn,t)
                    target = raw[torch.randint(len(raw),(256,))].float().flatten(1)/127.5-1
                    target = interpolate(target,torch.randn_like(target),(k+1)/10)
                    value, info = _riesz_topk_softmax_loss_impl(gen[None],target[None],
                        tau_list=(.2,.05,.02),no_field_rms_normalization=True,epsilon=1e-6,return_field=True)
                    field = info['frozen_velocity'][0]
                    forces.append(dict(surrogate=float(value), rms=float(field.square().mean().sqrt()),
                        positive_ess=float(info['riesz_topk_softmax_positive_ess']),
                        generated_ess=float(info['riesz_topk_softmax_generated_ess']),
                        radial_dot=float((field*gen).mean())))
                row = dict(transition=k, mode=mode, repeats=forces,
                           **{key:sum(f[key] for f in forces)/len(forces) for key in forces[0]})
                results['field_controls'].append(row)
                print('field',row,flush=True)
                save()
    print('OUTPUT', output, flush=True)


if __name__ == '__main__':
    main()
