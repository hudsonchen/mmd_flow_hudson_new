"""H100 correctness and timing gate using a real Tiny ImageNet checkpoint/batch."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import torch
import cifar_riesz_velocity
from riesz_loss import riesz_loss
from test_riesz_vectorized import compare, reference
from imagenet64_jit import JiTVelocity, sample_velocity
from imagenet64_data import ImageNet64, RandomImageBatches
from training_objectives import training_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(passed=False, stage='initializing', job_id=os.environ.get('JOB_ID'))
    def record(stage):
        report['stage'] = stage
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output)
        print(stage, flush=True)
    record('initializing')
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    # Strict FP32 GEMMs; do not trade field accuracy for TF32 throughput.
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    assert 'H100' in torch.cuda.get_device_name(device)
    report.update(gpu=torch.cuda.get_device_name(device), torch=torch.__version__,
                  tf32=torch.backends.cuda.matmul.allow_tf32,
                  checkpoint=str(args.checkpoint),
                  checkpoint_sha256=hashlib.file_digest(args.checkpoint.open('rb'),'sha256').hexdigest(),
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                 for name in ['riesz_loss.py','training_objectives.py','tests/riesz_loss_reference.py']})
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert state['objective'] == 'riesz' and not state['lineage_has_flow_matching']
    report['checkpoint_update'] = state['update']
    config = state['config']
    model = JiTVelocity(4, 'x', .05, True).to(device)
    model.load_state_dict(state['model'])
    dataset = ImageNet64(config['data_root'])
    batch = RandomImageBatches(dataset,1024,7,dropout=config['cfg_dropout'])[state['attempts']]
    clean = batch['images'].to(device).float()/127.5-1
    labels = torch.full((1024,),int(batch['label']),device=device,dtype=torch.long)
    torch.manual_seed(319)
    noise = torch.randn_like(clean)
    report['parity'] = {}
    record('field_parity')
    for t in [.001, .2, .77, .999]:
        h = min(.01, 1-t)
        source = (1-t)*noise+t*clean
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            velocity = model(source,source.new_tensor(t),labels).float()
        query, target = source+h*velocity, (1-t-h)*noise+(t+h)*clean
        report['parity'][str(t)] = compare(query.flatten(1),target.flatten(1))
        record('field_parity_'+str(t))
    # Include degenerate/duplicate particles on the allocated GPU.
    duplicates = torch.randn(33,96,device=device)
    duplicates[1] = duplicates[0]
    compare(duplicates,duplicates.clone())
    compare(torch.ones(9,64,device=device),torch.ones(7,64,device=device))
    t = .2
    source = (1-t)*noise+t*clean
    gradients = []
    record('model_gradient_parity')
    for implementation in [reference.riesz_loss, riesz_loss]:
        cifar_riesz_velocity.riesz_loss = implementation
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            velocity = model(source,source.new_tensor(t),labels)
        loss = training_loss(velocity,source,clean,noise,t,.01,'riesz')
        loss.backward()
        gradients.append(torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]))
    gradient_error = float((gradients[1]-gradients[0]).double().norm()/gradients[0].double().norm())
    report['model_gradient_relative_l2'] = gradient_error
    # BF16 backward adds rounding at intermediate layers; bound its aggregate impact.
    assert gradient_error < .005, gradient_error
    del gradients, velocity, loss
    report['timings'] = {}
    for name,implementation,objective in [('reference',reference.riesz_loss,'riesz'),
                                         ('vectorized',riesz_loss,'riesz'),
                                         ('flow_matching',riesz_loss,'flow_matching')]:
        record('benchmark_'+name)
        cifar_riesz_velocity.riesz_loss = implementation
        model.load_state_dict(state['model'])
        optimizer = torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.95))
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        durations = []
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                velocity = model(source,source.new_tensor(t),labels)
            loss = training_loss(velocity,source,clean,noise,t,.01,objective)
            assert bool(loss.isfinite())
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            if iteration:
                durations.append(time.perf_counter()-started)
        report['timings'][name] = dict(seconds_per_step=statistics.median(durations),
            samples_seconds=durations,peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9)
        del optimizer,velocity,loss
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    report['speedup'] = report['timings']['reference']['seconds_per_step']/report['timings']['vectorized']['seconds_per_step']
    assert report['speedup'] > 1.2, report['speedup']
    cifar_riesz_velocity.riesz_loss = riesz_loss
    samples = sample_velocity(model,noise[:2],steps=2,labels=labels[:2],cfg=1.,amp=True)
    assert bool(samples.isfinite().all())
    report['finite_sampler'] = True
    report['passed'] = True
    record('complete')


if __name__ == '__main__':
    main()
