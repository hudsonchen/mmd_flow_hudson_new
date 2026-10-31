"""Validate and time the ImageNet64 field on the actual four-H100 DDP model."""
import os
# Isolate allocated devices before importing torch, as in the production entrypoint.
devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
slot, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
assert world == len(devices) == 4
os.environ['MMD_ALLOCATED_GPU'] = devices[slot]
os.environ['CUDA_VISIBLE_DEVICES'] = devices[slot]
os.environ['LOCAL_RANK'] = '0'

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cifar_riesz_velocity
from riesz_loss import riesz_loss
from test_riesz_vectorized import compare, reference
from imagenet64_jit import JiTVelocity, sample_velocity
from imagenet64_data import ImageNet64, RandomImageBatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device,timeout=datetime.timedelta(minutes=15))
    rank = dist.get_rank()
    assert 'H100' in torch.cuda.get_device_name(device)
    report = dict(passed=False,job_id=os.environ['JOB_ID'],world_size=world,
                  global_batch=1024,local_batch=256,torch=torch.__version__,
                  tf32=torch.backends.cuda.matmul.allow_tf32,parity={},timings={})
    def record(stage):
        if rank == 0:
            report['stage'] = stage
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(report,indent=2)+'\n')
            temporary.replace(args.output)
            print(stage,flush=True)
    record('loading_checkpoint')
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert state['objective']=='coupled' and not state['lineage_has_flow_matching']
    assert len(state['rank_rng'])==4 and state['config']['checkpoint_every']==10000
    config = state['config']
    if rank == 0:
        report.update(checkpoint=str(args.checkpoint),checkpoint_update=state['update'],
            checkpoint_sha256=hashlib.file_digest(args.checkpoint.open('rb'),'sha256').hexdigest(),
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                for name in ['riesz_loss.py','cifar_riesz_velocity.py','tests/riesz_loss_reference.py']})
    hardware = dict(rank=rank,gpu=torch.cuda.get_device_name(device),
        allocated_gpu=os.environ['MMD_ALLOCATED_GPU'])
    ranks = [None]*world
    dist.all_gather_object(ranks,hardware)
    report['hardware'] = ranks
    model = JiTVelocity(4,'x',.05,True).to(device)
    model.load_state_dict(state['model'])
    ddp = DDP(model,device_ids=[0],broadcast_buffers=False,gradient_as_bucket_view=True)
    dataset = ImageNet64(config['data_root'],args.study/'training/dataset_index.json')
    batch = RandomImageBatches(dataset,256,config['seed']+7,rank,config['cfg_dropout'])[state['attempts']]
    clean = batch['images'].to(device).float()/127.5-1
    labels = torch.full((256,),int(batch['label']),device=device,dtype=torch.long)
    torch.manual_seed(319+rank)
    noise = torch.randn_like(clean)
    def gather(value):
        values = [torch.empty_like(value) for _ in range(world)]
        dist.all_gather(values,value.detach())
        return values
    for t in [.001,.2,.77,.999]:
        h = min(.01,1-t)
        source = (1-t)*noise+t*clean
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            velocity = model(source,source.new_tensor(t),labels).float()
        query = torch.cat(gather(source+h*velocity)).flatten(1)
        target = torch.cat(gather((1-t-h)*noise+(t+h)*clean)).flatten(1)
        if rank == 0:
            report['parity'][str(t)] = compare(query,target)
            record('field_parity_'+str(t))
        dist.barrier()
    del query,target,velocity
    t = .2
    source = (1-t)*noise+t*clean
    target = (1-t-.01)*noise+(t+.01)*clean
    def loss_step():
        with torch.autocast('cuda',dtype=torch.bfloat16):
            velocity = ddp(source,source.new_tensor(t),labels)
        prediction = source+.01*velocity.float()
        predictions, targets = gather(prediction),gather(target)
        loss,_ = cifar_riesz_velocity.global_particle_loss(
            prediction,predictions,targets,rank,t,.01,
            config['distance_powers'],config['tau'],config['reference_time_power'])
        assert bool(loss.isfinite())
        (loss*world).backward()
        return loss.detach()
    record('distributed_gradient_parity')
    gradients = []
    for implementation in [reference.riesz_loss,riesz_loss]:
        cifar_riesz_velocity.riesz_loss = implementation
        model.zero_grad(set_to_none=True)
        loss_step()
        gradients.append(torch.cat([p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]))
    error = (gradients[1]-gradients[0]).double().norm()/gradients[0].double().norm()
    dist.all_reduce(error,op=dist.ReduceOp.MAX)
    report['ddp_model_gradient_relative_l2'] = float(error)
    assert float(error)<.005,float(error)
    del gradients
    for name,implementation in [('reference',reference.riesz_loss),('vectorized',riesz_loss)]:
        record('benchmark_'+name)
        cifar_riesz_velocity.riesz_loss = implementation
        model.load_state_dict(state['model'])
        optimizer = torch.optim.Adam(model.parameters(),lr=.0002,betas=(.9,.95))
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        durations = []
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(5):
            dist.barrier();torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss_step()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter()-started,device=device,dtype=torch.float64)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            if iteration:durations.append(float(elapsed))
        peak = torch.tensor(torch.cuda.max_memory_allocated()/1e9,device=device)
        dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        report['timings'][name] = dict(seconds_per_step=statistics.median(durations),
            samples_seconds=durations,max_rank_peak_allocated_gb=float(peak))
        del optimizer
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    report['speedup'] = report['timings']['reference']['seconds_per_step']/report['timings']['vectorized']['seconds_per_step']
    assert report['speedup']>1.2,report['speedup']
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
    master = flat.clone();dist.broadcast(master,src=0)
    difference = (flat-master).abs().max();dist.all_reduce(difference,op=dist.ReduceOp.MAX)
    report['ddp_parameter_max_difference'] = float(difference)
    assert float(difference)<1e-6
    del flat,master
    if rank == 0:
        samples = sample_velocity(model,noise[:2],steps=2,labels=labels[:2],cfg=1.,amp=True)
        assert bool(samples.isfinite().all())
        report['finite_sampler'] = True
        report['passed'] = True
        record('complete')
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
