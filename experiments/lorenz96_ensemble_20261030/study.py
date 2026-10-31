"""Reproducible exploratory L96 ensemble comparison; outputs never enter source."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from core import ForecastFlow, integrate, objective, sample, scores

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def status(out, stage, **kwargs):
    record = dict(stage=stage, updated_at=dt.datetime.now(dt.timezone.utc).isoformat(), **kwargs)
    write_json(out/'status.json', record)
    print(json.dumps(record), flush=True)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def prepare(out, c):
    path = out/'data.npz'
    if path.exists():
        info = json.loads((out/'data_identity.json').read_text())
        if sha(path) != info['sha256']:
            raise RuntimeError('Dataset hash mismatch')
        return
    start = time.perf_counter()
    rng = np.random.default_rng(c['data_seed'])
    arrays = {}
    for split in ['train', 'validation', 'test']:
        status(out, 'prepare_data', split=split)
        count, refs = c[split+'_contexts'], c[split+'_references']
        # Each context starts from an independent random state and its own
        # spin-up trajectory. Splits never share segments of one trajectory.
        initial = c['forcing']+rng.normal(size=(count, c['dimension']))
        context = integrate(initial, [c['burn_time']], .01, c['forcing'])[:, 0]
        futures, controls = [], []
        for lo in range(0, count, 32):
            ctx = context[lo:lo+32]
            perturbed = ctx[:, None]+c['initial_sigma']*rng.normal(size=(len(ctx), refs, c['dimension']))
            futures.append(integrate(perturbed, c['lead_times'], c['dt'], c['forcing']).transpose(0, 2, 1, 3))
            controls.append(integrate(ctx, c['lead_times'], c['dt'], c['forcing']))
        arrays[split+'_context'] = context.astype('float32')
        arrays[split+'_future'] = np.concatenate(futures).astype('float32')
        arrays[split+'_control'] = np.concatenate(controls).astype('float32')
    # One scalar normalization preserves the cyclic spatial symmetry.
    arrays['center'] = np.asarray(arrays['train_future'].mean(), np.float32)
    arrays['scale'] = np.asarray(arrays['train_future'].std(), np.float32)
    arrays['event_threshold'] = np.quantile(arrays['train_future'].max(-1), .95, axis=(0,2))
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError('Nonfinite simulator data')
    # Independent numerical validation on perturbed held-out conditions.
    check = arrays['test_context'][:4].astype('float64')+rng.normal(scale=c['initial_sigma'],size=(4,c['dimension']))
    coarse = integrate(check, c['lead_times'], c['dt'], c['forcing'])
    fine = integrate(check, c['lead_times'], c['dt']/2, c['forcing'])
    finer = integrate(check, c['lead_times'], c['dt']/4, c['forcing'])
    error = float(np.max(np.abs(coarse-fine)))
    if error > 1e-3 or np.max(np.abs(fine-finer)) > 1e-4:
        raise RuntimeError('Reference integrator refinement failed')
    from scipy.integrate import solve_ivp
    from core import rhs
    oracle = solve_ivp(lambda t,x: rhs(x,c['forcing']), (0,max(c['lead_times'])), check[0],
        method='DOP853',t_eval=c['lead_times'],rtol=1e-11,atol=1e-12)
    oracle_error = float(np.max(np.abs(coarse[0]-oracle.y.T)))
    if not oracle.success or oracle_error > 1e-3:
        raise RuntimeError('Independent DOP853 comparison failed')
    np.savez(path, **arrays)
    write_json(out/'data_identity.json', dict(sha256=sha(path), config=c,
        shapes={k:list(v.shape) for k,v in arrays.items()},seconds=time.perf_counter()-start,
        rk4_halving_max_abs=error, dop853_max_abs=oracle_error,
        uncertainty='X_initial | context ~ N(context, initial_sigma^2 I); deterministic Lorenz-96 thereafter',
        split='Independent initial draws and spin-up trajectories; no trajectory shared between splits',
        test_partition='First half reference outcomes; next ensemble_size independent oracle members'))


def training_batch(data, c, rng, device):
    b, n = c['contexts_per_batch'], c['particles_per_context']
    ci = torch.randint(c['train_contexts'], (b,), generator=rng, device=device)
    li = torch.randint(len(c['lead_times']), (b,), generator=rng, device=device)
    ri = torch.randint(c['train_references'], (b,n), generator=rng, device=device)
    clean = data['future'][ci[:,None],li[:,None],ri]
    context = data['context'][ci]
    # Exactly matched random cyclic augmentation for all methods in each seed.
    shifts = torch.randint(c['dimension'], (b,1), generator=rng, device=device)
    indices = (torch.arange(c['dimension'],device=device)[None]+shifts)%c['dimension']
    context = context.gather(1,indices)
    clean = clean.gather(2,indices[:,None].expand(b,n,-1))
    noise = torch.randn(clean.shape, generator=rng,device=device)
    t = float(torch.sigmoid(c['time_mean']+c['time_std']*torch.randn((),generator=rng,device=device)).clamp(.0001,.999))
    return clean,noise,context,data['leads'][li],t


def checkpoint(path, model, ema, opt, rng, step, seconds, c, init_hash):
    temp = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
        generator=rng.get_state(),step=step,training_seconds=seconds,config=c,initialization_sha256=init_hash),temp)
    temp.replace(path)


def train(out, c, method, seed, device):
    run = out/'runs'/f'{method}_seed{seed}'
    run.mkdir(parents=True,exist_ok=True)
    done = run/'training_complete.json'
    if done.exists():
        return
    torch.manual_seed(seed)
    model = ForecastFlow(c['dimension'],c['width'],c['depth']).to(device)
    init_hash = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(),lr=c['learning_rate'],betas=(.9,.95),eps=1e-8)
    rng = torch.Generator(device=device).manual_seed(10000+seed)
    with np.load(out/'data.npz') as d:
        center,scale = float(d['center']),float(d['scale'])
        data = dict(context=torch.tensor((d['train_context']-center)/scale,device=device),
                    future=torch.tensor((d['train_future']-center)/scale,device=device),
                    leads=torch.tensor(c['lead_times'],device=device))
    existing = sorted(run.glob('checkpoint_*.pt'))
    start_step, elapsed = 0,0.
    if existing:
        state = torch.load(existing[-1],map_location=device,weights_only=False)
        model.load_state_dict(state['model']); ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer']); rng.set_state(state['generator'].cpu())
        start_step,elapsed = state['step'],state['training_seconds']
        if state['config']!=c or state['initialization_sha256']!=init_hash:
            raise RuntimeError('Resume identity mismatch')
    # Initialize objective-specific CUDA kernels before timing. Warm-up uses a
    # separate RNG and no optimizer updates, preserving paired initialization,
    # optimizer state and every production minibatch. Report this cost separately.
    warm_rng = torch.Generator(device=device).manual_seed(78787)
    sync(device)
    warm_start = time.perf_counter()
    for _ in range(5):
        clean,noise,context,lead,t = training_batch(data,c,warm_rng,device)
        opt.zero_grad(set_to_none=True)
        objective(model,clean,noise,context,lead,t,method,c).backward()
    opt.zero_grad(set_to_none=True)
    sync(device)
    write_json(run/'warmup.json',dict(seconds=time.perf_counter()-warm_start,
        updates=0,backward_passes=5,separate_rng=True))
    if not existing:
        checkpoint(run/'checkpoint_000000.pt',model,ema,opt,rng,0,0.,c,init_hash)
    started = time.perf_counter()
    excluded = 0.
    for step in range(start_step+1,c['updates']+1):
        model.train()
        clean,noise,context,lead,t = training_batch(data,c,rng,device)
        lr = c['learning_rate']*min(step/c['warmup_updates'],1)
        for g in opt.param_groups:
            g['lr']=lr
        opt.zero_grad(set_to_none=True)
        loss=objective(model,clean,noise,context,lead,t,method,c)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite training loss')
        opt.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),model.parameters()):
                ep.lerp_(p,1-c['ema_decay'])
        if step%100==0 or step==c['updates']:
            sync(device)
            seconds=elapsed+time.perf_counter()-started-excluded
            record=dict(method=method,seed=seed,step=step,target=c['updates'],loss=float(loss),grad=float(grad),
                        training_seconds=seconds,lr=lr)
            with (run/'training.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            write_json(run/'status.json',record)
            status(out,'training',**record)
        if step%c['checkpoint_every']==0 or step==c['updates']:
            sync(device)
            seconds=elapsed+time.perf_counter()-started-excluded
            saving=time.perf_counter()
            checkpoint(run/f'checkpoint_{step:06d}.pt',model,ema,opt,rng,step,seconds,c,init_hash)
            excluded+=time.perf_counter()-saving
    write_json(done,dict(method=method,seed=seed,step=c['updates'],training_seconds=seconds,
                        initialization_sha256=init_hash,final_checkpoint=f'checkpoint_{c["updates"]:06d}.pt'))


def evaluate(out,c,method,seed,step,device,tag,solver_steps=None,split='test'):
    run=out/'runs'/f'{method}_seed{seed}'
    path=run/f'{tag}.npz'
    if path.exists() and path.with_suffix('.json').exists():
        return
    solver_steps=solver_steps or c['sample_steps']
    state=torch.load(run/f'checkpoint_{step:06d}.pt',map_location=device,weights_only=False)
    model=ForecastFlow(c['dimension'],c['width'],c['depth']).to(device)
    model.load_state_dict(state['ema']);model.eval()
    d=np.load(out/'data.npz')
    context=d[split+'_context'];future=d[split+'_future'];center,scale=float(d['center']),float(d['scale'])
    refs=future.shape[2]//2
    # Evaluation randomness is shared across methods AND training seeds, so
    # differences across seeds reflect training, not different evaluation sets.
    rng=np.random.default_rng(88991 if split=='test' else 88992)
    ensembles=[]
    sampling_seconds=0.
    for li,lead in enumerate(c['lead_times']):
        pieces=[]
        for start in range(0,len(context),8):
            ctx=torch.tensor((context[start:start+8]-center)/scale,device=device)
            z=torch.tensor(rng.normal(size=(len(ctx),c['ensemble_size'],c['dimension'])).astype('float32'),device=device)
            lt=torch.full((len(ctx),),lead,device=device)
            sync(device);beg=time.perf_counter()
            draws=sample(model,z,ctx,lt,solver_steps,c['denominator_floor'])
            sync(device);sampling_seconds+=time.perf_counter()-beg
            pieces.append(draws.cpu().numpy()*scale+center)
        ensembles.append(np.concatenate(pieces))
    ensemble=np.stack(ensembles,axis=1)
    values={}
    for li,lead in enumerate(c['lead_times']):
        result=scores(ensemble[:,li],future[:,li,:refs],d['event_threshold'][li])
        for key,value in result.items():
            values.setdefault(key,[]).append(value)
    values={key:np.stack(value,axis=1) for key,value in values.items()}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise RuntimeError('Nonfinite evaluation scores')
    np.savez(path,ensemble=ensemble,**values)
    write_json(path.with_suffix('.json'),dict(method=method,seed=seed,step=step,tag=tag,split=split,
        checkpoint_sha256=sha(run/f'checkpoint_{step:06d}.pt'),data_sha256=sha(out/'data.npz'),
        sha256=sha(path),training_seconds=state['training_seconds'],sampling_seconds=sampling_seconds,
        solver_steps=solver_steps,nfe=2*solver_steps-1,ensemble_size=c['ensemble_size'],reference_size=refs,
        mean_by_lead={k:v.mean(0).tolist() for k,v in values.items()}))
    status(out,'evaluated',method=method,seed=seed,tag=tag,step=step,
           crps=values['crps'].mean(0).tolist(),energy_score=values['energy_score'].mean(0).tolist())


def baselines(out,c):
    path=out/'baselines.npz'
    if path.exists():
        return
    d=np.load(out/'data.npz'); y=d['test_future'];refs=y.shape[2]//2
    values={}
    for label in ['simulator_ensemble','unperturbed_forecast']:
        for li in range(len(c['lead_times'])):
            x=y[:,li,refs:refs+c['ensemble_size']] if label=='simulator_ensemble' else d['test_control'][:,li,None]
            for key,v in scores(x,y[:,li,:refs],d['event_threshold'][li]).items():
                values.setdefault(label+'__'+key,[]).append(v)
    np.savez(path,**{k:np.stack(v,1) for k,v in values.items()})


def report(out,c):
    records=[];comparisons=[]
    rng=np.random.default_rng(91919)
    for tag in ['equal_updates','equal_time','refinement']:
        all_values={}
        for method in c['methods']:
            members=[]
            for seed in c['seeds']:
                p=out/'runs'/f'{method}_seed{seed}'/f'{tag}.npz'
                with np.load(p) as a:
                    members.append({k:a[k] for k in a.files if k!='ensemble'})
            all_values[method]={k:np.stack([v[k] for v in members]) for k in members[0]}
            for li,lead in enumerate(c['lead_times']):
                records.append(dict(comparison=tag,method=method,lead=lead,
                    **{k:float(v[:,:,li].mean()) for k,v in all_values[method].items()}))
        for method in c['methods'][1:]:
            for key in ['crps','energy_score','coverage90','width90','correlation_rmse','brier_extreme']:
                delta=all_values[method][key]-all_values['flow_matching'][key]
                # Two-way paired bootstrap: same sampled contexts for all seeds,
                # preserving shared test-set dependence rather than treating all
                # seed-context cells as independent observations.
                draws=[]
                for _ in range(c['bootstrap_repetitions']):
                    si=rng.integers(len(c['seeds']),size=len(c['seeds']))
                    ci=rng.integers(c['test_contexts'],size=c['test_contexts'])
                    draws.append(delta[si][:,ci].mean((0,1)))
                lower,upper=np.quantile(draws,[.025,.975],axis=0)
                for li,lead in enumerate(c['lead_times']):
                    comparisons.append(dict(comparison=tag,method=method,baseline='flow_matching',metric=key,lead=lead,
                        difference=float(delta[:,:,li].mean()),ci95=[float(lower[li]),float(upper[li])],
                        per_seed_difference=delta[:,:,li].mean(1).tolist()))
    import csv
    for name,rows in [('summary.csv',records),('paired_comparisons.csv',comparisons)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'summary.json',dict(records=records,comparisons=comparisons,
        limitations=['Exploratory fixed-hyperparameter comparison, three seeds; not a weather benchmark.',
        'CI resamples seeds and held-out contexts; three seeds give limited uncertainty resolution.',
        'No-repulsion ablation changes the target field and need not retain consistency.',
        'Equal-time checkpoints are rounded down to checkpoint cadence; actual times are reported.',
        'Reference-conditioned training guarantee is not automatically a generated-trajectory guarantee.']))
    # Required identities, coverage and finite artifacts; ready follows audit.
    identities=[];hashes={}
    for seed in c['seeds']:
        initial=[]
        for method in c['methods']:
            run=out/'runs'/f'{method}_seed{seed}'
            completion=json.loads((run/'training_complete.json').read_text())
            if completion['step']!=c['updates']:
                raise RuntimeError('Incomplete training')
            initial.append(completion['initialization_sha256'])
            for tag in ['equal_updates','equal_time','refinement','validation']:
                p=run/f'{tag}.npz';meta=json.loads(p.with_suffix('.json').read_text())
                if sha(p)!=meta['sha256'] or sha(out/'data.npz')!=meta['data_sha256']:
                    raise RuntimeError('Artifact hash mismatch')
                if sha(run/f'checkpoint_{meta["step"]:06d}.pt')!=meta['checkpoint_sha256']:
                    raise RuntimeError('Checkpoint hash mismatch')
                with np.load(p) as a:
                    expected=c['test_contexts'] if tag!='validation' else c['validation_contexts']
                    if a['ensemble'].shape!=(expected,len(c['lead_times']),c['ensemble_size'],c['dimension']):
                        raise RuntimeError('Wrong evaluation coverage')
                    if not all(np.isfinite(a[k]).all() for k in a.files):
                        raise RuntimeError('Nonfinite artifact')
                hashes[str(p.relative_to(out))]=meta['sha256']
        if len(set(initial))!=1:
            raise RuntimeError('Unpaired model initialization')
        identities.append(initial[0])
    write_json(out/'audit.json',dict(passed=True,expected_runs=len(c['methods'])*len(c['seeds']),
        evaluation_artifacts=len(hashes),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        data_sha256=sha(out/'data.npz'),paired_initializations=identities,sha256=hashes))
    write_json(out/'ready.json',dict(complete=True,audit_sha256=sha(out/'audit.json'),summary_sha256=sha(out/'summary.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=json.loads((ROOT/'config.json').read_text())
    if args.smoke:
        c.update(train_contexts=16,validation_contexts=4,test_contexts=8,train_references=16,
            validation_references=64,test_references=64,seeds=[0],updates=40,checkpoint_every=2,
            contexts_per_batch=2,particles_per_context=8,width=32,depth=2,ensemble_size=16,
            sample_steps=5,refinement_steps=10,bootstrap_repetitions=50,warmup_updates=5)
    existing=out/'config.json'
    if existing.exists() and json.loads(existing.read_text())!=c:
        raise RuntimeError('Output directory already has a different configuration')
    write_json(existing,c)
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        if sha(ROOT/name)!=digest:
            raise RuntimeError('Frozen source mismatch: '+name)
    torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('Exactly one allocated CUDA GPU is required')
    torch.use_deterministic_algorithms(True)
    write_json(out/'runtime.json',dict(job_id=os.environ.get('JOB_ID'),host=os.uname().nodename,
        device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        torch=torch.__version__,numpy=np.__version__,allocated_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_manifest=manifest,smoke=args.smoke))
    try:
        prepare(out,c)
        if args.prepare_only:
            status(out,'data_ready');return
        for seed in c['seeds']:
            # Rotate order to reduce systematic hardware-temperature/time effects.
            methods=c['methods'][seed%3:]+c['methods'][:seed%3]
            for method in methods:
                train(out,c,method,seed,device)
                evaluate(out,c,method,seed,c['updates'],device,'validation',split='validation')
            budgets=[]
            for method in methods:
                meta=json.loads((out/'runs'/f'{method}_seed{seed}'/'training_complete.json').read_text())
                budgets.append(meta['training_seconds'])
            budget=min(budgets)
            for method in methods:
                run=out/'runs'/f'{method}_seed{seed}'
                selected=None
                for p in sorted(run.glob('checkpoint_*.pt')):
                    state=torch.load(p,map_location='cpu',weights_only=False)
                    if state['training_seconds']<=budget+1e-6:
                        selected=state['step']
                if selected is None:
                    raise RuntimeError('No checkpoint fits common time budget')
                write_json(run/'equal_time_budget.json',dict(common_budget_seconds=budget,selected_step=selected))
                evaluate(out,c,method,seed,c['updates'],device,'equal_updates')
                evaluate(out,c,method,seed,selected,device,'equal_time')
                evaluate(out,c,method,seed,c['updates'],device,'refinement',c['refinement_steps'])
        baselines(out,c);report(out,c)
        status(out,'complete',ready=True)
    except BaseException as e:
        status(out,'failed',error=repr(e),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    main()
