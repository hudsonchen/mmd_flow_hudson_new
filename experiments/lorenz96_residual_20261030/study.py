"""Validation-selected hybrid residual ensemble study; no test-driven tuning."""
import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import torch
from core import ForecastNet, objective, sample, scores, integrate

ROOT=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def status(out,stage,**kwargs):
    value=dict(stage=stage,updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs)
    write_json(out/'status.json',value);print(json.dumps(value),flush=True)


def sync(device):
    if device.type=='cuda': torch.cuda.synchronize(device)


def identity(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()


def load_data(path,c,device):
    raw=dict(np.load(path))
    center,scale=float(raw['center']),float(raw['scale'])
    residual=raw['train_future']-raw['train_control'][:,:,None,:]
    residual_scale=np.sqrt((residual.astype('float64')**2).mean((0,2,3))).astype('float32')
    assert np.all(residual_scale>0)
    data={}
    for split in ['train','validation','test']:
        data[split]={
            'context':torch.tensor((raw[split+'_context']-center)/scale,device=device),
            'control':torch.tensor((raw[split+'_control']-center)/scale,device=device),
            'future':torch.tensor((raw[split+'_future']-raw[split+'_control'][:,:,None,:])/
                residual_scale[None,:,None,None],device=device)}
    data['leads']=torch.tensor(c['lead_times'],device=device)
    data['residual_scale']=residual_scale
    return raw,data


def batch(data,c,rng,device):
    tr=data['train'];b,n=c['contexts_per_batch'],c['particles_per_context']
    ci=torch.randint(len(tr['context']),(b,),generator=rng,device=device)
    li=torch.randint(len(c['lead_times']),(b,),generator=rng,device=device)
    ri=torch.randint(tr['future'].shape[2],(b,n),generator=rng,device=device)
    clean=tr['future'][ci[:,None],li[:,None],ri]
    ctx,ctrl=tr['context'][ci],tr['control'][ci,li]
    noise=torch.randn(clean.shape,generator=rng,device=device)
    t=float(torch.sigmoid(c['time_mean']+c['time_std']*torch.randn((),generator=rng,device=device)).clamp(.0001,.999))
    return clean,noise,ctx,ctrl,data['leads'][li],t


@torch.inference_mode()
def draw(model,method,split,raw,data,c,device,count,members,steps):
    # Shared noise across methods, training seeds, checkpoints and solver grids.
    rng=np.random.default_rng(76123 if split=='validation' else 76124)
    ensembles=[];sync(device);start=time.perf_counter()
    for li,lead in enumerate(c['lead_times']):
        pieces=[]
        for lo in range(0,count,8):
            hi=min(lo+8,count)
            z=torch.tensor(rng.normal(size=(hi-lo,members,c['dimension'])).astype('float32'),device=device)
            residual=sample(model,z,data[split]['context'][lo:hi],data[split]['control'][lo:hi,li],
                torch.full((hi-lo,),lead,device=device),method,steps)
            pieces.append(residual.cpu().numpy()*data['residual_scale'][li]+raw[split+'_control'][lo:hi,li,None])
        ensembles.append(np.concatenate(pieces))
    sync(device)
    return np.stack(ensembles,1),time.perf_counter()-start


def evaluate_arrays(ensemble,raw,c,split,refs,extended=True):
    result={}
    for li in range(len(c['lead_times'])):
        value=scores(ensemble[:,li],raw[split+'_future'][:len(ensemble),li,:refs],
            raw['event_threshold'][li],extended=extended)
        for key,val in value.items():result.setdefault(key,[]).append(val)
    result={key:np.stack(val,1) for key,val in result.items()}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise RuntimeError('Nonfinite scores')
    return result


def validate(model,method,raw,data,c,device):
    model.eval()
    ensemble,seconds=draw(model,method,'validation',raw,data,c,device,
        c['validation_contexts'],c['validation_ensemble'],c['validation_steps'])
    values=evaluate_arrays(ensemble,raw,c,'validation',c['validation_references'],extended=False)
    crps=values['crps'].mean(0)
    # Equal weighting across leads in training-only residual-scale units.
    return dict(selection_score=float((crps/data['residual_scale']).mean()),
        crps=crps.tolist(),coverage90=values['coverage90'].mean(0).tolist(),
        mean_rmse=values['mean_rmse'].mean(0).tolist(),sampling_seconds=seconds)


def train(out,run,method,seed,lr,updates,raw,data,c,device):
    run.mkdir(parents=True,exist_ok=True)
    if (run/'complete.json').exists():return json.loads((run/'complete.json').read_text())
    torch.manual_seed(seed);model=ForecastNet(c['width'],c['depth']).to(device)
    init=identity(model);ema=copy.deepcopy(model).eval()
    opt=torch.optim.Adam(model.parameters(),lr=lr,betas=(.9,.95))
    rng=torch.Generator(device=device).manual_seed(10000+seed)
    run_identity=dict(method=method,seed=seed,lr=lr,updates=updates,config=c,
        initialization_sha256=init,parameter_count=sum(p.numel() for p in model.parameters()))
    if (run/'identity.json').exists():
        assert json.loads((run/'identity.json').read_text())==run_identity
    write_json(run/'identity.json',run_identity)
    records=[];elapsed=0.;start_step=0
    existing=sorted(run.glob('checkpoint_*.pt'))
    if existing:
        st=torch.load(existing[-1],map_location=device,weights_only=False)
        model.load_state_dict(st['model']);ema.load_state_dict(st['ema']);opt.load_state_dict(st['optimizer'])
        rng.set_state(st['rng'].cpu());start_step=st['step'];elapsed=st['training_seconds']
        records=st['validation_records'];assert st['identity']==run_identity
    warm=torch.Generator(device=device).manual_seed(99999)
    sync(device);beg=time.perf_counter()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        objective(model,*batch(data,c,warm,device),method,c).backward()
    opt.zero_grad(set_to_none=True);sync(device)
    write_json(run/'warmup.json',dict(seconds=time.perf_counter()-beg,updates=0))

    def save(step):
        path=run/f'checkpoint_{step:06d}.pt';temp=path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
            rng=rng.get_state(),step=step,training_seconds=elapsed,identity=run_identity,
            validation_records=records),temp);temp.replace(path)

    if not existing:
        val=validate(ema,method,raw,data,c,device)
        records.append(dict(step=0,training_seconds=0.,**val));save(0)
    sync(device);started=time.perf_counter();base_elapsed=elapsed
    for step in range(start_step+1,updates+1):
        model.train();opt.zero_grad(set_to_none=True)
        for group in opt.param_groups:group['lr']=lr*min(step/c['warmup_updates'],1.)
        loss=objective(model,*batch(data,c,rng,device),method,c)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-c['ema_decay'])
        if step%100==0 or step==updates:
            sync(device);elapsed=base_elapsed+time.perf_counter()-started
            record=dict(run=str(run.relative_to(out)),method=method,seed=seed,lr=lr,step=step,
                target=updates,loss=float(loss),gradient_norm=float(grad),training_seconds=elapsed)
            with (run/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            status(out,'training',**record)
        if step%c['checkpoint_every']==0 or step==updates:
            sync(device);elapsed=base_elapsed+time.perf_counter()-started
            val=validate(ema,method,raw,data,c,device)
            records.append(dict(step=step,training_seconds=elapsed,**val))
            write_json(run/'validation.json',records);save(step)
            status(out,'validation',run=str(run.relative_to(out)),step=step,**val)
            base_elapsed=elapsed;sync(device);started=time.perf_counter()
    # Step zero remains a transparent baseline but is not a trained checkpoint.
    best=min((r for r in records if r['step']>0),key=lambda r:r['selection_score'])
    complete=dict(**run_identity,training_seconds=elapsed,best=best,validation_records=records)
    write_json(run/'complete.json',complete)
    return complete


def evaluate(out,run,method,step,tag,raw,data,c,device,steps=None):
    path=run/f'{tag}.npz'
    if path.exists() and path.with_suffix('.json').exists():return
    cp=run/f'checkpoint_{step:06d}.pt';state=torch.load(cp,map_location=device,weights_only=False)
    model=ForecastNet(c['width'],c['depth']).to(device);model.load_state_dict(state['ema']);model.eval()
    steps=steps or c['sample_steps']
    ensemble,seconds=draw(model,method,'test',raw,data,c,device,len(raw['test_context']),c['ensemble_size'],steps)
    values=evaluate_arrays(ensemble,raw,c,'test',raw['test_future'].shape[2]//2)
    np.savez(path,ensemble=ensemble,**values)
    write_json(path.with_suffix('.json'),dict(method=method,step=step,tag=tag,sha256=sha(path),
        checkpoint_sha256=sha(cp),training_seconds=state['training_seconds'],sampling_seconds=seconds,
        solver_steps=steps,nfe=1 if method=='energy_score' else 2*steps,
        mean_by_lead={k:v.mean(0).tolist() for k,v in values.items()}))
    status(out,'evaluated',method=method,tag=tag,step=step,crps=values['crps'].mean(0).tolist())


def baselines(out,raw,data,c):
    if (out/'baselines.npz').exists():return
    y=raw['test_future'];r=y.shape[2]//2
    rng=np.random.default_rng(76124)
    gaussian=raw['test_control'][:,:,None]+rng.normal(size=(len(y),len(c['lead_times']),c['ensemble_size'],c['dimension']))*data['residual_scale'][None,:,None,None]
    ensembles={'simulator_ensemble':y[:,:,r:r+c['ensemble_size']],
        'unperturbed_forecast':raw['test_control'][:,:,None], 'gaussian_residual':gaussian}
    values={}
    for method,ensemble in ensembles.items():
        for k,v in evaluate_arrays(ensemble,raw,c,'test',r).items():values[method+'__'+k]=v
    np.savez(out/'baselines.npz',**values)
    start=time.perf_counter()
    fresh=integrate(raw['test_context'],c['lead_times'],.005,8.)
    elapsed=time.perf_counter()-start
    np.testing.assert_allclose(fresh,raw['test_control'],atol=2e-4,rtol=2e-4)
    write_json(out/'physics_cost.json',dict(cpu_seconds_all_test_contexts=elapsed,
        contexts=len(y),leads=c['lead_times'],dt=.005,
        note='Shared deterministic forecast cost, additional to neural sampling. CPU and GPU timings are reported separately.'))


def report(out,raw,data,c):
    records=[];comparisons=[];hashes={};initializations=[]
    tags=['selected','equal_updates','equal_time','refinement']
    for seed in c['seeds']:
        ids=[json.loads((out/'runs'/f'{m}_seed{seed}'/'identity.json').read_text()) for m in c['methods']]
        assert len({x['initialization_sha256'] for x in ids})==1
        assert len({x['parameter_count'] for x in ids})==1
        initializations.append(ids[0]['initialization_sha256'])
    for tag in tags:
        all_values={}
        for method in c['methods']:
            members=[]
            for seed in c['seeds']:
                run=out/'runs'/f'{method}_seed{seed}'
                done=json.loads((run/'complete.json').read_text());assert done['updates']==c['updates']
                p=run/f'{tag}.npz';meta=json.loads(p.with_suffix('.json').read_text())
                assert sha(p)==meta['sha256']
                assert sha(run/f'checkpoint_{meta["step"]:06d}.pt')==meta['checkpoint_sha256']
                with np.load(p) as v:
                    assert v['ensemble'].shape==(len(raw['test_context']),len(c['lead_times']),c['ensemble_size'],c['dimension'])
                    assert all(np.isfinite(v[k]).all() for k in v.files)
                    # Replay fixed context at every lead directly from saved draws.
                    actual=evaluate_arrays(v['ensemble'][:1],raw,c,'test',raw['test_future'].shape[2]//2)
                    for k,a in actual.items():np.testing.assert_allclose(a,v[k][:1],rtol=1e-9,atol=1e-9)
                    members.append({k:v[k] for k in v.files if k!='ensemble'})
                hashes[str(p.relative_to(out))]=sha(p)
            all_values[method]={k:np.stack([v[k] for v in members]) for k in members[0]}
            for li,lead in enumerate(c['lead_times']):
                records.append(dict(comparison=tag,method=method,lead=lead,
                    **{k:float(v[:,:,li].mean()) for k,v in all_values[method].items()}))
        rng=np.random.default_rng(91919)
        for method in c['methods'][1:]:
            for key in ['crps','energy_score','coverage90','mean_rmse','variogram','shuffle_variogram_difference']:
                delta=all_values[method][key]-all_values['flow_matching'][key]
                draws=[]
                for _ in range(c['bootstrap_repetitions']):
                    si=rng.integers(len(c['seeds']),size=len(c['seeds']))
                    ci=rng.integers(len(raw['test_context']),size=len(raw['test_context']))
                    draws.append(delta[si][:,ci].mean((0,1)))
                lower,upper=np.quantile(draws,[.025,.975],axis=0)
                for li,lead in enumerate(c['lead_times']):
                    comparisons.append(dict(comparison=tag,method=method,metric=key,lead=lead,
                        difference=float(delta[:,:,li].mean()),ci95=[float(lower[li]),float(upper[li])]))
    for name,rows in [('summary.csv',records),('paired_comparisons.csv',comparisons)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(out/'summary.json',dict(records=records,comparisons=comparisons,
        limitations=['Hybrid known-physics residual task, not a reproduction of the two-scale JMLR benchmark.',
        'Old pilot test set reused for continuity; improved protocol fixed before new test evaluation.',
        'Validation selects learning rates and checkpoints; three training seeds remain exploratory.',
        'All methods share exact architecture and conditioning; direct generator versus velocity semantics differ.',
        'Per-lead conditional ensembles are not coherent multilead trajectories.',
        'Compute-matched checkpoints rounded down; tuning and shared physics costs reported separately.']))
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():assert sha(ROOT/name)==digest
    hashes['baselines.npz']=sha(out/'baselines.npz')
    write_json(out/'audit.json',dict(passed=True,runs=len(c['seeds'])*len(c['methods']),
        evaluation_artifacts=len(tags)*len(c['seeds'])*len(c['methods']),sha256=hashes,
        data_sha256=sha(out/'data.npz'),source_manifest_sha256=sha(ROOT/'source_manifest.json'),
        paired_initializations=initializations,architecture_identical=True,score_replay=True))
    write_json(out/'ready.json',dict(complete=True,audit_sha256=sha(out/'audit.json'),summary_sha256=sha(out/'summary.json')))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda');ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(out/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=json.loads((ROOT/'config.json').read_text())
    if args.smoke:c.update(seeds=[0],learning_rates=[.001],tuning_updates=10,updates=20,
        checkpoint_every=10,contexts_per_batch=2,particles_per_context=8,width=16,depth=1,
        validation_contexts=2,validation_ensemble=8,validation_references=8,
        validation_steps=2,ensemble_size=8,sample_steps=2,refinement_steps=4,bootstrap_repetitions=20,warmup_updates=2)
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==c
    write_json(out/'config.json',c)
    manifest=json.loads((ROOT/'source_manifest.json').read_text())
    for name,digest in manifest['sha256'].items():assert sha(ROOT/name)==digest,name
    parent=json.loads(args.data.with_name('data_identity.json').read_text())
    assert sha(args.data)==parent['sha256']
    if not (out/'data.npz').exists():shutil.copy2(args.data,out/'data.npz')
    assert sha(out/'data.npz')==parent['sha256']
    write_json(out/'data_identity.json',parent)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    device=torch.device(args.device)
    if device.type=='cuda':assert torch.cuda.is_available() and torch.cuda.device_count()==1
    write_json(out/'runtime.json',dict(job_id=os.environ.get('JOB_ID'),host=os.uname().nodename,
        device=str(device),gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None,
        allocated_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,numpy=np.__version__,
        source_manifest=manifest,smoke=args.smoke))
    try:
        raw,data=load_data(out/'data.npz',c,device)
        if args.smoke:
            # Keep full training-derived normalization but shrink audit/test coverage.
            for key in ['test_context','test_future','test_control']:raw[key]=raw[key][:2]
            raw['test_future']=raw['test_future'][:,:,:16]
        write_json(out/'normalization.json',dict(center=float(raw['center']),scale=float(raw['scale']),
            residual_rms_by_lead=data['residual_scale'].tolist(),derived_from='training only'))
        selection={}
        for method in c['methods']:
            trials=[]
            for lr in c['learning_rates']:
                run=out/'tuning'/f'{method}_lr{lr:g}'
                trials.append(train(out,run,method,c['tuning_seed'],lr,c['tuning_updates'],raw,data,c,device))
            best=min(trials,key=lambda r:r['best']['selection_score'])
            selection[method]=dict(lr=best['lr'],best=best['best'],
                tuning_training_seconds=sum(t['training_seconds'] for t in trials))
            write_json(out/'selection.json',selection)
        fm=selection['flow_matching']['best'];n=c['validation_contexts']
        control=raw['validation_control'][:n,:,None]
        deterministic=evaluate_arrays(control,raw,c,'validation',c['validation_references'],False)['crps'].mean(0)
        competent=bool(fm['crps'][0]<deterministic[0] and .7<fm['coverage90'][0]<.98)
        write_json(out/'competence.json',dict(passed=competent,smoke=args.smoke,
            criterion='Validation FM short-lead CRPS beats deterministic control and 90% coverage is between 70% and 98%',
            fm=fm,deterministic_crps=deterministic.tolist()))
        if not competent and not args.smoke:
            status(out,'diagnostic_gate_failed',reason='Baseline not yet competent; production and test evaluation were not started')
            return
        # Freeze all learning-rate decisions before accessing test scores.
        for seed in c['seeds']:
            methods=c['methods'][seed%len(c['methods']):]+c['methods'][:seed%len(c['methods'])]
            completed={}
            for method in methods:
                completed[method]=train(out,out/'runs'/f'{method}_seed{seed}',method,seed,
                    selection[method]['lr'],c['updates'],raw,data,c,device)
            budget=min(r['training_seconds'] for r in completed.values())
            for method in methods:
                run=out/'runs'/f'{method}_seed{seed}';record=completed[method]
                eligible=[r for r in record['validation_records'] if r['training_seconds']<=budget+1e-8]
                equal=max(eligible,key=lambda r:r['step'])
                write_json(run/'equal_time_budget.json',dict(common_budget_seconds=budget,selected_step=equal['step']))
                evaluate(out,run,method,record['best']['step'],'selected',raw,data,c,device)
                evaluate(out,run,method,c['updates'],'equal_updates',raw,data,c,device)
                evaluate(out,run,method,equal['step'],'equal_time',raw,data,c,device)
                evaluate(out,run,method,record['best']['step'],'refinement',raw,data,c,device,c['refinement_steps'])
        baselines(out,raw,data,c);report(out,raw,data,c);status(out,'complete',ready=True)
    except BaseException as e:
        status(out,'failed',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
