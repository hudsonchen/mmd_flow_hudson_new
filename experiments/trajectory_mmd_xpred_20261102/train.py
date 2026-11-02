"""Train evolved-particle laws with Gaussian MMD; no velocity or image labels."""
import argparse
import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import shutil
import sys
import time
import numpy as np
import torch
from torchvision.utils import save_image
from transport import CleanPredictor,integrate
from kernel import objective

ROOT=Path(__file__).resolve().parents[2]
PREVIOUS=ROOT/'experiments/trajectory_mmd_20261102'
sys.path.insert(0,str(PREVIOUS))
sys.path.insert(0,str(ROOT))
from run import Classifier,features,metrics,write,sha
from data import download_mnist,load_split


@torch.no_grad()
def generate(model,z,steps,epsilon,chunk=64):
    return torch.cat([integrate(model,b,steps=steps,epsilon=epsilon) for b in z.split(chunk)])


def run(args):
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device.startswith('cuda'):
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark=True
    device=torch.device(args.device)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=bool(args.resume))
    if not args.resume:
        source=out/'source';source.mkdir()
        for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,source/p.name)
        for p in [ROOT/'data.py',PREVIOUS/'run.py',PREVIOUS/'core.py']:
            shutil.copy2(p,source/('dependency_'+p.name))
        config=vars(args).copy()
        config.update(objective=f'{1-args.endpoint_probability} * time_integral_MMD + {args.endpoint_probability} * endpoint_MMD',
            reference_reference_constant_omitted=True,
            training_loss_label='model-dependent U-statistic terms, not full MMD squared',
            kernel='Gaussian full pixels, fixed average pooling, aligned random raw-pixel crops',
            velocity='(tanh_predictor(x,t)-x)/(1-t+epsilon)',
            solver='second-order exponential midpoint in logarithmic noise time',
            labels_used_for_training=False,velocity_targets=False,pretrained_weights=False,
            source_hashes={p.name:sha(p) for p in source.iterdir()},
            torch_version=torch.__version__,device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
        if args.init_checkpoint:config['initial_checkpoint_sha256']=sha(Path(args.init_checkpoint))
        write(out/'config.json',config)
    else:
        config=json.loads((out/'config.json').read_text())
        for key in ['seed','width','epsilon','train_steps','endpoint_probability','patch_weight','batch','reference_batch','ema','schedule_updates','amp']:
            assert config[key]==getattr(args,key),(key,config[key],getattr(args,key))
        assert not (out/'ready.json').exists(),'Use a fresh output directory for a new completed-run stage'
    raw=download_mnist(Path(args.data_root))
    train,_=load_split(raw,True);test,labels=load_split(raw,False)
    train=train.to(device).float().reshape(-1,1,28,28)/127.5-1
    test=test.to(device).float().reshape(-1,1,28,28)/127.5-1
    classifier=Classifier().to(device).eval()
    classifier.load_state_dict(torch.load(args.classifier,map_location=device,weights_only=True))
    classifier.requires_grad_(False)
    if not args.resume:
        logits,_=features(classifier,(test+1)/2)
        write(out/'data_identity.json',{'mnist':{p.name:sha(p) for p in raw.glob('*.gz')},
            'classifier_sha256':sha(Path(args.classifier)),
            'classifier_accuracy':float((logits.argmax(1).cpu()==labels).float().mean()),
            'classifier_evaluation_only':True})
    model=CleanPredictor(args.width).to(device)
    if args.init_checkpoint:
        state=torch.load(args.init_checkpoint,map_location=device,weights_only=False)
        model.load_state_dict(state['ema_model'])
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.95))
    sg=torch.Generator(device=device).manual_seed(11000+args.seed)
    rg=torch.Generator(device=device).manual_seed(21000+args.seed)
    tg=torch.Generator().manual_seed(31000+args.seed)
    pg=torch.Generator().manual_seed(51000+args.seed)
    eg=torch.Generator(device=device).manual_seed(41000+args.seed)
    order=torch.randperm(len(test),device=device,generator=eg)
    target=test[order[:args.eval_n]];other=test[order[args.eval_n:2*args.eval_n]]
    noise=torch.randn(args.eval_n,1,28,28,device=device,generator=eg)
    history=[];first=0;elapsed=0.;best=float('inf')
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema_model'])
        optimizer.load_state_dict(state['optimizer']);first=state['update'];elapsed=state['seconds'];best=state['best_frechet']
        for name,generator in [('source',sg),('reference',rg),('time',tg),('patch',pg)]:generator.set_state(state[name+'_rng'].cpu())
        history=json.loads((out/'history.json').read_text())
    else:
        write(out/'real_real.json',metrics(other,target,'mnist',classifier))
        save_image((target[:64]+1)/2,out/'real_samples.png',nrow=8)
    started=time.monotonic()

    def save_checkpoint(update,durable=False):
        payload={'model':model.state_dict(),'ema_model':ema.state_dict(),'optimizer':optimizer.state_dict(),
            'update':update,'seconds':elapsed+time.monotonic()-started,'config':config,'best_frechet':best,
            'source_rng':sg.get_state(),'reference_rng':rg.get_state(),'time_rng':tg.get_state(),'patch_rng':pg.get_state()}
        temporary=out/'checkpoint_latest.tmp'
        torch.save(payload,temporary);temporary.replace(out/'checkpoint_latest.pt')
        if durable:shutil.copy2(out/'checkpoint_latest.pt',out/f'checkpoint_{update:06d}.pt')

    def evaluate(update,loss=None):
        nonlocal best
        x=generate(ema,noise,args.eval_steps,args.epsilon)
        result={'update':update,'seconds':elapsed+time.monotonic()-started,
            'training_loss_model_terms':loss,**metrics(x,target,'mnist',classifier)}
        result['max_abs_pixel']=float(x.abs().max())
        result['rms_above_2_fraction']=float((x.square().flatten(1).mean(1).sqrt()>2).float().mean())
        history.append(result);write(out/'history.json',history)
        save_image(((x[:64]+1)/2).clamp(0,1),out/f'samples_{update:06d}.png',nrow=8)
        np.savez_compressed(out/f'samples_{update:06d}.npz',generated=x.cpu().numpy(),reference=target.cpu().numpy(),noise=noise.cpu().numpy())
        improved=result['classifier_frechet']<best
        best=min(best,result['classifier_frechet'])
        save_checkpoint(update,durable=True)
        if improved:shutil.copy2(out/'checkpoint_latest.pt',out/'checkpoint_best.pt')
        write(out/'evaluation_status.json',result)
        print(json.dumps(result),flush=True)

    if not args.resume:evaluate(0)
    for update in range(first+1,args.updates+1):
        t=1. if float(torch.rand((),generator=tg))<args.endpoint_probability else float(torch.rand((),generator=tg))
        z=torch.randn(args.batch,1,28,28,device=device,generator=sg)
        clean=train[torch.randint(len(train),(args.reference_batch,),device=device,generator=rg)]
        steps=max(2,math.ceil(args.train_steps*math.log((1+args.epsilon)/(1-t+args.epsilon))/math.log((1+args.epsilon)/args.epsilon)))
        context=torch.autocast('cuda',dtype=torch.bfloat16) if args.amp and device.type=='cuda' else nullcontext()
        with context:
            x=integrate(model,z,t,steps,args.epsilon,args.checkpoint_steps)
        loss=objective(x.float(),clean,t,pg,args.patch_weight)
        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at update {update}')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        lr=args.lr*min(1.,update/max(1,args.warmup))*(.1+.9*.5*(1+math.cos(math.pi*(update-1)/args.schedule_updates)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step()
        with torch.no_grad():
            for target_p,current in zip(ema.parameters(),model.parameters()):target_p.lerp_(current,1-args.ema)
        if update%10==0 or update==first+1:
            write(out/'status.json',{'state':'training','update':update,'updates':args.updates,
                'seconds':elapsed+time.monotonic()-started,'time':t,'solver_steps':steps,
                'training_loss_model_terms':float(loss.detach()),'gradient_norm':float(gradient),
                'peak_cuda_memory_gb':torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else None})
        if update%args.eval_every==0 or update==args.updates:evaluate(update,float(loss.detach()))
    write(out/'status.json',{'state':'refinement_evaluation','update':args.updates,'updates':args.updates})
    refinement={};previous=None
    for steps in sorted(set([args.train_steps,args.eval_steps,2*args.eval_steps])):
        x=generate(ema,noise,steps,args.epsilon)
        refinement[str(steps)]=metrics(x,target,'mnist',classifier)
        if previous is not None:refinement[str(steps)]['paired_rms_change']=float((x-previous).square().mean().sqrt())
        previous=x
    write(out/'solver_refinement.json',refinement)
    marginal=[];arrays={}
    target_noise=torch.randn(target.shape,device=device,generator=eg)
    from run import exact_mmd
    for t in (0.,.25,.5,.75,1.):
        with torch.no_grad():
            x=torch.cat([integrate(ema,b,t,args.eval_steps,args.epsilon) for b in noise.split(64)])
        reference=t*target+(1-t)*target_noise
        bands=tuple(28*((t*b)**2+(1-t)**2)**.5 for b in (.05,.1,.2,.4))
        marginal.append({'t':t,'full_pixel_mmd2':exact_mmd(x,reference,bands)})
        arrays[f'generated_{t}']=x.cpu().numpy();arrays[f'reference_{t}']=reference.cpu().numpy()
    write(out/'marginals.json',marginal);np.savez_compressed(out/'marginals.npz',**arrays)
    write(out/'status.json',{'state':'complete','updates':args.updates,'seconds':elapsed+time.monotonic()-started})
    write(out/'audit.json',{'passed':True,'meaning':'finite run and artifact identities, not quality acceptance',
        'files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    write(out/'ready.json',{'ready':True,'quality_accepted':False})


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--data-root',default=str(ROOT.parent/'data'))
    p.add_argument('--classifier',default=str(ROOT/'results/mnist/independent_mmd_20261031/assets/classifier.pt'))
    p.add_argument('--seed',type=int,default=0);p.add_argument('--width',type=int,default=24)
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--schedule-updates',type=int,default=12000)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--reference-batch',type=int,default=2048)
    p.add_argument('--lr',type=float,default=.0002);p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--epsilon',type=float,default=.001);p.add_argument('--train-steps',type=int,default=8)
    p.add_argument('--eval-steps',type=int,default=32);p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--eval-n',type=int,default=1024);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--ema',type=float,default=.995);p.add_argument('--endpoint-probability',type=float,default=.5)
    p.add_argument('--patch-weight',type=float,default=.25)
    p.add_argument('--amp',action='store_true');p.add_argument('--checkpoint-steps',action='store_true')
    p.add_argument('--resume');p.add_argument('--init-checkpoint')
    args=p.parse_args()
    if not 0<=args.endpoint_probability<=1 or not 0<=args.patch_weight<1:p.error('Invalid mixture weights')
    if not 0<args.updates<=args.schedule_updates:p.error('Invalid training budget')
    if not 2<=args.eval_n<=5000:p.error('Require two disjoint held-out populations')
    if args.resume and args.init_checkpoint:p.error('Resume and initial checkpoint are mutually exclusive')
    return args


if __name__=='__main__':
    args=arguments()
    try:run(args)
    except Exception as error:
        path=Path(args.output)
        if path.exists() and not isinstance(error,FileExistsError):write(path/'failure.json',{'error':repr(error)})
        raise
