"""Train actual ODE marginals, first MoG then MNIST, with Gaussian MMD."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data import download_mnist, load_split
from core import Velocity, centers, mixture, integrate, objective, gaussian_mmd


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Classifier(nn.Module):
    """Compatible with the existing MNIST classifier; strictly evaluation-only."""
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,3)
        self.conv2=nn.Conv2d(32,64,3)
        self.fc1=nn.Linear(64*12*12,128)
        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        x=F.relu(self.conv1((x-.1307)/.3081))
        x=F.max_pool2d(F.relu(self.conv2(x)),2)
        feature=F.relu(self.fc1(x.flatten(1)))
        return self.fc2(feature),feature


@torch.no_grad()
def features(model, images):
    values = [model(chunk) for chunk in images.split(128)]
    return torch.cat([v[0] for v in values]), torch.cat([v[1] for v in values])


def frechet(a,b):
    a,b=a.double().cpu().numpy(),b.double().cpu().numpy()
    ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False)
    eig,vec=np.linalg.eigh(ca)
    root=(vec*np.maximum(eig,0)**.5)@vec.T
    cross=root@cb@root
    return float(max(0, np.sum((a.mean(0)-b.mean(0))**2)+np.trace(ca)+np.trace(cb)-
        2*np.sqrt(np.maximum(np.linalg.eigvalsh((cross+cross.T)/2),0)).sum()))


@torch.no_grad()
def exact_mmd(x,y,bands):
    # Blocked summation avoids quadratic evaluation memory.
    def total(a,b):
        value=0.
        same = a is b
        for i,aa in enumerate(a.split(256)):
            for j,bb in enumerate(b.split(256)):
                # Double precision and exact self diagonals matter for escaped tails.
                d=torch.cdist(aa.flatten(1).double(),bb.flatten(1).double()).square()
                if same and i == j:
                    d.diagonal().zero_()
                value+=float(torch.stack([(-d/(2*s*s)).exp().sum() for s in bands]).mean())
        return value
    return (total(x,x)-len(x))/(len(x)*(len(x)-1)) + \
        (total(y,y)-len(y))/(len(y)*(len(y)-1)) - 2*total(x,y)/(len(x)*len(y))


@torch.no_grad()
def metrics(x,y,dataset,classifier=None):
    if not torch.isfinite(x).all():
        raise FloatingPointError('Nonfinite generated samples')
    result={'exact_mmd2':exact_mmd(x,y,(.2,.4,.8,1.6) if dataset=='mog' else (1.4,2.8,5.6,11.2))}
    if dataset=='mog':
        angle=torch.arange(128,device=x.device)*math.pi/128
        directions=torch.stack([angle.cos(),angle.sin()])
        result['sliced_w2']=float(((x@directions).sort(0).values-(y@directions).sort(0).values).square().mean().sqrt())
        distance,label=torch.cdist(x,centers(x.device)).min(1)
        result['within_3sigma']=float((distance<.45).float().mean())
        result['mode_fractions']=(torch.bincount(label,minlength=8)/len(x)).tolist()
    else:
        result['outside_pixel_range']=float(((x < -1)|(x > 1)).float().mean())
        result['pixel_mean']=float(x.mean())
        result['pixel_std']=float(x.std())
        if classifier is not None:
            # Classifier diagnostics use the same display-range clipping for all samples.
            logits,fx=features(classifier,((x+1)/2).clamp(0,1))
            _,fy=features(classifier,((y+1)/2).clamp(0,1))
            prob=logits.softmax(1)
            result['classifier_frechet']=frechet(fx,fy)
            result['classifier_confidence']=float(prob.max(1).values.mean())
            result['digit_fractions']=(torch.bincount(prob.argmax(1),minlength=10)/len(x)).tolist()
    return result


@torch.no_grad()
def generate(model, noise, steps):
    return torch.cat([integrate(model,z,steps=steps) for z in noise.split(128)])


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    out=Path(args.output)/args.dataset/f'seed_{args.seed}'
    out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,source/path.name)
    shutil.copy2(ROOT/'data.py',source/'data.py')
    config=vars(args).copy()
    config.update(objective='E_uniform_t MMD^2(Law(ODE_theta(t,Z)), p_t)',
        reference_path='(1-t)N(0,I)+t*data', estimator='unbiased U-statistic',
        solver='Heun, backpropagate through every step',
        training_time='one independent Uniform(0,1) draw per update',
        kernel='Gaussian mixture; MNIST full pixels plus fixed 2x and 4x average pooling',
        reference_noise='analytic Gaussian integration for MNIST',
        initial_weights='random, zero output layer', velocity_labels=False,
        pretrained_training_weights=False, endpoint_extra_weight=0,
        source_hashes={p.name:sha(p) for p in source.glob('*.py')},
        torch_version=torch.__version__, device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU')
    write(out/'config.json',config)
    model=Velocity(args.dataset,args.width,args.linear_skip,args.stable_field).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.95))
    sg=torch.Generator(device=device).manual_seed(11000+args.seed)
    rg=torch.Generator(device=device).manual_seed(21000+args.seed)
    tg=torch.Generator().manual_seed(31000+args.seed)
    eg=torch.Generator(device=device).manual_seed(41000+args.seed)
    shape=(2,) if args.dataset=='mog' else (1,28,28)
    classifier=None
    train=None
    if args.dataset=='mnist':
        raw=download_mnist(Path(args.data_root))
        train,_=load_split(raw,True)
        test,labels=load_split(raw,False)
        train=train.to(device).float().reshape(-1,1,28,28)/127.5-1
        test=test.to(device).float().reshape(-1,1,28,28)/127.5-1
        write(out/'data_identity.json',{p.name:sha(p) for p in raw.glob('*.gz')})
        target=test[torch.randperm(len(test),device=device,generator=eg)[:args.eval_n]]
        real_other=test[torch.randperm(len(test),device=device,generator=eg)[:args.eval_n]]
        if args.classifier:
            classifier=Classifier().to(device).eval()
            classifier.load_state_dict(torch.load(args.classifier,map_location=device,weights_only=True))
            classifier.requires_grad_(False)
            logits,_=features(classifier,(test+1)/2)
            write(out/'classifier_identity.json',{'sha256':sha(Path(args.classifier)),
                'test_accuracy':float((logits.argmax(1).cpu()==labels).float().mean()),'evaluation_only':True})
    else:
        target=mixture(args.eval_n,eg)
        real_other=mixture(args.eval_n,eg)
    noise=torch.randn(args.eval_n,*shape,device=device,generator=eg)
    target_noise=torch.randn(target.shape,device=device,generator=eg)
    write(out/'real_real.json',metrics(real_other,target,args.dataset,classifier))
    history=[]
    start=time.monotonic()

    def evaluate(update, loss=None):
        model.eval()
        samples=generate(model,noise,args.eval_steps)
        record={'update':update,'seconds':time.monotonic()-start,
            'training_loss':loss,**metrics(samples,target,args.dataset,classifier)}
        history.append(record)
        write(out/'history.json',history)
        np.savez_compressed(out/f'samples_{update:06d}.npz',generated=samples.cpu().numpy(),
            reference=target.cpu().numpy(),noise=noise.cpu().numpy())
        if args.dataset=='mnist':
            save_image(((samples[:64]+1)/2).clamp(0,1),out/f'samples_{update:06d}.png',nrow=8)
            if update==0:
                save_image((target[:64]+1)/2,out/'real_samples.png',nrow=8)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'update':update,'config':config,'source_rng':sg.get_state(),
            'reference_rng':rg.get_state(),'time_rng':tg.get_state(),
            'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None},out/'checkpoint_latest.pt')
        print(json.dumps(record),flush=True)
        model.train()

    evaluate(0)
    for update in range(1,args.updates+1):
        t=float(torch.rand((),generator=tg))
        z=torch.randn(args.batch,*shape,device=device,generator=sg)
        clean=mixture(args.reference_batch,rg) if train is None else train[
            torch.randint(len(train),(args.reference_batch,),device=device,generator=rg)]
        generated=integrate(model,z,end=t,steps=max(1,math.ceil(t*args.train_steps)),
            checkpoint_steps=args.checkpoint_steps)
        loss=objective(args.dataset,generated,clean,t,rg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at {update}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        for group in optimizer.param_groups:
            group['lr']=args.lr*(.1+.9*.5*(1+math.cos(math.pi*(update-1)/args.updates)))
        optimizer.step()
        if update%25==0 or update==1:
            write(out/'status.json',{'state':'running','update':update,'updates':args.updates,
                'seconds':time.monotonic()-start,'loss':float(loss.detach()),'gradient_norm':float(grad),
                'time':t})
        if update%args.eval_every==0 or update==args.updates:
            evaluate(update,float(loss.detach()))

    model.eval()
    refinements={}
    previous=None
    for steps in sorted(set([args.train_steps,args.eval_steps,2*args.eval_steps])):
        generated=generate(model,noise,steps)
        refinements[str(steps)]=metrics(generated,target,args.dataset,classifier)
        if previous is not None:
            refinements[str(steps)]['paired_rms_change']=float((generated-previous).square().mean().sqrt())
        previous=generated
    write(out/'solver_refinement.json',refinements)
    marginals=[]
    arrays={}
    for t in (0.,.25,.5,.75,1.):
        with torch.no_grad():
            x=torch.cat([integrate(model,z,end=t,steps=max(1,math.ceil(t*2*args.eval_steps))) for z in noise.split(128)])
            reference=t*target+(1-t)*target_noise
            bands=(.2,.4,.8,1.6) if args.dataset=='mog' else tuple(
                28*((t*b)**2+(1-t)**2)**.5 for b in (.05,.1,.2,.4))
            marginals.append({'t':t,'heldout_full_space_mmd2':exact_mmd(x,reference,bands)})
            arrays[f'generated_{t}']=x.cpu().numpy()
            arrays[f'reference_{t}']=reference.cpu().numpy()
    write(out/'marginals.json',marginals)
    np.savez_compressed(out/'marginals.npz',**arrays)
    finite=all(math.isfinite(v['exact_mmd2']) for v in refinements.values())
    write(out/'status.json',{'state':'complete','updates':args.updates,'seconds':time.monotonic()-start})
    write(out/'audit.json',{'passed':finite,'meaning':'artifact and finite-value checks; not a quality guarantee',
        'updates_completed':args.updates,'source_hashes_verified':all(sha(source/k)==v for k,v in config['source_hashes'].items()),
        'files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    write(out/'ready.json',{'ready':True,'quality_accepted':False,'note':'Pilot requires visual and independent-seed review'})


def arguments():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['mog','mnist'],required=True)
    p.add_argument('--output',default=str(ROOT/'results/trajectory_mmd_20261102/pilot'))
    p.add_argument('--data-root',default=str(ROOT.parent/'data'))
    p.add_argument('--classifier')
    p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--updates',type=int,default=4000)
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--reference-batch',type=int,default=512)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--train-steps',type=int,default=16)
    p.add_argument('--eval-steps',type=int,default=64)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--eval-n',type=int,default=2048)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--checkpoint-steps',action='store_true')
    p.add_argument('--linear-skip',action='store_true')
    p.add_argument('--stable-field',action='store_true')
    a=p.parse_args()
    if a.stable_field:
        a.linear_skip=True
    if min(a.updates,a.train_steps,a.eval_steps,a.eval_every)<1 or min(a.batch,a.reference_batch,a.eval_n)<2:
        p.error('Positive budgets and batches >= 2 required')
    if a.dataset=='mnist' and a.eval_n>10000:
        p.error('MNIST evaluation uses at most 10000 held-out samples')
    return a


if __name__=='__main__':
    args=arguments()
    try:
        run(args)
    except Exception as error:
        path=Path(args.output)/args.dataset/f'seed_{args.seed}'
        # Never mutate an earlier completed run when overwrite protection fires.
        if path.exists() and not isinstance(error,FileExistsError):
            write(path/'failure.json',{'error':repr(error)})
        raise
