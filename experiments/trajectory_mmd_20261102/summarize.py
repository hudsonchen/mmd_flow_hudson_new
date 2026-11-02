"""Aggregate completed pilots and check the MNIST output-subspace diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from audit import audit
from run import ROOT, sha, write


def mean_sd(values):
    return {'mean':float(np.mean(values)),'sample_sd':float(np.std(values,ddof=1)),
        'per_seed':values}


def subspace_diagnostic(path):
    state=torch.load(path/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    weights=state['model']
    matrix=torch.cat([weights['net.6.weight'],weights['net.6.bias'][:,None]],1).double()
    u,s,_=torch.linalg.svd(matrix,full_matrices=False)
    rank=int((s > s.max()*1e-10).sum())
    q=u[:,:rank]
    data=np.load(path/f"samples_{state['update']:06d}.npz")
    def project(name):
        x=torch.from_numpy(data[name]).flatten(1).double()
        return x-(x@q)@q.T
    x,z,y=project('generated'),project('noise'),project('reference')
    multiplier=float((x*z).sum()/z.square().sum())
    degrees=784-rank
    return {'velocity_output_span_rank':rank,'orthogonal_dimensions':degrees,
        'projected_source_std':float((z.var(0).sum()/degrees).sqrt()),
        'projected_generated_std':float((x.var(0).sum()/degrees).sqrt()),
        'projected_reference_std':float((y.var(0).sum()/degrees).sqrt()),
        'fitted_noise_multiplier':multiplier,
        'projected_change_rms':float((x-z).square().mean().sqrt()),
        'residual_after_scalar_contraction_rms':float((x-multiplier*z).square().mean().sqrt())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT/'results/trajectory_mmd_20261102')
    args=p.parse_args();base=args.root
    torch.set_num_threads(1)
    mog_paths=[base/'cpu_development/mog/seed_0',base/'cpu_replication/mog/seed_1',base/'cpu_replication/mog/seed_2']
    mnist_paths={'baseline':base/'cpu_mnist_pilot/mnist/seed_0',
        'learned_linear_skip':base/'cpu_mnist_skip_pilot/mnist/seed_0',
        'stable_field':base/'cpu_mnist_stable_pilot/mnist/seed_0'}
    receipts=[audit(path) for path in mog_paths+list(mnist_paths.values())]
    fresh=[]
    for path in mog_paths:
        folder=path/'fresh_validation'
        receipt=json.loads((folder/'audit.json').read_text())
        for filename,expected in receipt['files'].items():
            assert sha(folder/filename)==expected
        result=json.loads((folder/'metrics.json').read_text())
        assert result['checkpoint_sha256']==sha(path/'checkpoint_latest.pt')
        fresh.append(result)
    mog={key:mean_sd([r['solver']['128'][key] for r in fresh])
        for key in ['exact_mmd2','sliced_w2','within_3sigma']}
    fractions=np.array([r['solver']['128']['mode_fractions'] for r in fresh])
    mog.update(mode_fraction_range=[float(fractions.min()),float(fractions.max())],
        real_real_within_3sigma=mean_sd([r['real_real']['within_3sigma'] for r in fresh]),
        max_64_to_128_mmd2_change=max(abs(r['solver']['128']['exact_mmd2']-r['solver']['64']['exact_mmd2']) for r in fresh),
        max_64_to_128_paired_rms=max(r['solver']['128']['paired_rms_change'] for r in fresh),
        interpretation='Eight modes recovered in all three seeds; excess tails remain. Seed 0 is development; seeds 1/2 are replication.')
    mnist={}
    for name,path in mnist_paths.items():
        history=json.loads((path/'history.json').read_text())
        mnist[name]={'initial':history[0],'final':history[-1],
            'real_real':json.loads((path/'real_real.json').read_text()),
            'refinement':json.loads((path/'solver_refinement.json').read_text()),
            'subspace_diagnostic':subspace_diagnostic(path) if name!='stable_field' else
                {'applicable':False,'reason':'Coordinatewise tanh removes the fixed linear output-span restriction.'}}
        diagnostic=path/'raw_tail_diagnostic'
        receipt=json.loads((diagnostic/'audit.json').read_text())
        for filename,expected in receipt['files'].items():
            assert sha(diagnostic/filename)==expected
        mnist[name]['raw_tail_diagnostic']=json.loads((diagnostic/'metrics.json').read_text())
    result={'scope':'completed exploratory CPU pilots, no GPU or cluster jobs submitted',
        'mog':mog,'mnist':mnist,'audits':receipts,
        'sources':{str(path.relative_to(base)):sha(path/'checkpoint_latest.pt') for path in mog_paths+list(mnist_paths.values())}}
    write(base/'study_summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
