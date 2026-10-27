"""Compare saved staged toy models, separating endpoints from partial rollouts."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from run_storage import new_run_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    base=new_run_path('toy').resolve()
    output=new_run_path(Path('toy')/args.name).resolve()
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(output/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    studies=[
        ('Riesz MMD\n5k per map','gaussian_cascade_8modes_3seeds_5k_20261026','mmd_actual'),
        ('Softmax Riesz, 3 temperatures\n1k per map','gaussian_cascade_8modes_3seeds_20261026','softmax_actual'),
        ('Gaussian MMD + Fourier\n2k per map; final map 10k','gaussian_staged_rbf_fourier_2k_last10k_3seeds_20261026','rbf_fourier_actual'),
    ]
    active=base/'gaussian_staged_softmax_tau01_50k_per_map_3seeds_20261026'
    source=active/'source/scripts/test_gaussian_cascade.py'
    spec=importlib.util.spec_from_file_location('frozen_gaussian',source)
    frozen=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    seeds=[0,1,2]
    statuses=[json.loads((active/f'softmax_actual_seed{s}/status.json').read_text()) for s in seeds]
    common_stage=min(max(int(p.stem.split('_')[-1]) for p in
        (active/f'softmax_actual_seed{s}').glob('checkpoint_stage_*.pt')) for s in seeds)
    manifest=dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        active_study=str(active),active_statuses=statuses,common_completed_stage=common_stage,
        completed_studies=[dict(label=label,path=str(base/name),recipe=recipe) for label,name,recipe in studies],
        plot_samples_per_panel=2048,seeds=seeds,
        caution='Different budgets and model capacities. Partial active rollouts are compared only at a common target time, not to final images.')
    shutil.copy2(__file__,output/'plot_source.py')
    manifest['plot_source_sha256']=hashlib.sha256((output/'plot_source.py').read_bytes()).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    metrics={}
    for label,name,recipe in studies:
        rows=[json.loads((base/name/f'{recipe}_seed{s}/final.json').read_text()) for s in seeds]
        metrics[recipe]={key:dict(mean=float(np.mean([r[key] for r in rows])),
            std=float(np.std([r[key] for r in rows],ddof=1)))
            for key in ['off_mode_fraction','sliced_w2','nearest_center_rms','mode_mass_l1']}
        metrics[recipe]['parameter_count']=rows[0]['parameter_count']
        metrics[recipe]['total_updates']=rows[0]['update']

    def panel(ax,points,title,seed,color):
        outside=float(np.any(np.abs(points)>3.5,axis=1).mean())
        ax.scatter(points[:,0],points[:,1],s=2,alpha=.4,color=color,rasterized=True)
        ax.set(xlim=(-3.5,3.5),ylim=(-3.5,3.5),aspect='equal',xticks=[-2,0,2],yticks=[-2,0,2])
        if seed==0:ax.set_title(title,fontsize=10)
        if outside:ax.text(.02,.02,f'{100*outside:.2f}% outside frame',transform=ax.transAxes,fontsize=7)
    colors=['#426b99','#a96051','#42926e','#9266a7']
    fig,axes=plt.subplots(3,4,figsize=(13,9.5),sharex=True,sharey=True)
    for seed in seeds:
        reference=np.load(base/studies[0][1]/f'mmd_actual_seed{seed}/samples.npz')['target_10']
        panel(axes[seed,0],reference,'Target mixture',seed,'#333333')
        axes[seed,0].set_ylabel(f'Seed {seed}')
        for j,(label,name,recipe) in enumerate(studies):
            points=np.load(base/name/f'{recipe}_seed{seed}/samples.npz')['stage_10']
            panel(axes[seed,j+1],points,label,seed,colors[j])
            off=json.loads((base/name/f'{recipe}_seed{seed}/final.json').read_text())['off_mode_fraction']
            axes[seed,j+1].text(.02,.94,f'Outside clusters: {off:.1%}',transform=axes[seed,j+1].transAxes,fontsize=8)
    fig.suptitle('Completed ten-map cascades: same samples and axes; training budgets and architectures differ')
    fig.tight_layout(rect=(0,0,1,.97))
    for ext in ['png','pdf']:fig.savefig(output/f'completed_staged_comparison.{ext}',dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(3,5,figsize=(16,9.5),sharex=True,sharey=True)
    active_metrics=[]
    for seed in seeds:
        reference,_=frozen.sample_path(2048,common_stage/10,
            torch.Generator().manual_seed(3000+seed*100+common_stage-1))
        panel(axes[seed,0],reference.numpy(),f'Target at t={common_stage/10:.1f}',seed,'#333333')
        axes[seed,0].set_ylabel(f'Seed {seed}')
        for j,(label,name,recipe) in enumerate(studies):
            points=np.load(base/name/f'{recipe}_seed{seed}/samples.npz')[f'stage_{common_stage:02d}']
            panel(axes[seed,j+1],points,label,seed,colors[j])
        checkpoint=active/f'softmax_actual_seed{seed}/checkpoint_stage_{common_stage:02d}.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        maps=[]
        for width,weights in zip(state['widths'],state['maps']):
            model=frozen.Map(width)
            model.load_state_dict(weights)
            model.eval().requires_grad_(False)
            maps.append(model)
        noise=torch.randn(2048,2,generator=torch.Generator().manual_seed(2000+seed))
        points=frozen.compose(maps,noise)
        active_metrics.append(dict(seed=seed,completed_stage=common_stage,
            checkpoint=str(checkpoint),**frozen.diagnostics(points,reference,common_stage/10)))
        panel(axes[seed,4],points.numpy(),'NEW softmax Riesz, fixed tau=0.1\n50k per completed map',seed,colors[3])
    fig.suptitle(f'Matched intermediate target: first {common_stage} map(s) only. The fixed-tau run is unfinished.')
    fig.tight_layout(rect=(0,0,1,.97))
    for ext in ['png','pdf']:fig.savefig(output/f'matched_stage_comparison.{ext}',dpi=180)
    plt.close(fig)
    metrics['active_matched_stage']=active_metrics
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),metrics=metrics),indent=2))


if __name__=='__main__':main()
