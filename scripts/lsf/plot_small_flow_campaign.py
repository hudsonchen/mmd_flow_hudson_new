"""Plot measured campaign results; no training or additional FID evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_storage import new_run_path


def main():
    destination=new_run_path('toy/reference_flow_campaign_20261026')
    destination.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(destination/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    inputs={}

    def read(path):
        payload=path.read_bytes()
        inputs[str(path)]=hashlib.sha256(payload).hexdigest()
        return json.loads(payload)

    audit=read(new_run_path('toy/mog_row_column_three_seed_audit_20261027/audit/audit.json'))
    groups=[audit['rows'][:3],audit['rows'][3:6]]
    panels=[('off_mode_fraction','Outside (%)',100),('mode_mass_l1','Mode mass L1',1),
            ('radial_ks','Radial KS',1),('sliced_w2_analytic_target','Sliced W2 to exact target',1)]
    fig,axes=plt.subplots(1,4,figsize=(12,3.5))
    for ax,(key,title,scale) in zip(axes,panels):
        upper=max(row[key]*scale for rows in groups for row in rows)*1.15
        for index,rows in enumerate(groups):
            values=np.array([r[key] for r in rows])*scale
            ax.scatter(index+np.linspace(-.08,.08,len(values)),values,s=30,color=['#2878b5','#8c4fb5'][index])
            ax.plot([index-.2,index+.2],[values.mean()]*2,color='black',linewidth=1.5)
        ax.set(xticks=[0,1],xticklabels=['FM\n100 NFE','Pure Riesz\n10 NFE'],ylabel=title,xlim=(-.45,1.45),ylim=(0,upper))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('MoG: 50,000 fresh samples per checkpoint; three training seeds; lower is better',fontsize=11)
    fig.text(.5,.01,'Dots: individual training seeds. Bars: means. Pure Riesz uses row/column refinement and terminal noise σ=0.05.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    for extension in ['png','pdf']: fig.savefig(destination/f'mog_comparison.{extension}',dpi=180)
    plt.close(fig)

    confirmation_path=new_run_path('toy/mog_tail_three_seed_independent_confirmation_20261027/audit/audit.json')
    if confirmation_path.exists():
        confirmation=read(confirmation_path)
        groups=[[r for r in confirmation['rows'] if r['study'].startswith(prefix)] for prefix in
            ['reference_fm_fourier_mog_','riesz_row_column_mog_refine_','riesz_row_column_mog_tail_weighted_refine_']]
        if confirmation['count']!=200000 or confirmation['noise_seed']!=8855123 or any(len(g)!=3 for g in groups):
            raise ValueError('Unexpected MoG confirmation protocol')
        fig,axes=plt.subplots(1,4,figsize=(13,3.8))
        for ax,(key,title,scale) in zip(axes,panels):
            upper=max(row[key]*scale for rows in groups for row in rows)*1.15
            if key=='off_mode_fraction':
                upper=max(upper,5.5)
                ax.axhline(5,color='#bb4444',ls='--',linewidth=.8)
            for index,rows in enumerate(groups):
                values=np.array([r[key] for r in rows])*scale
                ax.scatter(index+np.linspace(-.08,.08,len(values)),values,s=30,color=['#2878b5','#8c4fb5','#2ca02c'][index])
                ax.plot([index-.2,index+.2],[values.mean()]*2,color='black',linewidth=1.5)
            ax.set(xticks=[0,1,2],xticklabels=['FM\n100 NFE','Riesz parent\n10 NFE','Riesz weighted\n10 NFE'],
                ylabel=title,xlim=(-.45,2.45),ylim=(0,upper))
            ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
        fig.suptitle('MoG confirmation: 200,000 samples per checkpoint, new common noise, three training seeds',fontsize=11)
        fig.text(.5,.015,'Dots: training seeds; bars: means. Riesz uses terminal noise σ=0.05 and no FM weights. Red dashed line: 5% outside target.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.93))
        for extension in ['png','pdf']:fig.savefig(destination/f'mog_independent_confirmation.{extension}',dpi=180)
        plt.close(fig)

    configurations={
        'mnist':[
            ('reference_fm_mnist_deep16_seed0_20261026','FM, 100 NFE'),
            ('riesz_reference_hybrid_mnist_deep16_w01_seed0_20261027','Hybrid Riesz w=0.1, 20 NFE'),
            ('riesz_reference_hybrid_mnist_deep16_w05_seed0_20261027','Hybrid Riesz w=0.5, 20 NFE'),
            ('riesz_only_mnist_deep16_masked_features_seed0_20261026','Pure Riesz, primed features'),
            ('riesz_row_column_mnist_deep16_continue_seed0_20261027','Pure Riesz, row/column'),
            ('riesz_row_column_mnist_primed_decoder_seed0_20261027','Pure Riesz, frozen decoder'),
            ('riesz_denoiser_mnist_deep16_masked_seed0_20261027','Pure Riesz, denoiser'),
            ('riesz_denoiser_mnist_deep16_continue_seed0_20261027','Pure Riesz, denoiser continuation'),
            ('riesz_row_column_mnist_pixel_moments_seed0_20261027','Pure Riesz, local moments'),
            ('riesz_row_column_mnist_reference_whitened_seed0_20261027','Pure Riesz, reference scaling')],
        'fashionmnist':[
            ('reference_fm_fashionmnist_deep16_seed0_20261027','FM, 100 NFE'),
            ('riesz_reference_hybrid_fashionmnist_deep16_w01_seed0_20261027','Hybrid Riesz w=0.1, 20 NFE'),
            ('riesz_row_column_fashionmnist_primed_decoder_seed0_20261027','Pure Riesz, frozen decoder'),
            ('riesz_reference_whitened_fashionmnist_decoder_refine_seed0_20261027','Pure Riesz, decoder + reference scaling'),
            ('riesz_denoiser_fashionmnist_deep16_masked_seed0_20261027','Pure Riesz, denoiser'),
            ('riesz_denoiser_fashionmnist_local_moments_refine_seed0_20261027','Pure Riesz, denoiser + local moments (50 NFE)'),
            ('riesz_denoiser_fashionmnist_gradient_features_refine_seed0_20261027','Pure Riesz, denoiser + spatial derivatives (50 NFE)'),
            ('riesz_row_column_fashionmnist_pixel_moments_seed0_20261027','Pure Riesz, local moments'),
            ('riesz_row_column_fashionmnist_large_batch_refine_seed0_20261027','Pure Riesz, larger particle batch'),
            ('riesz_row_column_fashionmnist_coupled_reference_refine_seed0_20261027','Pure Riesz, coupled reference particles'),
            ('riesz_row_column_fashionmnist_coupled_loo_refine_seed0_20261027','Pure Riesz, coupled reference, self-target excluded'),
            ('riesz_local_loo_denoiser_fashionmnist_seed0_20261027','Pure Riesz, local reference denoiser (50 NFE)'),
            ('riesz_local_loo_velocity_fashionmnist_seed0_20261027','Pure Riesz, local reference velocity (50 NFE)')]}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,(dataset,studies) in zip(axes,configurations.items()):
        for name,label in studies:
            training=new_run_path(Path(dataset)/name/'training')
            files=sorted(training.glob('metrics_*.json'))
            if not files: continue
            config=read(training/'config.json')
            offset=0
            parent=config.get('fork')
            seen=set()
            while parent:
                path=Path(parent).resolve()
                if path in seen: raise ValueError('Cyclic checkpoint ancestry')
                seen.add(path)
                payload=path.read_bytes();inputs[str(path)]=hashlib.sha256(payload).hexdigest()
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                offset+=checkpoint['update'];parent=checkpoint['config'].get('fork')
            rows=[read(p) for p in files]
            ax.plot([offset+r['update'] for r in rows],[r['fid'] for r in rows],marker='o',markersize=4,label=label)
        ax.set(title=dataset.upper(),xlabel='Cumulative transport optimizer updates',ylabel='Training-reference FID-10K (log scale)',yscale='log')
        from matplotlib.ticker import FuncFormatter
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
        ax.grid(alpha=.2)
        ax.legend(fontsize=7,loc='best')
    fig.suptitle('Image training: measured checkpoints only; lower FID is better',fontsize=12)
    fig.text(.5,.01,'Updates exclude reconstruction priming; updates have different compute costs. Decoder-only postprocessing diagnostics are not plotted.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.94))
    for extension in ['png','pdf']:fig.savefig(destination/f'image_learning_curves.{extension}',dpi=180)
    plt.close(fig)

    fashion_refinements=[
        ('riesz_denoiser_fashionmnist_rollout20_refine_seed0_20261027','20 training steps'),
        ('riesz_denoiser_fashionmnist_spatial_direct_features_refine_seed0_20261027','10 steps + spatial features'),
        ('riesz_denoiser_fashionmnist_large_batch_refine_seed0_20261027','10 steps + larger particle batches'),
        ('riesz_denoiser_fashionmnist_rollout20_control_seed0_20261027','20 steps: unchanged continuation'),
        ('riesz_denoiser_fashionmnist_rollout20_time_weighted_seed0_20261027','20 steps + reference time weighting'),
        ('riesz_denoiser_fashionmnist_rollout20_spatial_seed0_20261027','20 steps + spatial features'),
        ('riesz_denoiser_fashionmnist_encoder20k_spatial_refine_seed0_20261027','20 steps + spatial features + 20k encoder priming'),
        ('riesz_denoiser_fashionmnist_spatial_time_weighted_refine_seed0_20261027','Spatial features + reference time weighting'),
        ('riesz_denoiser_fashionmnist_spatial_continuation_control_seed0_20261027','Spatial features: unchanged continuation')]
    fig,ax=plt.subplots(figsize=(9,5))
    for name,label in fashion_refinements:
        training=new_run_path(Path('fashionmnist')/name/'training')
        files=sorted(training.glob('metrics_*.json'))
        if not files:continue
        cfg=read(training/'config.json')
        parent=cfg['fork'];offset=0;parent_fid=None;seen=set()
        while parent:
            path=Path(parent).resolve()
            if path in seen:raise ValueError('Cyclic refinement ancestry')
            seen.add(path)
            payload=path.read_bytes();inputs[str(path)]=hashlib.sha256(payload).hexdigest()
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            if parent_fid is None:
                parent_fid=read(Path(checkpoint['config']['output'])/f"metrics_{checkpoint['update']:07d}.json")['fid']
            offset+=checkpoint['update'];parent=checkpoint['config'].get('fork')
        rows=[read(path) for path in files]
        ax.plot([offset]+[offset+row['update'] for row in rows],[parent_fid]+[row['fid'] for row in rows],marker='o',label=label)
    fm=read(new_run_path('fashionmnist/reference_fm_fashionmnist_deep16_seed0_20261027/training/best.json'))
    ax.axhline(fm['fid'],color='black',linestyle='--',label=f"FM baseline: {fm['fid']:.2f} (100 NFE)")
    ax.set(xlabel='Cumulative transport optimizer updates',ylabel='Training-reference FID-10K',title='FashionMNIST refinements: measured parent and child checkpoints')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.text(.5,.015,'Riesz: 50 NFE, reconstruction-only ancestry. Priming excluded; the longer-encoder branch adds 20k updates. Pending results omitted.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,1))
    for extension in ['png','pdf']:fig.savefig(destination/f'fashion_refinement_curves.{extension}',dpi=180)
    plt.close(fig)

    sampler_audits=[
        'fashion_fm_matched_sampler_reference_20261027',
        'fashion_rollout20_time_weighting_sampler_audit_20261027',
        'fashion_rollout20_spatial_sampler_audit_20261027',
        'fashion_encoder20k_spatial_sampler_audit_20261027']
    sampler_groups={};signature=None
    for name in sampler_audits:
        path=new_run_path(Path('fashionmnist')/name/'audit/audit.json')
        if not path.exists():continue
        for row in read(path)['rows']:
            current=(row['dataset'],row['reference_split'],row['noise_seed'],row['generated_count'],row['reference_count'],row['fid_batch_size'])
            if signature is not None and current!=signature:raise ValueError('Sampler audit protocols differ')
            signature=current
            sampler_groups.setdefault(row['study'],[]).append(row)
    if sampler_groups:
        labels={
            'reference_fm_fashionmnist_deep16_seed0_20261027':'FM',
            'riesz_denoiser_fashionmnist_rollout20_control_seed0_20261027':'Riesz: unchanged-loss control',
            'riesz_denoiser_fashionmnist_rollout20_time_weighted_seed0_20261027':'Riesz: time weighting',
            'riesz_denoiser_fashionmnist_rollout20_spatial_seed0_20261027':'Riesz: spatial features',
            'riesz_denoiser_fashionmnist_encoder20k_spatial_refine_seed0_20261027':'Riesz: spatial + longer encoder priming'}
        fig,ax=plt.subplots(figsize=(8,5))
        for name,rows in sampler_groups.items():
            rows=sorted(rows,key=lambda row:row['network_evaluations'])
            ax.plot([row['network_evaluations'] for row in rows],[row['fid'] for row in rows],
                marker='D' if name.startswith('reference_fm') else 'o',label=labels.get(name,name))
        ax.set(xlabel='Network evaluations per sample',ylabel='Training-reference FID-10K',
            title='FashionMNIST: common fresh-noise sampling comparison',xticks=[20,50,100])
        ax.grid(alpha=.2);ax.legend(fontsize=9)
        fig.text(.5,.015,'Development audit; identical noise/reference protocol. Only completed measurements are shown; no held-out test selection.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,1))
        for extension in ['png','pdf']:fig.savefig(destination/f'fashion_sampler_comparison.{extension}',dpi=180)
        plt.close(fig)

    image_audit=new_run_path('mnist/mnist_pixel_moments_fresh_test_audit_20261027/audit/audit.json')
    final_image_audit=new_run_path('mnist/mnist_pixel_moments_final_fresh_test_audit_20261027/audit/audit.json')
    if final_image_audit.exists() and len(json.loads(final_image_audit.read_text())['rows'])==2:
        image_audit=final_image_audit
    if image_audit.exists():
        comparison=read(image_audit)
        fig,axes=plt.subplots(1,2,figsize=(9,5))
        for ax,row,label in zip(axes,comparison['rows'],['Flow matching','Pure Riesz + local pixel statistics']):
            path=image_audit.parent/row['study']/f"samples_{row['update']:07d}.npy"
            inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            samples=np.load(path)[:100].reshape(10,10,28,28)
            grid=samples.transpose(0,2,1,3).reshape(280,280)
            ax.imshow(grid,cmap='gray',vmin=0,vmax=255)
            ax.set_title(f"{label}\nTest FID {row['fid']:.2f}; {row['network_evaluations']} network evaluations",fontsize=11)
            ax.axis('off')
        fig.suptitle('MNIST: common fresh noise, held-out reference, 10,000 generated samples',fontsize=11)
        fig.text(.5,.02,'First 100 samples shown without selection. One training seed; Riesz uses reconstruction priming and no FM weights.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.94))
        for extension in ['png','pdf']:fig.savefig(destination/f'mnist_fresh_test_comparison.{extension}',dpi=180)
        plt.close(fig)

    fashion_final_paths=[new_run_path('fashionmnist/fashion_fm_independent_noise_test_audit_20261027/audit/audit.json'),
        new_run_path('fashionmnist/fashion_encoder20k_independent_noise_test_audit_20261027/audit/audit.json')]
    if all(p.exists() for p in fashion_final_paths):
        final_rows=[(row,path.parent) for path in fashion_final_paths for row in read(path)['rows']]
        expected={('reference_fm_fashionmnist_deep16_seed0_20261027',100),
                  ('riesz_denoiser_fashionmnist_encoder20k_spatial_refine_seed0_20261027',20),
                  ('riesz_denoiser_fashionmnist_encoder20k_spatial_refine_seed0_20261027',100)}
        if {(r['study'],r['network_evaluations']) for r,_ in final_rows}==expected:
            signatures={(r['dataset'],r['reference_split'],r['noise_seed'],r['generated_count'],r['reference_count'],r['fid_batch_size']) for r,_ in final_rows}
            if signatures!={('fashionmnist','test',8865432,10000,10000,32)}:
                raise ValueError('Final FashionMNIST audit protocols differ')
            final_rows.sort(key=lambda item:(not item[0]['study'].startswith('reference_fm'),-item[0]['network_evaluations']))
            fig,axes=plt.subplots(1,3,figsize=(12,4.8))
            for ax,(row,folder) in zip(axes,final_rows):
                tag=f"{row['study']}_{row['solver']}{row['sample_steps']}"
                path=folder/tag/f"samples_{row['update']:07d}.npy"
                inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                samples=np.load(path)[:100].reshape(10,10,28,28)
                grid=samples.transpose(0,2,1,3).reshape(280,280)
                ax.imshow(grid,cmap='gray',vmin=0,vmax=255)
                label='Flow matching' if row['study'].startswith('reference_fm') else 'Pure Riesz + reconstruction features'
                ax.set_title(f"{label}\nTest FID {row['fid']:.2f}; {row['network_evaluations']} evaluations",fontsize=10)
                ax.axis('off')
            fig.suptitle('FashionMNIST: independent common noise, held-out reference, 10,000 samples',fontsize=12)
            fig.text(.5,.035,'First 100 samples, no selection. One training seed; Riesz adds reconstruction-only priming and never uses FM weights.',ha='center',fontsize=8)
            fig.text(.5,.005,'100-evaluation comparison is primary; 20-evaluation Riesz is a prespecified efficiency check. Priming compute differs.',ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.07,1,.93))
            for extension in ['png','pdf']:fig.savefig(destination/f'fashion_independent_test_comparison.{extension}',dpi=180)
            plt.close(fig)

    reference_audit=new_run_path('toy/reference_gradient_diagnostic_20261027/diagnostic.json')
    loo_audit=new_run_path('toy/reference_gradient_loo_diagnostic_20261027/diagnostic.json')
    if reference_audit.exists() and loo_audit.exists():
        gradient_rows=read(reference_audit)['rows']+read(loo_audit)['rows']
        selected={(r['coupling'],r['perturbation']):r for r in gradient_rows if r['time']==.5}
        methods=['independent','paired','paired-loo']
        labels=['Independent','Paired','Paired, self-target excluded']
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        values=[selected[(m,'population_correct')]['gradient_mean'][1] for m in methods]
        errors=[1.96*selected[(m,'population_correct')]['gradient_mean_standard_error'][1] for m in methods]
        axes[0].errorbar(range(3),values,yerr=errors,fmt='o',capsize=5,color='#2878b5')
        axes[0].axhline(0,color='black',linewidth=.8)
        axes[0].set(xticks=range(3),xticklabels=labels,ylabel='Mean radial gradient ± 1.96 Monte Carlo SE',title='Bias at population-correct transport')
        for index,(perturbation,coordinate,label) in enumerate([('mean_error',0,'Mean error'),('scale_error',1,'Scale error')]):
            snr=[selected[(m,perturbation)]['gradient_mean'][coordinate]/selected[(m,perturbation)]['gradient_std'][coordinate] for m in methods]
            axes[1].bar(np.arange(3)+(index-.5)*.3,snr,width=.3,label=label)
        axes[1].set(xticks=range(3),xticklabels=labels,ylabel='Mean corrective gradient / per-batch SD',title='Corrective signal with known parameter error')
        axes[1].legend()
        for ax in axes:ax.tick_params(axis='x',labelsize=8);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Gaussian reference diagnostic: 16 dimensions, 128 particles, 128 trials, t=0.5',fontsize=11)
        fig.text(.5,.01,'Positive signal restores the known error; negative signal amplifies it. Analytic transport; no trained teacher or image-quality claim.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.92))
        for extension in ['png','pdf']:fig.savefig(destination/f'reference_gradient_diagnostic.{extension}',dpi=180)
        plt.close(fig)

    balance=new_run_path('fashionmnist/fashion_reference_gradient_balance_20261027/diagnostics.json')
    full_balance=new_run_path('fashionmnist/fashion_reference_gradient_balance_full_batch_20261027/diagnostics.json')
    if full_balance.exists():balance=full_balance
    if balance.exists():
        diagnostic=read(balance)
        times=sorted({r['time'] for r in diagnostic['rows']})
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for key,label in [('reference_to_endpoint_ratio','Recorded weighting'),('remaining_weighted_ratio','Proposed extra factor (1-t)')]:
            values=np.array([[r[key] for r in diagnostic['rows'] if r['time']==t] for t in times])
            axes[0].plot(times,np.median(values,axis=1),marker='o',label=label)
            axes[0].fill_between(times,np.quantile(values,.25,axis=1),np.quantile(values,.75,axis=1),alpha=.15)
        axes[0].axhline(1,color='black',linestyle='--',linewidth=.8)
        axes[0].set(yscale='log',xlabel='Reference time t',ylabel='Reference / endpoint gradient norm',title='Median and interquartile range')
        axes[0].legend(fontsize=8)
        values=np.array([[r['endpoint_reference_cosine'] for r in diagnostic['rows'] if r['time']==t] for t in times])
        axes[1].plot(times,np.median(values,axis=1),marker='o',color='#8c4fb5')
        axes[1].fill_between(times,np.quantile(values,.25,axis=1),np.quantile(values,.75,axis=1),alpha=.15,color='#8c4fb5')
        axes[1].axhline(0,color='black',linewidth=.8)
        axes[1].set(xlabel='Reference time t',ylabel='Gradient cosine',title='Endpoint–reference alignment',ylim=(-1,1))
        for ax in axes:ax.grid(alpha=.2)
        fig.suptitle(f"FashionMNIST: fixed pure-Riesz denoiser; {diagnostic['config']['repeats']} independent training batches of {diagnostic['config']['batch']}",fontsize=11)
        fig.text(.5,.01,'No optimizer updates. Fixed-checkpoint gradient measurements; proposed time weighting has not yet been tested for image quality.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.05,1,.93))
        for extension in ['png','pdf']:fig.savefig(destination/f'fashion_reference_gradient_balance.{extension}',dpi=180)
        plt.close(fig)

    source=Path(__file__).read_bytes();source_hash=hashlib.sha256(source).hexdigest()
    archive=destination/'plot_sources'/source_hash
    script=archive/'scripts/lsf/plot_small_flow_campaign.py'
    script.parent.mkdir(parents=True,exist_ok=True)
    script.write_bytes(source)
    storage_source=(ROOT/'run_storage.py').read_bytes()
    (archive/'run_storage.py').write_bytes(storage_source)
    (destination/'plot_inputs.json').write_text(json.dumps(dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source_hash,run_storage_sha256=hashlib.sha256(storage_source).hexdigest(),
        runtime=dict(torch=torch.__version__,matplotlib=matplotlib.__version__),inputs=inputs),indent=2)+'\n')
    print(destination)


if __name__=='__main__':main()
