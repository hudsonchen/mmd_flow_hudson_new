"""Write a live, constraint-aware summary of the small reference-flow study."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_storage import new_run_path


def main():
    destination=new_run_path('toy/reference_flow_campaign_20261026')
    destination.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((destination/'protocol.json').read_text())
    stopped={r['study'] for r in json.loads((destination/'early_stops.json').read_text())}
    rows=[]
    studies=[]
    for dataset,folder in [('mog','toy'),('mnist','mnist'),('fashionmnist','fashionmnist')]:
        root=new_run_path(folder)
        if root.exists(): studies.extend((dataset,p) for p in sorted(root.iterdir()))
    for dataset,study in studies:
        if not (study.name.startswith(('riesz_','reference_fm_')) or study.name=='reference_riesz_marginal_scratch_seed0_20261026'):
            continue
        training=study/'training'
        if not (training/'config.json').exists():continue
        config=json.loads((training/'config.json').read_text())
        status=json.loads((training/'status.json').read_text()) if (training/'status.json').exists() else {}
        best=json.loads((training/'best.json').read_text()) if (training/'best.json').exists() else {}
        early_stop=json.loads((study/'early_stop.json').read_text()) if (study/'early_stop.json').exists() else None
        role={'fm':'baseline_only','reference-hybrid':'hybrid_reference_riesz'}.get(config['objective'],'riesz_objective')
        row=dict(dataset=dataset,study=study.name,objective=config['objective'],seed=config['seed'],role=role,
            status='stopped_early' if early_stop else 'stopped_mode_collapse' if study.name in stopped else status.get('status','starting'),
            early_stop_reason=early_stop.get('reason') if early_stop else None,
            current_update=status.get('update'),selected_update=best.get('update'),
            outside_percent=100*best['off_mode_fraction'] if 'off_mode_fraction' in best else None,covered_modes=best.get('covered_modes'),
            component_std=best.get('component_coordinate_std'),mode_mass_l1=best.get('mode_mass_l1'),
            radial_ks=best.get('radial_ks'),fid=best.get('fid'),
            sliced_w2=best.get('sliced_w2'),training_rollout_steps=config.get('train_rollout_steps'),
            sampling_steps=config['sample_steps'],solver=config['solver'],spatial_fourier=config.get('spatial_fourier',False),
            affinity=config.get('affinity','row'),feature_space=config.get('feature_space'),
            endpoint_tail_scale=config.get('endpoint_tail_scale',0.),endpoint_tail_max_weight=config.get('endpoint_tail_max_weight',4.),
            reference_coupling=config.get('reference_coupling','paired' if config['objective']=='coupled' else 'independent'),
            parameterization=config.get('parameterization','velocity'),decode_primed=config.get('decode_primed',False),
            pixel_statistics_weight=config.get('pixel_statistics_weight',0.),reference_whitening_floor=config.get('reference_whitening_floor',0.),
            pixel_gradient_weight=config.get('pixel_gradient_weight',0.),
            spatial_feature_weight=config.get('spatial_feature_weight',0.),spatial_feature_locations=config.get('spatial_feature_locations',16),
            reference_time_power=config.get('reference_time_power',0.),
            reprime_features=config.get('reprime_features',False),priming_updates_in_stage=config.get('prime_steps',0),
            reference_weight=config.get('reference_weight') if config['objective']=='reference-hybrid' else None,
            riesz_weight=config.get('riesz_weight'),
            parent=config.get('fork'),path=str(study))
        rows.append(row)
    report=dict(initialization_constraint=protocol['constraint'],excluded_fm_initialized_studies=protocol['excluded_fm_initialized_studies'],
        rows=rows,interpretation='Best validation checkpoints, not a fresh-sample or ancestry audit. Roles describe the current objective; audit parent chains before pure-Riesz claims. Check mode coverage and spread alongside outside rate. FID uses the fixed training-image reference; held-out test audits are separate.')
    (destination/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    if rows:
        with (destination/'progress.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    for row in rows:
        if row['outside_percent'] is not None:
            print(f"{row['study']}: {row['outside_percent']:.2f}% outside, {row['covered_modes']} modes, {row['status']}")
        elif row['fid'] is not None:
            print(f"{row['study']}: FID {row['fid']:.2f}, {row['status']}")


if __name__=='__main__':main()
