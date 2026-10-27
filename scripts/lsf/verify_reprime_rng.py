"""Verify matched saved RNG states after an extra reconstruction-encoder stage."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from run_storage import new_run_path
from small_flow import validate_riesz_parent,verify_reconstruction_encoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    output=new_run_path(args.output);output.mkdir(parents=True,exist_ok=True)
    payloads=[path.read_bytes() for path in [args.reference,args.candidate]]
    checkpoints=[torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False) for payload in payloads]
    for checkpoint in checkpoints:validate_riesz_parent(checkpoint)
    reference,candidate=checkpoints
    keys=['objective','parameterization','affinity','seed','steps','lr','batch','target_batch','endpoint_batch',
        'width','deep','spatial_fourier','feature_space','decode_primed','pixel_statistics_weight','pixel_gradient_weight',
        'spatial_feature_weight','spatial_feature_locations','reference_time_power','reference_whitening_floor',
        'train_rollout_steps','sample_steps','solver','tau','normalization','repulsion_weight','horizon','riesz_weight','marginal_weight']
    differences={key:[reference['config'].get(key),candidate['config'].get(key)] for key in keys
        if reference['config'].get(key)!=candidate['config'].get(key)}
    parent_hashes=[hashlib.sha256(Path(cp['config']['fork']).read_bytes()).hexdigest() for cp in checkpoints]
    equal={key:torch.equal(reference[key],candidate[key]) for key in ['torch_rng','cuda_rng']}
    report=dict(reference=str(args.reference),candidate=str(args.candidate),
        checkpoint_sha256=[hashlib.sha256(payload).hexdigest() for payload in payloads],
        updates=[cp['update'] for cp in checkpoints],parent_sha256=parent_hashes,
        sampling_configuration_differences=differences,rng_state_equal=equal,
        encoder_provenance=[verify_reconstruction_encoder(cp) for cp in checkpoints],
        interpretation='Exact saved RNG-state comparison at equal update counts with matched sampling configuration and generator parent. No per-draw tensor hashes or quality claim.')
    report['passed']=all(equal.values()) and not differences and reference['update']==candidate['update'] and parent_hashes[0]==parent_hashes[1]
    snapshot=output/f"candidate_update{candidate['update']}.pt"
    if snapshot.exists():raise FileExistsError(snapshot)
    snapshot.write_bytes(payloads[1])
    hashes={}
    for name in ['small_flow.py','backbone.py','loss.py','sinkhorn.py','run_storage.py',str(Path(__file__).relative_to(ROOT))]:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    report['source_hashes']=hashes
    (output/'comparison.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(report,indent=2,default=str))
    if not report['passed']:raise RuntimeError('Matched transport RNG verification failed')


if __name__=='__main__':main()
