"""Render a recorded comparison without evaluating models or changing metrics."""
import argparse
import os
from pathlib import Path
import json
import shutil
import sys

sys.path.append('/appl/python-3.12/lib/python3.12/site-packages')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from run_storage import new_run_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=new_run_path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    os.environ['MPLCONFIGDIR']=str(output/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    comparison=json.loads(args.comparison.read_text())
    rows=comparison['rows']
    scores=[row['metrics']['fid'] for row in rows]
    assert len(rows)==4 and all(row['metrics']['update']==5000 for row in rows)
    fig,ax=plt.subplots(figsize=(8.8,5))
    bars=ax.bar(range(4),scores,color=['#64748b','#b07d46','#22856c','#53a8a1'],width=.62)
    ax.bar_label(bars,labels=[f'{value:.2f}' for value in scores],padding=5,fontsize=12)
    ax.set_xticks(range(4),['Flow matching','Original paired\nRiesz','Symmetric paired\nRiesz','Symmetric smooth\nno RMS'])
    ax.set_ylabel('Training-reference FID-10K (lower is better)')
    ax.set_title('MNIST at 5,000 optimizer updates',fontsize=15,pad=15)
    ax.set_ylim(0,max(scores)*1.22)
    ax.set_axisbelow(True);ax.yaxis.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    best=comparison['fm_best']
    ax.axhline(best['fid'],color='#64748b',linestyle='--',linewidth=1)
    ax.text(1.55,best['fid']+.35,f"FM at {best['update']//1000}K: {best['fid']:.2f}",ha='left',fontsize=9,color='#475569')
    fig.text(.08,.045,'Single seed; 100 sampling network evaluations. FM uses width 16 / batch 128; Riesz uses width 32 / batch 256.\nRiesz variants use direct paired reference information. Development scores; independent test audit is separate.',fontsize=8,color='#475569')
    fig.tight_layout(rect=(0,.12,1,1))
    fig.savefig(output/'fid_comparison.png',dpi=180)
    plt.close(fig)
    shutil.copy2(args.comparison,output/'comparison.json')
    shutil.copy2(__file__,output/'plot_source.py')
    shutil.copy2(ROOT/'run_storage.py',output/'run_storage_source.py')
    print(output/'fid_comparison.png')


if __name__=='__main__':main()
