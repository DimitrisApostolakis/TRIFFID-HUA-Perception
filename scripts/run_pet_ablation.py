"""Controlled Swin-T input-resolution/support study; no absent-backbone results."""
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).absolute().parents[1]
destination=ROOT/'archive/samples_legacy_20261106/pet_ablation_20261105_v2'
destination.mkdir(exist_ok=True)
runs=[('swin_t_800_1shot',800,1333,False),
      ('swin_t_480_all',480,800,True),
      ('swin_t_640_all',640,1066,True),
      ('swin_t_800_all',800,1333,True)]
results=[]
for name,size,maximum,all_supports in runs:
    output=destination/name
    if output.exists() and not (output/'summary.json').exists():
        raise ValueError(f'Incomplete run exists: {output}; inspect before restarting')
    if not (output/'summary.json').exists():
        args=['bash',str(ROOT/'scripts/run_pet_baseline.sh'),'--output',str(output),
              '--image-size',str(size),'--max-size',str(maximum),'--confidence','0.35','--precision','float32']
        if all_supports:args+=['--use-all-supports']
        print('Starting',name,flush=True)
        subprocess.run(args,check=True,cwd=ROOT)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src/triffid_ugv_perception'),PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(['/home/oracle/miniforge3/envs/fewshot-dev/bin/python',str(ROOT/'scripts/audit_fewshot_baseline.py'),
                    str(output),'--reference',str(ROOT/'archive/samples_legacy_20261106/pet_baseline/component_supports')],check=True,env=env)
    summary=json.loads((output/'summary.json').read_text())
    results.append(dict(name=name,output=str(output),ways=summary['ways'],shots=summary['shots'],timing=summary['timing'],
        preparation_s=summary['preparation_s'],peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes']))
    (destination/'timing_matrix.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
