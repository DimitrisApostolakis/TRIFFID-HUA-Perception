"""Run the authorised offline COSINE comparison once the checkpoint is complete."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).absolute().parents[1]
checkpoint=ROOT/'ugv_models/cosine_multiscale.bin'
expected_size=2764208701
while not checkpoint.exists() or checkpoint.stat().st_size<expected_size:
    time.sleep(5)
if checkpoint.stat().st_size!=expected_size:raise ValueError('Unexpected checkpoint size')

environ=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',
    PYTHONPATH=':'.join(str(ROOT/path) for path in ('ugv_models/cosine_runtime','ugv_models/devit_runtime','src/triffid_ugv_perception')),
    OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',YOLO_CONFIG_DIR=str(ROOT/'ugv_models/ultralytics_config'))
python='/home/oracle/miniforge3/envs/fewshot-dev/bin/python'
reference=ROOT/'archive/samples_legacy_20261106/sine_baseline/ugv_only_20261105'
results=[]
for label,extra in [('native_896',['--image-size','896']),('efficient_644',['--image-size','640']),
                    ('five_shot_644',['--image-size','640','--shots','5'])]:
    base=ROOT/'archive/samples_legacy_20261106/cosine_baseline'/label
    output=base;attempt=1
    while output.exists() and not (output/'summary.json').exists():
        attempt+=1;output=base.with_name(base.name+f'_attempt{attempt}')
    if not (output/'summary.json').exists():
        args=[python,str(ROOT/'scripts/run_cosine_baseline.py'),'--output',str(output),*extra]
        if label=='efficient_644':
            args+=['--registry',str(ROOT/'archive/samples_legacy_20261106/pui_registry/cosine_v1.json')]
        print('Running',label,'output',output,flush=True)
        subprocess.run(args,env=environ,check=True,cwd=ROOT)
    subprocess.run([python,str(ROOT/'scripts/audit_fewshot_baseline.py'),str(output),'--reference',str(reference)],
                   env=environ,check=True,cwd=ROOT)
    summary=json.loads((output/'summary.json').read_text())
    results.append(dict(label=label,output=str(output),frames=summary['frames'],ways=summary['ways'],shots=summary['shots'],timing=summary['timing']))
(ROOT/'archive/samples_legacy_20261106/cosine_baseline/validation_matrix.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2),flush=True)
