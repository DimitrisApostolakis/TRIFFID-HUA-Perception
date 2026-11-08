"""Run isolated GPU jobs sequentially with persistent logs, status and immutable outputs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import fcntl

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/evaluation_suite.json').read_text())
UGV=tuple(CONFIG['ugv']['models'])
UAV=tuple(CONFIG['uav']['models'])

def environment(model):
    from run_pui_comparison import environment as existing
    if model in ('DE-ViT','SINE','COSINE'):
        python,env=existing('PET-DINO')
        paths=['ugv_models/cosine_runtime','ugv_models/devit_runtime','src/triffid_ugv_perception']
        if model=='DE-ViT':paths.insert(2,'ugv_models/devit')
        env['PYTHONPATH']=':'.join(str(ROOT/p) for p in paths)
    elif model=='SANSA':
        python='/home/oracle/miniforge3/envs/sansa/bin/python';env=os.environ.copy()
        env['PYTHONPATH']=str(ROOT/'src/triffid_ugv_perception')
    else:
        python,env=existing(model)
    env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH']+=os.pathsep+str(ROOT/'src/triffid_uav_perception')
    return python,env

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--platform',choices=['ugv','uav'],required=True)
    p.add_argument('--models',nargs='+')
    p.add_argument('--run',required=True)
    p.add_argument('--after-status',type=Path,help='Wait for all jobs in an earlier suite before using the GPU')
    args=p.parse_args()
    if Path(args.run).name!=args.run:raise ValueError('Run must be a simple directory name')
    models=args.models or (UGV if args.platform=='ugv' else UAV)
    allowed=UGV if args.platform=='ugv' else UAV
    if set(models)-set(allowed):raise ValueError(f'Models outside the active {args.platform} suite: {set(models)-set(allowed)}')
    runroot=ROOT/'samples/evaluation'/args.platform/args.run
    runroot.mkdir(parents=True,exist_ok=False)
    states={m:dict(status='pending') for m in models}
    def save():
        tmp=runroot/'status.tmp'
        tmp.write_text(json.dumps(dict(platform=args.platform,run=args.run,expected_models=list(models),models=states),indent=2)+'\n')
        tmp.replace(runroot/'status.json')
    save()
    if args.after_status:
        print(f'Waiting for previous suite: {args.after_status}',flush=True)
        while True:
            dependency=json.loads(args.after_status.read_text())
            expected=dependency.get('expected_models',UGV if dependency['platform']=='ugv' else UAV)
            jobs=dependency['models']
            if set(expected)<=set(jobs) and all(s['status'] in ('complete','failed','interrupted') for s in jobs.values()):break
            time.sleep(5)
    # Separate invocations share this advisory GPU lock. Existing external
    # workloads must be handled by explicit queue dependencies.
    gpu_lock=open('/tmp/triffid-evaluation-gpu.lock','w')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX)
    for model in models:
        out=runroot/model
        states[model]=dict(status='running',started_at=datetime.now(timezone.utc).isoformat())
        save();start=time.perf_counter()
        if args.platform=='ugv':
            python,env=environment(model)
            command=[python,str(ROOT/'scripts/benchmark_ugv_holdout.py'),'--model',model,'--output',str(out)]
        else:
            command=['bash',str(ROOT/'scripts/run_uav_holdout.sh'),model,'--output',str(out)]
            env=os.environ.copy()
        env['PYTHONUNBUFFERED']='1'
        with (runroot/(model+'.environment.txt')).open('w') as frozen:
            runtime_python=python if args.platform=='ugv' else CONFIG['uav'].get('python_by_backend',{}).get(model,CONFIG['uav']['python'])
            subprocess.run([runtime_python,'-m','pip','freeze'],env=env,cwd=ROOT,stdout=frozen,stderr=subprocess.STDOUT,check=True)
        print(f'Running {args.platform}/{model}; log: {runroot/(model+".log")}',flush=True)
        with (runroot/(model+'.log')).open('w') as log:
            result=subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        states[model].update(status='complete' if result.returncode==0 and (out/'metrics.json').is_file() else 'failed',
            returncode=result.returncode,elapsed_s=time.perf_counter()-start,command=command)
        save();print(f'{model}: {states[model]["status"]}',flush=True)
    if any(s['status']!='complete' for s in states.values()):sys.exit(1)

if __name__=='__main__':main()
