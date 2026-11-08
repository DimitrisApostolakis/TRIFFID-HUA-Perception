"""Run frozen YOLOE and MASA once; keep a small complete replay cache, API off."""
import subprocess
from pathlib import Path
from run_pui_comparison import environment

ROOT=Path('/mnt/d/projects/triffid-hua-perception')
WORK=ROOT/'ugv_models/comparison_work/class_correction/YOLOE'
WORK.mkdir(parents=True,exist_ok=True)
if not (WORK/'detector.json').exists():
    python,env=environment('YOLOE')
    subprocess.run([python,str(ROOT/'scripts/run_pui_comparison.py'),'--stage','detect','--model','YOLOE',
        '--output',str(WORK/'unused'),'--work',str(WORK)],env=env,cwd=ROOT,check=True)
python,env=environment('MASA')
subprocess.run([python,str(ROOT/'scripts/cache_yoloe_identity.py')],env=env,cwd=ROOT,check=True)
