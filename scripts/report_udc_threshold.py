"""Compare saved UDC proof-gate replays without repeating detector inference."""
import json
from pathlib import Path
from triffid_ugv_perception.io import write_json

ROOT=Path(__file__).absolute().parents[1]
old=ROOT/'archive/samples_legacy_20261106/udc_tracking_20261105/comparison'
new=ROOT/'archive/samples_legacy_20261106/udc_threshold_035_20261105'
name='udc_person_roles_low_recovery'
a=json.loads((old/name/'summary.json').read_text())
b=json.loads((new/name/'summary.json').read_text())
assert a['input_cache_sha256']==b['input_cache_sha256'] and a['input_floor']==b['input_floor']==.1
assert a['frames']==b['frames']==65 and a['audit_passed'] and b['audit_passed']
settings_a=dict(a['tracker']['engine']['settings']);settings_b=dict(b['tracker']['engine']['settings'])
assert settings_a.pop('proof_thr')==.5 and settings_b.pop('proof_thr')==.35 and settings_a==settings_b
fa=[json.loads(l) for l in (old/name/'frames.jsonl').read_text().splitlines()]
fb=[json.loads(l) for l in (new/name/'frames.jsonl').read_text().splitlines()]
assert [f['source'] for f in fa]==[f['source'] for f in fb]
write_json(new/'threshold_comparison.json',dict(old=a,new=b,same_cached_detections=True,only_setting_changed='proof_thr',
    limits='Output counts and overlap pairs are not ground-truth accuracy or physical identity metrics'))
options=''.join(f'<option value="{i}"'+(' selected' if i==1 else '')+f'>{i:04d}</option>' for i in range(65))
html='''<!doctype html><meta charset="utf-8"><title>UDC threshold 50% versus 35%</title><style>body{font-family:system-ui;margin:24px;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}select{font-size:18px}</style><h1>UDC confirmation gate: 50% versus 35%</h1><p>Identical cached PET-DINO Visual-G outputs: 800 resolution, all 141 supports, 28 classes. Same UDC person-role grouping and 10% recovery floor. Grey boxes: raw detections. Coloured boxes: current accepted tracks. Frame 0000 still awaits repeated-observation confirmation; frame 0001 is shown initially.</p><label>Frame <select id="frame">OPTIONS</select></label><main><figure><figcaption>Previous UDC proof gate: 50%</figcaption><a target="_blank"><img data-path="../udc_tracking_20261105/comparison/udc_person_roles_low_recovery/overlays"></a></figure><figure><figcaption>New UDC proof gate: 35%</figcaption><a target="_blank"><img data-path="udc_person_roles_low_recovery/overlays"></a></figure></main><script>function update(){const f=document.getElementById('frame').value.padStart(4,'0');document.querySelectorAll('img').forEach(i=>{i.src=i.dataset.path+'/'+f+'.jpg';i.parentElement.href=i.src;});}document.getElementById('frame').onchange=update;update();</script>'''
(new/'threshold_comparison.html').write_text(html.replace('OPTIONS',options))
print(json.dumps(dict(old_observations=a['observations'],new_observations=b['observations'],same_inputs_verified=True)))
