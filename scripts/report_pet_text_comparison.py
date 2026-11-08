"""Compare native PET-DINO visual and class-name text modes on identical frames."""
from collections import Counter
import json
from pathlib import Path
import cv2
import numpy as np
from triffid_ugv_perception.io import write_json

ROOT=Path(__file__).absolute().parents[1]
base=ROOT/'archive/samples_legacy_20261106/pet_text_20261105'
visual=ROOT/'archive/samples_legacy_20261106/udc_tracking_20261105/udc_pipeline'
text=base/'text_class_names'
data=ROOT/'data/PUI/rgb_frames'
base.mkdir(parents=True,exist_ok=True)
rows=[];sequences=[];classes=[];checkpoints=[];configs=[];frames_by_mode={}
for mode,run in [('visual',visual),('text',text)]:
    frames=[json.loads(line) for line in (run/'frames.jsonl').read_text().splitlines()]
    summary=json.loads((run/'summary.json').read_text())
    cfg=json.loads((run/'config.json').read_text())
    assert summary['api_enabled'] is False and summary['frames']==summary['tracker_updates']==65
    assert summary['tracker']['name']=='udc' and summary['tracker']['person_roles'] is True
    sequences.append([f['source'] for f in frames]);classes.append(cfg['classes']);checkpoints.append(summary['detector']['checkpoint_sha256'])
    configs.append((cfg['image_size'],cfg['pet_max_size'],cfg['pet_precision'],cfg['confidence_threshold']))
    counts=Counter();seen=Counter();out=base/(mode+'_raw');out.mkdir(exist_ok=True)
    for f in frames:
        image=cv2.imread(str(data/f['source']['file']))
        for d in f['detections']:
            counts[d['class_name']]+=1
            x1,y1,x2,y2=map(int,d['bbox'])
            color=(0,210,255)
            cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            cv2.putText(image,f"{d['class_name']} {d['confidence']:.2f}",(max(0,x1),max(16,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        seen.update({d['class_name'] for d in f['detections']})
        assert cv2.imwrite(str(out/f"{f['frame']:04d}.jpg"),image)
    times=[f['total_s'] for f in frames];warm=times[1:]
    rows.append(dict(mode=mode,run=str(run),detector=summary['detector'],detections=summary['detections'],
        raw_class_observations=dict(counts),frames_with_class=dict(seen),tracked_observations=summary['tracked_observations'],
        registry_objects=summary['unique_objects'],timing=summary['timing'],preparation_s=summary['preparation_s'],
        warm_complete_mean_s=float(np.mean(warm)),warm_complete_p95_s=float(np.percentile(warm,95))))
    frames_by_mode[mode]=frames
assert sequences[0]==sequences[1] and classes[0]==classes[1] and checkpoints[0]==checkpoints[1] and configs[0]==configs[1]
assert rows[1]['detector']['prompt_mode']=='text' and rows[1]['detector']['support_bank'] is None
assert all(n==0 for n in rows[1]['detector']['shots_per_class'].values())
report=dict(study='Native visual-G versus native class-name text prompts; not simultaneous fusion',frames=65,ways=28,
    controls=dict(same_frames=True,same_classes=True,same_checkpoint=True,same_resolution_precision_threshold=True,
        tracker='UDC with person-role grouping and low-confidence recovery'),runs=rows,
    accuracy_status='Pending independent query ground truth',
    limitations=['Detection counts are not accuracy scores','Text run also writes optional tracking cache; total latency includes its export cost',
        'Roles can still be ambiguous; tracker proof gating can hide raw detections'])
write_json(base/'comparison_results.json',report)
options=''.join(f'<option value="{i}">{i:04d}</option>' for i in range(65))
tiles='''<figure><figcaption>Visual-G: raw detections, score ≥ 0.35</figcaption><a target="_blank"><img data-path="visual_raw"></a></figure>
<figure><figcaption>Exact class-name text: raw detections, score ≥ 0.35</figcaption><a target="_blank"><img data-path="text_raw"></a></figure>
<figure><figcaption>Visual-G + UDC: accepted tracks</figcaption><a target="_blank"><img data-path="../udc_tracking_20261105/udc_pipeline/overlays"></a></figure>
<figure><figcaption>Text + UDC: accepted tracks</figcaption><a target="_blank"><img data-path="text_class_names/overlays"></a></figure>'''
html='''<!doctype html><meta charset="utf-8"><title>PET-DINO visual / text comparison</title><style>body{font-family:system-ui;margin:24px;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}select{font-size:18px}</style><h1>PET-DINO: visual supports versus class-name text</h1><p>Same 28 classes, Swin-T checkpoint and 65 one-Hz frames. Top row: labelled raw detector outputs. Bottom row: UDC accepted tracks, where grey boxes are raw hypotheses. Text mode uses no visual supports. Numeric accuracy is unmeasured.</p><label>Frame <select id="frame">OPTIONS</select></label><main>TILES</main><script>function update(){const f=document.getElementById('frame').value.padStart(4,'0');document.querySelectorAll('img').forEach(i=>{i.src=i.dataset.path+'/'+f+'.jpg';i.parentElement.href=i.src;});}document.getElementById('frame').onchange=update;update();</script>'''
(base/'comparison.html').write_text(html.replace('OPTIONS',options).replace('TILES',tiles))
print(json.dumps([dict(mode=r['mode'],detections=r['detections'],raw_class_observations=r['raw_class_observations'],timing=r['timing']) for r in rows],indent=2))
