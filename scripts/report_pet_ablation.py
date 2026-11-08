"""Summarize completed controlled runs and build a local prediction comparison."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from triffid_ugv_perception.io import write_json

ROOT=Path(__file__).absolute().parents[1]
base=ROOT/'archive/samples_legacy_20261106/pet_ablation_20261105_v2'
matrix=json.loads((base/'timing_matrix.json').read_text())
labels=[];means=[];p95=[];report=[];bank_hashes=set();sequences=[];class_sets=[];checkpoints=set()
for row in matrix:
    output=Path(row['output'])
    summary=json.loads((output/'summary.json').read_text())
    audit=json.loads((output/'audit.json').read_text())
    frames=[json.loads(line) for line in (output/'frames.jsonl').read_text().splitlines()]
    bank_hashes.add(summary['detector']['support_bank']['manifest_sha256'])
    checkpoints.add(summary['detector']['checkpoint_sha256'])
    config=json.loads((output/'config.json').read_text())
    assert config['pet_precision']=='float32' and config['confidence_threshold']==.35
    class_sets.append(json.loads((output/'class_availability.json').read_text())['selected'])
    sequences.append([frame['source'] for frame in frames])
    assert audit['passed'] and len(frames)==65
    assert all('candidate_detections' in frame for frame in frames)
    label=row['name'].replace('swin_t_','').replace('_',' / ')
    labels.append(label);means.append(summary['timing']['mean_s']);p95.append(summary['timing']['p95_s'])
    report.append(dict(row,export_audit_passed=True,accuracy_status='pending independent query ground truth',
        class_detections=audit['class_detections'],candidate_registry_ids=summary['unique_objects'],
        shots_per_class=summary['detector']['shots_per_class']))
assert len(matrix)==4 and len(bank_hashes)==1 and len(checkpoints)==1
assert all(sequence==sequences[0] for sequence in sequences)
assert all(classes==class_sets[0] for classes in class_sets)
write_json(base/'study_results.json',dict(study='Swin-T support-count and input-resolution ablation',
    backbone_ablation_status='Not measurable: only trained PET-DINO Swin-T weights are publicly available',
    controls=dict(bank_sha256=next(iter(bank_hashes)),frames=65,ways=28,precision='float32',score_threshold=.35),runs=report))
fig,axis=plt.subplots(figsize=(8,4))
x=np.arange(len(labels));width=.36
axis.bar(x-width/2,means,width,label='Mean complete frame')
axis.bar(x+width/2,p95,width,label='95th percentile')
axis.axhline(1,color='red',linestyle='--',label='1 second budget')
axis.set_xticks(x,labels);axis.set_ylabel('Seconds per frame');axis.set_xlabel('Input setting / support count')
axis.set_title('PET-DINO Swin-T · RTX 3060 · 28 classes · 65 frames');axis.legend(fontsize=8)
fig.tight_layout();fig.savefig(base/'timing_plot.png',dpi=160);plt.close(fig)
options=''.join(f'<option value="{i}">{i:04d}</option>' for i in range(65))
tiles=''.join(f'<figure><figcaption>{row["name"]}</figcaption><a target="_blank"><img data-run="{row["name"]}"></a></figure>' for row in matrix)
html='''<!doctype html><meta charset="utf-8"><title>PET-DINO comparison</title>
<style>body{font-family:system-ui;margin:24px;background:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%}select{font-size:18px}</style>
<h1>PET-DINO Swin-T comparison</h1><p>Same 28 classes and source frames. Accuracy is unmeasured; boxes and IDs are predictions. Different resolutions and support counts can change detections. Click an image to enlarge.</p>
<label>Frame <select id="frame">OPTIONS</select></label><main>TILES</main>
<script>function update(){const frame=document.getElementById('frame').value.padStart(4,'0');document.querySelectorAll('img').forEach(image=>{image.src=image.dataset.run+'/overlays/'+frame+'.jpg';image.parentElement.href=image.src;});}document.getElementById('frame').onchange=update;update();</script>'''.replace('OPTIONS',options).replace('TILES',tiles)
(base/'comparison.html').write_text(html)
print(json.dumps(dict(runs=len(matrix),controls_verified=True,report=str(base/'study_results.json')),indent=2))
