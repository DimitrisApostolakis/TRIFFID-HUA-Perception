"""Prepare a deterministic, unlabelled query-frame review set; never invent GT."""
import json
from pathlib import Path
import shutil
from triffid_ugv_perception.ontology import CLASSES
from triffid_ugv_perception.io import write_json

ROOT=Path(__file__).absolute().parents[1]
output=ROOT/'archive/samples_legacy_20261106/pet_ablation_20261105_v2/review'
(output/'images').mkdir(parents=True,exist_ok=True)
frames=json.loads((ROOT/'archive/samples_legacy_20261106/pet_baseline/component_supports/selected_frames.json').read_text())
indices=[0,5,10,15,20,25,30,35,40,45,50,55,60,64]
images=[];tasks=[]
for index in indices:
    row=frames[index];source=ROOT/'data/PUI/rgb_frames'/row['file']
    destination=output/'images'/source.name
    if not destination.exists():shutil.copyfile(source,destination)
    images.append(dict(id=index,file_name=destination.name,width=row['width'],height=row['height']))
    tasks.append(dict(id=index,data=dict(image='images/'+destination.name),meta=dict(source=row,selected_frame=index)))
write_json(output/'query_ground_truth_TEMPLATE_UNLABELLED.json',dict(annotation_status='unlabelled',
    images=images,categories=[dict(id=c.id,name=c.name) for c in CLASSES],annotations=[]))
write_json(output/'labelstudio_tasks.json',tasks)
(output/'README.md').write_text('These 14 query frames have no ground truth yet. Label all target objects, including citizen and first responder separately. Do not evaluate the empty template as negatives. Import image files into Label Studio, export completed annotations as COCO boxes, then run scripts/evaluate_pet_ablation.py RUN COCO_JSON. Predictions are intentionally excluded from initial labelling to reduce model bias.\n')
print('Prepared',len(images),'unlabelled query images in',output)
