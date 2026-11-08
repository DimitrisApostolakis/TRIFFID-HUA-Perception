import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('pet_evaluation',Path(__file__).resolve().parents[3]/'scripts/evaluate_pet_ablation.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_duplicate_and_wrong_class_are_false_positives(tmp_path):
    pytest.importorskip('pycocotools')
    run=tmp_path/'run';run.mkdir()
    (run/'class_availability.json').write_text(json.dumps(dict(selected=['citizen','first responder'])))
    def box(name,score):return dict(class_name=name,confidence=score,bbox=[10,10,20,20])
    row=dict(source=dict(file='frame.png'),detections=[],candidate_detections=[box('citizen',.9),box('citizen',.7),box('first responder',.8)])
    (run/'frames.jsonl').write_text(json.dumps(row)+'\n')
    gt=tmp_path/'gt.json';gt.write_text(json.dumps(dict(images=[dict(id=0,file_name='hash1234-frame.png',height=30,width=30)],
        categories=[dict(id=23,name='citizen'),dict(id=6,name='first responder')],
        annotations=[dict(id=1,image_id=0,category_id=23,bbox=[10,10,10,10],area=100,iscrowd=0)])))
    result=module.evaluate(run,gt)
    assert result['per_class']['citizen']==dict(tp=1,fp=1,fn=0,precision=.5,recall=1.)
    assert result['per_class']['first responder']['fp']==1
    assert result['evaluated_images']==1
    assert result['bbox_AP50']>.99


def test_unlabelled_review_template_cannot_be_scored(tmp_path):
    pytest.importorskip('pycocotools')
    gt=tmp_path/'template.json';gt.write_text(json.dumps(dict(annotation_status='unlabelled')))
    with pytest.raises(ValueError,match='unlabelled'):module.evaluate(tmp_path,gt)
