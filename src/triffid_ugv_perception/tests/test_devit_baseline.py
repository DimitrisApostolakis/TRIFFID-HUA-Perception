"""Baseline contracts independent of released checkpoints or ROS traffic."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def test_one_hz_sampling_preserves_first_frame_per_source_second():
    path=Path(__file__).resolve().parents[3]/'scripts/run_devit_baseline.py'
    spec=importlib.util.spec_from_file_location('devit_runner',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    rows=[dict(bag='camera',topic='rgb',bag_timestamp_ns=t) for t in [10,100,999_999_999,1_000_000_010,1_700_000_000,3_000_000_010]]
    result=list(runner.sample_frames(reversed(rows)))
    assert [r['bag_timestamp_ns'] for r in result]==[10,1_000_000_010,3_000_000_010]


def test_devit_maps_contiguous_predictions_to_original_ontology():
    torch=pytest.importorskip('torch')
    from triffid_ugv_perception.models.devit import DevitDetector
    from triffid_ugv_perception.ontology import classes_for
    instance=SimpleNamespace(pred_boxes=SimpleNamespace(tensor=torch.tensor([[-1.,2.,12.,15.],[0.,0.,1.,1.]])),
        pred_classes=torch.tensor([0,1]),scores=torch.tensor([.9,.1]))
    instance.to=lambda device:instance
    captured=[]
    def model(inputs):
        captured.append(inputs[0]['image'])
        return [dict(instances=instance)]
    adapter=DevitDetector.__new__(DevitDetector)
    adapter.torch=torch;adapter.classes=classes_for('civilian vehicle,cone')
    adapter.resize=SimpleNamespace(get_transform=lambda image:SimpleNamespace(apply_image=lambda x:x))
    adapter.model=model;adapter.device='cpu';adapter.score=.35;adapter.nms_iou=.3;adapter.min_box_area=1
    adapter._appearance=lambda bbox,shape:[1.,0.]
    image=np.zeros((10,10,3),np.uint8);image[...,0]=255
    result=adapter.detect(image)
    assert len(result)==1
    assert result[0]['class_id']==21 and result[0]['class_name']=='civilian vehicle'
    assert result[0]['bbox']==(0.,2.,10.,10.) and result[0]['mask'] is None
    assert captured[0][2,0,0].item()==255  # BGR converted to RGB.


def test_botsort_updates_empty_frames_and_preserves_two_same_class_objects():
    pytest.importorskip('ultralytics')
    from triffid_ugv_perception.tracking.botsort import BotSortTracker
    tracker=BotSortTracker(fps=1,lost_seconds=10)
    image=np.zeros((200,400,3),np.uint8)
    def detection(x,cls=21):
        return dict(bbox=[x,20,x+50,80],confidence=.9,class_id=cls,class_name=str(cls),appearance=[1.,0.])
    first=tracker.update([detection(10),detection(210)],image)
    assert len(first)==2 and len({t['track_id'] for t in first})==2
    assert tracker.update([],image)==[]
    again=tracker.update([detection(12),detection(212)],image)
    assert {t['track_id'] for t in first}=={t['track_id'] for t in again}
    assert tracker.calls==3 and tracker.tracker.max_time_lost==10


def test_botsort_does_not_relabel_same_location_as_same_identity():
    pytest.importorskip('ultralytics')
    from triffid_ugv_perception.tracking.botsort import BotSortTracker
    tracker=BotSortTracker()
    image=np.zeros((120,120,3),np.uint8)
    def d(cls):return dict(bbox=[10,10,60,80],confidence=.9,class_id=cls,class_name=str(cls),appearance=[1.,0.])
    first=tracker.update([d(21)],image)
    tracker.update([d(39)],image)
    second=tracker.update([d(39)],image)
    assert first[0]['track_id']!=second[0]['track_id']
