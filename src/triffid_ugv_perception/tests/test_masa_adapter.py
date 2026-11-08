"""Check the adapter boundary against MASA's negative-label rejection."""
from types import ModuleType, SimpleNamespace
import sys
import numpy as np
import torch
from triffid_ugv_perception.tracking.masa import MasaTracker


def test_person_group_survives_native_validity_filter_and_empty_frames_expire_memory(monkeypatch):
    structures=ModuleType('mmdet.structures')
    structures.DetDataSample=SimpleNamespace
    structures.TrackDataSample=SimpleNamespace
    monkeypatch.setitem(sys.modules,'mmdet.structures',structures)
    monkeypatch.setattr(torch.Tensor,'cuda',lambda tensor:tensor)
    class CpuTorch:
        def __getattr__(self,name):return getattr(torch,name)
        def tensor(self,*args,**kwargs):
            kwargs.pop('device',None)
            return torch.tensor(*args,**kwargs)
    tracker=object.__new__(MasaTracker)
    tracker.torch=CpuTorch();tracker.memory_seconds=60
    tracker.calls=tracker.image_encodings=0
    tracker.first_timestamp=tracker.last_timestamp=None
    ticks=[]
    def predict(inputs,samples):
        sample=samples[0].video_data_samples[0]
        ticks.append(sample.metainfo['frame_id'])
        # Exact upstream precondition: negative labels are invalid detections.
        valid=sample.det_labels>-1
        tracker.embeddings=torch.eye(len(sample.det_labels),256)
        pred=SimpleNamespace(bboxes=sample.det_bboxes[valid,:4],scores=sample.det_bboxes[valid,4],
            labels=sample.det_labels[valid],instances_id=torch.arange(int(valid.sum())))
        return [[SimpleNamespace(pred_track_instances=pred)]]
    tracker.model=SimpleNamespace(predict=predict,tracker=SimpleNamespace(tracks={}))
    rows=[dict(class_id=-1,class_name='person',bbox=[2,2,5,7],confidence=.8,
               role_evidence=[dict(class_id=6,class_name='first responder',confidence=.8)]),
          dict(class_id=21,class_name='civilian vehicle',bbox=[9,2,15,7],confidence=.9)]
    output=tracker.update(rows,np.zeros((8,16,3),np.uint8),0)
    assert [r['class_id'] for r in output]==[-1,21]
    assert output[0]['role_evidence']==rows[0]['role_evidence']
    assert output[0]['appearance'][0]==1 and output[1]['appearance'][1]==1
    tracker.update(rows,np.zeros((8,16,3),np.uint8),67_000_000)
    assert ticks==[0,1]  # Do not trigger the native reset for the second image.
    tracker.model.tracker.tracks={3:dict(last_frame=0)}
    assert tracker.update([],np.zeros((8,16,3),np.uint8),61*10**9)==[]
    assert not tracker.model.tracker.tracks
    assert tracker.calls==3 and tracker.image_encodings==2
