import threading
from uuid import uuid4
import numpy as np
import pytest
from triffid_ugv_perception.evidence import encode_packet,decode_packet
from triffid_ugv_perception.latest_frame import LatestFrameWorker

def track(session):
    return dict(track_id=session+':7',class_id=23,class_name='citizen',confidence=.9,
        bbox=[1,2,4,5],position=[2.,0.,0.],extent=[1.,1.,2.],appearance=[1.,0.],
        mask=np.ones((10,10),bool),identity_class_id=-1)

def test_evidence_preserves_reid_and_position_but_excludes_masks():
    session=uuid4().hex
    result=decode_packet(encode_packet([track(session)],10,'b2/base_link','MASA-R50:abc',session))
    t=result['tracks'][0]
    assert t['appearance']==[1.,0.] and t['bbox']==[1,2,4,5] and t['identity_class_id']==-1
    assert 'mask' not in t and result['timestamp_ns']==10

@pytest.mark.parametrize('change',[
    {'appearance':[float('nan')]},{'appearance':[0.,0.]},{'appearance':[1.]*513},
    {'confidence':float('inf')},{'bbox':[4,2,1,5]},{'track_id':'unscoped'},
    {'class_id':23,'class_name':'first responder'},{'extent':[-1,1,1]}])
def test_bad_evidence_is_rejected(change):
    session=uuid4().hex;t=track(session);t.update(change)
    with pytest.raises(ValueError):encode_packet([t],10,'base','MASA-R50:abc',session)

def test_duplicate_ids_and_payload_limits_fail_closed():
    session=uuid4().hex;t=track(session)
    with pytest.raises(ValueError):encode_packet([t,t],10,'base','MASA-R50:abc',session)
    with pytest.raises(ValueError):encode_packet([t],10,'base','MASA-R50:abc',session,max_bytes=10)
    with pytest.raises(ValueError):decode_packet('x'*20,max_bytes=10)

def test_pending_frames_are_replaced_while_one_frame_is_inflight():
    started=threading.Event();release=threading.Event();finished=threading.Event();seen=[]
    def process(item,received):
        seen.append(item)
        if item==1:started.set();assert release.wait(2)
        if item==3:finished.set()
    worker=LatestFrameWorker(process,period_s=0,max_age_s=10)
    worker.start()
    try:
        worker.submit(1);assert started.wait(2)
        worker.submit(2);worker.submit(3)
        assert worker.stats()['pending']==1
        release.set();assert finished.wait(2)
    finally:release.set();worker.close()
    assert seen==[1,3] and worker.stats()['replaced']==1
    assert worker.submit(4) is False

def test_worker_recovers_after_frame_failure():
    error=threading.Event();finished=threading.Event()
    def process(item,received):
        if item==1:raise ValueError('frame error')
        finished.set()
    worker=LatestFrameWorker(process,period_s=0,on_error=lambda e:error.set())
    worker.start()
    try:
        worker.submit(1);assert error.wait(2)
        worker.submit(2);assert finished.wait(2)
    finally:worker.close()
    assert worker.stats()['failed']==1 and worker.stats()['processed']==1
