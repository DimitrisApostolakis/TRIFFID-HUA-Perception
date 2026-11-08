import numpy as np
import pytest
from triffid_ugv_perception.tracking.udc import UdcTracker
from triffid_ugv_perception.tracking.factory import TrackingPipeline, merge_person_hypotheses
from triffid_ugv_perception.discoveries import ObjectRegistry


def detection(cid=23,name='citizen',box=(5,5,35,85),score=.9):
    return dict(class_id=cid,class_name=name,bbox=list(box),confidence=score,appearance=[1.,0.])


def frame():
    image=np.zeros((100,180,3),np.uint8)
    image[5:85,5:35]=(0,0,255)
    image[5:85,100:130]=(255,0,0)
    return image


def test_role_duplicates_merge_but_separate_people_and_other_classes_survive():
    rows=[detection(),detection(6,'first responder'),detection(box=(100,5,130,85)),detection(39,'cone')]
    result=merge_person_hypotheses(rows)
    assert len(result)==3
    people=[d for d in result if d['class_id']==-1]
    assert len(people)==2 and len(people[0]['role_evidence'])==2
    assert result[0]['class_name']=='cone'


def test_udc_confirmation_low_recovery_empty_frames_and_independent_instances():
    a,b=UdcTracker(),UdcTracker()
    assert not a.update([detection()],frame(),0)
    current=a.update([detection()],frame(),10**9)
    assert len(current)==1 and current[0]['track_id']==1
    weak=a.update([detection(score=.2)],frame(),2*10**9)
    assert len(weak)==1 and weak[0]['recovered'] and weak[0]['track_id']==1
    assert a.update([],frame(),3*10**9)==[]
    b.update([detection()],frame(),0)
    assert b.update([detection()],frame(),10**9)[0]['track_id']==1
    with pytest.raises(ValueError):a.update([],frame(),3*10**9)


def test_udc_duplicate_birth_suppression_and_class_separation():
    t=UdcTracker()
    rows=[detection(),detection(),detection(39,'cone')]
    t.update(rows,frame(),0)
    result=t.update(rows,frame(),10**9)
    assert len(result)==2 and {d['class_id'] for d in result}=={23,39}


def test_udc_missing_timestamp_ticks_expire_gallery():
    t=UdcTracker(lost_seconds=2)
    t.update([detection()],frame(),0)
    old=t.update([detection()],frame(),10**9)[0]['track_id']
    assert not t.update([detection()],frame(),100*10**9)
    new=t.update([detection()],frame(),101*10**9)[0]['track_id']
    assert old!=new


def test_udc_reidentifies_after_active_track_expiry():
    t=UdcTracker(lost_seconds=2)
    t.update([detection()],frame(),0)
    old=t.update([detection()],frame(),10**9)[0]['track_id']
    for second in range(2,6):assert not t.update([],frame(),second*10**9)
    assert not t.update([detection()],frame(),6*10**9)
    recovered=t.update([detection()],frame(),7*10**9)
    assert recovered[0]['track_id']==old and t.revivals


def test_role_flip_keeps_registry_uuid_and_one_creation_event(tmp_path):
    tracker=TrackingPipeline('udc',person_roles=True)
    registry=ObjectRegistry(tmp_path/'registry.json')
    ids=set();events=[]
    for second in range(12):
        rows=[detection()] if second<4 else [detection(6,'first responder')]
        tracks=tracker.update(rows,frame(),second*10**9)
        observations,created=registry.update(tracks,second*10**9)
        ids.update(t['object_id'] for t in observations if t['object_id'])
        events.extend(created)
    assert len(ids)==len(events)==1
    assert observations[0]['class_id']==6
    # Restart with a new role must still reuse the saved physical-person UUID.
    restored=ObjectRegistry(tmp_path/'registry.json')
    track=dict(tracks[0],track_id=999)
    observations,created=restored.update([track],20*10**9)
    assert not created and observations[0]['object_id'] in ids
    restored.write_creation_events(tmp_path/'events')
    assert len(list((tmp_path/'events').glob('*.json')))==1


def test_ambiguous_roles_do_not_create_two_person_tracks():
    tracker=TrackingPipeline('udc',person_roles=True)
    for second in range(4):
        tracks=tracker.update([detection(),detection(6,'first responder',score=.89)],frame(),second*10**9)
    assert len(tracks)==1 and tracks[0]['role_ambiguous']
    assert tracks[0]['identity_class_id']==-1


def test_udc_weak_persistent_false_positive_never_clears_proof_gate():
    tracker=UdcTracker(proof_threshold=.5)
    for second in range(10):
        assert not tracker.update([detection(score=.49)],frame(),second*10**9)


def test_gallery_does_not_reuse_different_colour_object():
    tracker=UdcTracker(lost_seconds=2)
    tracker.update([detection()],frame(),0)
    old=tracker.update([detection()],frame(),10**9)[0]['track_id']
    for second in range(2,6):tracker.update([],frame(),second*10**9)
    different=detection(box=(100,5,130,85))
    tracker.update([different],frame(),6*10**9)
    result=tracker.update([different],frame(),7*10**9)
    assert result[0]['track_id']!=old and not tracker.revivals


def test_lower_proof_gate_accepts_vehicle_score_but_still_requires_confirmation():
    tracker=UdcTracker()
    assert tracker.settings.proof_thr==.35
    assert not tracker.update([detection(score=.438)],frame(),0)
    assert len(tracker.update([detection(score=.438)],frame(),10**9))==1
    assert not UdcTracker().update([detection(score=.2)],frame(),0)
    with pytest.raises(ValueError):UdcTracker(proof_threshold=.2)


def test_association_gates_invalid_edges_before_global_assignment():
    from triffid_ugv_perception.tracking.vendor.udc_core import ByteTracker
    # Unconstrained assignment chooses (0,0),(1,1), then discards (1,1).
    # Both cross-pairs are valid at 0.3 and must remain available to the solver.
    tracks = [np.array([0, 0, 100, 100]), np.array([50, 0, 145, 100])]
    detections = [np.array([0, 0, 100, 100]), np.array([-100, 0, 230, 100])]
    matches, unmatched_tracks, unmatched_detections = ByteTracker()._associate(tracks, detections, .3)
    assert set(matches) == {(0, 1), (1, 0)}
    assert unmatched_tracks == unmatched_detections == []


def test_filtered_boxes_do_not_steal_another_persons_role_or_appearance(monkeypatch):
    from triffid_ugv_perception.tracking.vendor.udc_core import KalmanBoxTracker
    # A Kalman-filtered box can lag behind its measurement near a crossing.
    # Identity metadata must follow the actual association, not another IoU lookup.
    monkeypatch.setattr(KalmanBoxTracker, 'current_bbox', lambda self: np.array([5., 5., 35., 85.]))
    tracker = TrackingPipeline('udc', person_roles=True)
    rows = [detection(), dict(detection(6, 'first responder', (100, 5, 130, 85)), appearance=[0., 1.])]
    assert tracker.update(rows, frame(), 0) == []
    result = tracker.update(rows, frame(), 10**9)
    assert [(r['class_name'], r['appearance']) for r in result] == [
        ('citizen', [1., 0.]), ('first responder', [0., 1.])]


@pytest.mark.parametrize('backend', ['udc', 'botsort'])
def test_wrappers_preserve_person_metadata_through_low_confidence_recovery(backend):
    if backend == 'botsort':
        pytest.importorskip('ultralytics')
    tracker = TrackingPipeline(backend, person_roles=True)
    a = detection()
    b = dict(detection(6, 'first responder', (100, 5, 130, 85)), appearance=[0., 1.])
    for second in range(3):
        tracks = tracker.update([a, b], frame(), second * 10**9)
    original = {t['class_name']: t['track_id'] for t in tracks}
    tracks = tracker.update([a, dict(b, confidence=.2)], frame(), 3 * 10**9)
    assert {t['class_name']: t['track_id'] for t in tracks} == original
    responder = next(t for t in tracks if t['class_name'] == 'first responder')
    assert responder['appearance'] == [0., 1.]
    assert responder['role_evidence'][0]['confidence'] == .2


def test_botsort_weak_stage_cannot_turn_a_vehicle_into_a_cone():
    pytest.importorskip('ultralytics')
    from triffid_ugv_perception.tracking.botsort import BotSortTracker
    tracker = BotSortTracker()
    vehicle = detection(21, 'civilian vehicle')
    original = tracker.update([vehicle], frame())[0]['track_id']
    assert tracker.update([detection(39, 'cone', score=.2)], frame()) == []
    assert tracker.update([vehicle], frame())[0]['track_id'] == original


def test_botsort_ids_are_local_and_unique_across_class_pools():
    pytest.importorskip('ultralytics')
    from triffid_ugv_perception.tracking.botsort import BotSortTracker
    a, b = BotSortTracker(), BotSortTracker()
    rows = [detection(21, 'civilian vehicle'), detection(39, 'cone', (100, 5, 130, 85))]
    first = a.update(rows, frame())
    assert len({t['track_id'] for t in first}) == 2
    assert [t['track_id'] for t in b.update(rows, frame())] == [t['track_id'] for t in first]
    assert [t['track_id'] for t in a.update(rows, frame())] == [t['track_id'] for t in first]


def test_botsort_timestamp_gaps_expire_tracks_without_extra_observations():
    pytest.importorskip('ultralytics')
    from triffid_ugv_perception.tracking.botsort import BotSortTracker
    tracker = BotSortTracker(lost_seconds=2)
    rows = [detection()]
    original = tracker.update(rows, frame(), 0)[0]['track_id']
    assert tracker.update(rows, frame(), 10**9)[0]['track_id'] == original
    assert tracker.update(rows, frame(), 100*10**9) == []
    assert tracker.update(rows, frame(), 101*10**9)[0]['track_id'] != original
    assert tracker.calls == 4
    with pytest.raises(ValueError):
        tracker.update(rows, frame(), 101*10**9)
