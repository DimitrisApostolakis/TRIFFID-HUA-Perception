import json
import pytest
from triffid_ugv_perception.discoveries import ObjectRegistry


def track(tid=1, cls=21, box=(0,0,100,100), appearance=(1.,0.)):
    return dict(track_id=tid,class_id=cls,class_name=str(cls),bbox=box,appearance=appearance,confidence=.9)


def test_creation_only_and_confirmation(tmp_path):
    r=ObjectRegistry(tmp_path/'registry.json',confirm_frames=2)
    assert r.update([track()],0)[1]==[]
    observations,new=r.update([track()],10**9)
    assert len(new)==1
    oid=observations[0]['object_id']
    for second in range(2,8):
        observations,new=r.update([track()],second*10**9)
        assert observations[0]['object_id']==oid and not new
        r.write_creation_events(tmp_path/'events')
    assert len(list((tmp_path/'events').glob('*.json')))==1
    assert json.loads(next((tmp_path/'events').glob('*.json')).read_text())['timestamp_ns']==10**9


def test_empty_frame_resets_unconfirmed_candidate(tmp_path):
    r=ObjectRegistry(tmp_path/'registry.json',confirm_frames=2)
    assert not r.update([track()],0)[1]
    r.update([],10**9)
    assert not r.update([track()],2*10**9)[1]
    assert len(r.update([track()],3*10**9)[1])==1


def test_reacquire_with_new_track_id_and_no_repeat(tmp_path):
    r=ObjectRegistry(tmp_path/'registry.json',confirm_frames=1)
    first,new=r.update([track()],0)
    for second in range(1,15):r.update([],second*10**9)
    again,new=r.update([track(tid=99,box=(5,0,105,100))],15*10**9)
    assert not new and again[0]['object_id']==first[0]['object_id']


def test_distinct_objects_same_class_and_one_to_one(tmp_path):
    r=ObjectRegistry(tmp_path/'registry.json',confirm_frames=1)
    initial,_=r.update([track()],0)
    current,new=r.update([track(),track(tid=2,box=(110,0,210,100))],10**9)
    assert len(new)==1 and len({o['object_id'] for o in current})==2
    # Two fresh track IDs must not both bind to the same saved object.
    current,new=r.update([track(tid=3),track(tid=4)],2*10**9)
    assert len({o['object_id'] for o in current})==2


def test_reload_does_not_reuse_tracker_ids_blindly(tmp_path):
    path=tmp_path/'registry.json'
    r=ObjectRegistry(path,confirm_frames=1)
    previous,_=r.update([track()],0)
    restored=ObjectRegistry(path,confirm_frames=1)
    different,new=restored.update([track(box=(300,0,400,100))],10**9)
    assert len(new)==1 and different[0]['object_id']!=previous[0]['object_id']
    restored.write_creation_events(tmp_path/'events')
    assert len(list((tmp_path/'events').glob('*.json')))==2


def test_reid_rejects_different_class_or_appearance(tmp_path):
    r=ObjectRegistry(tmp_path/'registry.json',confirm_frames=1)
    r.update([track()],0)
    assert len(r.update([track(tid=2,appearance=(0.,1.))],10**9)[1])==1
    assert len(r.update([track(tid=3,cls=39)],2*10**9)[1])==1
    with pytest.raises(ValueError):r.update([track(),track()],3*10**9)


def test_registry_rejects_different_experiment_or_embedding_space(tmp_path):
    path=tmp_path/'registry.json'
    context=dict(experiment='pui',checkpoint='sine')
    registry=ObjectRegistry(path,confirm_frames=1,context=context)
    registry.update([track()],0)
    restored=ObjectRegistry(path,context=context)
    assert len(restored.objects)==1
    with pytest.raises(ValueError,match='context differs'):
        ObjectRegistry(path,context=dict(experiment='pui',checkpoint='cosine'))
    with pytest.raises(ValueError,match='context differs'):
        ObjectRegistry(path,context=dict(experiment='another-bag',checkpoint='sine'))


def test_rebinding_retires_stale_alias(tmp_path):
    registry=ObjectRegistry(tmp_path/'registry.json',confirm_frames=1)
    first,_=registry.update([track(tid=1)],0)
    rebound,_=registry.update([track(tid=2)],10**9)
    assert rebound[0]['object_id']==first[0]['object_id']
    assert (21,1) not in registry.aliases
    rows,_=registry.update([track(tid=1,box=(300,0,400,100)),track(tid=2)],2*10**9)
    assert len({row['object_id'] for row in rows})==2


def test_crash_before_outbox_write_recovers_immutable_creation_event(tmp_path):
    path=tmp_path/'registry.json';outbox=tmp_path/'outbox'
    registry=ObjectRegistry(path,confirm_frames=1)
    _,created=registry.update([track()],0)
    # Simulate a crash after committing the registry but before writing outbox.
    restored=ObjectRegistry(path,confirm_frames=1)
    restored.write_creation_events(outbox)
    event_path=outbox/(created[0]['object_id']+'.json')
    original=event_path.read_bytes()
    _,again=restored.update([track(tid=9,box=(2,0,102,100))],10**9)
    assert not again
    restored.write_creation_events(outbox)
    assert event_path.read_bytes()==original
    assert len(list(outbox.glob('*.json')))==1
