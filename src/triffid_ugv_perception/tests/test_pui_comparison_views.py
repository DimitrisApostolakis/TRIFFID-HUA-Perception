"""Ensure the three videos separate actual stages and never invent exports."""
import importlib.util
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest

SCRIPT=Path('/mnt/d/projects/triffid-hua-perception/scripts/pui_comparison_views.py')
spec=importlib.util.spec_from_file_location('pui_comparison_views_test',SCRIPT)
views=importlib.util.module_from_spec(spec);spec.loader.exec_module(views)


def row(name='citizen',score=.8):
    return dict(class_name=name,class_id=23,confidence=score,bbox=[20.,90.,100.,200.])


def packet(frame=0):
    track=dict(row(),track_id=7,object_id='1'*32,discovery_state='new')
    return dict(frame=frame,source=dict(bag_timestamp_ns=(frame+1)*10**9,file=f'{frame}.png'),detections=[row(),row('first responder')],tracks=[track],
        new_object_ids=[track['object_id']],api_ready_event_ids=[],api_queue_pending=0)


def test_geojson_discoveries_are_separate_from_api_ready_events():
    p=packet()
    assert len(views.rows_for_view(p,'detections'))==2
    assert len(views.rows_for_view(p,'tracking'))==1
    assert views.rows_for_view(p,'exports')==p['tracks']
    assert p['api_ready_event_ids']==[]  # Offline GeoJSON does not require API readiness.
    p['new_object_ids']=[]
    p['api_ready_event_ids']=[p['tracks'][0]['object_id']]
    assert views.rows_for_view(p,'exports')==[]  # A later API event is not a new GeoJSON discovery.


def test_ready_event_preview_is_once_per_queue_admission_not_every_sighting():
    seen=set()
    assert views.new_ready_events([],seen)==[]
    assert views.new_ready_events(['a'],seen)==['a']
    assert views.new_ready_events(['a'],seen)==[]
    assert views.new_ready_events([],seen)==[]
    assert views.new_ready_events(['a','b'],seen)==['b']


def test_competing_roles_are_distinguished_from_duplicate_display_layers():
    p=packet();d=views.competing_labels(p['detections'],p['tracks'])
    assert len(d['raw_cross_class_pairs'])==1
    assert d['raw_cross_class_pairs'][0]['person_roles']
    assert d['raw_cross_class_pairs'][0]['classes']==['citizen','first responder']
    # One raw prediction and its same-labelled track is only two display layers.
    d=views.competing_labels([row()],p['tracks'])
    assert d['raw_cross_class_pairs']==[] and d['raw_track_role_disagreements']==[]
    other=dict(row('first responder'),bbox=[150.,90.,230.,200.])
    assert views.competing_labels([row(),other],[])['raw_cross_class_pairs']==[]


def test_api_queue_read_is_observational_and_excludes_blocked_events(tmp_path):
    from triffid_ugv_perception.discovery import DiscoveryStore,DiscoverySettings
    from triffid_ugv_perception.discovery.features import identified_feature
    store=DiscoveryStore(tmp_path/'state.db',DiscoverySettings())
    try:
        oid=uuid4().hex
        record=dict(object_id=oid,identity_class='person',class_name='citizen',confidence=.8,
                    first_seen_ns=1,last_seen_ns=2,bbox=None,appearance=None)
        store.register(record)
        assert store.ready_event_ids()==set()
        feature=dict(type='Feature',geometry=dict(type='Point',coordinates=[23.7,37.9]),
                     properties=dict(location_valid=True,heading_valid=True,local_frame=False))
        store.update_records([(record,identified_feature(feature,oid,store.settings.namespace))])
        assert store.ready_event_ids()=={oid}
        assert store.db.execute('SELECT state FROM outbox').fetchone()[0]=='pending'
        event=store.claim(0)
        assert store.ready_event_ids()=={oid}
        store.fail(oid,event['lease_token'],0,RuntimeError('rejected'),permanent=True)
        assert store.ready_event_ids()==set() and store.stats()['blocked']==1
    finally:store.close()


def write_staged(directory,count=3):
    videos=views.VideoSet(directory,'test',count)
    try:
        for i in range(count):videos.write(np.zeros((240,320,3),np.uint8),packet(i))
    finally:videos.close()
    for name in ('timings.csv','observations.jsonl','detections.geojson','audit.json'):(directory/name).write_text('{}')


def test_incomplete_replacement_preserves_original_video(tmp_path):
    output=tmp_path/'output';output.mkdir();(output/'video.mp4').write_bytes(b'original preserved')
    staged=tmp_path/'staged';write_staged(staged)
    writer=cv2.VideoWriter(str(staged/'exports.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),1.,(320,240))
    writer.write(np.zeros((240,320,3),np.uint8));writer.release()
    with pytest.raises(ValueError,match='Incomplete exports'):
        views.install_replacement(staged,output,3)
    assert (output/'video.mp4').read_bytes()==b'original preserved'


def test_complete_replacement_removes_only_obsolete_combined_artifacts(tmp_path):
    output=tmp_path/'output';(output/'overlays').mkdir(parents=True)
    (output/'video.mp4').write_bytes(b'old')
    (output/'overlays/0000.jpg').write_bytes(b'old')
    (output/'keep.txt').write_text('unrelated')
    staged=tmp_path/'staged';write_staged(staged)
    views.install_replacement(staged,output,3)
    assert not (output/'video.mp4').exists() and not (output/'overlays/0000.jpg').exists()
    assert (output/'keep.txt').read_text()=='unrelated'
    for name in views.VIDEO_NAMES:
        assert (output/f'{name}.mp4').is_file()
        assert len(list((output/'overlays'/name).glob('*.jpg')))==3


def test_geojson_contains_new_objects_once_even_without_location_or_api_events():
    first=packet(0);later=packet(1);later['new_object_ids']=[]
    collection=views.geojson_collection([first,later],'test')
    assert len(collection['features'])==1
    feature=collection['features'][0]
    assert feature['geometry'] is None
    assert feature['properties']['bbox_px']==first['tracks'][0]['bbox']
    assert feature['properties']['class']=='citizen'
    assert 'bbox' not in feature  # Pixel boxes must not masquerade as geographic bounds.
    assert first['api_ready_event_ids']==[]
    assert collection['metadata']['api_enabled'] is False


def test_duplicate_discovery_event_is_rejected_instead_of_exported_twice():
    with pytest.raises(ValueError,match='Repeated GeoJSON'):
        views.geojson_collection([packet(0),packet(1)],'test')


def test_discovery_without_current_observation_is_not_silently_dropped():
    p=packet();p['tracks']=[]
    with pytest.raises(ValueError,match='exactly one'):
        views.rows_for_view(p,'exports')


def test_corrected_geojson_updates_same_feature_and_keeps_discovery_and_best_snapshot():
    first=packet(0);first['class_correction']=True
    later=packet(1);later.update(class_correction=True,new_object_ids=[],updated_object_ids=['1'*32])
    t=later['tracks'][0]
    t.update(class_name='bag',class_id=35,confidence=.71,first_class='citizen',class_revision=1,
        best_observation=dict(confidence=.78,bbox=[22.,92.,102.,202.],timestamp_ns=2*10**9))
    last=packet(2);last.update(class_correction=True,new_object_ids=[],updated_object_ids=[],tracks=[dict(t,bbox=[25.,95.,105.,205.])])
    collection=views.geojson_collection([first,later,last],'test')
    assert len(collection['features'])==1
    f=collection['features'][0];p=f['properties']
    assert f['id']=='1'*32 and p['discovered_frame']==0
    assert p['class']=='bag' and p['confidence']==.78 and p['class_confidence']==.71
    assert p['timestamp_ns']==2*10**9 and p['image']=='1.png'
    assert p['last_seen_ns']==3*10**9 and p['bbox_px']==[22.,92.,102.,202.]
    assert views.rows_for_view(later,'exports')==later['tracks']
    assert not views.rows_for_view(last,'exports')
