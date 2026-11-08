"""Durable PUT/PATCH delivery against a fake API; never contacts production."""
from copy import deepcopy
from dataclasses import replace
import json
import logging
import sqlite3

import pytest

from triffid_ugv_perception.discovery import DiscoverySettings, DiscoveryStore, PersistentRegistry
from triffid_ugv_perception.discovery.features import encode, identified_feature, meaningful_change
from triffid_ugv_perception.discovery.outbox import OutboxWorker
from triffid_ugv_perception.discovery.transport import FeatureHTTPTransport, PermanentDeliveryError


def observation(lon=23.7, altitude=397.):
    return dict(type='Feature', geometry=dict(type='Point',coordinates=[lon,37.9]),
        properties=dict(location_valid=True,heading_valid=True,local_frame=False,
                        altitude_m=altitude,height_m=1.7,confidence=.9,**{'class':'person'}))


def record(oid='persistent-person'):
    return dict(object_id=oid,identity_class='person',class_name='person',confidence=.9,
        first_seen_ns=1,last_seen_ns=2,bbox=None,appearance=None)


@pytest.fixture
def mission(tmp_path):
    settings=DiscoverySettings(updates_enabled=True)
    store=DiscoveryStore(tmp_path/'state.db',settings)
    yield store
    store.close()


def put(store, lon=23.7, oid='persistent-person'):
    feature=identified_feature(observation(lon),oid,store.settings.namespace)
    store.register(record(oid),feature)
    event=store.claim(0)
    assert event['operation']=='create'
    store.acknowledge(oid,event['lease_token'],'poi-'+oid,now=0)
    return feature


def refresh(store, lon, oid='persistent-person'):
    f=identified_feature(observation(lon),oid,store.settings.namespace)
    store.update_records([(record(oid),f)])
    return f


def test_static_jitter_is_silent_and_movement_coalesces_with_rate_limit(mission):
    put(mission)
    refresh(mission,23.700001)  # about 9 cm
    assert mission.claim(1) is None
    refresh(mission,23.7001)
    assert mission.claim(1) is None
    refresh(mission,23.7002)
    event=mission.claim(2)
    assert event['operation']=='update' and event['remote_id']=='poi-persistent-person'
    assert json.loads(event['payload'])['geometry']['coordinates'][0]==23.7002
    mission.acknowledge(event['object_id'],event['lease_token'],event['remote_id'],now=2)
    assert mission.claim(100) is None
    refresh(mission,23.7003)
    assert mission.claim(3.99) is None
    assert mission.claim(4)['operation']=='update'


def test_return_to_previous_position_cancels_unattempted_update(mission):
    put(mission);refresh(mission,23.7001)
    assert mission.claim(1) is None
    refresh(mission,23.7)
    assert mission.claim(2) is None
    assert mission.stats()['pending']==0


def test_observation_during_create_is_patched_after_ack(mission):
    oid='person'
    mission.register(record(oid),identified_feature(observation(),oid,mission.settings.namespace))
    create=mission.claim(0)
    refresh(mission,23.7001,oid)
    assert json.loads(create['payload'])['geometry']['coordinates'][0]==23.7
    mission.acknowledge(oid,create['lease_token'],'remote',now=0)
    assert mission.claim(1) is None
    update=mission.claim(2)
    assert update['remote_id']=='remote'
    assert json.loads(update['payload'])['geometry']['coordinates'][0]==23.7001


def test_failed_patch_is_immutable_and_newer_observation_survives_restart(mission):
    put(mission);refresh(mission,23.7001)
    first=mission.claim(2)
    mission.fail(first['object_id'],first['lease_token'],2,TimeoutError('lost acknowledgement'))
    refresh(mission,23.7002)
    path,settings=mission.path,mission.settings
    mission.close()
    restored=DiscoveryStore(path,settings)
    try:
        assert restored.claim(3) is None
        retry=restored.claim(4)
        assert retry['payload']==first['payload'] and retry['operation_id']==first['operation_id']
        assert not restored.acknowledge(first['object_id'],first['lease_token'],'stale',now=4)
        assert restored.acknowledge(retry['object_id'],retry['lease_token'],retry['remote_id'],now=4)
        assert restored.claim(5) is None
        latest=restored.claim(6)
        assert latest['operation_id']!=first['operation_id']
        assert json.loads(latest['payload'])['geometry']['coordinates'][0]==23.7002
    finally:
        restored.close()


def test_crashed_inflight_patch_reclaims_same_payload_after_lease(mission):
    put(mission);refresh(mission,23.7001)
    first=mission.claim(2)
    refresh(mission,23.7002)
    assert mission.claim(3) is None
    retry=mission.claim(2+mission.settings.lease_s)
    assert retry['payload']==first['payload'] and retry['operation_id']==first['operation_id']


def test_capacity_and_permanent_failure_preserve_latest_and_poi(mission):
    put(mission);refresh(mission,23.7001)
    event=mission.claim(2)
    mission.fail(event['object_id'],event['lease_token'],2,PermanentDeliveryError('404'),permanent=True)
    refresh(mission,23.7002)
    assert mission.claim(100) is None
    assert mission.stats()['pending_updates']==1 and mission.stats()['blocked']==1
    assert mission.records()[0]['remote_id']=='poi-persistent-person'


def test_full_feature_transport_and_worker_acknowledge_real_api_response_shape(mission):
    calls=[];now=[0.]
    class Response:
        status=200
        def __init__(self,body):self.body=body
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return encode(self.body)
    def opener(request,timeout):
        calls.append(request)
        return Response({'id':'poi-42'} if request.method=='PUT' else {'success':True})
    client=FeatureHTTPTransport('https://example.invalid/features',mission.settings.namespace,True,opener=opener)
    oid='persistent-person'
    mission.register(record(oid),identified_feature(observation(),oid,mission.settings.namespace))
    worker=OutboxWorker(mission,client,logging.getLogger('test'),clock=lambda:now[0])
    assert worker.pump_once()
    assert json.loads(calls[0].data)['id']==oid
    refresh(mission,23.7001)
    now[0]=1;assert not worker.pump_once()
    now[0]=2;assert worker.pump_once()
    patch=calls[-1];body=json.loads(patch.data)
    assert patch.method=='PATCH' and patch.full_url.endswith('/poi-42')
    assert body['id']=='poi-42' and body['properties']['client_object_id']==oid
    assert patch.get_header('Idempotency-key')!=calls[0].get_header('Idempotency-key')
    assert mission.stats()['sent']==1 and mission.stats()['pending']==0
    assert not worker.pump_once()


def test_reidentified_track_updates_original_poi(mission):
    reg=PersistentRegistry(mission,mission.settings)
    def track(tid):
        return dict(track_id=tid,class_name='person',class_id=1,confidence=.9,
                    appearance=[1.,0.],bbox=[0.,0.,10.,10.])
    reg.update([track('session-1:1')],[observation()],10**9)
    oid=reg.update([track('session-1:1')],[observation()],2*10**9).created_ids[0]
    event=mission.claim(0);mission.acknowledge(oid,event['lease_token'],'original-poi',now=0)
    reg.flush()
    reg=PersistentRegistry(mission,mission.settings)
    result=reg.update([track('session-2:7')],[observation(lon=23.70002)],3*10**9)
    assert result.observations[0]['object_id']==oid and not result.created_ids
    reg.flush()
    update=mission.claim(2)
    assert update['operation']=='update' and update['remote_id']=='original-poi'
    assert mission.stats()['objects']==1


@pytest.mark.parametrize('body',[{}, {'success':False}, {'success':1}])
def test_update_requires_explicit_success_ack(mission,body):
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return encode(body)
    client=FeatureHTTPTransport('https://example.invalid/features',mission.settings.namespace,True,
        opener=lambda *a,**kw:Response())
    payload=encode(identified_feature(observation(),'person',mission.settings.namespace))
    with pytest.raises(RuntimeError,match='acknowledge'):
        client.update('person','remote',payload,'operation')


def test_change_thresholds_include_altitude_class_and_geometry(mission):
    original=observation()
    small=deepcopy(original);small['properties'].update(altitude_m=397.1,height_m=1.8,confidence=.89)
    assert not meaningful_change(original,small,mission.settings)
    for key,value in [('altitude_m',398),('class','first responder'),('marker-color','#ffffff')]:
        changed=deepcopy(original);changed['properties'][key]=value
        assert meaningful_change(original,changed,mission.settings)
    invalid=deepcopy(original);invalid['properties']['location_valid']=False
    assert not meaningful_change(original,invalid,mission.settings)


def test_update_queue_respects_shared_capacity(tmp_path):
    settings=DiscoverySettings(updates_enabled=True,max_pending=1)
    store=DiscoveryStore(tmp_path/'capacity.db',settings)
    try:
        put(store,oid='one');put(store,oid='two')
        refresh(store,23.7001,'one');refresh(store,23.7001,'two')
        first=store.claim(2)
        assert store.stats()['pending']==1
        store.acknowledge(first['object_id'],first['lease_token'],first['remote_id'],now=2)
        second=store.claim(2)
        assert second['object_id']!=first['object_id']
    finally:store.close()


def test_additive_migration_preserves_existing_create(tmp_path):
    path=tmp_path/'legacy.db';settings=DiscoverySettings()
    store=DiscoveryStore(path,settings)
    store.register(record(),identified_feature(observation(),'persistent-person',settings.namespace))
    original=bytes(store.db.execute('SELECT payload FROM outbox').fetchone()[0]);store.close()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE metadata SET value='triffid.discovery.sqlite.v1' WHERE key='schema'")
        for table,columns in [('objects',['ack_payload','last_delivery']),('outbox',['operation','operation_id'])]:
            for column in columns:db.execute(f'ALTER TABLE {table} DROP COLUMN {column}')
    store=DiscoveryStore(path,replace(settings,updates_enabled=True))
    try:
        assert store.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()[0]=='triffid.discovery.sqlite.v2'
        event=store.claim(0)
        assert event['operation']=='create' and event['payload']==original
        store.acknowledge(event['object_id'],event['lease_token'],'legacy-poi',now=0)
        refresh(store,23.7001)
        assert store.claim(2)['remote_id']=='legacy-poi'
    finally:store.close()
