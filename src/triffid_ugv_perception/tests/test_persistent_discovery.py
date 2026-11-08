"""Mission identity/delivery contracts: synthetic observations and fake servers only."""
from dataclasses import replace
import json
import logging
from uuid import uuid4

import numpy as np
import pytest

from triffid_ugv_perception.discovery import (DiscoverySettings, DiscoveryStore,
    PersistentRegistry, DiscoveryService, CapacityExceeded)
from triffid_ugv_perception.discovery.features import encode, identified_feature
from triffid_ugv_perception.discovery.outbox import OutboxWorker
from triffid_ugv_perception.discovery.transport import CreateOnlyHTTPTransport, PermanentDeliveryError
from triffid_ugv_perception.storage import _atomic_write_geojson, BoundedImageWriter


def feature(ready=True, lon=23.7):
    return dict(type='Feature', geometry=dict(type='Point',coordinates=[lon,37.9]),
                properties=dict(location_valid=ready,heading_valid=ready,local_frame=not ready))


def track(tid=1, name='civilian', appearance=(1.,0.), box=(0.,0.,10.,10.), confidence=.9):
    return dict(track_id=tid,class_name=name,class_id=1,confidence=confidence,
                appearance=appearance,bbox=box)


def record(oid=None):
    return dict(object_id=oid or uuid4().hex,identity_class='person',class_name='civilian',
                confidence=.9,first_seen_ns=1,last_seen_ns=2,bbox=None,appearance=None)


@pytest.fixture
def store(tmp_path):
    store=DiscoveryStore(tmp_path/'mission.sqlite3',DiscoverySettings())
    yield store
    store.close()


def confirm(reg, tracks=None, features=None, second=2):
    tracks=tracks or [track()]
    features=features or [feature() for _ in tracks]
    reg.update(tracks,features,int((second-1)*1e9))
    return reg.update(tracks,features,int(second*1e9))


def test_one_create_only_despite_repeated_observations_and_class_changes(store):
    reg=PersistentRegistry(store,store.settings)
    result=confirm(reg);oid=result.created_ids[0]
    original=store.claim(0)
    for second in range(3,20):
        result=reg.update([track(name='first responder')],[feature(lon=23.70001)],second*10**9)
        assert result.observations[0]['object_id']==oid
        assert not result.created_ids
    reg.flush()
    assert original['payload']==store.db.execute('SELECT payload FROM outbox').fetchone()[0]
    assert store.acknowledge(oid,original['lease_token'],'remote-1')
    for second in range(20,30):
        reg.update([track()],[feature()],second*10**9)
    reg.flush()
    assert store.stats()['sent']==1 and store.stats()['pending']==0
    assert store.db.execute('SELECT feature FROM objects').fetchone()[0] is None


def test_invalid_location_waits_then_queues_once(store):
    reg=PersistentRegistry(store,store.settings)
    oid=confirm(reg,features=[feature(False)]).created_ids[0]
    assert store.stats()['waiting']==1 and store.claim(0) is None
    reg.update([track()],[feature()],3*10**9)
    assert store.stats()['pending']==1
    assert store.claim(0)['object_id']==oid


def test_restart_reidentification_and_changed_local_id(tmp_path):
    path=tmp_path/'state.db';settings=DiscoverySettings()
    store=DiscoveryStore(path,settings);reg=PersistentRegistry(store,settings)
    oid=confirm(reg).created_ids[0];reg.flush();store.close()
    store=DiscoveryStore(path,settings)
    try:
        reg=PersistentRegistry(store,settings)
        result=reg.update([track(99)],[feature()],10**9)
        assert result.observations[0]['object_id']==oid and not result.created_ids
        assert store.stats()['objects']==1
        # New object with different visual and spatial evidence can reuse a local number.
        result=confirm(reg,[track(2,appearance=(0.,1.),box=(100,100,110,110))],
                       [feature(lon=24.)],second=4)
        assert len(result.created_ids)==1 and result.created_ids[0]!=oid
    finally: store.close()


def test_ambiguous_reappearance_is_held_and_simultaneous_objects_are_distinct(store):
    reg=PersistentRegistry(store,store.settings)
    result=confirm(reg,[track(1),track(2)])
    assert len(set(result.created_ids))==2
    reg.flush()
    reg=PersistentRegistry(store,store.settings)
    result=confirm(reg,[track(8)])
    assert result.observations[0]['discovery_state']=='uncertain'
    assert not result.created_ids
    result=confirm(reg,[track(9,appearance=None,box=None)],[None],second=4)
    assert result.observations[0]['discovery_state']=='uncertain'
    assert store.stats()['objects']==2


@pytest.mark.parametrize('interrupt', ['empty','low','gap','duplicate'])
def test_confirmation_requires_distinct_consecutive_good_frames(store,interrupt):
    reg=PersistentRegistry(store,store.settings)
    reg.update([track()],[feature()],10**9)
    if interrupt=='empty':
        reg.update([],[],2*10**9);result=reg.update([track()],[feature()],3*10**9)
    elif interrupt=='low':
        reg.update([track(confidence=.1)],[feature()],2*10**9)
        result=reg.update([track()],[feature()],3*10**9)
    elif interrupt=='gap': result=reg.update([track()],[feature()],5*10**9)
    else:
        result=reg.update([track()],[feature()],10**9)
        assert result.stale_frame
    assert not result.created_ids and store.stats()['objects']==0


def test_duplicate_local_id_or_uuid_never_assigns_one_object_twice(store):
    reg=PersistentRegistry(store,store.settings)
    oid=confirm(reg).created_ids[0]
    a=track(2);a['object_id']=oid
    b=track(3);b['object_id']=oid
    result=reg.update([a,b],[feature(),feature()],3*10**9)
    assert sum(t['object_id']==oid for t in result.observations)==1
    assert result.observations[1]['discovery_state']=='identity_conflict'


def test_queue_capacity_preserves_waiting_and_sent_ids(tmp_path):
    settings=replace(DiscoverySettings(),max_pending=1,max_objects=2)
    store=DiscoveryStore(tmp_path/'state.db',settings)
    try:
        a,b=record(),record()
        for item in (a,b): store.register(item,identified_feature(feature(),item['object_id'],settings.namespace))
        assert store.stats()['waiting_ready']==1 and store.stats()['pending']==1
        event=store.claim(0);store.acknowledge(event['object_id'],event['lease_token'],'server')
        store.promote_ready();assert store.stats()['pending']==1
        with pytest.raises(CapacityExceeded):store.register(record())
        assert store.stats()['objects']==2 and store.stats()['sent']==1
    finally: store.close()


def test_payload_and_byte_caps_are_atomic(tmp_path):
    settings=replace(DiscoverySettings(),max_payload_bytes=400,max_pending_bytes=400)
    store=DiscoveryStore(tmp_path/'state.db',settings)
    try:
        a=record();f=identified_feature(feature(),a['object_id'],settings.namespace)
        store.register(a,f)
        b=record();store.register(b,identified_feature(feature(),b['object_id'],settings.namespace))
        assert store.stats()['pending']==1 and store.stats()['waiting_ready']==1
        f['properties']['oversize']='x'*500
        with pytest.raises(CapacityExceeded):store.register(record(),f)
        assert store.stats()['objects']==2
    finally: store.close()


def test_lease_survives_restart_and_stale_ack_is_ignored(tmp_path):
    path=tmp_path/'state.db';settings=DiscoverySettings()
    store=DiscoveryStore(path,settings);a=record()
    store.register(a,identified_feature(feature(),a['object_id'],settings.namespace))
    event=store.claim(0);store.close()
    store=DiscoveryStore(path,settings)
    try:
        assert store.claim(1) is None
        retry=store.claim(settings.lease_s+1)
        assert retry['payload']==event['payload'] and retry['object_id']==event['object_id']
        assert retry['lease_token']!=event['lease_token']
        assert not store.acknowledge(a['object_id'],event['lease_token'],'old')
        assert store.acknowledge(a['object_id'],retry['lease_token'],'remote')
    finally: store.close()


def test_ambiguous_timeout_replays_same_key_and_server_creates_once(store):
    reg=PersistentRegistry(store,store.settings);oid=confirm(reg).created_ids[0]
    calls=[];server={};now=[0.]
    class Server:
        def create(self,key,payload):
            server.setdefault(key,'server-1');calls.append((key,payload))
            if len(calls)==1:raise TimeoutError('response lost after commit')
            return server[key]
    worker=OutboxWorker(store,Server(),logging.getLogger('test'),clock=lambda:now[0])
    assert worker.pump_once() and store.stats()['pending']==1
    assert not worker.pump_once()
    now[0]=1.;assert worker.pump_once()
    assert calls[0]==calls[1] and len(server)==1 and store.stats()['sent']==1
    assert store.records()[0]['object_id']==oid


def test_permanent_failure_is_retained_for_review_and_backoff_is_bounded(store):
    confirm(PersistentRegistry(store,store.settings))
    for n in range(20):
        event=store.claim(n*100.)
        store.fail(event['object_id'],event['lease_token'],n*100.,RuntimeError('x'*1000))
        row=store.db.execute('SELECT * FROM outbox').fetchone()
        assert row['next_attempt']-n*100 <= store.settings.retry_max_s
        assert len(row['last_error'])==256
    event=store.claim(2100)
    store.fail(event['object_id'],event['lease_token'],2100,PermanentDeliveryError('bad'),permanent=True)
    assert store.claim(10000) is None and store.stats()['blocked']==1


def test_api_disabled_and_contract_gate(tmp_path):
    class Forbidden:
        def create(self,*args):raise AssertionError('API was called')
    with pytest.raises(ValueError):
        DiscoveryService(tmp_path/'not-created.db',DiscoverySettings(),api_enabled=True)
    assert not (tmp_path/'not-created.db').exists()
    service=DiscoveryService(tmp_path/'state.db',DiscoverySettings(),transport=Forbidden())
    try:
        confirm(service.registry)
        assert service.worker is None and service.status()['pending']==1
        _,collection=service.process([track()],[feature(False)],3*10**9)
        assert collection['features'][0]['geometry'] is None
    finally:service.close()


def test_network_io_does_not_hold_registry_lock(store):
    import threading
    started,release=threading.Event(),threading.Event()
    class Slow:
        def create(self,*args):started.set();release.wait(2);return 'remote'
    reg=PersistentRegistry(store,store.settings);confirm(reg)
    worker=OutboxWorker(store,Slow(),logging.getLogger('test'));worker.start()
    try:
        assert started.wait(1)
        # A separate thread would block if the worker held the SQLite lock while waiting.
        finished=threading.Event()
        def update():
            reg.update([track()],[feature()],3*10**9);reg.flush();finished.set()
        thread=threading.Thread(target=update);thread.start()
        assert finished.wait(1)
        thread.join()
    finally:release.set();assert worker.stop()


def test_transport_is_create_only_and_reuses_immutable_uuid():
    calls=[]
    class Response:
        status=201
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return b'{"id":42}'
    def opener(request,timeout):calls.append(request);return Response()
    transport=CreateOnlyHTTPTransport('https://example.invalid/points','PUI/front',True,opener=opener)
    oid=uuid4().hex;payload=encode(identified_feature(feature(),oid,'PUI/front'))
    assert transport.create(oid,payload)=='42';transport.create(oid,payload)
    assert all(r.method=='PUT' for r in calls)
    assert calls[0].data==calls[1].data
    assert calls[0].get_header('Idempotency-key')=='PUI/front:'+oid
    with pytest.raises(PermanentDeliveryError):transport.create(uuid4().hex,payload)
    with pytest.raises(ValueError):CreateOnlyHTTPTransport('https://example.invalid','PUI/front')


def test_bounded_snapshot_and_debug_rotation(tmp_path):
    path=tmp_path/'snapshot.geojson'
    _atomic_write_geojson(path,dict(ok=True),max_bytes=100)
    with pytest.raises(ValueError):_atomic_write_geojson(path,dict(large='x'*101),max_bytes=100)
    assert json.loads(path.read_text())==dict(ok=True)
    writer=BoundedImageWriter(tmp_path/'rgb','detection',max_files=2,max_bytes=10000)
    image=np.zeros((10,10,3),np.uint8)
    for i in range(10):assert writer.write(f'detection_{i}.png',image)
    assert len(list((tmp_path/'rgb').glob('*.png')))==2
    other=tmp_path/'rgb'/'unrelated.txt';other.write_text('preserve')
    writer.write('detection_11.png',image)
    assert other.read_text()=='preserve'
    writer.max_bytes=1
    assert not writer.write('detection_12.png',image)
    with pytest.raises(ValueError):writer.write('../detection_bad.png',image)


def test_database_budget_and_single_owner(tmp_path):
    settings=replace(DiscoverySettings(),max_database_bytes=256*1024,max_objects=10000)
    path=tmp_path/'state.db';store=DiscoveryStore(path,settings)
    try:
        with pytest.raises(RuntimeError):DiscoveryStore(path,settings)
        count=0
        with pytest.raises(CapacityExceeded):
            for i in range(10000):
                a=record();store.register(a,identified_feature(feature(),a['object_id'],settings.namespace));count+=1
        assert store.stats()['objects']==count and count>0
        with store.transaction():
            store.db.execute('UPDATE objects SET confidence=.8')
            active_size=sum(p.stat().st_size for p in tmp_path.glob('state.db*'))
            assert active_size <= settings.max_database_bytes
        size=sum(p.stat().st_size for p in tmp_path.glob('state.db*'))
        assert size <= settings.max_database_bytes
        assert not path.with_name(path.name+'-wal').exists()
    finally:store.close()
    with pytest.raises(ValueError):DiscoveryStore(path,replace(settings,namespace='another-mission'))


@pytest.mark.parametrize('values',[dict(retry_max_s=float('inf')),dict(confirmation_gap_s=float('nan')),
    dict(namespace='x'*129),dict(max_objects=0),dict(max_pending=1.5)])
def test_invalid_settings_rejected(values):
    with pytest.raises(ValueError):DiscoverySettings(**values)


def test_rotated_logs_and_bounded_registry_snapshot(tmp_path):
    settings=replace(DiscoverySettings(),log_bytes=512)
    service=DiscoveryService(tmp_path/'mission.db',settings)
    try:
        for i in range(100):service.logger.warning('Failure %s: %s',i,'x'*200)
        files=list((tmp_path/'logs').glob('discovery.log*'))
        assert len(files)==3 and all(p.stat().st_size<=512 for p in files)
        confirm(service.registry,[track(i,name=f'class-{i}') for i in range(8)])
        snapshot=service.store.snapshot(limit=3,max_bytes=800)
        assert len(snapshot['features'])<=3 and len(encode(snapshot))<=800
        assert service.store.stats()['objects']==8
    finally:service.close()


def test_queue_byte_pressure_report_and_recovery(tmp_path):
    settings=replace(DiscoverySettings(),max_pending_bytes=400)
    service=DiscoveryService(tmp_path/'mission.db',settings)
    try:
        confirm(service.registry,[track(1,name='cone'),track(2,name='helmet')])
        assert service.status()['queue_capacity_reached'] and service.status()['paused_reason']
        event=service.store.claim(0);service.store.acknowledge(event['object_id'],event['lease_token'],'remote')
        service.store.promote_ready()
        assert not service.status()['queue_capacity_reached']
        assert service.status()['paused_reason'] is None
    finally:service.close()


def test_http_permanent_rejection_and_missing_ack_never_mark_sent(store):
    from urllib.error import HTTPError
    confirm(PersistentRegistry(store,store.settings))
    def rejected(request,timeout):raise HTTPError(request.full_url,403,'denied',{},None)
    client=CreateOnlyHTTPTransport('https://example.invalid/points',store.settings.namespace,True,opener=rejected)
    worker=OutboxWorker(store,client,logging.getLogger('test'),clock=lambda:0.)
    assert worker.pump_once()
    assert store.stats()['blocked']==1 and store.stats()['sent']==0
    # A status 200 response without an object acknowledgement remains retryable.
    class Response:
        status=200
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return b'{}'
    client.opener=lambda *a,**kw:Response()
    row=store.db.execute('SELECT object_id,payload FROM outbox').fetchone()
    with pytest.raises(RuntimeError,match='acknowledge'):client.create(row[0],row[1])


def test_store_rejects_oversized_identity_state(store):
    a=record();a['appearance']=b'x'*4096
    with pytest.raises(ValueError):store.register(a)
    a=record();a['class_name']='x'*129
    with pytest.raises(ValueError):store.register(a)
    assert store.stats()['objects']==0
