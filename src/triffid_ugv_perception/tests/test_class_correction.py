"""Identity continuity, ambiguous neighbours and confidence-based local correction."""
import numpy as np
import torch
import pytest
from triffid_ugv_perception.tracking.class_correction import ClassChangeAssociation,TemporalClassEvidence
from triffid_ugv_perception.discovery import DiscoverySettings,DiscoveryStore,PersistentRegistry


def observation(name='chainsaw',score=.6,tid=3):
    return dict(track_id=tid,class_id=59 if name=='chainsaw' else 35,class_name=name,
                confidence=score,bbox=[10.,10.,30.,30.],appearance=[1.,0.])


def test_one_confident_wrong_label_cannot_override_consistent_history():
    policy=TemporalClassEvidence();state=None
    for second in (1,2,3):state,_=policy.update(state,observation(),second*10**9)
    state,event=policy.update(state,observation('bag',.99),4*10**9)
    assert state['class_name']=='chainsaw' and not (event and event['kind']=='class')
    state,event=policy.update(state,observation('bag',.9),5*10**9)
    assert state['class_name']=='bag' and event['kind']=='class'


def test_low_confidence_recovery_does_not_vote_for_class_switch():
    policy=TemporalClassEvidence();state,_=policy.update(None,observation(),10**9)
    for second in range(2,7):state,event=policy.update(state,observation('bag',.29),second*10**9)
    assert state['class_name']=='chainsaw' and state['revision']==0
    assert len(state['history'])<=5


def test_duplicate_hypotheses_are_one_observation_with_one_vote_per_class():
    p=ClassChangeAssociation()
    ds=[observation(),observation('bag',.55)];p.set_sources(ds)
    output=p.prepare(torch.tensor([d['bbox'] for d in ds]),torch.tensor([59,35]),torch.tensor([.6,.55]),torch.tensor([[1.,0.],[1.,0.]]),[])
    assert len(output[0])==1
    assert len(next(iter(p.evidence.values())))==2


def test_cross_class_match_needs_distinctive_appearance_and_geometry():
    p=ClassChangeAssociation()
    boxes=torch.tensor([[10.,10.,30.,30.]])
    result=p.apply(torch.tensor([[.9]]),torch.tensor([[.95]]),boxes,torch.tensor([35]),boxes,torch.tensor([59]),2,torch.tensor([1]))
    assert result.item()>.5
    # A nearby helmet with a different appearance must remain a separate object.
    p=ClassChangeAssociation()
    result=p.apply(torch.tensor([[.9]]),torch.tensor([[.2]]),boxes,torch.tensor([3]),boxes,torch.tensor([59]),2,torch.tensor([1]))
    assert result.item()<0 and p.filter_new(torch.tensor([True])).item()
    # Two equally plausible identities are held instead of producing a duplicate.
    p=ClassChangeAssociation()
    result=p.apply(torch.tensor([[.9,.9]]),torch.tensor([[.95,.94]]),boxes,torch.tensor([35]),boxes.repeat(2,1),torch.tensor([59,59]),2,torch.tensor([1,1]))
    assert (result<0).all() and not p.filter_new(torch.tensor([True])).item()


def test_same_uuid_survives_class_correction_and_restart(tmp_path):
    settings=DiscoverySettings(class_correction=True)
    path=tmp_path/'mission.db';store=DiscoveryStore(path,settings);registry=PersistentRegistry(store,settings)
    registry.update([observation()],[None],10**9)
    oid=registry.update([observation()],[None],2*10**9).created_ids[0]
    first=registry.update([observation('bag',.64)],[None],3*10**9)
    assert not first.created_ids and first.observations[0]['class_name']=='chainsaw'
    second=registry.update([observation('bag',.78)],[None],4*10**9)
    assert not second.created_ids and second.updated_ids==[oid]
    assert second.observations[0]['class_name']=='bag' and second.observations[0]['object_id']==oid
    # Meaningful label changes are persisted immediately, before periodic flush.
    store.close()
    store=DiscoveryStore(path,settings)
    try:
        assert store.records()[0]['label_state']['class_name']=='bag'
        registry=PersistentRegistry(store,settings)
        output=registry.update([observation('bag',.8,tid=99)],[None],5*10**9)
        assert not output.created_ids and output.observations[0]['object_id']==oid
        assert store.stats()['objects']==1 and store.stats()['pending']==0
    finally:store.close()


def test_one_hz_motion_without_box_overlap_can_keep_identity_but_far_objects_cannot():
    p=ClassChangeAssociation();old=torch.tensor([[0.,0.,20.,20.]])
    shifted=torch.tensor([[22.,0.,42.,20.]])
    result=p.apply(torch.tensor([[.9]]),torch.tensor([[.96]]),shifted,torch.tensor([35]),old,torch.tensor([59]),2,torch.tensor([1]))
    assert result.item()>.5
    far=shifted+200
    result=p.apply(torch.tensor([[.9]]),torch.tensor([[.99]]),far,torch.tensor([35]),old,torch.tensor([59]),2,torch.tensor([1]))
    assert result.item()<0 and p.filter_new(torch.tensor([True])).item()


def test_class_correction_does_not_recreate_acknowledged_api_event(tmp_path):
    feature=dict(type='Feature',geometry=dict(type='Point',coordinates=[23.7,37.9]),
        properties=dict(location_valid=True,heading_valid=True,local_frame=False))
    settings=DiscoverySettings(class_correction=True)
    store=DiscoveryStore(tmp_path/'state.db',settings);registry=PersistentRegistry(store,settings)
    try:
        registry.update([observation()],[feature],10**9)
        oid=registry.update([observation()],[feature],2*10**9).created_ids[0]
        pending=store.claim(0);store.acknowledge(oid,pending['lease_token'],'remote-one')
        registry.update([observation('bag',.64)],[feature],3*10**9)
        result=registry.update([observation('bag',.78)],[feature],4*10**9)
        assert result.updated_ids==[oid] and not result.created_ids
        assert store.stats()['objects']==1 and store.stats()['pending']==0
        assert store.claim(100) is None
    finally:store.close()


def test_moderate_appearance_recovery_requires_a_clear_two_way_lead():
    old=torch.tensor([[0.,0.,20.,20.]])
    enlarged=torch.tensor([[10.,0.,40.,30.]])
    p=ClassChangeAssociation()
    result=p.apply(torch.tensor([[.4]]),torch.tensor([[.68]]),enlarged,torch.tensor([35]),old,torch.tensor([59]),3,torch.tensor([1]))
    assert result.item()>.5
    p=ClassChangeAssociation()
    result=p.apply(torch.tensor([[.4,.4]]),torch.tensor([[.68,.64]]),enlarged,torch.tensor([35]),old.repeat(2,1),torch.tensor([59,59]),3,torch.tensor([1,1]))
    assert (result<0).all() and not p.filter_new(torch.tensor([True])).item()


def test_class_update_cannot_mutate_or_duplicate_an_existing_creation_payload(tmp_path):
    feature=dict(type='Feature',geometry=dict(type='Point',coordinates=[23.7,37.9]),
        properties=dict(location_valid=True,heading_valid=True,local_frame=False,**{'class':'chainsaw'}))
    settings=DiscoverySettings(class_correction=True)
    store=DiscoveryStore(tmp_path/'state.db',settings);registry=PersistentRegistry(store,settings)
    try:
        registry.update([observation()],[feature],10**9)
        oid=registry.update([observation()],[feature],2*10**9).created_ids[0]
        payload=bytes(store.db.execute('SELECT payload FROM outbox').fetchone()[0])
        registry.update([observation('bag',.64)],[feature],3*10**9)
        result=registry.update([observation('bag',.78)],[feature],4*10**9)
        assert result.updated_ids==[oid] and store.stats()['pending']==1
        assert bytes(store.db.execute('SELECT payload FROM outbox').fetchone()[0])==payload
    finally:store.close()


def test_legacy_class_aware_objects_cannot_silently_enter_new_identity_policy(tmp_path):
    path=tmp_path/'legacy.db';settings=DiscoverySettings()
    store=DiscoveryStore(path,settings);registry=PersistentRegistry(store,settings)
    registry.update([observation()],[None],10**9)
    registry.update([observation()],[None],2*10**9)
    registry.flush()
    store.db.execute("DELETE FROM metadata WHERE key='identity_policy'")
    store.close()
    with pytest.raises(ValueError,match='Legacy class-aware'):
        DiscoveryStore(path,DiscoverySettings(class_correction=True))
    store=DiscoveryStore(path,settings)
    try:assert store.stats()['objects']==1
    finally:store.close()
