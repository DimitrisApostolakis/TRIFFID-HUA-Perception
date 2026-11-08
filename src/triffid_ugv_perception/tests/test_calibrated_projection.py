from types import SimpleNamespace as NS
import numpy as np
import pytest

from triffid_ugv_perception.calibration import (
    calibrated_intrinsics, rigid_matrix, resolve_static_transform, transform_points)
from triffid_ugv_perception.pipeline import localize_detections


def test_pinhole_with_recorded_mount_and_inverse_chain():
    info=NS(width=3,height=3,k=[2.,0.,1.,0.,2.,1.,0.,0.,1.],d=[0.]*5)
    intrinsics=calibrated_intrinsics(info,(3,3))
    # A single center pixel at 2 m is [0,0,2] in the optical frame.
    depth=np.zeros((3,3),np.uint16);depth[1,1]=2000
    mask=depth>0
    record=dict(parent='base',child='optical',translation_m=[.4,.02,.28],
                quaternion_xyzw=[-.5,.5,-.5,.5])
    matrix=resolve_static_transform([record],'base','optical')
    detection=dict(bbox=[1,1,2,2],mask=mask,class_id=23,class_name='citizen',confidence=.8)
    result=localize_detections([detection],depth,(3,3),intrinsics,camera_to_target=matrix)[0]
    np.testing.assert_allclose(result['position'],[2.4,.02,.28],atol=1e-12)
    inverse=resolve_static_transform([record],'optical','base')
    np.testing.assert_allclose(transform_points(result['position'],inverse),[0,0,2],atol=1e-12)


def test_transform_composition_rotates_child_translation():
    q=np.sqrt(.5)
    records=[dict(parent='base',child='mount',translation_m=[1,0,0],quaternion_xyzw=[0,0,q,q]),
             dict(parent='mount',child='camera',translation_m=[1,0,0],quaternion_xyzw=[0,0,0,1])]
    matrix=resolve_static_transform(records,'base','camera')
    np.testing.assert_allclose(transform_points([0,0,0],matrix),[1,1,0],atol=1e-12)
    with pytest.raises(ValueError,match='No recorded transform'):
        resolve_static_transform(records,'base','unrelated_camera')


def test_projection_rejects_wrong_calibration_distortion_and_nonrigid_transform():
    info=NS(width=3,height=3,k=[2.,0.,1.,0.,2.,1.,0.,0.,1.],d=[0.]*5)
    with pytest.raises(ValueError,match='resolution'):
        calibrated_intrinsics(info,(6,6))
    info.d=[.01,0,0,0,0]
    with pytest.raises(ValueError,match='distortion'):
        calibrated_intrinsics(info,(3,3))
    with pytest.raises(ValueError,match='unit length'):
        rigid_matrix([0,0,0],[0,0,0,0])
    with pytest.raises(ValueError,match='rigid'):
        transform_points([0,0,1],np.diag([2.,1.,1.,1.]))


def test_saturated_depth_is_not_a_65_metre_object_and_input_is_preserved():
    depth=np.array([[65535,2000,65535]],np.uint16)
    detection=dict(bbox=[0,0,3,1],class_id=23,class_name='citizen',confidence=.8)
    result=localize_detections([detection],depth,(1,3),(1.,1.,1.,0.),camera_to_target=np.eye(4))
    assert len(result)==1 and result[0]['n_depth_pts']==1
    np.testing.assert_allclose(result[0]['position'],[0.,0.,2.])
    np.testing.assert_array_equal(depth,[[65535,2000,65535]])
    assert localize_detections([detection],np.full((1,3),65535,np.uint16),
        (1,3),(1.,1.,1.,0.),camera_to_target=np.eye(4))==[]

def test_localization_preserves_distinct_tracked_instances_and_reid_evidence():
    depth=np.full((3,3),2000,np.uint16)
    a=dict(bbox=[0,0,2,2],class_id=23,class_name='citizen',confidence=.9,
        track_id='session:1',appearance=[1.,0.],identity_class_id=-1)
    b=dict(a,track_id='session:2',bbox=[1,0,3,2],appearance=[0.,1.])
    rows=localize_detections([a,b],depth,(3,3),(100.,100.,1.,1.),camera_to_target=np.eye(4),deduplicate=False)
    assert len(rows)==2 and rows[0]['track_id']=='session:1'
    assert rows[1]['appearance']==[0.,1.] and rows[0]['identity_class_id']==-1
