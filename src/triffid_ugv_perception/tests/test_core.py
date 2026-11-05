"""Behavioral regressions and preserved-methodology checks; no ROS/model weights needed."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from triffid_ugv_perception import geometry
from triffid_ugv_perception.config import NODE_PARAMETERS, CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY
from triffid_ugv_perception.pipeline import localize_detections
from triffid_ugv_perception.ontology import CLASSES, BY_ID, classes_for, geometry_type_for_class
from triffid_ugv_perception.tracker import ByteTracker
from triffid_ugv_perception.models import create_detector, register_detector
from triffid_ugv_perception.models.base import ClassPrediction
from triffid_ugv_perception.models.semantic import SemanticDetector
from triffid_ugv_perception.supports import BANK_SCHEMA, SupportBank, bank_template, initialize_bank, add_example
from triffid_ugv_perception.spatial import _merge_features

class Normalize(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            if node.attr in ('node', 'position'):
                return node.value
            if node.attr.startswith('_'):
                return ast.Name(id=node.attr[1:], ctx=node.ctx)
        return node

def test_algorithm_bodies_preserved():
    baseline = json.loads(Path(__file__).with_name('algorithm_baseline.json').read_text())
    root = Path(__file__).parents[1] / 'triffid_ugv_perception'
    for record in baseline['checks']:
        node = ast.parse((root / record['path']).read_text())
        for name in record['name'].split('.'):
            node = next(n for n in node.body if getattr(n, 'name', None) == name)
        statements = node.body[1:] if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str) else node.body
        body = Normalize().visit(ast.Module(body=statements, type_ignores=[]))
        digest = hashlib.sha256(ast.dump(body, include_attributes=False).encode()).hexdigest()
        assert digest == record['sha256'], record['name']

def test_depth_sampling_axes_and_invalid_depth():
    det = dict(bbox=(0., 0., 3., 2.), mask=np.ones((2, 3), bool))
    depth = np.array([[1000, 0, 2000], [3000, 4000, 5000]], np.uint16)
    points = geometry.sample_depth_for_detection(det, depth, 2., 2., 1., 0.)
    np.testing.assert_array_equal(points, [[-.5, 0., 1.], [1., 0., 2.], [-1.5, 1.5, 3.], [0., 2., 4.], [2.5, 2.5, 5.]])
    assert geometry.camera_to_base((1., 2., 3.)) == (3., -1., -2.)
    assert geometry.sample_depth_for_detection(det, np.zeros((2, 3), np.uint16), 2, 2, 1, 0) is None

def test_localization_and_existing_cross_class_nms():
    mask = np.zeros((480, 640), bool); mask[240:245, 320:325] = True
    depth = np.full(mask.shape, 2000, np.uint16)
    raw = [dict(bbox=(320.,240.,325.,245.), mask=mask, class_id=i, class_name=BY_ID[i].name, confidence=c)
           for i, c in ((14,.7),(15,.9))]
    localized = localize_detections(raw, depth, mask.shape, (CAMERA_FX,CAMERA_FY,CAMERA_CX,CAMERA_CY))
    assert len(localized) == 1 and localized[0]['class_id'] == 15
    np.testing.assert_allclose(localized[0]['position'], [2., -(322-CAMERA_CX)*2/CAMERA_FX, -(242-CAMERA_CY)*2/CAMERA_FY])
    assert localized[0]['n_depth_pts'] == 25
    assert localize_detections(raw, depth*0, mask.shape, (CAMERA_FX,CAMERA_FY,CAMERA_CX,CAMERA_CY)) == []

def test_tracker_confirmation_and_persistent_ids():
    tracker = ByteTracker(n_init=3, max_age=2)
    d = dict(bbox=(0.,0.,20.,20.), position=(1.,0.,0.), extent=(1.,1.,1.), class_id=23, class_name='citizen', confidence=.9)
    assert tracker.update([d]) == []
    assert tracker.update([d]) == []
    first = tracker.update([d]); assert len(first) == 1
    identity = first[0]['track_id']
    assert tracker.update([d])[0]['track_id'] == identity
    for _ in range(4): tracker.update([])
    for _ in range(3): last = tracker.update([d])
    assert last[0]['track_id'] > identity

def test_bank_is_ugv_only_and_preserves_roles(tmp_path):
    assert len(CLASSES) == 63 and len(bank_template()['classes']) == 63
    assert [BY_ID[i].name for i in (6,23,43)] == ['first responder','citizen','military personnel']
    path = initialize_bank(tmp_path/'ugv_supports')
    bank = SupportBank(path)
    assert not bank.validate(classes_for('citizen'))['valid']
    with pytest.raises(ValueError): initialize_bank(path.parent)
    data = bank_template(); data['schema_version'] = 'triffid.uav.support-bank.v1'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='schema'): SupportBank(path)

@pytest.mark.parametrize('shots',[1,5])
def test_support_selection_and_metadata(bank_selection, shots):
    classes = classes_for('citizen,debris')
    selected = bank_selection(classes, shots)
    assert len(selected.for_class(BY_ID[23])) == shots
    assert selected.metadata()['schema_version'] == BANK_SCHEMA
    assert set(selected.metadata()['classes']) == {'citizen','debris'}

class Backend:
    def prepare(self, classes, supports): supports.require_classes(classes)
    def predict(self, rgb, classes):
        assert rgb[0,0].tolist() == [30,20,10]  # BGR converted once at boundary.
        mask = np.zeros(rgb.shape[:2],bool)
        mask[1:3,1:3] = True; mask[1:3,5:7] = True
        scores = np.where(mask,.8,.1).astype(np.float32)
        yield ClassPrediction(classes[0].id, mask, scores)
    def metadata(self): return {'name':'stub'}

def test_semantic_components_remain_separate(bank_selection):
    classes = classes_for('citizen')
    detector = SemanticDetector(Backend(), classes, bank_selection(classes,1))
    result = detector.detect(np.full((8,8,3),[10,20,30],np.uint8))
    assert len(result) == 2
    assert result[0]['bbox'] == (1.,1.,3.,3.) and result[1]['bbox'] == (5.,1.,7.,3.)
    assert result[0]['mask'].sum() == 4 and result[0]['confidence'] == pytest.approx(.8)
    assert 'not calibrated' in detector.metadata()['score_type']

def test_model_replacement_and_dummy_need_no_weights():
    assert NODE_PARAMETERS['backend'] == 'sansa'
    detector = create_detector(dict(NODE_PARAMETERS,use_dummy_detections=True))
    assert len(detector.detect(np.zeros((2,3,3),np.uint8))) == 1
    marker = object()
    register_detector('test-backend',lambda config: marker)
    assert create_detector(dict(NODE_PARAMETERS,backend='test-backend')) is marker
    with pytest.raises(ValueError): create_detector(dict(NODE_PARAMETERS,backend='missing'))

def test_local_feature_accumulation_keeps_newest():
    a = dict(type='Feature',geometry=dict(type='Point',coordinates=[1,2]),properties=dict(local_frame=True,**{'class':'citizen','id':'1','confidence':.9}))
    b = json.loads(json.dumps(a)); b['geometry']['coordinates'] = [3,4]
    assert _merge_features([a],[b],3.) == [b]
    assert geometry_type_for_class('citizen') == 'Point'
    assert geometry_type_for_class('wall') == 'LineString'
    assert geometry_type_for_class('building') == 'Point'
    for item in CLASSES:
        assert geometry_type_for_class(item.name) == ('LineString' if item.name in ('fence','wall') else 'Point')

def test_export_routing_and_line_extents_for_every_class():
    from triffid_ugv_perception.geojson import GeoJSONFormatter
    class LocalPosition:
        gps_valid = False
        def body_to_gps(self, x, y, z): return x, y, z
    detections = [dict(coordinates=(10.,20.,3.), position=(10.,20.,3.), size=(4.,2.,1.),
                       class_name=item.name, track_id=str(item.id), confidence=.8) for item in CLASSES]
    result = GeoJSONFormatter(LocalPosition()).detections_to_geojson(detections)
    assert len(result['features']) == 63
    for item, feature in zip(CLASSES,result['features']):
        if item.name in ('fence','wall'):
            assert feature['geometry'] == dict(type='LineString',coordinates=[[12.,20.],[8.,20.]])
            assert feature['properties']['stroke-width'] == 2
        else:
            assert feature['geometry'] == dict(type='Point',coordinates=[10.,20.])
            assert 'stroke' not in feature['properties'] and 'fill' not in feature['properties']
    detections[1]['size'] = (0.,0.,0.)
    assert GeoJSONFormatter(LocalPosition()).detections_to_geojson([detections[1]])['features'][0]['geometry']['coordinates'] == [[10.15,20.],[9.85,20.]]
    detections[1]['size'] = (2.,6.,1.)
    assert GeoJSONFormatter(LocalPosition()).detections_to_geojson([detections[1]])['features'][0]['geometry']['coordinates'] == [[10.,23.],[10.,17.]]

def test_managed_bank_add_preserves_existing_example(tmp_path):
    import cv2
    initialize_bank(tmp_path/'ugv_supports')
    image=tmp_path/'reference.png';mask=tmp_path/'mask.png'
    cv2.imwrite(str(image),np.full((8,10,3),100,np.uint8))
    foreground=np.zeros((8,10),np.uint8);foreground[2:6,3:8]=7
    cv2.imwrite(str(mask),foreground)
    output=add_example(tmp_path/'ugv_supports','citizen',1,image,mask,foreground_value=7)
    selected=SupportBank(tmp_path/'ugv_supports').select(classes_for('citizen'),shots=1)
    assert selected.for_class(BY_ID[23])[0].load()[1].sum() == 20
    before=(output/'image.png').read_bytes()
    with pytest.raises(ValueError,match='already exists'):
        add_example(tmp_path/'ugv_supports','citizen',1,image,mask,foreground_value=7)
    assert (output/'image.png').read_bytes() == before
    with pytest.raises(ValueError): SupportBank(tmp_path/'ugv_supports').select(classes_for('citizen'),shots=5)
