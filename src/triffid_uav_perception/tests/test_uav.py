"""Contract/IO tests use synthetic media and an injected backend, no GPU/weights."""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from triffid_uav_perception.export import write_json
from triffid_uav_perception.frame_selection import angular_distance, select_views
from triffid_uav_perception.masks import decode_rle, encode_rle, observations_from_prediction
from triffid_uav_perception.models.base import ClassPrediction
from triffid_uav_perception.ontology import BY_ID, CLASSES, classes_for
from triffid_uav_perception.pipeline import detect_frames, detect_video, segment_top_view, select_video_frames
from triffid_uav_perception.srt_metadata import SrtFrameMeta, parse_srt
from triffid_uav_perception.supports import SupportBank
from triffid_uav_perception.uav_node import build_parser


def circle_records(pitch=None, start=0):
    records = []
    for i, degrees in enumerate(range(0, 360, 45)):
        angle = math.radians(degrees)
        records.append(SrtFrameMeta(i, start + i, 38 + .001 * math.cos(angle),
                                    23 + .001 * math.sin(angle) / math.cos(math.radians(38)),
                                    150, 50, (degrees + 180) % 360, pitch, 0))
    return records


def test_ontology_matches_user_decisions():
    assert len(CLASSES) == 38
    assert len(classes_for('Point')) == 18
    assert classes_for('Point', 'furniture')[0].id == 50
    assert len(classes_for('Polygon')) == 20
    assert BY_ID[16].geometry == 'Point'
    assert BY_ID[23].name == 'citizen'
    assert BY_ID[23].source_ids == (23,)
    assert [(BY_ID[i].name,BY_ID[i].source_ids) for i in (6,23,43)] == [
        ('first responder',(6,)),('citizen',(23,)),('military personnel',(43,))]
    assert classes_for('Point', 'citizen,16,citizen') == (BY_ID[23], BY_ID[16])
    with pytest.raises(ValueError):
        classes_for('Point', 'road')


def test_orbit_angles_are_position_not_camera_yaw():
    selected, settings = select_views(circle_records(), 10, 80, centre=(38, 23), tolerance=.01)
    assert len({s.frame_index for s in selected}) == 8
    assert [s.frame_index for s in selected] == list(range(0, 80, 10))
    assert selected[0].actual_angle_deg == pytest.approx(0)
    assert selected[0].telemetry.gimbal_yaw == 180
    assert settings['centre_source'] == 'explicit'


def test_heading_wrap_and_negative_yaw():
    records = [SrtFrameMeta(i, i, gimbal_yaw=i * 45 - 180) for i in range(8)]
    selected, _ = select_views(records, 10, 80, mode='heading')
    assert angular_distance(-1, 359) == 0
    assert selected[0].frame_index == 40


def test_low_rate_srt_uses_time_not_subtitle_number():
    selected, _ = select_views(circle_records(), 30, 240)
    assert selected[1].frame_index == 30  # SRT index is only 1


def test_missing_angles_fail_instead_of_reusing_view():
    with pytest.raises(ValueError, match='distinct views'):
        select_views(circle_records()[:3], 10, 80, centre=(38, 23))
    with pytest.raises(ValueError, match='distinct views'):
        select_views([SrtFrameMeta(0, 0, gimbal_yaw=0)], 10, 80, mode='heading')


def test_four_views_at_two_pitches():
    records = circle_records(-90) + circle_records(-45, start=8)
    selected, _ = select_views(records, 10, 160, views=4, pitches=(-90, -45))
    assert len(selected) == len({s.frame_index for s in selected}) == 8
    assert [s.target_pitch_deg for s in selected] == [-90] * 4 + [-45] * 4


def test_one_circle_time_window_and_estimated_centre():
    records = circle_records() + circle_records(start=8)
    selected, settings = select_views(records, 10, 160, start_s=8, end_s=16)
    assert all(s.timestamp_s >= 8 for s in selected)
    assert settings['centre_source'] == 'gps_bounds_estimate'


@pytest.mark.parametrize('kwargs', [dict(fps=0), dict(views=0), dict(tolerance=180),
    dict(start_s=5, end_s=1), dict(pitches=()), dict(centre=(float('nan'), 0))])
def test_invalid_selection_parameters(kwargs):
    parameters = dict(fps=10, frame_count=80)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        select_views(circle_records(), **parameters)


@pytest.mark.parametrize('shape', [(1, 1), (7, 13), (20, 5)])
def test_exact_rle_roundtrip(shape):
    random = np.random.default_rng(17)
    for mask in [np.zeros(shape, bool), np.ones(shape, bool), random.random(shape) > .5]:
        encoded = encode_rle(mask)
        assert np.array_equal(decode_rle(encoded), mask)
        assert sum(encoded['counts']) == mask.size


def test_mask_centroid_bbox_and_foreground_anchor():
    mask = np.zeros((20, 30), bool)
    mask[2:10, 3:11] = True
    mask[4:8, 5:9] = False  # centroid lies in the hole
    mask[15, 25] = True
    result = observations_from_prediction(ClassPrediction(16, mask), BY_ID[16], 'view_00')
    assert len(result) == 2
    first = result[0]
    assert first['bbox_xyxy'] == [3, 2, 11, 10]
    assert first['centroid_xy'] == [6.5, 5.5]
    assert first['area_px'] == 48
    x, y = first['raycast_point_xy']
    assert mask[y, x]
    assert np.array_equal(decode_rle(first['mask']), mask[2:10, 3:11])
    assert first['mask']['offset_xy'] == [3, 2]
    assert first['score'] is None


def test_empty_mask_minimum_area_and_probability_semantics():
    mask = np.zeros((10, 10), bool)
    assert observations_from_prediction(ClassPrediction(16, mask), BY_ID[16], 'a') == []
    mask[1:3, 1:3] = True
    prediction = ClassPrediction(16, mask, np.full(mask.shape, .75))
    result = observations_from_prediction(prediction, BY_ID[16], 'a')
    assert result[0]['score'] == .75
    assert 'uncalibrated' in result[0]['score_type']
    assert observations_from_prediction(prediction, BY_ID[16], 'a', min_area_px=5) == []


@pytest.mark.parametrize('mask', [np.zeros((5, 5), np.uint8), np.zeros((1, 5, 5), bool)])
def test_backend_shape_contract(mask):
    with pytest.raises(ValueError):
        observations_from_prediction(ClassPrediction(16, mask), BY_ID[16], 'x')


def test_srt_telemetry_parser_modern_and_legacy():
    records = parse_srt('1\n00:00:00,000 --> 00:00:00,100\n'
                        'FrameCnt: 1 [latitude: 38 longitude: 23] '
                        '[abs_alt: 150 rel_alt: 50 gb_yaw: -90 gb_pitch: -45 gb_roll: 0]\n\n'
                        '2\n00:00:01,000 --> 00:00:01,100\nGPS(23,38,150) H 50m')
    assert len(records) == 2
    assert records[0].gimbal_yaw == -90
    assert records[0].gimbal_pitch == -45
    assert records[1].lat == 38 and records[1].lon == 23


def test_support_manifest_relative_paths_multiclass_mask(tmp_path):
    cv2.imwrite(str(tmp_path / 'rgb.png'), np.zeros((5, 5, 3), np.uint8))
    mask = np.zeros((5, 5), np.uint8)
    mask[1:3, 1:3] = 16
    mask[4, 4] = 14
    cv2.imwrite(str(tmp_path / 'mask.png'), mask)
    path = tmp_path / 'supports.json'
    entries = []
    for i in range(5):
        cv2.imwrite(str(tmp_path / f'rgb_{i}.png'), np.full((5, 5, 3), i, np.uint8))
        entries.append({'image': f'rgb_{i}.png', 'mask': 'mask.png', 'foreground_value': 16})
    data = {'schema_version': 'triffid.uav.supports.v1', 'classes': {'debris': entries}}
    path.write_text(json.dumps(data))
    support = SupportBank(path).for_class(BY_ID[16])[0]
    _, foreground = support.load()
    assert foreground.sum() == 4
    with pytest.raises(ValueError, match='exactly 5'):
        SupportBank(path).for_class(BY_ID[23])


class FakeBackend:
    def __init__(self):
        self.calls = 0
    def prepare(self, classes, supports):
        self.classes = tuple(classes)
        supports.require_classes(classes)
        self.support_ids = {item.name: [s.example_id for s in supports.for_class(item)] for item in classes}
    def predict(self, rgb, classes):
        self.calls += 1
        mask = np.zeros(rgb.shape[:2], bool)
        mask[4:9, 7:13] = True
        return [ClassPrediction(c.id, mask.copy()) for c in classes]
    def metadata(self):
        return {'name': 'test_backend'}


@pytest.fixture
def synthetic_flight(tmp_path):
    video = tmp_path / 'flight.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 48))
    assert writer.isOpened()
    for i in range(80):
        writer.write(np.full((48, 64, 3), i, np.uint8))
    writer.release()
    srt = tmp_path / 'flight.srt'
    blocks = []
    for i, record in enumerate(circle_records()):
        blocks.append(f'{i+1}\n00:00:{i:02},000 --> 00:00:{i:02},100\n'
                      f'[latitude: {record.lat} longitude: {record.lon}] '
                      f'[gb_yaw: {record.gimbal_yaw} gb_pitch: -45 gb_roll: 0]')
    srt.write_text('\n\n'.join(blocks))
    return video, srt


def test_video_to_unity_json_end_to_end(synthetic_flight, tmp_path, bank_selection):
    video, srt = synthetic_flight
    backend = FakeBackend()
    classes = classes_for('Point', 'debris,citizen')
    output = detect_video(video, srt, tmp_path / 'run', backend, classes, supports=bank_selection(classes))
    result = json.loads(output.read_text())
    assert backend.calls == 8
    assert len(result['sources']) == 8
    assert result['coordinate_space'] == 'original_image_pixels'
    assert result['task'] == 'point_detections'
    assert result['support_bank']['shots'] == 5
    assert [s['example_id'] for s in result['support_bank']['classes']['citizen']] == backend.support_ids['citizen']
    ids = []
    for source in result['sources']:
        frame = cv2.imread(str(output.parent / source['image_file']))
        assert frame.shape == (48, 64, 3)
        assert float(frame.mean()) == pytest.approx(source['frame_index'], abs=2)
        assert len(source['observations']) == 2
        for observation in source['observations']:
            assert observation['bbox_xyxy'] == [7, 4, 13, 9]
            assert observation['target_geometry'] == 'Point'
            assert decode_rle(observation['mask']).sum() == 30
            ids.append(observation['observation_id'])
    assert len(ids) == len(set(ids))


def test_selection_only_needs_no_model(synthetic_flight, tmp_path):
    video, srt = synthetic_flight
    output = select_video_frames(video, srt, tmp_path / 'selected')
    assert len(json.loads(output.read_text())['sources']) == 8


def test_top_view_is_separate_polygon_task(tmp_path, bank_selection):
    image = tmp_path / 'top.png'
    cv2.imwrite(str(image), np.zeros((32, 64, 3), np.uint8))
    projection = tmp_path / 'projection.json'
    projection.write_text('{"projection_name":"test"}')
    backend = FakeBackend()
    classes = classes_for('Polygon', 'road,water')
    output = segment_top_view(image, tmp_path / 'top_run', backend, classes,
                              supports=bank_selection(classes), projection_json=projection)
    result = json.loads(output.read_text())
    assert backend.calls == 1
    assert result['task'] == 'polygon_segmentation'
    assert result['sources'][0]['projection_context'] == {'projection_name': 'test'}
    assert all(o['target_geometry'] == 'Polygon' for o in result['sources'][0]['observations'])


def test_failed_export_does_not_replace_existing_file(tmp_path):
    path = tmp_path / 'result.json'
    path.write_text('{"old": true}')
    with pytest.raises(ValueError):
        write_json(path, {'bad': float('nan')})
    assert json.loads(path.read_text()) == {'old': True}
    assert not list(tmp_path.glob('*.tmp'))


def test_select_cli_can_parse_without_sansa_options():
    args = build_parser().parse_args(['select', 'flight.mp4', '--output', 'out'])
    assert args.views == 8 and args.selection_mode == 'orbit'


def test_detect_preexported_frames_does_not_need_video_or_srt(synthetic_flight, tmp_path, bank_selection):
    video, srt = synthetic_flight
    manifest = select_video_frames(video, srt, tmp_path / 'selected')
    original = json.loads(manifest.read_text())
    video.unlink()
    srt.unlink()
    classes = classes_for('Point', 'civilian vehicle,window')
    backend = FakeBackend()
    result = detect_frames(manifest, tmp_path / 'inference', backend, classes,
                           supports=bank_selection(classes, shots=1))
    exported = json.loads(result.read_text())
    assert backend.calls == 8
    assert len(exported['sources']) == 8
    assert exported['support_bank']['shots'] == 1
    for before, after in zip(original['sources'], exported['sources']):
        assert before['frame_index'] == after['frame_index']
        assert before['timestamp_s'] == after['timestamp_s']
        assert before['telemetry'] == after['telemetry']
        assert Path(after['image_file']).is_file()
        assert len(after['observations']) == 2


@pytest.mark.parametrize('mutation', ['missing_file', 'duplicate_source', 'duplicate_frame', 'dimensions'])
def test_invalid_exported_frames(synthetic_flight, tmp_path, bank_selection, mutation):
    video, srt = synthetic_flight
    manifest = select_video_frames(video, srt, tmp_path / 'selected')
    payload = json.loads(manifest.read_text())
    if mutation == 'missing_file':
        payload['sources'][0]['image_file'] = 'missing.png'
    elif mutation == 'duplicate_source':
        payload['sources'][1]['source_id'] = payload['sources'][0]['source_id']
    elif mutation == 'duplicate_frame':
        payload['sources'][1]['frame_index'] = payload['sources'][0]['frame_index']
    else:
        payload['sources'][0]['width'] = 999
    manifest.write_text(json.dumps(payload))
    classes = classes_for('Point', 'citizen')
    with pytest.raises(ValueError):
        detect_frames(manifest, tmp_path / 'inference', FakeBackend(), classes,
                      supports=bank_selection(classes))
    assert not (tmp_path / 'inference/point_detections.json').exists()
