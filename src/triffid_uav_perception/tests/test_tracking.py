import copy
import json

import cv2
import numpy as np
import pytest

from triffid_uav_perception.export import SCHEMA_VERSION
from triffid_uav_perception.masks import encode_rle
from triffid_uav_perception.tracking import TrackingSettings, track_document


class FakeTracker:
    def __init__(self, identities):
        self.identities = iter(identities)
        self.times = []

    def update(self, detections, image, timestamp_ns):
        self.times.append(timestamp_ns)
        return [dict(d, track_id=tid) for d, tid in zip(detections, next(self.identities))]

    def metadata(self):
        return {'name': 'test'}


def observations(tmp_path, scores=(.6, .9, .7)):
    sources = []
    for i, score in enumerate(scores):
        frame = tmp_path/f'frame{i}.png'
        assert cv2.imwrite(str(frame), np.zeros((32, 64, 3), np.uint8))
        mask = encode_rle(np.ones((4, 4), bool))
        x = 4 + i*8
        mask['offset_xy'] = [x, 4]
        obs = dict(observation_id=f'o{i}', class_id=13, class_name='window',
                   target_geometry='Point', score=score, mask=mask,
                   bbox_xyxy=[x, 4, x+4, 8], centroid_xy=[x+1.5, 5.5],
                   raycast_point_xy=[x+1, 5], area_px=16)
        sources.append(dict(source_id=f's{i}', source_type='video_frame', frame_index=i*30,
                            timestamp_s=float(i), width=64, height=32,
                            image_file=str(frame), observations=[obs],
                            telemetry={'frame_index': i, 'timestamp_s': float(i)}))
    payload = dict(schema_version=SCHEMA_VERSION, task='point_detections', sources=sources)
    path = tmp_path/'detections.json'
    path.write_text(json.dumps(payload))
    return path, payload


def test_confirm_once_best_original_mask_and_point(tmp_path):
    path, raw = observations(tmp_path)
    original = path.read_bytes()
    output = tmp_path/'tracking'
    report = track_document(path, output, FakeTracker([[7], [7], [7]]))
    assert report['tracked_observations'] == 3
    assert report['confirmed_objects'] == 1
    tracked = json.loads((output/'tracked_points.json').read_text())
    annotated = [s['observations'][0] for s in tracked['sources']]
    assert [o['tracking']['is_new'] for o in annotated] == [False, True, False]
    assert len({o['tracking']['object_id'] for o in annotated}) == 1
    unique = json.loads((output/'point_detections.json').read_text())
    chosen = [o for s in unique['sources'] for o in s['observations']]
    assert len(chosen) == 1
    for field in ('mask', 'raycast_point_xy', 'score', 'class_id'):
        assert chosen[0][field] == raw['sources'][1]['observations'][0][field]
    assert path.read_bytes() == original


def test_match_survives_missed_frame_and_time_gap(tmp_path):
    path, raw = observations(tmp_path)
    raw['sources'][1]['observations'] = []
    raw['sources'][2]['timestamp_s'] = 12.
    path.write_text(json.dumps(raw))
    tracker = FakeTracker([[2], [], [2]])
    report = track_document(path, tmp_path/'tracking', tracker)
    assert tracker.times == [0, 10**9, 12*10**9]
    assert report['confirmed_objects'] == 1
    assert report['timings'][2]['new_objects'] == 1


def test_two_same_class_objects_stay_separate(tmp_path):
    path, raw = observations(tmp_path, (.7, .8))
    for i, source in enumerate(raw['sources']):
        extra = copy.deepcopy(source['observations'][0])
        extra['observation_id'] += '_other'
        source['observations'].append(extra)
    path.write_text(json.dumps(raw))
    report = track_document(path, tmp_path/'tracking', FakeTracker([[4, 5], [4, 5]]))
    assert report['confirmed_objects'] == 2


def test_duplicate_track_in_same_frame_is_error(tmp_path):
    path, raw = observations(tmp_path, (.7,))
    extra = copy.deepcopy(raw['sources'][0]['observations'][0])
    extra['observation_id'] = 'other'
    raw['sources'][0]['observations'].append(extra)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='one-to-one'):
        track_document(path, tmp_path/'tracking', FakeTracker([[4, 4]]))


def test_single_frame_track_not_exported(tmp_path):
    path, _ = observations(tmp_path, (.8,))
    report = track_document(path, tmp_path/'tracking', FakeTracker([[3]]))
    assert report['confirmed_objects'] == 0
    assert report['tentative_objects'] == 1


def test_sort_timestamp_and_reject_reused_time(tmp_path):
    path, raw = observations(tmp_path)
    raw['sources'].reverse()
    path.write_text(json.dumps(raw))
    tracker = FakeTracker([[0], [0], [0]])
    track_document(path, tmp_path/'tracking', tracker)
    assert tracker.times == [0, 10**9, 2*10**9]
    raw['sources'][0]['timestamp_s'] = raw['sources'][1]['timestamp_s']
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='distinct'):
        track_document(path, tmp_path/'invalid', FakeTracker([]))


def test_bad_frame_fails_before_association(tmp_path):
    path, raw = observations(tmp_path)
    raw['sources'][2]['image_file'] = 'missing.png'
    path.write_text(json.dumps(raw))
    tracker = FakeTracker([])
    with pytest.raises(ValueError, match='Missing tracking frame'):
        track_document(path, tmp_path/'tracking', tracker)
    assert tracker.times == []


def test_null_scores_and_tiny_actual_masks_are_filtered(tmp_path):
    path, raw = observations(tmp_path, (None, .8))
    # Misreported area cannot bypass the decoded-mask area filter.
    raw['sources'][1]['observations'][0]['area_px'] = 100000
    path.write_text(json.dumps(raw))
    report = track_document(path, tmp_path/'tracking', FakeTracker([[], []]),
                            settings=TrackingSettings(min_area_px=17))
    assert sum(t['candidates'] for t in report['timings']) == 0


def test_capacity_is_fail_closed_and_existing_output_not_overwritten(tmp_path):
    path, _ = observations(tmp_path)
    with pytest.raises(ValueError, match='capacity'):
        track_document(path, tmp_path/'tracking', FakeTracker([[0], [1], [2]]),
                       settings=TrackingSettings(max_identities=1))
    out = tmp_path/'existing'
    out.mkdir()
    (out/'keep.txt').write_text('keep')
    with pytest.raises(ValueError, match='must be empty'):
        track_document(path, out, FakeTracker([]))
    assert (out/'keep.txt').read_text() == 'keep'


def test_identity_namespace_includes_tracker_settings(tmp_path):
    path, _ = observations(tmp_path)
    sessions = []
    for index, hits in enumerate((2, 2, 1)):
        output = tmp_path/f'tracking{index}'
        track_document(path, output, FakeTracker([[1], [1], [1]]),
                       settings=TrackingSettings(confirmation_hits=hits))
        sessions.append(json.loads((output/'tracked_points.json').read_text())['tracking']['session_id'])
    assert sessions[0] == sessions[1]
    assert sessions[0] != sessions[2]


def test_binary_mask_tracking_never_invents_a_model_confidence(tmp_path):
    path, _ = observations(tmp_path, (None, None))
    out = tmp_path/'tracking'
    track_document(path, out, FakeTracker([[9], [9]]),
                   settings=TrackingSettings(accept_unscored_masks=True))
    tracked = json.loads((out/'tracked_points.json').read_text())
    assert [s['observations'][0]['score'] for s in tracked['sources']] == [None, None]
    assert [s['observations'][0]['tracking']['is_new'] for s in tracked['sources']] == [False, True]
    unique = json.loads((out/'point_detections.json').read_text())
    selected = [o for s in unique['sources'] for o in s['observations']]
    assert len(selected) == 1 and selected[0]['score'] is None
    assert selected[0]['tracking']['representative'] == 'first_unscored_observation'
