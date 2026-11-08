"""Track existing point masks and retain one best observation per confirmed ID.

All associations use images; this module never invents geographic positions.
The all-frame audit and the deduplicated Unity input are distinct documents.
"""
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import logging
import math
from pathlib import Path
import time
from uuid import NAMESPACE_URL, uuid5

import cv2

from ..export import SCHEMA_VERSION, write_json
from ..visualize import validate_observation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingSettings:
    confirmation_hits: int = 2
    min_score: float = .5
    min_area_px: int = 16
    memory_seconds: int = 900
    max_identities: int = 4096
    accept_unscored_masks: bool = False

    def __post_init__(self):
        for name in ('confirmation_hits', 'min_area_px', 'memory_seconds', 'max_identities'):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(self.min_score) or not 0 <= self.min_score <= 1:
            raise ValueError('min_score must be in [0, 1]')
        if not isinstance(self.accept_unscored_masks, bool):
            raise ValueError('accept_unscored_masks must be boolean')


def _sources(document):
    if document.get('schema_version') != SCHEMA_VERSION or document.get('task') != 'point_detections':
        raise ValueError('Tracking requires a point_detections observations document')
    sources = document.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Tracking requires at least one frame')
    seen_ids, seen_times, seen_frames = set(), set(), set()
    for source in sources:
        if source.get('source_type') != 'video_frame':
            raise ValueError('Tracking requires video-frame sources')
        sid, timestamp, frame = source.get('source_id'), source.get('timestamp_s'), source.get('frame_index')
        if not isinstance(sid, str) or not sid or sid in seen_ids:
            raise ValueError('Source IDs must be distinct and nonempty')
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Every frame needs a finite nonnegative timestamp_s')
        tick = round(timestamp * 1e9)
        if tick in seen_times or not isinstance(frame, int) or isinstance(frame, bool) or frame < 0 or frame in seen_frames:
            raise ValueError('Frame timestamps and indices must be distinct')
        if any(not isinstance(source.get(k), int) or isinstance(source[k], bool) or source[k] <= 0
               for k in ('width', 'height')):
            raise ValueError('Source dimensions must be positive integers')
        seen_ids.add(sid)
        seen_times.add(tick)
        seen_frames.add(frame)
    ordered = sorted(sources, key=lambda source: source['timestamp_s'])
    if any(a['frame_index'] >= b['frame_index'] for a, b in zip(ordered, ordered[1:])):
        raise ValueError('Frame indices must increase with timestamps within one video')
    return ordered


def track_document(input_json, output_dir, tracker, *, settings=None, render=None):
    """Offline, deterministic session. `tracker` and `render` are replaceable.

    An object is confirmed after distinct-frame matches, including across gaps.
    The selected raycast point/mask comes from its highest-scoring observation;
    it is never averaged across camera views. A run has bounded identity memory.
    """
    settings = settings or TrackingSettings()
    input_json, output_dir = Path(input_json).resolve(), Path(output_dir).resolve()
    if output_dir == input_json.parent or input_json.is_relative_to(output_dir):
        raise ValueError('Use a separate tracking output directory; preserve detector results')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Tracking output directory must be empty; old artifacts are preserved')
    raw = input_json.read_bytes()
    document = json.loads(raw.decode('utf-8-sig'))
    sources = _sources(document)
    # Validate masks and image availability before advancing tracking state.
    ids, areas = set(), {}
    for source in sources:
        image = Path(source.get('image_file', ''))
        source['_image_path'] = str(image if image.is_absolute() else input_json.parent / image)
        if not Path(source['_image_path']).is_file():
            raise ValueError(f'Missing tracking frame: {source["_image_path"]}')
        for observation in source['observations']:
            oid = observation.get('observation_id')
            if not isinstance(oid, str) or not oid or oid in ids:
                raise ValueError('Observation IDs must be distinct and nonempty')
            ids.add(oid)
            crop, _, _ = validate_observation(observation, source['width'], source['height'])
            areas[oid] = int(crop.sum())
            if observation.get('target_geometry') != 'Point':
                raise ValueError('Image tracking only accepts Point observations')
            cid = observation.get('class_id')
            if not isinstance(cid, int) or isinstance(cid, bool) or cid < 0:
                raise ValueError('Observations need nonnegative integer class IDs')
            score = observation.get('score')
            if score is not None and (not isinstance(score, (int, float)) or isinstance(score, bool)
                                      or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError('Observation scores must be null or finite values in [0, 1]')
    fingerprint = hashlib.sha256(raw).hexdigest()
    backend_context = {key: value for key, value in tracker.metadata().items()
                       if key not in ('image_encodings',)}
    identity_context = json.dumps(dict(input_sha256=fingerprint, settings=asdict(settings),
                                      backend=backend_context), sort_keys=True)
    session_id = uuid5(NAMESPACE_URL, 'triffid:uav:image-tracking:' + identity_context).hex
    identities, rows, timings = {}, [], []
    exports = 0
    for source in sources:
        start = time.perf_counter()
        image = cv2.imread(source['_image_path'])
        if image is None or image.shape[:2] != (source['height'], source['width']):
            raise ValueError(f'Frame dimensions differ from observations JSON: {source["source_id"]}')
        detections = []
        by_id = {obs['observation_id']: obs for obs in source['observations']}
        for observation in source['observations']:
            score = observation.get('score')
            area = areas[observation['observation_id']]
            if area < settings.min_area_px or (score is None and not settings.accept_unscored_masks):
                continue
            if score is not None and score < settings.min_score:
                continue
            # MASA requires a proposal weight. This is not written into the source
            # model score or GeoJSON confidence when the backend exposes binary masks.
            proposal_weight = 1.0 if score is None else float(score)
            detections.append(dict(bbox=observation['bbox_xyxy'], confidence=proposal_weight,
                                   class_id=observation['class_id'], class_name=observation['class_name'],
                                   observation_id=observation['observation_id']))
        begin_tracking = time.perf_counter()
        tracks = tracker.update(detections, image, round(source['timestamp_s'] * 1e9))
        tracker_s = time.perf_counter() - begin_tracking
        accepted_ids = {d['observation_id'] for d in detections}
        used_ids, used_tracks, annotated = set(), set(), []
        for track in tracks:
            oid, tid = track.get('observation_id'), track.get('track_id')
            if oid not in accepted_ids or oid in used_ids or not isinstance(tid, int) or tid < 0 or tid in used_tracks:
                raise ValueError('Tracker violated one-to-one detection/identity association')
            observation = by_id[oid]
            if track['class_id'] != observation['class_id']:
                raise ValueError('Tracker changed the source detection class')
            used_ids.add(oid)
            used_tracks.add(tid)
            if tid not in identities:
                if len(identities) >= settings.max_identities:
                    raise ValueError('Identity capacity reached; refusing to forget IDs and duplicate exports')
                identities[tid] = dict(hits=0, class_id=observation['class_id'], emitted=False,
                                       object_id=uuid5(NAMESPACE_URL, f'{session_id}:{tid}').hex,
                                       first_seen_s=source['timestamp_s'], best=None)
            state = identities[tid]
            if state['class_id'] != observation['class_id']:
                raise ValueError('A persistent track ID changed class')
            state['hits'] += 1
            state['last_seen_s'] = source['timestamp_s']
            confirmed = state['hits'] >= settings.confirmation_hits
            new = confirmed and not state['emitted']
            if new:
                state['emitted'] = True
                exports += 1
            item = copy.deepcopy(observation)
            item['tracking'] = dict(track_id=tid, object_id=state['object_id'],
                                    state='confirmed' if confirmed else 'tentative',
                                    is_new=new, hits=state['hits'],
                                    unscored_proposal=observation['score'] is None)
            annotated.append(item)
            # Strict > retains the earlier measurement on ties.
            if state['best'] is None or (observation['score'] is not None and (
                    state['best'][1]['score'] is None or observation['score'] > state['best'][1]['score'])):
                state['best'] = (source['source_id'], copy.deepcopy(item))
        record = {key: copy.deepcopy(value) for key, value in source.items()
                  if key not in ('observations', '_image_path')}
        record['image_file'] = source['_image_path']
        record['observations'] = annotated
        rows.append(record)
        timing = dict(source_id=source['source_id'], timestamp_s=source['timestamp_s'],
                      detections=len(source['observations']), candidates=len(detections),
                      tracked=len(annotated), new_objects=sum(o['tracking']['is_new'] for o in annotated),
                      tracker_s=tracker_s)
        if render:
            render(image, source, annotated)
        timing['total_s'] = time.perf_counter() - start
        timings.append(timing)
        log.info('%s: candidates=%d tracks=%d new=%d tracker=%.3fs', source['source_id'],
                 len(detections), len(annotated), timing['new_objects'], tracker_s)
    tracked = {key: copy.deepcopy(value) for key, value in document.items() if key != 'sources'}
    tracked['sources'] = rows
    tracked['tracking'] = dict(session_id=session_id, input_sha256=fingerprint,
                               settings=asdict(settings), backend=tracker.metadata(),
                               coordinate_space='image_pixels', persistence_scope='one input video/run',
                               unscored_policy='Fixed internal proposal weight 1.0 only when explicitly enabled; exported model scores remain null')
    tracked['deduplication'] = 'All tracked observations; repeats retained for audit, not unique export'
    unique = copy.deepcopy(tracked)
    for source in unique['sources']:
        source['observations'] = []
    lookup = {source['source_id']: source for source in unique['sources']}
    for state in identities.values():
        if not state['emitted']:
            continue
        sid, observation = state['best']
        observation['tracking'].update(state='confirmed', is_new=False, hits=state['hits'],
                                        representative='highest_score' if observation['score'] is not None else 'first_unscored_observation',
                                        first_seen_s=state['first_seen_s'], last_seen_s=state['last_seen_s'])
        lookup[sid]['observations'].append(observation)
    unique['deduplication'] = 'One highest-score original observation per confirmed ID; first observation when scores are unavailable'
    unique_ids = [o['tracking']['object_id'] for source in unique['sources'] for o in source['observations']]
    assert len(unique_ids) == len(set(unique_ids)) == exports
    active_times = [t['tracker_s'] for t in timings if t['candidates']]
    summary = dict(frames=len(sources), raw_detections=sum(len(s['observations']) for s in sources),
                   tracked_observations=sum(len(s['observations']) for s in rows),
                   confirmed_objects=exports, tentative_objects=len(identities)-exports,
                   tracker_mean_s=sum(t['tracker_s'] for t in timings)/len(timings),
                   tracker_first_nonempty_s=active_times[0] if active_times else None,
                   tracker_warm_nonempty_mean_s=(sum(active_times[1:])/len(active_times[1:])
                                                if len(active_times)>1 else None),
                   timings=timings, accuracy_evaluated=False, api_enabled=False,
                   limitations=['Image IDs are not guaranteed physical identities across an orbit.',
                                'View changes and repeated lookalikes can split or merge tracks.',
                                ('Unscored masks use a fixed internal proposal weight, not an inferred confidence.'
                                 if settings.accept_unscored_masks else
                                 'Null-score and below-threshold detections are not tracker candidates.'),
                                'Confirmation needs distinct observed frames; no coordinates are inferred.'])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir/'tracked_points.json', tracked)
    write_json(output_dir/'point_detections.json', unique)
    write_json(output_dir/'timings.json', summary)
    return summary
