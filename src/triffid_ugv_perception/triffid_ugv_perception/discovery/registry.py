"""Bounded identity association; ambiguous reappearances do not create uploads."""
from dataclasses import dataclass, field
import math
import time
from uuid import UUID, uuid4

import numpy as np

from ..spatial import _haversine_m
from .features import identity_class, identified_feature, location_ready, centroid
from .store import CapacityExceeded


def overlap(a, b):
    if a is None or b is None:
        return 0.
    width = max(0.,min(a[2],b[2])-max(a[0],b[0]))
    height = max(0.,min(a[3],b[3])-max(a[1],b[1]))
    intersection = width*height
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


@dataclass
class FrameResult:
    observations: list
    created_ids: list
    stale_frame: bool = False
    updated_ids: list = field(default_factory=list)


class PersistentRegistry:
    def __init__(self, store, settings, clock=time.monotonic):
        self.store, self.settings, self.clock = store, settings, clock
        self.known = {row['object_id']:row for row in store.records()}
        self.aliases, self.candidates, self.dirty = {}, {}, {}
        self.frame, self.last_timestamp = 0, None
        self.last_flush = clock()
        self.paused_reason = None
        if settings.class_correction:
            from ..tracking.class_correction import TemporalClassEvidence
            self.label_policy=TemporalClassEvidence()

    def _appearance(self, values):
        array = np.asarray(values if values is not None else [],dtype=np.float32)
        if array.ndim != 1 or len(array) > self.settings.max_appearance_dims or not np.isfinite(array).all():
            raise ValueError('Invalid or oversized appearance descriptor')
        norm = np.linalg.norm(array)
        return (array/norm).astype(np.float32).tobytes() if norm > 0 else None

    def _prepare(self, track):
        group = 'instance' if self.settings.class_correction else identity_class(track)
        tid = str(track['track_id'])
        confidence = float(track['confidence'])
        if len(str(track['class_name'])) > 128 or not group or len(group) > 128 or not tid or len(tid) > 128 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid track identity/confidence')
        box = track.get('bbox')
        if box is not None:
            box = list(map(float,box))
            if len(box) != 4 or not all(map(math.isfinite,box)) or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid image box')
        appearance = self._appearance(track.get('appearance'))
        external = None
        if track.get('object_id'):
            external = UUID(str(track['object_id'])).hex
        return (group,tid),box,appearance,confidence,external

    def _reassociate(self, group, box, appearance, feature, used, class_name=None, timestamp_ns=None):
        matches, uncertain = [], False
        point = centroid(feature) if location_ready(feature) else None
        for oid, known in self.known.items():
            if oid in used or known['identity_class'] != group:
                continue
            similarity = -1.
            if appearance and known['appearance'] and len(appearance) == len(known['appearance']):
                similarity = float(np.frombuffer(appearance,np.float32) @ np.frombuffer(known['appearance'],np.float32))
            iou = overlap(box,known['bbox'])
            previous_point = centroid(dict(geometry=known['geometry'])) if known['geometry'] else None
            nearby = bool(point and previous_point and _haversine_m(*point,*previous_point) <= self.settings.reid_radius_m)
            if self.settings.class_correction and class_name != known['class_name'] and not nearby:
                if box is None or known['bbox'] is None or timestamp_ns-known['last_seen_ns'] > 3e9:
                    continue
                a,b=np.asarray(box),np.asarray(known['bbox'])
                areas=np.maximum(1.,[(a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1])])
                shift=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/np.sqrt(max(areas))
                if not .4 <= areas[0]/areas[1] <= 2.5 or shift > 1.5:
                    continue
            if box is None and not appearance and point is None:
                uncertain = True
            if similarity >= self.settings.reid_similarity and (iou >= self.settings.reid_iou or nearby):
                matches.append((similarity+.1*iou,oid))
            elif (nearby or similarity >= self.settings.reid_uncertain_similarity
                  or (iou >= self.settings.reid_iou and (not appearance or not known['appearance']))):
                uncertain = True
        matches.sort(reverse=True)
        if matches:
            if len(matches)>1 and matches[0][0]-matches[1][0] < self.settings.reid_margin:
                return None, True
            return matches[0][1], False
        return None, uncertain

    def flush(self):
        if self.dirty:
            self.store.update_records(list(self.dirty.values()))
            self.dirty.clear()
        self.last_flush = self.clock()

    def update(self, tracks, features, timestamp_ns):
        if len(tracks) != len(features):
            raise ValueError('One feature entry is required per track')
        timestamp_ns = int(timestamp_ns)
        if not 0 <= timestamp_ns < 2**63:
            raise ValueError('Timestamp is outside SQLite nanosecond range')
        if self.last_timestamp is not None and timestamp_ns <= self.last_timestamp:
            return FrameResult([dict(t,object_id=None,discovery_state='stale_frame') for t in tracks],[],True)
        self.paused_reason = None
        self.last_timestamp = timestamp_ns
        self.frame += 1
        ttl = self.settings.alias_ttl_s*1e9
        self.aliases = {k:v for k,v in self.aliases.items() if timestamp_ns-v[1] <= ttl}
        prepared = []
        keys = set()
        for track in tracks:
            try:
                item = self._prepare(track)
                if item[0] in keys:
                    raise ValueError('Repeated physical track identity in a frame')
                keys.add(item[0])
                prepared.append(item)
            except (ValueError,TypeError,KeyError):
                prepared.append(None)
        # Reserve already associated objects before processing fresh tracks.
        used = {self.aliases[item[0]][0] for item in prepared if item and item[0] in self.aliases}
        used.update(item[4] for item in prepared if item and item[4] in self.known)
        observations, created, updated, current_keys, assigned = [], [], [], set(), set()
        for track, feature, item in zip(tracks,features,prepared):
            oid, state = None, 'invalid'
            if item is None:
                observations.append(dict(track,object_id=None,discovery_state=state))
                continue
            key, box, appearance, confidence, external = item
            current_keys.add(key)
            candidate = self.candidates.get(key)
            if (candidate and candidate['frame'] == self.frame-1 and
                    timestamp_ns-candidate['timestamp_ns'] <= self.settings.confirmation_gap_s*1e9):
                hits, first = candidate['hits']+1,candidate['first_ns']
            else:
                hits, first = 1,timestamp_ns
            if confidence < self.settings.min_confidence:
                hits, first = 0,timestamp_ns
            if key in self.aliases:
                oid = self.aliases[key][0]
            elif external in self.known:
                if self.known[external]['identity_class'] != key[0]:
                    observations.append(dict(track,object_id=None,discovery_state='identity_conflict'))
                    continue
                oid = external
            else:
                oid, uncertain = self._reassociate(key[0],box,appearance,feature,used,track['class_name'],timestamp_ns)
                state = 'uncertain' if uncertain else 'confirming'
                if oid is None and not uncertain and hits >= self.settings.confirm_frames:
                    candidate_oid = external or uuid4().hex
                    record = dict(object_id=candidate_oid,identity_class=key[0],
                        class_name=str(track['class_name']),class_id=track.get('class_id'),confidence=confidence,
                        first_seen_ns=first,last_seen_ns=timestamp_ns,bbox=box,appearance=appearance,
                        geometry=(feature or {}).get('geometry') if location_ready(feature) else None)
                    payload = identified_feature(feature,candidate_oid,self.settings.namespace) if feature else None
                    try:
                        self.store.register(record,payload)
                    except CapacityExceeded as error:
                        self.paused_reason = str(error)
                        state = 'capacity_paused'
                    else:
                        oid = candidate_oid
                        self.known[oid] = record
                        created.append(oid)
                        state = 'new'
            if len(self.candidates) < self.settings.max_candidates or key in self.candidates:
                self.candidates[key] = dict(frame=self.frame,timestamp_ns=timestamp_ns,first_ns=first,
                                            hits=min(hits,self.settings.confirm_frames))
            elif oid is None:
                state = 'capacity_paused'
                self.paused_reason = 'Unconfirmed-track capacity reached'
            if oid is not None:
                if oid in assigned:
                    observations.append(dict(track,object_id=None,discovery_state='identity_conflict'))
                    continue
                assigned.add(oid)
                # One current tracker alias per object, including after re-ID.
                self.aliases = {k:v for k,v in self.aliases.items() if v[0] != oid}
                self.aliases[key] = (oid,timestamp_ns)
                used.add(oid)
                known = self.known[oid]
                if appearance and known.get('appearance') and len(appearance) == len(known['appearance']):
                    prototype = .9*np.frombuffer(known['appearance'],np.float32)+.1*np.frombuffer(appearance,np.float32)
                    appearance = (prototype/max(float(np.linalg.norm(prototype)),1e-12)).astype(np.float32).tobytes()
                label_event=None
                if self.settings.class_correction:
                    label_state,label_event=self.label_policy.update(known.get('label_state'),track,timestamp_ns)
                    known['label_state']=label_state
                    track=dict(track,raw_class_name=track['class_name'],observation_confidence=confidence,
                        class_name=label_state['class_name'],class_id=label_state['class_id'],
                        confidence=label_state['confidence'],class_revision=label_state['revision'],
                        best_observation=label_state['best'],class_update=label_event,
                        first_class=label_state['first_class'])
                    if label_event and state!='new':updated.append(oid)
                known.update(class_name=str(track['class_name']),class_id=track.get('class_id'),
                             confidence=float(track['confidence']),last_seen_ns=timestamp_ns,bbox=box,
                             appearance=appearance or known.get('appearance'))
                ready_feature = feature if confidence >= self.settings.min_confidence else None
                first_location = known.get('geometry') is None and location_ready(ready_feature)
                if location_ready(ready_feature):
                    known['geometry'] = ready_feature['geometry']
                payload = identified_feature(ready_feature,oid,self.settings.namespace) if ready_feature else None
                self.dirty[oid] = (dict(known),payload)
                if first_location or (self.settings.class_correction and (state=='new' or label_event)):
                    try:
                        self.store.update_records([self.dirty[oid]])
                    except CapacityExceeded as error:
                        self.paused_reason = str(error)
                    else:
                        self.dirty.pop(oid,None)
                state = state if state == 'new' else 'known'
            observations.append(dict(track,object_id=oid,discovery_state=state))
        self.candidates = {k:v for k,v in self.candidates.items() if k in current_keys}
        # Alias and candidate memory have independent fixed upper bounds.
        if len(self.aliases) > self.settings.max_objects:
            oldest = sorted(self.aliases,key=lambda k:self.aliases[k][1])
            for key in oldest[:len(self.aliases)-self.settings.max_objects]:
                del self.aliases[key]
        if self.clock()-self.last_flush >= self.settings.flush_interval_s:
            try:
                self.flush()
            except CapacityExceeded as error:
                self.paused_reason = str(error)
        return FrameResult(observations,created,updated_ids=updated)
