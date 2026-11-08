"""Compact feature encoding and location checks; no network or ROS imports."""
import copy
import json
import math

PERSON_ROLES = frozenset(('citizen', 'civilian', 'first responder', 'military personnel', 'person'))


def identity_class(track):
    name = str(track['class_name']).strip().lower()
    return 'person' if name in PERSON_ROLES else name


def encode(value):
    return json.dumps(value, separators=(',', ':'), sort_keys=True, allow_nan=False).encode('utf-8')


def centroid(feature):
    geometry = (feature or {}).get('geometry') or {}
    coordinates = geometry.get('coordinates')
    kind = geometry.get('type')
    if kind == 'Point' and isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        pairs = [coordinates]
    elif kind == 'LineString' and isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
        pairs = coordinates
    else:
        return None
    if not all(isinstance(p, (list, tuple)) and len(p) >= 2 and
               all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p[:2])
               and -180 <= p[0] <= 180 and -90 <= p[1] <= 90 for p in pairs):
        return None
    return sum(p[0] for p in pairs)/len(pairs), sum(p[1] for p in pairs)/len(pairs)


def location_ready(feature):
    props = (feature or {}).get('properties', {})
    return (props.get('location_valid') is True and props.get('heading_valid') is True
            and props.get('local_frame') is False and centroid(feature) is not None)


def identified_feature(feature, object_id, namespace):
    feature = copy.deepcopy(feature)
    feature['type'] = 'Feature'
    feature['id'] = object_id
    props = feature.setdefault('properties', {})
    props.update(id=object_id, object_id=object_id, client_object_id=object_id, namespace=namespace)
    return feature


def meaningful_change(previous, latest, settings):
    """Compare to the last acknowledged Feature, ignoring per-frame jitter."""
    if not location_ready(latest):
        return False
    if not previous:
        return True
    a, b = previous.get('properties', {}), latest.get('properties', {})
    ignored = {'confidence', 'altitude_m', 'height_m', 'location_valid', 'heading_valid',
               'local_frame', 'discovery_state', 'delivery_state', 'timestamp',
               'timestamp_ns', 'last_seen', 'last_seen_ns', 'track_id'}
    if {k:v for k,v in a.items() if k not in ignored} != {k:v for k,v in b.items() if k not in ignored}:
        return True
    for key, threshold in (('confidence', settings.update_confidence_delta),
                           ('altitude_m', settings.update_altitude_m),
                           ('height_m', settings.update_altitude_m)):
        if (key in a) != (key in b):
            return True
        if key in a and abs(float(a[key])-float(b[key])) >= threshold:
            return True
    ga, gb = previous['geometry'], latest['geometry']
    if ga['type'] != gb['type']:
        return True
    pa = [ga['coordinates']] if ga['type']=='Point' else ga['coordinates']
    pb = [gb['coordinates']] if gb['type']=='Point' else gb['coordinates']
    if len(pa) != len(pb):
        return True
    for p, q in zip(pa, pb):
        dx = math.radians(q[0]-p[0])*6371000*math.cos(math.radians((q[1]+p[1])/2))
        dy = math.radians(q[1]-p[1])*6371000
        if math.hypot(dx, dy) >= settings.update_distance_m:
            return True
        if len(p) != len(q) or (len(p)>2 and abs(q[2]-p[2]) >= settings.update_altitude_m):
            return True
    return False
