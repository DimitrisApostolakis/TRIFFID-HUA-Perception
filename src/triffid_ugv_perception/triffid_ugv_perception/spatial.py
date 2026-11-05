"""Existing geographic deduplication and accumulation rules."""

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return the Haversine distance in metres between two GPS points."""
    import math as _math
    r = 6378137.0
    d_lat = _math.radians(lat2 - lat1)
    d_lon = _math.radians(lon2 - lon1)
    a = (_math.sin(d_lat / 2) ** 2
         + _math.cos(_math.radians(lat1)) * _math.cos(_math.radians(lat2))
         * _math.sin(d_lon / 2) ** 2)
    return r * 2.0 * _math.asin(_math.sqrt(a))

def _feature_centroid(feature: dict):
    """Return (lon, lat) centroid of a GeoJSON feature, or None."""
    geom = feature.get('geometry') or {}
    geom_type = geom.get('type')
    coords = geom.get('coordinates')
    if not coords:
        return None
    if geom_type == 'Point':
        return (coords[0], coords[1])
    if geom_type == 'LineString':
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return (sum(lons) / len(lons), sum(lats) / len(lats))
    if geom_type == 'Polygon' and coords:
        ring = coords[0]
        lons = [c[0] for c in ring]
        lats = [c[1] for c in ring]
        return (sum(lons) / len(lons), sum(lats) / len(lats))
    return None

def _deduplicate_features(features: list, radius_m: float) -> list:
    """Keep the highest-confidence feature per class within radius_m of
    each other; local_frame features (no GPS) pass through unchanged."""
    if len(features) <= 1:
        return features

    # Sort descending by confidence so greedy keep is always the best
    def _conf(f):
        return f.get('properties', {}).get('confidence', 0.0)

    sorted_feats = sorted(features, key=_conf, reverse=True)
    kept = []
    for candidate in sorted_feats:
        props = candidate.get('properties', {})
        if props.get('local_frame', False):
            kept.append(candidate)
            continue
        centroid = _feature_centroid(candidate)
        if centroid is None:
            kept.append(candidate)
            continue
        cls = props.get('class', '')
        suppressed = False
        for existing in kept:
            if existing.get('properties', {}).get('class', '') != cls:
                continue
            existing_centroid = _feature_centroid(existing)
            if existing_centroid is None:
                continue
            dist = _haversine_m(
                centroid[0], centroid[1],
                existing_centroid[0], existing_centroid[1],
            )
            if dist < radius_m:
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)
    return kept

def _merge_features(accumulated: list, new: list, radius_m: float) -> list:
    """Merge new features into the accumulated set, deduplicated. GPS
    features collapse to the best within radius_m; local-frame features
    (no GPS) collapse to the newest per (class, track id)."""
    latest_local = {}
    gps_feats = []
    for feat in accumulated + new:
        props = feat.get('properties', {})
        if props.get('local_frame', False):
            key = (props.get('class', ''), props.get('id', ''))
            latest_local[key] = feat        # later (newer) entries win
        else:
            gps_feats.append(feat)
    return _deduplicate_features(gps_feats, radius_m) + list(latest_local.values())
