"""Existing TELESTO upsert policy; no implicit calls during import/construction."""
import json
from urllib.request import Request, urlopen
from urllib.error import URLError
from .spatial import _haversine_m, _feature_centroid

class FeatureDelivery:
    def __init__(self, api_url, radius_m, logger):
        self.api_url = api_url
        self._dedup_radius_m = radius_m
        self.logger = logger
        self._uploaded_local_ids = set()

    def _api_request(self, url: str, method: str, feature=None):
        """One JSON request to the TELESTO Map Manager. Returns the parsed
        response body (dict) or None on failure."""
        data = None
        headers = {}
        if feature is not None:
            data = json.dumps({
                'geometry': feature.get('geometry'),
                'properties': feature.get('properties'),
            }).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        req = Request(url, data=data, method=method, headers=headers)
        with urlopen(req, timeout=5) as resp:
            if resp.status not in (200, 201):
                self.logger.warn(
                    f'API {method} returned {resp.status}'
                )
                return None
            try:
                return json.loads(resp.read().decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {}

    def _send_to_api(self, collection: dict):
        """Upsert to TELESTO: PUT always creates a new POI, so blindly
        re-PUTting every flush would duplicate features server-side.
        Instead: GET remote, PUT if nothing nearby of the same class,
        PATCH if ours is better, else skip. Local-frame features (no GPS
        yet) can't be geo-compared, so they PUT once per track and get
        remembered to avoid repeat uploads.
        """
        features = collection.get('features', [])
        if not features:
            return

        try:
            remote = self._api_request(self.api_url, 'GET') or {}
            remote_features = remote.get('features', [])
        except (URLError, OSError) as e:
            self.logger.warn(
                f'API GET failed — skipping this flush (no safe dedup): {e}',
                throttle_duration_sec=10.0,
            )
            return

        # (class, centroid, confidence, poi_id) snapshot of the remote set;
        # successful PUTs are appended so one flush can't self-duplicate.
        remote_index = []
        for rf in remote_features:
            centroid = _feature_centroid(rf)
            if centroid is None:
                continue
            props = rf.get('properties', {})
            remote_index.append((
                str(props.get('class', '')).lower(),
                centroid,
                float(props.get('confidence', 0.0)),
                rf.get('id'),
            ))

        created = updated = skipped = errors = 0
        for feature in features:
            props = feature.get('properties', {})
            centroid = _feature_centroid(feature)
            try:
                if props.get('local_frame', False) or centroid is None:
                    key = (props.get('class', ''), props.get('id', ''))
                    if key in self._uploaded_local_ids:
                        skipped += 1
                        continue
                    self._api_request(self.api_url, 'PUT', feature)
                    self._uploaded_local_ids.add(key)
                    created += 1
                    continue

                cls = str(props.get('class', '')).lower()
                conf = float(props.get('confidence', 0.0))
                best = None
                best_dist = float('inf')
                for i, (r_cls, r_centroid, r_conf, r_id) in enumerate(remote_index):
                    if r_cls != cls:
                        continue
                    d = _haversine_m(centroid[0], centroid[1],
                                     r_centroid[0], r_centroid[1])
                    if d < self._dedup_radius_m and d < best_dist:
                        best_dist = d
                        best = i

                if best is None:
                    resp = self._api_request(self.api_url, 'PUT', feature)
                    poi_id = (resp or {}).get('id')
                    remote_index.append((cls, centroid, conf, poi_id))
                    created += 1
                elif conf > remote_index[best][2]:
                    poi_id = remote_index[best][3]
                    if poi_id:
                        self._api_request(
                            f'{self.api_url}/{poi_id}', 'PATCH', feature,
                        )
                    remote_index[best] = (cls, centroid, conf, poi_id)
                    updated += 1
                else:
                    skipped += 1
            except (URLError, OSError) as e:
                errors += 1
                self.logger.warn(
                    f'API upsert failed for a feature: {e}',
                    throttle_duration_sec=10.0,
                )
            except Exception as e:
                errors += 1
                self.logger.error(f'API upsert error: {e}')

        self.logger.info(
            f'API upsert: created={created} updated={updated} '
            f'skipped={skipped} errors={errors}',
            throttle_duration_sec=10.0,
        )
