"""Export confirmed tracked objects without inventing geographic coordinates."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    registry = json.loads((args.run / 'object_registry.json').read_text())
    observations = defaultdict(list)
    for line in (args.run / 'tracks.jsonl').read_text().splitlines():
        row = json.loads(line)
        for track in row['tracks']:
            if not track.get('object_id'):
                continue
            observations[track['object_id']].append({
                'frame': row['frame'],
                'timestamp_ns': row['source']['bag_timestamp_ns'],
                'image': row['source']['file'],
                'bbox_xyxy_px': track['bbox'],
                'confidence': track['confidence'],
                'class_name': track['class_name'],
                'track_id': track['track_id'],
                'recovered': track.get('recovered', False),
                'role_ambiguous': track.get('role_ambiguous', False),
            })
    features = []
    for object_id, obj in registry['objects'].items():
        seen = observations[object_id]
        assert seen, f'No observations for {object_id}'
        event = obj['created_event']
        features.append({
            'type': 'Feature',
            'id': object_id,
            'geometry': None,
            'properties': {
                'object_id': object_id,
                'class_name': obj['class_name'],
                'class_id': obj['class_id'],
                'created_timestamp_ns': event['timestamp_ns'],
                'created_at_utc': datetime.fromtimestamp(
                    event['timestamp_ns'] / 1e9, timezone.utc).isoformat(),
                'last_seen_timestamp_ns': obj['last_seen_ns'],
                'confidence_at_creation': event['confidence'],
                'max_observed_confidence': max(x['confidence'] for x in seen),
                'observation_count': len(seen),
                'location_status': 'not_georeferenced',
                'location_reason': 'Offline RGB tracking did not project detections using depth, camera calibration and georeferenced robot pose.',
                'observations': seen,
            },
        })
    collection = {
        'type': 'FeatureCollection',
        'name': 'PET-DINO MASA confirmed tracked objects',
        'description': 'One feature per confirmed UUID, not independently verified physical-object identities. Null geometry means object coordinates have not been calculated; pixel boxes are properties, not geographic geometry. Observations start at UUID confirmation.',
        'context': registry['context'],
        'features': features,
    }
    path = args.run / 'detections.geojson'
    path.write_text(json.dumps(collection, indent=2, allow_nan=False) + '\n')
    print(f'{path}: {len(features)} confirmed UUIDs, {sum(len(v) for v in observations.values())} observations; geometry=null')


if __name__ == '__main__':
    main()
