"""UGV Point/LineString geometry and properties, independent of ROS transport."""
from .ontology import geometry_type_for_class, class_color, class_category, class_symbol


_MIN_LINE_EXTENT = 0.3


class GeoJSONFormatter:
    def __init__(self, position):
        self.position = position

    def detections_to_geojson(self, detections, source='ugv', detection_type='seg'):
        """Keep 2D coordinates; store altitude and object height as properties."""
        features = []
        for detection in detections:
            lon, lat, alt = detection['coordinates']
            sx, sy, sz = detection.get('size', (0.0, 0.0, 0.0))
            x, y, z = detection.get('position', (0.0, 0.0, 0.0))
            name = detection.get('class_name', 'unknown')
            kind = geometry_type_for_class(name)
            if kind == 'LineString':
                # Preserve the prior longest-extent line construction and ordering.
                half_long = max(sx, sy, _MIN_LINE_EXTENT) / 2.0
                if sx >= sy:
                    offsets = [(x + half_long, y, z), (x - half_long, y, z)]
                else:
                    offsets = [(x, y + half_long, z), (x, y - half_long, z)]
                coordinates = [list(self.position.body_to_gps(*point))[:2] for point in offsets]
            else:
                coordinates = [lon, lat]

            properties = {
                'class': name,
                'id': detection.get('track_id', ''),
                'confidence': detection.get('confidence', 0.0),
                'category': class_category(name),
                'detection_type': detection_type,
                'source': source,
                'local_frame': not self.position.gps_valid,
                'altitude_m': round(alt, 2),
                'height_m': round(float(sz), 2),
                'marker-color': class_color(name),
                'marker-size': 'medium',
                'marker-symbol': class_symbol(name),
            }
            if kind == 'LineString':
                properties.update({
                    'stroke': properties['marker-color'],
                    'stroke-width': 2,
                    'stroke-opacity': 1.0,
                })
            features.append({
                'type': 'Feature',
                'id': detection.get('track_id', ''),
                'geometry': {'type': kind, 'coordinates': coordinates},
                'properties': properties,
            })
        return {'type': 'FeatureCollection', 'features': features}
