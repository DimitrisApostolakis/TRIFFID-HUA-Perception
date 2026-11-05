# TRIFFID Perception — Interface Spec

## UGV Topics

| Direction | Topic | Type | Notes |
|---|---|---|---|
| in | `rgb_image_topic` (default `/b2/camera_front_435i/realsense_front_435i/color/image_raw_broadcasted`) | `sensor_msgs/Image` | bgr8 or YUYV |
| in | `depth_image_topic` (default `/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw_broadcasted`) | `sensor_msgs/Image` | 16UC1, mm, aligned to RGB |
| in | `thermal_image_topic` (default `/b2/camera_flir/image_raw_broadcasted`) | `sensor_msgs/Image` | debug snapshots only |
| in | `/b2/ugv_status` | `ugv_status_interfaces/UnitreeStatus` | primary position when installed, valid and fresh |
| in | `/fix` | `sensor_msgs/NavSatFix` | filtered position fallback |
| in | `/b2/nicla/magnetometer/heading_broadcasted` | `std_msgs/Float32` | compass heading in degrees |
| out | `/ugv/detections/front/detections_3d` | `vision_msgs/Detection3DArray` | frame `b2/base_link` |
| out | `/ugv/detections/front/segmentation` | `sensor_msgs/Image` | mono8, class ID + 1; 0 is background |
| out | `/ugv/detections/front/debug_image` | `sensor_msgs/Image` | bgr8 overlays |
| out | `/ugv/detections/front/geojson` | `std_msgs/String` | per-frame FeatureCollection |

Calibration and geometry retain the topic-update branch's fixed RealSense
intrinsics and optical-to-body axis mapping. Position priority is fresh ROS status,
HTTP position API, filtered `/fix`, then parameter seed. See the
[UGV module documentation](src/triffid_ugv_perception/README.md).

## Detection3DArray Fields

`bbox.center.position` = 3D centroid, `bbox.size` = extent, both in `b2/base_link` (X=fwd, Y=left, Z=up).
`results[0].hypothesis.class_id` = class name (string), `.score` = backend confidence (SANSA: mean foreground probability per connected component, not calibrated YOLO confidence). `id` = persistent track id (string).

## GeoJSON Feature Schema

Both UGV (`/ugv/detections/front/geojson`) and UAV output use this shape.

```json
{
  "type": "Feature",
  "id": "<track_id>",
  "geometry": { "type": "Point|LineString|Polygon", "coordinates": [...] },
  "properties": {
    "class": "building",
    "id": "<track_id>",
    "confidence": 0.85,
    "category": "infrastructure",
    "detection_type": "seg",
    "source": "ugv",
    "local_frame": false,
    "altitude_m": 409.1,
    "height_m": 3.2,
    "marker-color": "#708090",
    "marker-size": "medium",
    "marker-symbol": "building",
    "stroke": "#708090", "stroke-width": 2, "stroke-opacity": 1.0,
    "fill": "#708090", "fill-opacity": 0.25
  }
}
```

`stroke*` on Polygon/LineString, `fill*` on Polygon only. `coordinates` are
`[lon, lat]` (UGV — RFC 7946) or `[x, y]` pixels (UAV — see below).

## UAV Feature Differences

- `geometry.type` is `Point` (1 sample) or `MultiPoint` (multiple independent
  raycast samples — never `LineString`/`Polygon`, no closure/ordering implied).
- `coordinates` are pixels in the original frame, origin top-left.
- `properties` additionally carry `class_id`, `frame_index`, `timestamp_s`,
  `coordinate_space: "pixel"`, `classes_txt_geometry` (the table below,
  informational — controls sample count, not `geometry.type`), and, when an
  SRT sidecar is present: `drone_latitude`, `drone_longitude`,
  `drone_altitude_m`, `drone_rel_altitude_m`, `gimbal_yaw`, `gimbal_pitch`,
  `gimbal_roll` (camera pose for that frame, not the detection's position).
- No `stroke`/`fill`/`local_frame`/`height_m`/`altitude_m`.

## Class List (63 classes)

`Geometry` below describes UGV output: fence and wall are LineString; all
other classes are Point. UAV geometry is independent of this UGV policy.

| ID | Class | Category | Geometry |
|---|---|---|---|
| 0 | water | obstacle | Point |
| 1 | fence | obstacle | LineString |
| 2 | green tree | nature | Point |
| 3 | helmet | equipment | Point |
| 4 | flame | hazard | Point |
| 5 | smoke | hazard | Point |
| 6 | first responder | person | Point |
| 7 | destroyed vehicle | vehicle | Point |
| 8 | fire hose | equipment | Point |
| 9 | scba | equipment | Point |
| 10 | boot | equipment | Point |
| 11 | green plant | nature | Point |
| 12 | mask | equipment | Point |
| 13 | window | infrastructure | Point |
| 14 | building | infrastructure | Point |
| 15 | destroyed building | infrastructure | Point |
| 16 | debris | obstacle | Point |
| 17 | ladder | equipment | Point |
| 18 | dirt road | infrastructure | Point |
| 19 | dry tree | nature | Point |
| 20 | wall | infrastructure | LineString |
| 21 | civilian vehicle | vehicle | Point |
| 22 | road | infrastructure | Point |
| 23 | citizen | person | Point |
| 24 | green grass | nature | Point |
| 25 | pole | infrastructure | Point |
| 26 | boat | vehicle | Point |
| 27 | pavement | infrastructure | Point |
| 28 | dry grass | nature | Point |
| 29 | animal | nature | Point |
| 30 | excavator | equipment | Point |
| 31 | door | infrastructure | Point |
| 32 | mud | obstacle | Point |
| 33 | barrier | obstacle | Point |
| 34 | hole in the ground | obstacle | Point |
| 35 | bag | equipment | Point |
| 36 | burnt tree | hazard | Point |
| 37 | ambulance | vehicle | Point |
| 38 | fire truck | vehicle | Point |
| 39 | cone | obstacle | Point |
| 40 | bicycle | vehicle | Point |
| 41 | tower | infrastructure | Point |
| 42 | silo | infrastructure | Point |
| 43 | military personnel | person | Point |
| 44 | burnt grass | hazard | Point |
| 45 | ax | equipment | Point |
| 46 | glove | equipment | Point |
| 47 | crane | equipment | Point |
| 48 | stairs | infrastructure | Point |
| 49 | dry plant | nature | Point |
| 50 | furniture | equipment | Point |
| 51 | tank | equipment | Point |
| 52 | protective glasses | equipment | Point |
| 53 | barrel | equipment | Point |
| 54 | shovel | equipment | Point |
| 55 | fire hydrant | equipment | Point |
| 56 | police vehicle | vehicle | Point |
| 57 | burnt plant | hazard | Point |
| 58 | army vehicle | vehicle | Point |
| 59 | chainsaw | equipment | Point |
| 60 | aerial vehicle | vehicle | Point |
| 61 | lifesaver | equipment | Point |
| 62 | extinguisher | equipment | Point |

## TELESTO Backend

`https://crispres.com/wp-json/map-manager/v1/features` — `GET` (list),
`PUT` (create, body `{"geometry", "properties"}`, returns server `id`),
`PATCH /features/{id}` (update), `DELETE /features/{id}`.
