# TRIFFID Perception — Interface Spec

## UGV Topics

| Direction | Topic | Type | Notes |
|---|---|---|---|
| in | `rgb_image_topic` param (default `/camera_front/raw_image`) | `sensor_msgs/Image` | bgr8 or yuv422 |
| in | `depth_image_topic` param (default `/camera_front/realsense_front/depth/image_rect_raw`) | `sensor_msgs/Image` | 16UC1, mm, pixel-aligned to RGB |
| in | `camera_info_topic` param (default `/camera_front/camera_info`) | `sensor_msgs/CameraInfo` | shared RGB/depth intrinsics |
| in | `/fix` | `sensor_msgs/NavSatFix` | optional; `local_frame:true` in output until received |
| in | `/dog_odom` | `nav_msgs/Odometry` | optional; heading for GeoJSON rotation |
| in | `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | camera_optical_frame → `b2/base_link` required |
| out | `/ugv/detections/front/detections_3d` | `vision_msgs/Detection3DArray` | frame_id `b2/base_link` |
| out | `/ugv/detections/front/segmentation` | `sensor_msgs/Image` | mono8, pixel = class id + 1 (0 = bg) |
| out | `/ugv/detections/front/debug_image` | `sensor_msgs/Image` | bgr8, lazy (only if subscribed) |
| out | `/ugv/detections/front/geojson` | `std_msgs/String` | GeoJSON FeatureCollection, per frame |

## Detection3DArray Fields

`bbox.center.position` = 3D centroid, `bbox.size` = extent, both in `b2/base_link` (X=fwd, Y=left, Z=up).
`results[0].hypothesis.class_id` = class name (string), `.score` = confidence. `id` = persistent track id (string).

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

`Geometry` is the point/line/polygon rule from the table below: it decides
sample-point count for the UAV pipeline and geometry type for the UGV bridge.

| ID | Class | Category | Geometry |
|---|---|---|---|
| 0 | water | obstacle | Polygon |
| 1 | fence | obstacle | Line |
| 2 | green tree | nature | Polygon |
| 3 | helmet | equipment | Point |
| 4 | flame | hazard | Polygon |
| 5 | smoke | hazard | Polygon |
| 6 | first responder | person | Point |
| 7 | destroyed vehicle | vehicle | Point |
| 8 | fire hose | equipment | Point |
| 9 | scba | equipment | Point |
| 10 | boot | equipment | Point |
| 11 | green plant | nature | Polygon |
| 12 | mask | equipment | Point |
| 13 | window | infrastructure | Point |
| 14 | building | infrastructure | Polygon |
| 15 | destroyed building | infrastructure | Polygon |
| 16 | debris | obstacle | Polygon |
| 17 | ladder | equipment | Polygon |
| 18 | dirt road | infrastructure | Polygon |
| 19 | dry tree | nature | Polygon |
| 20 | wall | infrastructure | Line |
| 21 | civilian vehicle | vehicle | Point |
| 22 | road | infrastructure | Polygon |
| 23 | citizen | person | Point |
| 24 | green grass | nature | Polygon |
| 25 | pole | infrastructure | Point |
| 26 | boat | vehicle | Polygon |
| 27 | pavement | infrastructure | Polygon |
| 28 | dry grass | nature | Polygon |
| 29 | animal | nature | Point |
| 30 | excavator | equipment | Polygon |
| 31 | door | infrastructure | Point |
| 32 | mud | obstacle | Polygon |
| 33 | barrier | obstacle | Polygon |
| 34 | hole in the ground | obstacle | Point |
| 35 | bag | equipment | Point |
| 36 | burnt tree | hazard | Polygon |
| 37 | ambulance | vehicle | Point |
| 38 | fire truck | vehicle | Point |
| 39 | cone | obstacle | Point |
| 40 | bicycle | vehicle | Polygon |
| 41 | tower | infrastructure | Polygon |
| 42 | silo | infrastructure | Polygon |
| 43 | military personnel | person | Point |
| 44 | burnt grass | hazard | Polygon |
| 45 | ax | equipment | Point |
| 46 | glove | equipment | Point |
| 47 | crane | equipment | Polygon |
| 48 | stairs | infrastructure | Point |
| 49 | dry plant | nature | Polygon |
| 50 | furniture | equipment | Polygon |
| 51 | tank | equipment | Polygon |
| 52 | protective glasses | equipment | Point |
| 53 | barrel | equipment | Polygon |
| 54 | shovel | equipment | Point |
| 55 | fire hydrant | equipment | Point |
| 56 | police vehicle | vehicle | Point |
| 57 | burnt plant | hazard | Polygon |
| 58 | army vehicle | vehicle | Point |
| 59 | chainsaw | equipment | Point |
| 60 | aerial vehicle | vehicle | Point |
| 61 | lifesaver | equipment | Point |
| 62 | extinguisher | equipment | Point |

## TELESTO Backend

`https://crispres.com/wp-json/map-manager/v1/features` — `GET` (list),
`PUT` (create, body `{"geometry", "properties"}`, returns server `id`),
`PATCH /features/{id}` (update), `DELETE /features/{id}`.
