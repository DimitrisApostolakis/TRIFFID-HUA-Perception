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

## UGV GeoJSON Feature Schema

UGV (`/ugv/detections/front/geojson`) output uses this shape.

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
`[lon, lat]` (UGV — RFC 7946).

## UAV Unity JSON (v1)

The UAV node now exports `triffid.uav.observations.v1`, **not GeoJSON**:

- `point_detections.json`: 15 point classes in selected MP4/SRT views (8 by default).
- `polygon_segmentations.json`: 20 polygon classes in a separate top-view image.
- Both contain exact cropped binary masks with row-major RLE, mask-derived
  `bbox_xyxy` (maximum exclusive), `centroid_xy`, a foreground `raycast_point_xy`,
  class identity and intended final `target_geometry`.
- Coordinates refer to original image pixels, top-left origin, x right/y down.
- Video records contain frame/time and available camera telemetry. Top-view
  records carry optional external projection context. No object world positions
  or camera intrinsics are inferred by perception.
- Person (ID 23) merges original IDs 6, 23 and 43. Debris is a **Point** class.
- Observation IDs are local to a document, not persistent cross-view object IDs.
- Unity performs world projection and duplicate merging before generating final GeoJSON.

Full schema, RLE decoding and examples:
[UAV interface documentation](src/triffid_uav_perception/README.md#unity-json-contract).
The old Point/MultiPoint pixel-GeoJSON UAV format and CLI are retired.

## Original UGV Class List (63 classes)

`Geometry` below is the existing UGV rule. The reduced UAV class list and
geometry rules are documented separately in the UAV README.

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

