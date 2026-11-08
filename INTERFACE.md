# TRIFFID Perception — Interface Spec

## UGV Topics

| Direction | Topic | Type | Notes |
|---|---|---|---|
| in | `rgb_image_topic` (default `/b2/camera_front_435i/realsense_front_435i/color/image_raw_broadcasted`) | `sensor_msgs/Image` | bgr8 or YUYV |
| in | `depth_image_topic` (default `/b2/camera_front_435i/realsense_front_435i/aligned_depth_to_color/image_raw_broadcasted`) | `sensor_msgs/Image` | 16UC1, mm, aligned to RGB |
| in | `thermal_image_topic` (default `/b2/camera_flir/image_raw_broadcasted`) | `sensor_msgs/Image` | debug snapshots only |
| in | `/b2/ugv_status` | `ugv_status_interfaces/UnitreeStatus` | primary position when installed, valid and fresh |
| in | `/fix` | `sensor_msgs/NavSatFix` | filtered position fallback |
| in | `/b2/nicla/magnetometer/heading` | `std_msgs/Float32` | compass heading in degrees |
| out | `/ugv/detections/front/detections_3d` | `vision_msgs/Detection3DArray` | frame `b2/base_link` |
| out | `/ugv/detections/front/segmentation` | `sensor_msgs/Image` | mono8, class ID + 1; 0 is background |
| out | `/ugv/detections/front/debug_image` | `sensor_msgs/Image` | bgr8 overlays |
| out | `/ugv/detections/front/geojson` | `std_msgs/String` | per-frame FeatureCollection; persistent object UUID after confirmation |
| out | `/ugv/detections/front/discovery_status` | `std_msgs/String` | bounded registry/outbox counts and capacity pressure |
| out | `/ugv/detections/front/tracked_observations` | `std_msgs/String` | live MASA profile: bounded image boxes, 256-D appearance, body-frame positions and session-scoped track IDs |
| out | `/ugv/detections/front/pipeline_status` | `std_msgs/String` | live worker timing, received/replaced/stale/failed counters and bounded pending-frame count |

Localization requires matching CameraInfo, synchronized aligned depth and recorded
optical-to-body TF. Position priority is fresh ROS status, optional HTTP position
API (disabled by default), filtered `/fix`, then parameter seed. Creation eligibility
requires fresh GPS and heading; per-frame geometry is null while they are missing. See the
[UGV module documentation](src/triffid_ugv_perception/README.md).

## Detection3DArray Fields

`bbox.center.position` = 3D centroid, `bbox.size` = extent, both in `b2/base_link` (X=fwd, Y=left, Z=up).
`results[0].hypothesis.class_id` = class name (string), `.score` = backend confidence (SANSA: mean foreground probability per connected component, not calibrated YOLO confidence). `id` = tracker-local ID (string); the discovery registry assigns a separate persistent object UUID.

The YOLOE/MASA live profile uses native YOLOE object confidence. Its tracker IDs are
opaque `<session_uuid>:<masa_id>` strings. The richer observation topic uses schema
`triffid.ugv.tracked-observations.v1`, original-image `bbox` in xyxy pixels, body-frame
`position`/`extent` in metres, capture-header `timestamp_ns`, and an explicit
`MASA-R50:<checkpoint_sha256>` embedding space. Masks are excluded from this topic.
GeoJSONBridge consumes this topic exclusively in the live profile so appearance
re-identification evidence is preserved and the parallel Detection3DArray stream
does not create duplicate registry updates. See [live deployment](docs/UGV_LIVE_YOLOE_MASA.md).

## GeoJSON Feature Schema

UGV `/ugv/detections/front/geojson` uses this Feature shape. UAV Python
output has a separate pixel-observation schema described below.

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

`stroke*` applies to Polygon/LineString and `fill*` to Polygon. Geographic
coordinates use `[longitude, latitude]`. UGV geometry is null while its valid
location/heading gate fails.

## UAV Observation Schema

The Python exporter writes `triffid.uav.observations.v1`, not geographic GeoJSON.
The document identifies `coordinate_space: "original_image_pixels"`, a top-left
pixel origin, a class catalogue, model metadata and `sources` containing the
image dimensions, telemetry/projection context and observations.

Each observation contains `observation_id`, `class_id`, `class_name`,
`target_geometry` (`Point` or `Polygon`), `bbox_xyxy`, `centroid_xy`,
`raycast_point_xy`, `area_px`, and an exact cropped binary mask. The mask uses
`binary_rle_row_major` with `size_hw`, alternating background/foreground
`counts`, and `offset_xy`; it is not COCO column-major RLE. `score` is null
when unavailable; an exposed mean foreground probability is uncalibrated.

`point_detections.json` supplies point observations; `polygon_segmentations.json`
supplies surface masks. Optional offline MASA output adds tracking metadata and
can export one representative observation per confirmed identity. Observation
IDs alone are not persistent world-object IDs. A separate Unity projector uses
the matching capture camera and mesh collider to produce geographic GeoJSON.
See [UAV architecture](docs/UAV_ARCHITECTURE.md) and the
[UAV CLI/output guide](src/triffid_uav_perception/README.md).

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

Automatic UGV delivery defaults to disabled. The live profile supports full-Feature
PUT creation and PATCH updates of the mapped POI. Re-ID retains the persistent
client UUID; meaningful changes coalesce with a two-second per-object PATCH limit.
The legacy profile remains creation-only. Network creation retries require the
server to enforce unique `(namespace, client_object_id)` and replay the original
remote ID; this contract has not been verified on the backend. See
[UGV API delivery](docs/UGV_API_DELIVERY.md) for request formats, thresholds,
durability and deployment gating.
