# TRIFFID UGV/UAV Perception Module (T4.1)

**Harokopio University of Athens, TRIFFID Research Project**

**Contact: amichailidou@hua.gr**

## General Information

Two independent modules, each in its own Docker container, with separate runtime dependencies:

| Module | Input | Output |
|---|---|---|
| **UGV** (`triffid_ugv_perception`) | Live RGB-D camera topics + GPS/odometry | `Detection3DArray` + GeoJSON on ROS topics, periodic GeoJSON snapshot to disk and/or upsert to the TELESTO backend |
| **UAV** (`triffid_uav_perception`) | MP4 + DJI SRT for selected views; separate top-view image; SANSA visual supports | Versioned Unity pixel JSON: exact masks, mask-derived boxes and centroids |

Full topic/message/GeoJSON spec: [INTERFACE.md](INTERFACE.md).

## 1. UGV Module

Real-time 3D detection: YOLOv11-seg on RGB, depth sampled at detection
pixels (pixel-aligned RGB-D), back-projection to 3D, TF to `b2/base_link`,
ByteTrack IDs, GeoJSON geo-referencing via GPS + heading (will be changed to be via tf2 tree once available).

### Build and start the container

```bash
docker compose up -d
docker exec -it triffid_perception bash
# inside the container:
cd /ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install
source /ws/install/setup.bash
```

Requires the NVIDIA container toolkit (GPU). The container uses
`network_mode: host`, so it sees your ROS 2 traffic directly
(`ROS_DOMAIN_ID=42`, CycloneDDS , see `cyclonedds.xml`).

### Run the perception node

Inside the container (one terminal per node, or append `&`):

```bash
ros2 run triffid_ugv_perception ugv_node --ros-args \
    -p model_path:=/ws/best.pt \
    -p rgb_image_topic:=/camera_front_435i/realsense_front_435i/color/image_raw \
    -p depth_image_topic:=/camera_front_435i/realsense_front_435i/depth/image_rect_raw \
    -p camera_info_topic:=/camera_front_435i/realsense_front_435i/color/camera_info
```

Set the three topic parameters to wherever the camera publishes. The
camera must be pixel-aligned RGB-D: `bgr8` (or YUYV) colour, `16UC1` depth
in millimetres, one shared `CameraInfo`. A TF from the camera optical
frame (read from `CameraInfo.header.frame_id`) to `b2/base_link` must be
available.

Other `ugv_node` parameters (defaults): `confidence_threshold` (0.35),
`yolo_imgsz` (1280), `target_frame` (`b2/base_link`), `nms_merge_dist_m`
(0.5), and tracker tuning , `tracker_iou_threshold` (0.30),
`tracker_iou_threshold_low` (0.15), `tracker_conf_high` (0.40),
`tracker_max_age` (30), `tracker_n_init` (3), `tracker_pos_gate` (2.0).

### Run the GeoJSON bridge

```bash
ros2 run triffid_ugv_perception geojson_bridge --ros-args \
    -p save_to_disk:=true \
    -p publish_to_api:=false \
    -p publish_period_s:=5.0 \
    -p output_dir:=/ws/samples
```

| Parameter | Default | Meaning |
|---|---|---|
| `save_to_disk` | `true` | Every period, write the accumulated deduplicated feature set atomically to `output_dir/ugv_detections.geojson` (host: `./samples/`) |
| `publish_to_api` | `false` | Every period, **upsert** the accumulated set to `api_url` (GET remote → PUT new / PATCH improved / skip unchanged |
| `publish_period_s` | `5.0` | Seconds between periodic flushes |
| `output_dir` | `/ws/samples` | Snapshot directory inside the container |
| `api_url` | `https://crispres.com/wp-json/map-manager/v1/features` | TELESTO Map Manager API endpoint |
| `dedup_radius_m` | `3.0` | Same-class features within this distance merge (highest confidence kept) |
| `gps_origin_lat/lon/alt` | `0.0` | Optional GPS seed used until the first `/fix` arrives |

Both flags are independent, enable either or both. The bridge also
publishes the per-frame GeoJSON on the ROS topic
`/ugv/detections/front/geojson` (`std_msgs/String`) regardless.

The bridge subscribes to `/fix` (`NavSatFix`) and `/dog_odom`
(`Odometry`) for geo-referencing. Until a GPS fix arrives, features are
emitted in body-frame coordinates with `"local_frame": true`.

### UGV topics

| Direction | Topic | Type |
|---|---|---|
| in | *( RGB / depth / CameraInfo topics)* | `sensor_msgs/Image`, `CameraInfo` |
| in | `/fix`, `/dog_odom`, `/tf`, `/tf_static` | `NavSatFix`, `Odometry`, TF |
| out | `/ugv/detections/front/detections_3d` | `vision_msgs/Detection3DArray` |
| out | `/ugv/detections/front/segmentation` | `sensor_msgs/Image` (mono8 label map) |
| out | `/ugv/detections/front/debug_image` | `sensor_msgs/Image` (lazy) |
| out | `/ugv/detections/front/geojson` | `std_msgs/String` (GeoJSON) |

**Alternatively, both nodes start together via the launch file:**

```bash
ros2 launch triffid_ugv_perception ugv_perception.launch.py \
    rgb_image_topic:=... depth_image_topic:=... camera_info_topic:=... \
    save_to_disk:=true publish_to_api:=false
```

---

## 2. UAV Module

The UAV node selects eight angular views from MP4 + DJI SRT telemetry, then
runs **SANSA** with a shared five-example-per-class support bank for the **15 point
classes**. A separate top-view image is segmented for the **20 polygon classes**.
Debris is a point class; Citizen/First responder/Military personnel become Person.

Each observation exports an exact binary mask, a mask-derived bounding box and
centroid, frame/time/camera telemetry, and intended final geometry. Outputs are
**original-image pixel JSON**, not geographic GeoJSON. Unity performs projection,
raycasting and duplicate merging afterwards. No continuous-video tracker is used.

Start by verifying selection without model weights:

```bash
docker compose -f docker-compose.uav.yml up -d --build
docker exec triffid_uav_perception python -m triffid_uav_perception.uav_node \
  select /app/videos/flight.mp4 --output /app/samples/selection
```

See [UAV setup, SANSA supports, CLI and Unity JSON contract](src/triffid_uav_perception/README.md).
Create the bank with `python -m triffid_uav_perception.uav_node bank init`.
It defines 35 classes x 5 example slots (175 pairs). Add real labelled examples
with `bank add`, then check them with `bank validate`. All backends receive the
same ordered supports; `--shots 5` is the default and `--shots 1` uses the first.
Use `detect-frames selected_frames.json` to infer already-exported frames without
reopening the video. A 1-shot test can run while slots 2–5 are still unfilled.
The former UAV YOLO/ByteTrack/pixel-GeoJSON and API-polling CLI has been replaced;
UGV inference, tracking and delivery are unchanged.

## Repository layout

```
├── docker-compose.yml / Dockerfile           # UGV container (ROS 2 Humble + CUDA)
├── docker-compose.uav.yml / Dockerfile.uav   # UAV container (Python + CUDA torch)
├── best.pt                                   # UGV YOLOv11-seg model (Git LFS)
├── INTERFACE.md                              # topic/message/GeoJSON/class-list spec
└── src/
    ├── triffid_ugv_perception/               # ROS 2 package (ugv_node, geojson_bridge, tracker)
    ├── triffid_uav_perception/               # modular UAV pipeline + SANSA backend
    └── triffid_telesto/                      # TELESTO REST client for UGV delivery
```

