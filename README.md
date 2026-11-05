# TRIFFID UGV/UAV Perception Module (T4.1)

**Harokopio University of Athens, TRIFFID Research Project**

**Contact: amichailidou@hua.gr**

## General Information

Two independent modules, each in its own Docker container, both running on
live data:

| Module | Input | Output |
|---|---|---|
| **UGV** (`triffid_ugv_perception`) | Live RGB-D camera topics + GPS/odometry | `Detection3DArray` + GeoJSON on ROS topics, periodic GeoJSON snapshot to disk and/or upsert to the TELESTO backend |
| **UAV** (`triffid_uav_perception`) | Drone video (MP4 + DJI SRT telemetry sidecar), polled live from API | One pixel-space GeoJSON per video, saved to disk, optional upload to the TELESTO backend |

Full topic/message/GeoJSON spec: [INTERFACE.md](INTERFACE.md).

## 1. UGV Module

The UGV code is modular and uses **SANSA** by default with its own
`ugv_supports` bank. The original depth projection, fixed calibration,
ByteTrack-style tracking, spatial filtering, thermal debugging and
position/GeoJSON delivery behavior are retained from the UGV topic-update branch.
UGV geometry is **LineString for fence/wall, Point for all other classes**.

See [UGV setup, independent support bank, launch commands and module responsibilities](src/triffid_ugv_perception/README.md).
All 63 UGV classes retain their original IDs and names. The UAV bank is not reused.

---

## 2. UAV Module

Drone-video detection producing pixel-space GeoJSON, with
persistent ByteTrack IDs, each detection reduced to a few independent
pixel sample points (`Point`/`MultiPoint`), DJI SRT
telemetry joined per frame so every feature also carries the camera pose
(`drone_latitude/longitude/altitude_m`, `gimbal_yaw/pitch/roll`). A
downstream consumer raycasts the pixel points against a 3DGS
reconstruction to obtain real-world coordinates.

### Build and start the container

```bash
docker compose -f docker-compose.uav.yml up -d
```

### Live mode - poll the API

Polls for newly uploaded videos (`.mp4`/`.mov`) and their `.srt`
telemetry files, processes each video, and writes one
GeoJSON per video:

```bash
docker exec -e FUTURISED_MEDIA_API_KEY='your-key' triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node --poll-api \
       --api-camera Wide --api-poll-interval 30 \
       --api-download-dir /app/videos --output /app/samples"
```

Outputs land in `./uav_samples/<video-stem>_detections.geojson` on the
host. Add `--post-telesto` to also upload each result to the TELESTO
backend (off by default).

### Local-file test mode

To verify the module place the mp4 file under
`./uav_videos/`, with its `.srt`, auto-detected by name):

```bash
docker exec triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node /app/videos/flight.mp4 \
       --output /app/samples"
```

Key flags (defaults): `--model /app/best.pt`* , `--confidence 0.5`,
`--stride 5` (every Nth frame), `--sample-seconds 1.0` (each track id
emitted at most once per second), `--classes id,label,...` (allowlist),
`--srt path` (override sidecar auto-detect), `--debug-video` (also save
`<stem>_debug.mp4` with boxes/points/labels drawn on top, for
`--debug-video-seconds 20` starting at `--debug-video-start 0` — useful
for eyeballing whether detections look right).

### UAV output schema

One RFC-7946 FeatureCollection per video (`metadata` member: video/model/
fps/stride/`srt_file`/counts). `geometry` is `Point` or `MultiPoint` in
**pixel** coordinates (origin top-left) — independent raycast samples, not
an outline. `drone_*`/`gimbal_*` properties (camera pose for that frame)
appear only when an SRT sidecar was found. Full schema: [INTERFACE.md](INTERFACE.md).

## Repository layout

```
├── docker-compose.yml / Dockerfile           # UGV container (ROS 2 Humble + CUDA)
├── docker-compose.uav.yml / Dockerfile.uav   # UAV container (Python + CUDA torch)
├── best.pt                                   # shared YOLOv11-seg model (Git LFS)
├── INTERFACE.md                              # topic/message/GeoJSON/class-list spec
└── src/
    ├── triffid_ugv_perception/               # ROS 2 package (ugv_node, geojson_bridge, tracker)
    ├── triffid_uav_perception/               # UAV pipeline (uav_node, srt_metadata, api_client)
    └── triffid_telesto/                      # TELESTO REST client (used by --post-telesto)
```
