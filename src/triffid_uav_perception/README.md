# TRIFFID UAV Perception

Standalone (no ROS2) drone-video perception pipeline: **video in, pixel-space
GeoJSON out**. Runs the shared TRIFFID YOLO-seg model with persistent track
IDs over a whole video, reduces every detection to a few boundary pixel
points, and saves one GeoJSON FeatureCollection per video. A downstream
consumer raycasts those pixel points against a **3D Gaussian Splatting
(3DGS)** reconstruction to recover real-world coordinates.

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Output GeoJSON](#output-geojson)
- [CLI Reference](#cli-reference)
- [SRT Telemetry Sidecar](#srt-telemetry-sidecar)
- [TELESTO Upload (optional)](#telesto-upload-optional)
- [FUTURISED Video-Poll Mode](#futurised-video-poll-mode---poll-api-poc)
- [Docker Setup](#docker-setup)

## Architecture

```
 video file (local, or polled from the FUTURISED API with --poll-api)
      │
      ▼
 ┌───────────────────────────────────────────────┐
 │ UAVPipeline.process_video()                   │
 │   every Nth frame:                            │
 │     YOLO-seg + ByteTrack (persistent IDs)     │
 │     per-track dedup (≤ 1 emission / second)   │
 │     boundary points per classes.txt geometry  │
 │     → GeoJSON Feature (pixel coordinates)     │
 └───────────────────┬───────────────────────────┘
                     │ FeatureCollection
        ┌────────────┴──────────────┐
        ▼                           ▼
 <stem>_detections.geojson    --post-telesto (optional, off by default;
 (always written to disk)      PUT per feature to the TELESTO backend —
                               meaningful only once coordinates are
                               geographic, i.e. after 3DGS raycasting)
```

The model is the same 63-class disaster-response YOLOv11-seg used by the UGV
pipeline (`best.pt` at the repo root). Class → geometry-type mapping comes
from `classes.txt` (repo root), the single source of truth.

## Quick Start

```bash
# Build + start the container (CUDA torch; falls back to CPU without a GPU)
docker compose -f docker-compose.uav.yml up -d

# Process a local video (place it under ./uav_videos/, its .srt next to it)
docker exec triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node /app/videos/clip.mp4 --output /app/samples"
# → ./uav_samples/clip_detections.geojson

# Live mode: poll FUTURISED for new video uploads (MP4 + SRT)
docker exec -e FUTURISED_MEDIA_API_KEY='your-key' triffid_uav_perception bash -c \
  "PYTHONPATH=/app/src/triffid_uav_perception:/app/src \
   python -m triffid_uav_perception.uav_node --poll-api \
       --api-download-dir /app/videos --output /app/samples"

# Stop
docker rm -f triffid_uav_perception
```

## How It Works

- Runs the YOLO-seg model with ultralytics `model.track(persist=True)`, so
  **track IDs persist across frames** (ByteTrack by default).
- Processes every `--stride`-th frame (default `5`); tracking always runs on
  every processed frame so IDs stay stable.
- **Per-track dedup:** a given track id is *emitted* at most once every
  `--sample-seconds` (default `1.0`). This is what shrinks the output from
  hundreds of thousands of records to a few thousand.
- **Sample point count by classes.txt "GeoJSON Type"** — this only controls
  *how many* independent pixel points get sampled per detection; it does
  **not** describe a shape. There is no outline, no implied ordering or
  connectivity between the points, and rings are never closed — each point
  is meant to be raycast independently against the 3DGS reconstruction, so
  forcing them into a polygon boundary would misrepresent what they are
  (and can actively hurt raycasting if a consumer treats them as an edge).

  | classes.txt Type | Points | How |
  |---|---|---|
  | `Point` | 1 | mask centroid (bbox centre fallback) |
  | `Line` | 4 | `cv2.minAreaRect` corners — spread across the structure's height, not just its ground line |
  | `Polygon` (compact) | 4 | `cv2.minAreaRect` corners |
  | `Polygon` (large-area) | 8–12 | simplified contour (`cv2.approxPolyDP`) for water, debris, dirt road, road, green/dry/burnt grass, pavement, mud — more samples across a bigger/irregular area |

- `--classes id,label,...` restricts output to an allowlist (default: all).

## Output GeoJSON

Written to `<output>/<video-stem>_detections.geojson` (default: next to the
input video; the runner always uses `./uav_samples/`). One RFC-7946
FeatureCollection per video with a top-level `metadata` foreign member
(permitted by RFC 7946 §6.1) carrying the run parameters.

**Coordinates are pixels** in the original frame resolution, origin top-left
(`x` = column, `y` = row) — flagged per feature via
`properties.coordinate_space: "pixel"`. Geometry is always **`Point`** (1
position) or **`MultiPoint`** (several independent positions, no closure,
no implied shape) — never `LineString`/`Polygon` — since these are raw
raycasting samples, not a detection outline.

```json
{
  "type": "FeatureCollection",
  "metadata": {
    "video": "clip.mp4",
    "model": "/app/best.pt",
    "fps": 29.97,
    "frame_width": 3840,
    "frame_height": 2160,
    "total_frames": 21071,
    "stride": 5,
    "tracker": "bytetrack.yaml",
    "confidence": 0.5,
    "sample_seconds": 1.0,
    "classes": "all",
    "geometry_source": "classes.txt",
    "processed_frames": 4215,
    "total_detections": 5821,
    "coordinate_space": "pixels in original frame resolution, origin top-left (x = column, y = row)",
    "generated_at": "2026-07-17T12:00:00+00:00"
  },
  "features": [
    {
      "type": "Feature",
      "id": "7",
      "geometry": {
        "type": "MultiPoint",
        "coordinates": [[1825.0, 1310.0], [1980.0, 1330.0],
                        [1972.0, 1700.0], [1817.0, 1680.0]]
      },
      "properties": {
        "class": "building",
        "class_id": 14,
        "id": "7",
        "confidence": 0.8123,
        "frame_index": 30,
        "timestamp_s": 1.001,
        "category": "infrastructure",
        "detection_type": "seg",
        "source": "uav",
        "coordinate_space": "pixel",
        "classes_txt_geometry": "Polygon",
        "marker-color": "#708090",
        "marker-size": "medium",
        "marker-symbol": "building"
      }
    }
  ]
}
```

| Property | Meaning |
|---|---|
| `id` (feature + property) | Persistent ByteTrack track ID (stable across frames; the same object emitted in a later window shares the id) |
| `class` / `class_id` | TRIFFID class name / index |
| `frame_index` / `timestamp_s` | Where in the video this emission happened |
| `coordinate_space` | Always `"pixel"` until the 3DGS raycast replaces coordinates |
| `classes_txt_geometry` | The original `classes.txt` classification (`Point`/`Line`/`Polygon`) — informational only, kept for traceability of *why* this detection got N points; does **not** describe `geometry.type` |
| `marker-*` | SimpleStyle marker styling, identical scheme to the UGV `geojson_bridge` output |

`geometry.type` is `Point` for single-sample classes, `MultiPoint` for
everything else — the point count still follows `classes.txt` (see table
above), it's just never packaged as an outline.

## CLI Reference

```bash
python -m triffid_uav_perception.uav_node VIDEO [options]
python -m triffid_uav_perception.uav_node --poll-api [options]   # PoC, see below
```

| Option | Default | Meaning |
|---|---|---|
| `--model` | `best.pt` | YOLO segmentation model |
| `--confidence` | `0.5` | Confidence threshold |
| `--imgsz` | `1280` | YOLO input size |
| `--stride` | `5` | Process every Nth frame |
| `--tracker` | `bytetrack.yaml` | Ultralytics tracker config |
| `--sample-seconds` | `1.0` | Per-track emission throttle (`0` = off) |
| `--classes` | *(all)* | Comma-separated class ids and/or labels |
| `--output` | *(video dir)* | Output directory for the GeoJSON |
| `--srt` | *(auto-detect)* | DJI SRT telemetry sidecar (default: `<video-stem>.srt` next to the video) |
| `--post-telesto` | off | Upload the collection to TELESTO after each video |
| `--telesto-base-url` | env `TELESTO_BASE_URL` / built-in | Backend override |
| `--poll-api` | off | Poll FUTURISED for new video uploads instead of a local file (PoC — see below) |
| `--api-media-key` | env `FUTURISED_MEDIA_API_KEY` | Media Files API key (required with `--poll-api`) |
| `--api-camera` | `Wide` | Camera filter: `Wide`/`Zoom`/`Thermal`/empty |
| `--api-poll-interval` | `30` | Seconds between polls |
| `--api-download-dir` | `./uav_videos` | Where downloaded videos land |
| `-v` | | Debug logging |

Exactly one of a positional `VIDEO` or `--poll-api` is required.

## SRT Telemetry Sidecar

FUTURISED delivers each drone video as an **MP4 + a sidecar `.SRT` file**
carrying per-frame telemetry (DJI drones write one subtitle block per
frame). When an SRT is available — `--srt path.srt`, or auto-detected as
`<video-stem>.srt`/`.SRT` next to the video — the pipeline joins it to
frames by frame number (nearest-timestamp fallback for lower-rate SRTs)
and attaches the **drone/camera** telemetry to every emitted feature:

| Property | Source (modern DJI SRT key) |
|---|---|
| `drone_latitude` / `drone_longitude` | `latitude` / `longitude` (or legacy `GPS(lon,lat,alt)`) |
| `drone_altitude_m` | `abs_alt` |
| `drone_rel_altitude_m` | `rel_alt` (or legacy `H <n>m`) |
| `gimbal_yaw` / `gimbal_pitch` / `gimbal_roll` | `gb_yaw` / `gb_pitch` / `gb_roll` |

These describe **where the camera was**, not where the detection is — the
detection geometry stays pixel-space. Together they give the downstream
3DGS raycast both halves of the problem: the pixel samples *and* the
camera pose per frame. Only fields actually present in the SRT are
attached; without an SRT the output is unchanged. The collection
`metadata` records `srt_file` and `srt_frames`.

Both known DJI SRT flavours are parsed (`srt_metadata.py`, stdlib-only):
the modern bracketed format (`FrameCnt: N`, `[latitude: …] [gb_yaw: …]`,
`<font>`-wrapped) and the legacy `GPS(lon,lat,alt)` format. Parsing is
tolerant — unknown keys ignored, telemetry-free blocks skipped.
**Caveat:** the parser is validated against synthetic samples of the two
known formats; the real FUTURISED SRT hasn't been seen yet, so regexes
may need a touch-up when the first real file arrives. Some DJI models
embed telemetry as a subtitle stream *inside* the MP4 instead of a
sidecar — extracting that needs ffmpeg (`ffmpeg -i in.mp4 -map 0:s:0
out.srt`), which is not in the image; only sidecar files are handled.

In `--poll-api` mode, `.SRT` files are polled and downloaded alongside
`.MP4`/`.MOV` into the same directory, where sidecar auto-detection pairs
them by basename. PoC limitation: an SRT that appears *after* its video
was already processed does not trigger reprocessing.

## TELESTO Upload (optional)

`--post-telesto` (or `POST_TELESTO=1` with the runner) PUTs every feature to
the TELESTO Map Manager REST API via `triffid_telesto.telesto_client` after
each video finishes. It is **off by default** and documented as
future-facing: the backend expects `[longitude, latitude]` coordinates, so
uploading makes sense only once the pixel coordinates have been converted
through the 3DGS raycast. The UGV↔TELESTO MQTT bridge is unaffected — the
UAV pipeline no longer publishes MQTT at all.

## FUTURISED Video-Poll Mode (`--poll-api`, PoC)

FUTURISED's Media Files API serves **uploaded files**, not a continuous
video stream — there's no true "live" feed to connect to yet, todo.txt
notes this may change. What `--poll-api` does today: repeatedly call
`FuturisedClient.poll_new_images(extensions={'.mp4', '.mov'})` (the same
polling mechanism the old still-image mode used, just pointed at video
extensions instead of image ones), download each newly-uploaded video, and
run the normal `process_video()` pipeline on it — same output, same flags,
same optional `--post-telesto`, one GeoJSON per video.

```bash
export FUTURISED_MEDIA_API_KEY='your-media-api-key'
python -m triffid_uav_perception.uav_node --poll-api \
    --api-camera Wide --api-poll-interval 30 --output ./uav_samples
```

This is a **proof of concept**: it hasn't been exercised against a real
FUTURISED account, and whether the platform actually serves `.mp4`/`.mov`
files through this endpoint yet is unconfirmed. If it doesn't, `list_media()`
will simply return no matching candidates each poll — nothing breaks, it
just never finds anything to download.

- **Media Files API** (`dji.getfuturised.com`) — `x-api-key` auth; lists
  uploaded media and returns temporary S3 download URLs. Camera suffixes:
  Wide (`_W`), Zoom (`_Z`), Thermal (`_T`), Split (`_S`).
- **Telemetry API** (`api.getfuturised.com/getDJIData`) — bearer-token auth;
  real-time drone state, not used by `--poll-api` (kept in `api_client.py`
  for future use). Coordinate values use inconsistent European number
  formatting (`parse_telemetry_coord` handles the known variants).

`api_client.py` has no third-party dependencies (stdlib `urllib`). The
full unit-test suite lives on the main development branch (this partner
build ships without tests).

## Docker Setup

Separate image from the UGV (no ROS2): Python 3.10 slim + CUDA PyTorch
(cu121 wheels — they fall back to CPU on hosts without a GPU). GPU access is
wired via `deploy.resources` in `docker-compose.uav.yml` and the
`nvidia-container-toolkit`; remove the `deploy` block to run CPU-only.

| Host Path | Container Path | Purpose |
|---|---|---|
| `./src/triffid_uav_perception/` | `/app/src/triffid_uav_perception/` | Source code (editable) |
| `./src/triffid_telesto/` | `/app/src/triffid_telesto/` | TELESTO REST client (for `--post-telesto`) |
| `./uav_videos/` | `/app/videos/` | Input videos |
| `./uav_data/` | `/app/uav_data/` | Additional input media |
| `./uav_samples/` | `/app/samples/` | Output GeoJSON |
| `./best.pt` | `/app/best.pt` | YOLO model (read-only) |
| `./classes.txt` | `/app/classes.txt` | Class table (geometry-type reference) |
