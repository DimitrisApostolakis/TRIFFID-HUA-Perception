# UAV perception: selected views + SANSA + Unity JSON

The UAV node has three independent commands:

1. `select`: MP4 + DJI SRT -> angularly selected PNG frames and a selection manifest.
2. `detect`: MP4 + DJI SRT + visual supports -> point-class observations in those views.
3. `segment-top-view`: one supplied top-view RGB image + visual supports -> polygon-class masks.

The node exports **pixel observations**, not geographic GeoJSON. Unity is responsible
for camera projection, raycasting, geographic conversion and cross-view deduplication.
The UGV node, dependencies and GeoJSON interface are unchanged.

## Setup

Python 3.10 or newer. From the repository root:

```bash
pip install -r src/triffid_uav_perception/requirements.txt
export PYTHONPATH="$PWD/src/triffid_uav_perception"
python -m triffid_uav_perception.uav_node --help
```

In Windows CMD use `set PYTHONPATH=%CD%\src\triffid_uav_perception` instead.
Selection works without PyTorch or model weights. Tests require `pip install pytest`.

For SANSA, install a matching PyTorch/torchvision pair for your GPU, then:

```bash
pip install -r src/triffid_uav_perception/requirements-sansa.txt
git clone https://github.com/ClaudiaCuttano/SANSA uav_models/SANSA
git -C uav_models/SANSA checkout f562b5642940e08a57e3da2608b804b99af3763b
```

Download the **SANSA Universal** weights and original **SAM2 Hiera Large** base
weights using the official SANSA README. Supply both local paths explicitly;
the adapter does not auto-download them. This backend uses the universal model
(`channel_factor=0.8`, stages 2/3), not a COCO-fold/FSS benchmark checkpoint.
Official reference: <https://github.com/ClaudiaCuttano/SANSA#1-sansa-universal-model->.
Reference runtime: torch 2.7.1 + torchvision 0.22.1. The Docker image uses CUDA 12.6;
native CPU execution is available via `--device cpu` with a CPU PyTorch installation.

### Support images and masks

Copy `examples/supports.template.json` into `uav_supports/supports.json`, and replace
its placeholder paths with your labelled examples. Paths are relative to the manifest.
Every **requested** class needs exactly **1 or 5** support image/mask pairs. A subset
can be run with `--classes "debris,person"`; unused classes need no supports.

```json
{
  "schema_version": "triffid.uav.supports.v1",
  "classes": {
    "debris": [{"image": "images/debris.jpg", "mask": "masks/debris.png"}],
    "person": [{"image": "images/person.jpg", "mask": "masks/person.png"}]
  }
}
```

Masks must be single-channel, match their reference image dimensions, and contain
foreground. For binary masks every nonzero pixel is foreground. For an indexed
multiclass mask specify `"foreground_value": 16` (or its actual pixel value) to
avoid treating other labelled classes as foreground. RGB colour-coded masks must
first be converted to binary/indexed masks. Person supports may show citizens,
first responders or military personnel; the output does not infer those roles.

## Selecting the eight views

Default: **8 orbital positions, 45 degrees apart**, clockwise from north. GPS
positions determine the bearing about the circle centre. Camera/gimbal yaw is
not the same as orbital position; for an inward-facing camera it differs by
approximately 180 degrees. Use `--selection-mode heading` explicitly if selecting
by `gb_yaw` instead. Heading mode needs yaw telemetry; orbit mode needs GPS.

```bash
python -m triffid_uav_perception.uav_node select uav_videos/flight.mp4 \
  --srt uav_videos/flight.srt --output uav_samples/selection \
  --start-seconds 10 --end-seconds 70
```

The time interval must contain **one complete circle**. This version does not
automatically split multi-circle flights. The circle centre is estimated from
the GPS bounds in that interval. For partial/irregular paths or a known target,
provide `--orbit-centre LAT LON`. Set `--angle-offset` to rotate the desired
views. Default angular acceptance is 12 degrees; it is an explicit configurable
selection tolerance, not a claim about camera/georeferencing accuracy.

If the eight views instead mean **4 positions at each of two camera pitches**:

```bash
python -m triffid_uav_perception.uav_node select uav_videos/flight.mp4 \
  --output uav_samples/selection --views 4 --pitches -90 -45
```

Pitch tolerance defaults to 10 degrees. A missing target or insufficient distinct
frames fails clearly; the selector never fills gaps by duplicating a frame.
It joins SRT timestamps to constant-frame-rate video using `round(time * fps)`;
subtitle block numbers are not assumed to equal video frame numbers. Variable
frame rate or a time-offset SRT must be aligned/converted before processing.

## Detect point classes in selected frames

```bash
python -m triffid_uav_perception.uav_node detect uav_videos/flight.mp4 \
  --srt uav_videos/flight.srt --output uav_samples/flight \
  --sansa-repo uav_models/SANSA \
  --checkpoint uav_models/sansa_universal.pth \
  --sam2-checkpoint uav_models/sam2_hiera_large.pt \
  --supports uav_supports/supports.json --device cuda
```

Outputs: `point_detections.json` and `frames/view_00.png` through `view_07.png`.
Frames are decoded/inferred one at a time. SANSA loads once, reference images
are preprocessed once per class, and gradients are disabled. Each class/query
runs the official model forward independently. This is **not native 15-way
inference**: encoder work is repeated across class episodes by the official API.
No cross-query memory or unsupported feature cache is injected into SANSA.

SANSA is a semantic segmenter. Eight-connected foreground components become
candidate objects; touching objects can merge and disconnected pieces can split.
IDs identify observations, not guaranteed object instances or persistent tracks.
`--min-area-px` can remove tiny components (default 1, no hidden area filtering).
Class masks may overlap; no arbitrary cross-class suppression is applied.

## Segment polygon classes in the top-view image

The top-view image is supplied separately, e.g. a Unity reconstruction render.
The node does not invent it from the orbit video or treat one nadir frame as an
orthomosaic.

```bash
python -m triffid_uav_perception.uav_node segment-top-view uav_data/top_view.png \
  --output uav_samples/top_view \
  --sansa-repo uav_models/SANSA \
  --checkpoint uav_models/sansa_universal.pth \
  --sam2-checkpoint uav_models/sam2_hiera_large.pt \
  --supports uav_supports/supports.json --device cuda
```

Output: `polygon_segmentations.json`. `--projection-json path.json` optionally
copies Unity's top-view camera/projection context into the source record; without
it Unity must retain the matching camera context itself. Polygon masks and holes
are kept exactly; geographic polygon rings are created downstream. Smoke/flame
projection requires separate interpretation and is not a measured ground footprint.

## UAV class catalogue

Original IDs are retained. Person uses ID 23 and merges source IDs 6/23/43.

| Final geometry | Classes |
|---|---|
| Point (15) | Destroyed vehicle, Window, **Debris**, Civilian vehicle, Person, Boat, Excavator, Door, Ambulance, Fire truck, Crane, Stairs, Tank, Police vehicle, Army vehicle |
| Polygon (20) | Water, Green tree, Flame, Smoke, Green plant, Building, Destroyed building, Dirt road, Dry tree, Road, Green grass, Pavement, Dry grass, Mud, Burnt tree, Tower, Silo, Burnt grass, Dry plant, Burnt plant |

The geometry is a **target for Unity**. Both tasks export binary masks in image
coordinates; no pixel coordinates are mislabelled as longitude/latitude GeoJSON.

## Unity JSON contract

Schema version: `triffid.uav.observations.v1`. Coordinates are original image pixels,
origin top-left, x right, y down. The envelope contains task, class catalogue,
backend settings, sources and (for video) selection/video/SRT metadata.

Each source contains original width/height and observations. Video sources also
contain zero-based frame index, timestamp, selected/actual angle, SRT telemetry
and telemetry-to-frame time difference. Telemetry is camera/drone pose, not object
location. Missing SRT fields remain null; camera intrinsics are not manufactured.

Each observation contains:

| Field | Meaning |
|---|---|
| `observation_id` | Unique within the output document: view + class + component |
| `class_id`, `class_name`, `source_class_ids` | UAV ontology identity |
| `target_geometry` | `Point` or `Polygon`, for downstream mapping |
| `mask` | Exact cropped binary mask, including holes; encoding below |
| `bbox_xyxy` | Tight mask box `[xmin,ymin,xmax,ymax]`, maximum bounds **exclusive** |
| `centroid_xy` | Mean x/y of foreground pixels, not the bounding-box centre |
| `raycast_point_xy` | Foreground pixel nearest centroid; safe for concave/hollow masks |
| `area_px` | Foreground pixel count at original image resolution |
| `score`, `score_type` | Mean foreground sigmoid probability, **not calibrated detection confidence** |

Mask example: `{"encoding":"binary_rle_row_major","size_hw":[2,3],
"offset_xy":[10,20],"counts":[1,2,2,1]}`. Decode by alternating runs of 0/1,
starting with 0. Traverse the crop **row by row** (x changes fastest), then place
the crop at `[10,20]` in the original image. Here the crop is `[[0,1,1],[0,0,1]]`.
This is intentionally a simple Unity-friendly RLE, **not COCO column-major RLE**.
Counts sum to crop height * width; a zero first count represents foreground at
the first pixel. The box and crop use the same offset/size. JSON writes atomically
and rejects non-finite values. See `masks.decode_rle` for a reference decoder.

### Migration from the previous UAV node

The old positional-video/`--poll-api` YOLO/ByteTrack CLI and pixel GeoJSON exporter
are replaced by these explicit task commands. No `best.pt` or tracking is used by
the UAV node. The API client remains isolated in `api_client.py` for a later ingest
adapter, but polling/upload are not connected to the new node. Raw pixel JSON is
not uploaded to TELESTO. Cross-view duplicate merging belongs after Unity projection.

## Docker

```bash
docker compose -f docker-compose.uav.yml up -d --build
docker exec triffid_uav_perception python -m triffid_uav_perception.uav_node \
  select /app/videos/flight.mp4 --output /app/samples/selection
```

Host mounts: `uav_videos` -> `/app/videos`, `uav_data` -> `/app/uav_data`,
`uav_samples` -> `/app/samples`, `uav_models` -> `/app/models`,
`uav_supports` -> `/app/supports`. Use the same task commands with container paths
to infer. The SANSA source directory is writable because its official builder may
create Python cache files; model weights are explicit local files.

## Validation

```bash
python -m pytest -q src/triffid_uav_perception/tests
```

Tests cover class/geometry decisions, angular selection and wraparound, two-pitch
selection, low-rate SRT, missing targets, mask holes/centroids/boxes/RLE, supports,
atomic output, and synthetic-video/top-view end-to-end export with an injected backend.
Real SANSA weights and CUDA inference require the local assets described above.
