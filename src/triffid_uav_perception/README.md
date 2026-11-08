# UAV perception: selected views + SANSA + Unity JSON

See the [UAV architecture and code guide](../../docs/UAV_ARCHITECTURE.md) for the
Python/Unity diagram, module structure, model runners and current live boundaries.

Based on latest UAV commit `285c68d`, imported into this workspace with human
roles preserved separately. PUI supports are under `uav_supports/PUI`; see
[UAV PUI supports](../../docs/UAV_PUI_SUPPORTS.md).

The UAV node provides video, exported-frame and top-view inference plus shared support-bank management:

1. `select`: MP4 + DJI SRT -> angularly selected PNG frames and a selection manifest.
2. `detect`: MP4 + DJI SRT + visual supports -> point-class observations in those views.
3. `segment-top-view`: one supplied top-view RGB image + visual supports -> polygon-class masks.
4. `detect-frames`: reuse PNG frames and telemetry from a previously exported selection manifest.
5. `bank init/add/validate`: create, fill and check the model-independent support bank.

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

Use the **SANSA Generalist** or **SANSA Universal** checkpoint and original
**SAM2 Hiera Large** base weights from the official SANSA README. Supply both
local paths explicitly; the adapter does not auto-download them. Both supported
checkpoints use `channel_factor=0.8`, stages 2/3, and mask prompts in this backend.
Do not use a COCO-fold/FSS benchmark checkpoint with different architecture settings.
Official reference: <https://github.com/ClaudiaCuttano/SANSA#1-sansa-universal-model->.
Reference runtime: torch 2.7.1 + torchvision 0.22.1. The Docker image uses CUDA 12.6;
native CPU execution is available via `--device cpu` with a CPU PyTorch installation.

## Shared support bank: five examples per class

One bank is shared across all models and both UAV tasks. It reserves **38 classes x
5 support slots = 190 slots**. Only slots with real annotated image/mask pairs
are usable examples; a reserved slot is not a populated support. The bank stores raw labelled data;
model checkpoints and model-specific feature caches are separate. New backends
receive a `SupportSelection` through `prepare(classes, supports)` and do not
select their own files or maintain a separate support manifest.

### Create the bank

```bash
python -m triffid_uav_perception.uav_node bank init --bank uav_supports
```

This creates `bank.json` and class directories. The manifest reserves five slots
per class, referencing `citizen/01/image.png`, `citizen/01/mask.png`, through slot
`05`, and similarly for each other class (spaces become underscores in folders).
No dummy images or masks are generated: these slots still need your annotations.
The bank can live outside the repository, e.g. `/mnt/d/Models/SupportBank/UAV`.

### Add a real example

```bash
python -m triffid_uav_perception.uav_node bank add \
  --bank uav_supports --class citizen --slot 1 \
  --image /path/to/person_reference.jpg --mask /path/to/person_mask.png
```

Repeat for slots **1 through 5** for each class. `bank add` copies the RGB image
and normalizes the mask to a lossless binary PNG (0 background, 255 foreground).
It validates shape/foreground and never overwrites an existing slot. You can also
place your own PNG pairs directly at the manifest paths. Empty directories are
not considered populated slots.

For an indexed multiclass mask, add `--foreground-value 23` (or the actual target
label value). Without it **all nonzero pixels** count as foreground, so only use
that default for already-binary masks. Masks must be single-channel and match the
reference image dimensions. RGB colour-coded masks need conversion first. Human roles have separate class
labels and supports: first responder (6), citizen (23), military personnel (43).
Their masks are never combined into a generic person class.

### Validate

```bash
python -m triffid_uav_perception.uav_node bank validate --bank uav_supports
```

The report checks all 190 support slots and lists all missing/invalid slots.
It rejects duplicate example IDs, repeated image paths, duplicate image-file
content within a class, empty masks, wrong sizes and non-grayscale masks. It exits
with code 2 if invalid. To start with a smaller test subset:

```bash
python -m triffid_uav_perception.uav_node bank validate \
  --bank uav_supports --classes "debris,citizen"
```

Detection validates only its requested point classes; top-view segmentation only
its requested polygon classes. A subset such as `--classes "debris,citizen"` needs
populated examples for each selected shot of those classes, not the rest of
the bank. Five-shot runs need all five; one-shot runs need the first slot only.

### Retrieval and reproducibility

Use `--support-bank uav_supports` (the default) on either inference command.
`--shots 5` retrieves all five examples per class. `--shots 1` deterministically
retrieves the first example, while the bank retains five slots per class. For an initial 1-shot test only the
first slot needs to be populated; `--shots 5` requires all five. `bank validate`
always checks all five slots, so an unfinished bank still reports incomplete.
No random sampling or duplication is used to fill missing examples. All classes
in one run use the same shot count. A bank directory or an explicit manifest path
is accepted; relative asset paths resolve against the manifest, not the shell's
working directory. `--supports` remains an alias for `--support-bank`.

The pipeline selects the example identities once, then injects the same selection
into the backend. SANSA only applies its own resizing, normalization and prompt
formatting. The JSON output's `support_bank` member records bank/asset SHA-256
hashes, selected example IDs, shot count and ordering policy so the supports used
by different models can be compared. Raw images are not cached permanently in RAM;
SANSA caches only its resized support tensors for the current run.

`examples/supports.template.json` contains the complete managed bank manifest if
you prefer to prepare files manually. Legacy `triffid.uav.supports.v1` manifests
are still accepted **only with five distinct valid pairs per requested class**;
expand old single-example manifests, then use `--shots 1` for 1-shot inference.

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
  --support-bank uav_supports --shots 5 --device cuda
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

## Reuse the eight frames already exported

Use the existing selection manifest so neither MP4 nor SRT is opened again:

```bash
python -m triffid_uav_perception.uav_node detect-frames \
  uav_samples/merged_cut_selection/selected_frames.json \
  --output uav_samples/merged_cut_detections \
  --sansa-repo uav_models/SANSA \
  --checkpoint /mnt/d/Models/Checkpoints/SANSA/adapter_generalist.pth \
  --sam2-checkpoint /mnt/d/Models/Checkpoints/SANSA/sam2_hiera_large.pt \
  --support-bank uav_supports --shots 1 --classes "civilian vehicle,window" \
  --device cuda
```

This initial test needs one filled support slot for each of those two classes.
The bank still has five slots per class. Switch to `--shots 5` after all five
examples are available. Other classes do not run unless requested or the class
filter is omitted. The manifest keeps the selected order, frame indices and camera
telemetry. The exported PNGs must be accessible at their recorded relative paths
and keep their original dimensions. Detection output uses resolved image paths
so it can live in a different directory. Results are in `point_detections.json`;
no model inference is performed by the `select` command itself.

A selected view used as its own support is a self-reference check, not an
independent test of generalization; use different support images for evaluation.

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
  --support-bank uav_supports --shots 5 --device cuda
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
backend settings, shared support-bank provenance, sources and (for video)
selection/video/SRT metadata.

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

## Visual detection debugger

After inference, render all saved observations without rerunning SANSA:

```bash
python -m triffid_uav_perception.visualize \
  uav_samples/merged_cut_detections/point_detections.json
```

By default the output directory is `debug/` next to the JSON. It contains:

- `00_overlay.png`, `01_overlay.png`, etc.: masks, exclusive-maximum boxes,
  centroid crosses and numbered class/score labels drawn on each source frame.
- `contact_sheet.jpg`: up to four views per row (the default eight views in two rows).
- `index.html`: offline browser gallery with clickable larger previews.
- `summary.json`: per-frame class counts and a mapping from displayed numbers
  to original observation IDs, areas, boxes, centroids and score semantics.

Civilian vehicles are cyan; windows are orange. Colors stay consistent across
frames. `p` is mean foreground probability, not calibrated detection confidence.
All observations are shown by default, including tiny components. The debugger
does not modify source images or detection JSON, filter inference results or
perform duplicate merging. It needs only the existing OpenCV/NumPy dependencies.
It also supports `polygon_segmentations.json` from the top-view command.

Optional controls:

```bash
# Only windows; suppress tiny regions in the preview, leaving results unchanged.
python -m triffid_uav_perception.visualize \
  uav_samples/merged_cut_detections/point_detections.json \
  --output uav_samples/debug_windows --classes window --min-area-px 100
```

Use `--alpha 0.3` to reduce mask opacity, `--max-width 0` for original-resolution
previews (default maximum width 1920), or `--no-labels`, `--no-masks`, `--no-boxes`,
`--no-centroids` to simplify the display. PNG dimensions do not change until the
optional final display resize. Source images must retain their original dimensions.
`--image-root /path/to/frames` relocates the source images by their exact basenames
if the paths recorded in the JSON are no longer accessible. Files are never
matched fuzzily. Open `index.html` or `contact_sheet.jpg` to inspect the results;
no display server or OpenCV GUI is needed, including under WSL.

## Validation checks

```bash
python -m pytest -q src/triffid_uav_perception/tests
```

Tests cover class/geometry decisions, angular selection and wraparound, two-pitch
selection, low-rate SRT, missing targets, mask holes/centroids/boxes/RLE, support banks,
atomic output, and synthetic-video/top-view end-to-end export with an injected backend.
Bank tests additionally check all 190 slots, add/validate commands, duplicate images,
model-independent retrieval, deterministic 1/5-shot selection and asset hashes.
Real SANSA weights and CUDA inference require the local assets described above.
# Image-space point tracking

The tracker is a separate stage after detection, before Unity georeferencing. It
reuses the pretrained MASA-R50 tracker already installed for UGV; no detector is
rerun and no model is trained. Human-role class names remain separate. The local
runner selects the existing MASA environment and imports the unchanged UGV engine:

```bash
bash scripts/run_uav_tracking.sh /path/point_detections.json \
  --output samples/FSS-SAM3/UAV/run/tracking_masa
```

Defaults: score at least 0.5, actual foreground area at least 16 pixels, confirmation
in two distinct frames, 900-second appearance memory, at most 4096 identities.
These are configurable CLI options. Null scores cannot establish a confident
track and are filtered. Timestamps are processed in chronological order; missed
frames still advance memory. Association is based on MASA appearance features,
with class compatibility and one-to-one IDs, rather than assuming fixed pixels
under a moving UAV camera.

Artifacts are `detections.mp4`, `tracking.mp4`, `exports.mp4`, JPEG overlays,
`timings.json`, `tracked_points.json` and `point_detections.json`. No HTML or copied
source-frame dataset is created. The videos play one selected frame per second
and display the original timestamp; they are not a resampled full-flight video.

`tracked_points.json` retains all matched frame observations, tentative/confirmed
state, persistent IDs and a first-confirmation `is_new` event exactly once per
ID. `exports.mp4` shows those first-confirmation events, without claiming API
readiness or world coordinates.

Feed the tracking output's **`point_detections.json`** into Unity PointProjector.
It contains exactly one highest-score, original mask/raycast measurement per
confirmed identity, with the original source telemetry. Later better observations
replace the representative in this snapshot rather than appending another point.
The snapshot is not an incremental API message. Geometry and camera coordinates
are never averaged across frames. PointProjector requires no schema changes.

Identity scope is one video/run, not permanent identity across flights or process
restarts. Sparse orbit views, lookalike windows and class changes can still split
or merge identities; the 900-second setting does not guarantee physical identity.
Use the overlays to review that behavior. The capacity limit fails instead of
silently evicting IDs and re-exporting them. No API is called by this stage.
