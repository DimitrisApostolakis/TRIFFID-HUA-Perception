# Modular UGV perception

This refactor starts from `feat/update-ugv-input-topics` at
`b1a41ed46e3f50847b9c1c0cb6d0357c7c5dbc36`. It keeps that UGV branch's
calibration, topic defaults, thermal debugging, tracker, position-source priority,
spatial filtering and GeoJSON/API delivery rules. The requested detector change is **SANSA**.
The requested UGV geometry policy is **fence/wall → LineString; all other classes → Point**.
It does not change the UAV module; UAV work remains on `feat/uav-sansa-perception`.

## Responsibilities

| Module | Owns |
|---|---|
| `ugv_node.py` | ROS image input, latest-depth state, pipeline orchestration |
| `config.py` | Central parameter defaults used by nodes and launch |
| `ontology.py` | Original 63 IDs/names/styles; fence/wall lines and all other classes points |
| `supports.py`, `bank.py` | UGV-only support storage, validation, selection and CLI |
| `models/base.py`, `models/__init__.py` | Detector/segmenter contracts and lazy factories |
| `models/sansa.py` | Frozen official SANSA model, preprocessing, support/query inference |
| `models/semantic.py` | Explicit class mask → connected-component observations |
| `models/yolo.py`, `models/dummy.py` | Optional legacy adapter and explicit dummy mode |
| `geometry.py`, `pipeline.py` | Existing depth sampling, back-projection, optical/body mapping, median position, extents and 3D NMS |
| `tracker.py`, `tracking/` | Existing ByteTrack-style coordinator, Kalman state and association metrics |
| `messages.py`, `debug.py` | Detection3DArray, mono8 labels, RGB overlays and thermal snapshots |
| `position.py` | ROS UGV status → HTTP position API → filtered `/fix` → seed, plus compass heading |
| `geojson.py`, `spatial.py` | Feature geometry/properties and geographic accumulation |
| `storage.py`, `delivery.py`, `geojson_bridge.py` | Atomic snapshots, TELESTO upsert policy, ROS/timer/bounded-worker transport |

Future instance detectors implement `DetectionBackend.detect(bgr)` and return
original-resolution `bbox` (xyxy, upper edge exclusive), `class_id`, `class_name`,
`confidence`, and an optional boolean mask. Register their factory with
`register_detector`; geometry, tracking and ROS messages do not need model imports.
A future semantic backend implements `SegmentationBackend` and can use the existing
`SemanticDetector` adapter. The selected ordered supports are model-independent.

## Independent UGV bank

UGV uses `ugv_supports/bank.json`, schema `triffid.ugv.support-bank.v1`.
UAV uses its separate `uav_supports` bank. UGV rejects UAV schemas and keeps all 63
classes: **first responder**, **citizen**, and **military personnel** remain separate.
The managed bank defines five image/mask slots per class (315 total). It creates
no fake examples. Paths are relative to the manifest, so the directory is portable.

From the repository root, with NumPy/OpenCV installed:

```bash
export PYTHONPATH="$PWD/src/triffid_ugv_perception${PYTHONPATH:+:$PYTHONPATH}"
python -m triffid_ugv_perception.bank init --bank ugv_supports
python -m triffid_ugv_perception.bank add --bank ugv_supports \
  --class citizen --slot 1 --image ground_view.png --mask citizen_mask.png
python -m triffid_ugv_perception.bank validate --bank ugv_supports --classes citizen
```

Masks must be single-channel, match image dimensions and contain foreground.
Use `--foreground-value N` for an indexed multiclass mask. Five distinct support
images are required per class; additions never overwrite an occupied slot.
`validate` checks all five slots. A **1-shot** run checks/uses only slot 1, so slots
2–5 can remain unfilled during initial testing. Default inference uses all classes;
set `classes:=citizen,window` to run only classes whose bank entries are ready.
No class is silently skipped. All models receive the same first-K ordering,
example IDs, and manifest/image/mask hashes. New query predictions never enter the bank.

## SANSA setup and execution

Place the official SANSA checkout and checkpoints on the host:

- `ugv_models/SANSA/` — official checkout, reference revision `f562b5642940e08a57e3da2608b804b99af3763b`.
- `ugv_models/adapter_generalist.pth` — SANSA generalist/universal adapter.
- `ugv_models/sam2_hiera_large.pt` — SAM2 Hiera Large base weights.
- `ugv_supports/` — the independently labelled UGV bank.

The default UGV Docker image remains **ROS 2 Jazzy** and installs torch 2.7.1 /
torchvision 0.22.1 with CUDA 12.6 and the SANSA runtime dependencies. Compose mounts
`ugv_models` at `/ws/models` and the UGV bank read-only at `/ws/ugv_supports`.
Build the workspace after pulling even if an old `install/` already exists:

```bash
docker compose up -d --build
docker exec -it triffid_perception bash
cd /ws
colcon build --symlink-install --packages-select triffid_ugv_perception
source /ws/install/setup.bash
ros2 launch triffid_ugv_perception ugv_perception.launch.py \
  backend:=sansa support_bank:=/ws/ugv_supports shots:=1 classes:=citizen
```

Use `shots:=5` once all five entries for the selected classes are valid. Model
parameters: `sansa_repo`, `checkpoint`, `sam2_checkpoint`, `device`, `image_size`
(default 640), `mask_threshold` (0.5), `min_area_px` (1). Full defaults live in
`config.py`; launch forwards typed parameters to the appropriate node. Missing
supports/weights or an unavailable backend fail explicitly without dummy fallback.
`use_dummy_detections:=true` intentionally bypasses model and bank loading.

For the optional legacy YOLO adapter, install `ultralytics` separately and select
`backend:=yolo model_path:=/path/to/best.pt`; its old inference/mask conversion is
retained. Compose no longer mounts YOLO weights by default, so supply the path or
an explicit mount when using that optional backend.

## Preserved methodology and model boundary

The processing remains: latest RGB + latest depth → detection → depth samples →
median camera point → fixed optical/body axis mapping → 3D extent → existing 3D
NMS → existing tracker → same ROS messages → geographic delivery.
Fence and wall use the existing longest-extent LineString construction. Every
other class exports its centre as a Point; UGV no longer emits Polygon features.

The inherited calibration is **fixed 640×480 RealSense intrinsics**, optical
`(right, down, forward)` mapped to body `(forward, left, up)`, with camera offset
neglected. There is no new CameraInfo subscription, TF lookup, image/depth
synchronizer, depth filter, new duplicate filter, or thermal detector. The old
scaled-depth fallback, deterministic depth subsampling, 0.5 m cross-class 3D NMS,
tracker confirmation/aging and 3 m geographic accumulation rules are retained.
The original empty-raw-detection early return is retained, including its tracker
update behavior. These choices are documented here rather than changed by refactoring.

SANSA's public forward, support masks, normalization, bilinear image/logit resize,
nearest mask resize and independent one-class 1/5-shot episodes are the same as
our existing SANSA adapter. It produces **semantic** masks: 8-connected components
provide observations, and touching people/vehicles may still become one observation.
No instance-splitting heuristic or tracker replacement is added.

The tracker/message confidence field receives **mean foreground probability** per
component. This is not calibrated YOLO detection confidence. Existing thresholds
are retained; changing the model can change detections, scores and tracking results.
No FPS or aerial/ground-scene accuracy claim is made without actual inference tests.

## Validation

```bash
python -m pytest -q src/triffid_ugv_perception/tests
```

Tests cover numeric geometry, preserved algorithm bodies against the pinned UGV
baseline, tracker IDs, bank separation and 1/5-shot selection, the real torch SANSA
adapter with a stub official model, component masks, ROS message/node doubles,
position fallback and staleness. CI runs the same checks. A one-off comparison also
matched 244 depth/localization, tracker-frame and per-class GeoJSON cases against
the original source before the explicitly requested Point/LineString routing update.
Tests additionally enforce the new routing for every UGV class. No model checkpoint, real GPU inference, Docker/ROS deployment,
robot/API traffic or rosbag playback was exercised in this environment.
