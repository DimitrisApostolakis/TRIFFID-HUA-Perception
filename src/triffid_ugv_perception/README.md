# Modular UGV perception

See [the live UGV architecture, diagrams and code map](../../docs/UGV_ARCHITECTURE.md)
for the selected YOLOE-v8-L/MASA pipeline, module responsibilities and output topics.

This refactor starts from `feat/update-ugv-input-topics` at
`b1a41ed46e3f50847b9c1c0cb6d0357c7c5dbc36`. It keeps that UGV branch's
topic defaults, thermal debugging, tracker, position-source priority,
spatial filtering and GeoJSON geometry. API delivery now follows the persistent creation-only policy below. The default ROS detector is **SANSA**.
RGB-D calibration and synchronization have since been corrected as described below.
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
| `discovery/` | Persistent UUID registry, bounded SQLite outbox, background creation-only delivery |
| `storage.py`, `delivery.py`, `geojson_bridge.py` | Bounded optional snapshots/debug files, API transport exports, ROS adapter |

Future instance detectors implement `DetectionBackend.detect(bgr)` and return
original-resolution `bbox` (xyxy, upper edge exclusive), `class_id`, `class_name`,
`confidence`, and an optional boolean mask. Register their factory with
`register_detector`; geometry, tracking and ROS messages do not need model imports.
A future semantic backend implements `SegmentationBackend` and can use the existing
`SemanticDetector` adapter. The selected ordered supports are model-independent.

## Independent UGV bank

UGV uses `ugv_supports/PUI/bank.json`, schema `triffid.ugv.support-bank.v1`.
UAV uses its separate `uav_supports` bank. UGV rejects UAV schemas and keeps all 63
classes: **first responder**, **citizen**, and **military personnel** remain separate.
The managed bank defines 16 image/mask slots per class (1,008 total). It creates
no fake examples. Paths are relative to the manifest, so the directory is portable.

From the repository root, with NumPy/OpenCV installed:

```bash
export PYTHONPATH="$PWD/src/triffid_ugv_perception${PYTHONPATH:+:$PYTHONPATH}"
python -m triffid_ugv_perception.bank init --bank ugv_supports/PUI
python -m triffid_ugv_perception.bank add --bank ugv_supports/PUI \
  --class citizen --slot 1 --image ground_view.png --mask citizen_mask.png
python -m triffid_ugv_perception.bank validate --bank ugv_supports/PUI --classes citizen --shots 1
```

Masks must be single-channel, match image dimensions and contain foreground.
Use `--foreground-value N` for an indexed multiclass mask. Each requested support
must be a distinct image; additions never overwrite an occupied slot.
`validate` checks all 16 slots unless `--shots N` is specified. Inference accepts
any integer shot count from 1 to 16 and uses the first N slots. A **1-shot** run
checks/uses only slot 1, so slots 2–16 can remain unfilled. Default inference uses all classes;
set `classes:=citizen,window` to run only classes whose bank entries are ready.
No class is silently skipped. All models receive the same first-K ordering,
example IDs, and manifest/image/mask hashes. New query predictions never enter the bank.

Legacy five-slot manifests remain readable. Expand one without changing its
existing examples using `python -m triffid_ugv_perception.bank expand --bank ugv_supports/PUI`.
The previous manifest is archived under `ugv_supports/PUI/history/`. The active bank
has already been expanded. Empty slots are paths awaiting real annotations.
Add slot 16 using `bank add --bank ugv_supports/PUI --class cone --slot 16 --image IMAGE --mask MASK`.
For PET-DINO, select `--shots 16` with `scripts/run_pet_baseline.sh` once the
selected classes have 16 distinct valid examples; increasing capacity does not
populate additional supports or guarantee improved accuracy.
For the current uneven bank, use PET-DINO's `--use-all-supports` to select each
class's populated examples up to 16 (currently 141 supports across 28 classes).
See the [updated-bank study and timing comparison](../../docs/PET_DINO_UPDATED_BANK_STUDY.md).
Tracking can be selected with `--tracker botsort` (default) or `--tracker udc`.
Add `--person-roles` to track person identity separately from citizen/responder
role predictions. See the [UDC integration and controlled comparison](../../docs/UDC_TRACKER_BENCHMARK.md).
UDC now defaults to a 35% confirmation proof gate; override it with
`--udc-proof-threshold`. See the [35% versus 50% overlay comparison](../../docs/UDC_THRESHOLD_035.md).
PET-DINO also supports `--pet-prompt-mode text` for exact class-name text
prompts instead of visual supports. Omit `--use-all-supports` in this mode.
See the [visual/text frame comparison and results](../../docs/PET_DINO_TEXT_COMPARISON.md).
The [foundation-model literature review](../../docs/research/FM_FEWSHOT_LITERATURE_20261105.md)
screens 44 detection, segmentation and adaptation families, including SPINO and
PASTEL, with a filterable CSV inventory and explicit multiclass/release caveats.

## SANSA setup and execution

Place the official SANSA checkout and checkpoints on the host:

- `ugv_models/SANSA/` — official checkout, reference revision `f562b5642940e08a57e3da2608b804b99af3763b`.
- `ugv_models/adapter_generalist.pth` — SANSA generalist/universal adapter.
- `ugv_models/sam2_hiera_large.pt` — SAM2 Hiera Large base weights.
- `ugv_supports/PUI/` — the independently labelled UGV bank.

The default UGV Docker image remains **ROS 2 Jazzy** and installs torch 2.7.1 /
torchvision 0.22.1 with CUDA 12.6 and the SANSA runtime dependencies. Compose mounts
`ugv_models` at `/ws/models` and the UGV bank read-only at `/ws/ugv_supports/PUI`.
Build the workspace after pulling even if an old `install/` already exists:

```bash
docker compose up -d --build
docker exec -it triffid_perception bash
cd /ws
colcon build --symlink-install --packages-select triffid_ugv_perception
source /ws/install/setup.bash
ros2 launch triffid_ugv_perception ugv_perception.launch.py \
  backend:=sansa support_bank:=/ws/ugv_supports/PUI shots:=1 classes:=citizen
```

Use `shots:=N` for any N from 1 to 16 once those entries for the selected classes are valid. Model
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

The ROS processing is: time-matched RGB + aligned depth → detection → valid depth
samples → median camera point → recorded optical/body TF → 3D extent → existing 3D
NMS → existing tracker → same ROS messages → geographic delivery.
Fence and wall use the existing longest-extent LineString construction. Every
other class exports its centre as a Point; UGV no longer emits Polygon features.

The ROS node now requires CameraInfo matching the RGB resolution/frame and a TF
from that optical frame to the robot frame. It matches RGB/depth headers within
50 ms by default and rejects resolution/frame mismatches and zero/saturated
65535 depth pixels. Missing calibration or TF produces no guessed 3D position.
Depth subsampling, 0.5 m cross-class 3D NMS and tracker confirmation/aging remain.
The live bridge now uses persistent UUID identities instead of geographic accumulation for API decisions. Valid GPS and absolute heading are still required
for reliable map placement; local calibration does not establish world accuracy.

The PUI model comparison setup and minimal future output policy are documented
in [MODEL_SETUP.md](../../docs/MODEL_SETUP.md). The newly prepared native model
environments have not run image inference or end-to-end comparison jobs yet.

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

## DE-ViT baseline

An offline 28-way, one-shot DE-ViT Small baseline is available with BoT-SORT updates on every sampled 1 Hz frame and a local creation-only object registry. See [setup, measured timings, outputs and known detection-quality failures](../../docs/DEVIT_BASELINE.md). It uses the released model unchanged and a separate legacy Python runtime; it does not change the default SANSA backend or enable API delivery.

## SINE baseline

The existing local SINE checkpoint remains available as a comparison baseline. See [SINE setup, results, mask exports and limitations](../../docs/SINE_BASELINE.md). Run `scripts/run_sine_baseline.sh` using the prepared Python environment.

## COSINE baseline (current offline default)

PET-DINO Visual-G is also available for comparison: see [PET-DINO results and limitations](../../docs/PET_DINO_BASELINE.md). Run `bash scripts/run_pet_baseline.sh --output archive/samples_legacy_20261106/pet_baseline/new_run --image-size 800 --max-size 1333 --precision float32 --confidence 0.35`.

The shared offline runner defaults to pretrained multi-scale COSINE with cached joint K-way visual/text prompts, instance masks, BoT-SORT and a persistent local registry. See [setup and results](../../docs/COSINE_BASELINE.md), [updated UGV supports](../../docs/UGV_SUPPORT_REFRESH_20261105.md), and [identity persistence limits](../../docs/PERSISTENT_DISCOVERIES.md). Run `bash scripts/run_cosine_baseline.sh --output archive/samples_legacy_20261106/cosine_baseline/new_run`. API delivery remains disabled; live ROS deployment has not been validated.

OPUS is available through `bash scripts/run_opus_baseline.sh` in its isolated
runtime. It supports joint K-way visual detection and native mixed text/visual
prompts with up to 16 available examples per class. See
[OPUS setup and evaluation](../../docs/OPUS_BASELINE.md) for commands, checkpoint
provenance and the raw-versus-tracked comparison. This backend outputs boxes;
it uses the existing tracker and local discovery registry.

## Persistent discoveries

The live bridge confirms objects across two frames, assigns a durable UUID and
queues one creation after fresh GPS and heading become available. Repeated
sightings update local state. API delivery, position-API polling, frame saving
and GeoJSON snapshots default to disabled. See [architecture, limits and backend
contract](../../docs/PERSISTENT_DISCOVERY_DELIVERY.md). Compose persists state in
`state/PUI/`; each mission/camera must use its own state directory and namespace.
The older offline JSON registry remains available for archived model runners.
