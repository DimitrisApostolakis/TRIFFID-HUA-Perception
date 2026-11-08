# Local SINE baseline

The support bank was refreshed on 5 November 2026 to exclude UAV tasks. The
original results below retain their original 28-class support fingerprints.
The new reference run is `samples/sine_baseline/ugv_only_20261105/`: 27-way
1-shot, 65 identical source frames, 215 masks, 15 candidate UUIDs, mean complete
frame time 0.439 s, p95 0.489 s, and no frames over one second. Its export audit
passed. The registry is under `samples/pui_registry/sine_v1.json` with a durable
outbox beside it. Neither run has labelled-query accuracy or physical identity
ground truth. See [bank refresh](UGV_SUPPORT_REFRESH_20261105.md) and
[persistent identity limits](PERSISTENT_DISCOVERIES.md).

The primary offline baseline now uses **SINE** from the existing checkout at `/mnt/d/Models/ThirdParty/SINE` and checkpoint `/mnt/d/Models/Checkpoints/SINE/pytorch_model.bin`. No download or fine-tuning was needed. It uses SINE's published few-shot instance-segmentation encoder, decoder, reference preprocessing and class-query integration.

## Completed run

Outputs: `samples/sine_baseline/final/`.

| Setting or result | Value |
|---|---|
| Query frames | The same 65 recorded RGB frames used by DE-ViT, sampled at 1 Hz |
| Support configuration | 28 classes, 1 shot per class |
| Model | Pretrained SINE, DINOv2 Large, 200 instance queries |
| Input | Long edge 644, normalized RGB, zero-padded to 644×644 |
| Precision | FP16 model and CUDA autocast; CPU reference-query integration in FP32 |
| Score threshold | 0.7, using the official instance-head filtering |
| Duplicate-query filtering | Class-agnostic mask IoU NMS at 0.5 |
| Mean inference time | 0.309 seconds/frame |
| Mean complete frame time | 0.445 seconds/frame |
| Mean after first frame | 0.398 seconds/frame |
| Complete-frame 95th percentile | 0.444 seconds |
| First / slowest frame | 3.462 seconds; the only frame over one second |
| Peak allocated CUDA memory | 2,225,754,112 bytes, approximately 2.07 GiB |
| Model/support preparation | 20.31 seconds, measured separately; warm support-query cache |
| Tracker / query encoder calls | 65 / 65 |
| Exported instance masks | 216 |
| Tracked observations | 114 |
| Candidate creation events | 14 distinct registry IDs, each emitted once |
| External API | Disabled |

Execution processes every selected frame without pacing; this is a replay of 1 Hz input, not a live hard-deadline test. Initialization and first-frame warmup need to complete before a deadline-sensitive stream starts. These measurements are for the desktop RTX 3060, not the Jetson.

The previous DE-ViT run averaged 0.660 seconds after its first frame. This comparison uses identical source frames and support identities, but **different model configurations**: DE-ViT used FP32 and a short-edge-640 / max-edge-1024 resize, while SINE uses FP16 and long-edge-644 square padding. Confidence thresholds and post-processing also differ because the scores have different meanings. The timings do not isolate architecture speed, and detection counts do not measure accuracy.

## What to inspect

- `overlays/0056.jpg`: an example of a first-responder-labelled mask aligned with a visible person.
- `overlays/0001.jpg`: the early cone mask and track.
- `summary.json`: configuration, checkpoint/support fingerprints and timings.
- `frames.jsonl`: all detections, tracker observations, creation events and source timestamps.
- `masks/NNNN.npz`: losslessly compressed instance masks; detections reference the file and mask index.
- `new_objects/<UUID>.json`: local creation events.
- `object_registry.json`: persistent candidate identities and appearance descriptors.
- `audit.json`: frame equivalence, mask/box checks and event-ID uniqueness checks.

To reconstruct masks:

```python
with np.load('masks/0056.npz') as data:
    shape = tuple(data['shape'])  # instances, height, width
    masks = np.unpackbits(data['packed_masks'], axis=-1, count=shape[-1]).astype(bool)
```

The audit checked all 216 exported masks against their original-resolution boxes and confirmed that the SINE and DE-ViT runs selected exactly the same frames. All 14 emitted IDs are distinct. Boxes use exclusive upper edges derived from masks, matching the common detector contract.

## Detection and identity limits

Manual inspection shows useful cone and person masks, but vehicle detections remain inconsistent and vehicles are missed in early views. No labelled-query mAP, mask IoU, identity-switch rate or duplicate-physical-object rate has been measured.

The 14 events represent registry identities, not 14 verified physical objects. They include three cone identities and six first-responder identities; these counts do not establish how many physical objects were present. Appearance and image overlap can rebind a lost track, but substantial camera motion, class changes and similar-looking objects can still defeat this RGB-only association. The registry must not be treated as a guarantee of physical deduplication. Capture-time pose, correct depth calibration and a world-frame object map remain relevant for strengthening that behavior.

The first smoke run exposed duplicate instance queries on the same cone. Mask IoU suppression removes strongly overlapping hypotheses before tracking. This is post-processing of SINE outputs, not a change to its learned model. As with any NMS, it can suppress genuinely overlapping instances; nearby nonoverlapping objects are explicitly tested.

## Implementation and compatibility

`models/sine.py` loads every encoder and decoder tensor strictly. It excludes only `criterion.empty_weight`, which belongs to the checkpoint's training loss and is absent from the official inference model. It constructs the same DINOv2 encoder, neck and MFormer decoder specified by `inference_fsod/configs/Base-SINE.yaml`.

The installed Detectron2 predates `get_norm('LN')`. During decoder construction, the adapter fills that dependency gap with **SINE's own LayerNorm2d**, then restores the construction hook. It has the same channel normalization and epsilon (1e-6) as modern Detectron2 LayerNorm. A numerical equivalence test and strict checkpoint loading validate this compatibility path. No local third-party source or weights were edited. The prepared runtime is Python 3.10 / PyTorch 1.13.1+cu117, with workspace-local Pillow/BoT-SORT dependencies already prepared for the preceding baseline; this differs from the authors' published PyTorch 2.0.1 training environment.

Support images are grouped so each unique image is encoded once. The official `do_preprocess` and `integrate_queries` methods build cached ID and class queries. The cache includes support hashes, checkpoint hash, source revision, precision and image size. Every query frame uses one encoder pass for all selected classes. The resulting neck features are reused for mask-pooled appearance descriptors, adding no separate ReID model.

BoT-SORT updates on every selected frame, including empty detections, with class-aware matching and duplicate removal. The existing confirmation and registry policy emits each known UUID only once into the local outbox. Remote delivery is not implemented by this runner; the legacy ROS API bridge must not be used as a substitute for a creation-event sender. Live ROS SINE deployment is not validated by this offline run.

## Run

```bash
SINE_PYTHON=/home/oracle/miniforge3/envs/fewshot-dev/bin/python \
  scripts/run_sine_baseline.sh --output samples/sine_baseline/new_run
```

Use a new output directory. The shared Python runner `scripts/run_fewshot_baseline.py` defaults to SINE. The old DE-ViT entrypoint explicitly selects DE-ViT for reproducibility.

The class set comes from the current bank, not a hardcoded 28-class list. `--classes 'civilian vehicle,cone'` selects a subset; `--shots 5` with no explicit class list selects the ten currently complete five-shot classes. Explicit incomplete classes fail validation. `--checkpoint`, `--repo`, `--precision float32`, `--image-size`, `--confidence` and `--mask-nms-iou` are available. CPU execution is selected explicitly with `--device cpu`; it uses FP32 and is not a silent fallback.

Image size is rounded up to a multiple of 14 (640 becomes 644). The adapter follows the official mapper's order: RGB normalization, then zero-padding of the normalized tensor. All query predictions remain separate from the support bank.

Provenance: source revision `5fb537fd576906e340560aaaf3675da3e98f1181`; checkpoint SHA256 `80b72bd9ec906f334c74c88bc77de26593a1781ca22f217fe4df09a8c389fb5a`.

Validation: 30 passing UGV tests, including mask-derived boxes, original ontology IDs, RGB normalization/padding, duplicate mask suppression, neighbouring instances, LayerNorm equivalence, tracking and registry tests. The completed GPU run and export audit provide integration checks; they do not establish accuracy or physical identity correctness.
