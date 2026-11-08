# YOLOE variant comparison — 7 November 2026

All three models processed the original rosbag's 65 RGB frames at 1 Hz, with
the same 29 classes, 200 visual supports (up to 16/class), 640-pixel target,
FP32 inference, 0.35 detection threshold and 0.10 tracking floor. Each class
uses the normalized mean of its visual support embeddings. No text prompts,
training, API requests or ground-truth metric evaluation were used.

All models use the same MASA-R50 tracker and persistent class-correction policy.
YOLOE-26 uses its NMS-free branch (`nms=False`), explicitly selected before
model fusion and checked during prediction. v8-L retains its original NMS
configuration. This compares complete model variants; changes from v8 to 26
also include weights, runtime and prediction heads, not only backbone size.

## Timings and event counts

RTX 3060 12 GiB; annotation helper stopped during measurement. Warm means
frames 2–65. Compute sums separately measured detector, GPU tracker and registry
times. It excludes support encoding, startup, image loading and overlay/video
rendering; it is not measured live end-to-end latency.

| Model | Detector mean | Tracker + registry mean | Compute mean | Raw accepted predictions | Confirmed UUIDs |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOE-v8-L | 63.9 ms | 140.6 ms | 204.4 ms | 382 | 27 |
| YOLOE-26-M | 46.0 ms | 128.3 ms | 174.2 ms | 248 | 15 |
| YOLOE-26-L | 44.8 ms | 127.1 ms | 171.9 ms | 158 | 11 |

The small M/L timing difference does not establish a reliable speed advantage.
Prediction and UUID counts are not recall, precision or duplicate-rate metrics.
All three fit the one-second compute budget in these warm frames.

## Visual check and outputs

At frame 15, v8-L detects the vehicle, cone, boots, extinguisher, chainsaw and
bag; 26-M reports two chainsaw boxes, and 26-L reports a chainsaw and bag.
The reviewed frame 50 shows different person/role predictions across models;
role confusion remains. These spot checks do not establish overall accuracy.
Retain v8-L as a baseline rather than assuming a newer checkpoint is better.

- `samples/YOLOE/original/yoloe26_comparison_v1/`
- `samples/YOLOE-26-M/original/class_correction_v1/`
- `samples/YOLOE-26-L/original/class_correction_v1/`

Each contains detections/tracking/exports videos, their overlays, timings,
observations, GeoJSON and audit. NEW and UPDATED retain the same established
semantics. Geometry remains null in this RGB-only experiment; image boxes are
GeoJSON properties. Previous v8-L results are preserved. Its fresh detector
predictions match the earlier run on 65/65 frames.

## Reproducibility and checks

YOLOE-26 runs in the separate `ugv_models/yoloe26_env` environment with Ultralytics
8.4.204. The original v8 environment is unchanged. Official pretrained segmentation
weights, download URLs, wheel checksum and checkpoint hashes are recorded in
`ugv_models/yoloe26_setup.json`; the support caches are distinct per variant and
runtime version. The adapter and model routing live in
`scripts/pui_comparison_detectors.py` and `scripts/run_pui_comparison.py`.

The 126-test suite passes. All nine videos decode to 65 frames, all 585 overlays
exist, and each GeoJSON matches its ledger with one feature per confirmed UUID.
Bank and dataset hashes agree across the three runs. API delivery remains off.
The annotation helper and Label Studio are running after the tests.

Official implementation reference: [Ultralytics YOLOE documentation](https://docs.ultralytics.com/models/yoloe).
