# UAV SANSA baseline — 7 November 2026

Input: `D:/Datasets/PUI/PUI - Raw_data/PUI DRONE/DJI_202610191547_004/DJI_20261019155500_0001_S.MP4`
with its matching SRT. The clip is 1920x1080, 29.97 FPS, 1578 frames (52.65 s).
The original latest UAV orbit selector chooses eight distinct views with angular
errors below 0.51 degrees. This is an eight-view baseline, not full-video-frame
inference or a continuous 1 Hz run. The preview orders views by timestamp at 1 FPS.

Base UAV code: `285c68d`. The UAV-only files were imported into this workspace;
the original worktree remains untouched. The local ontology preserves human roles
as separate Point classes: first responder 6, citizen 23, military personnel 43.
The PUI bank now has 27 supports over 10 populated classes. It retains five-slot
storage; the former person bank and old UAV files are backed up.

This video run uses the four available Point classes: **first responder, window,
citizen, fire truck**. Each uses its first labelled support image (one-shot).
Polygon classes remain in the separate supplied-top-view workflow. No support
exists for military personnel or the other unselected Point classes.

Model: SANSA Generalist (`adapter_generalist.pth`) plus original SAM2 Hiera-Large
(`sam2_hiera_large.pt`), official source revision `f562b5642940e08a57e3da2608b804b99af3763b`.
The original latest UAV SANSA adapter is unchanged. It runs independent per-class
episodes, re-encoding support/query for each class. This is not native joint K-way
inference. FP32, 640x640 external resize (SAM2 internally uses 1024), mask threshold
0.5, native minimum component area 1 pixel, no extra score threshold.

## Timing

RTX 3060; Torch 2.7.1+cu126. Annotation helper stopped during measurement and
restored afterwards. CUDA synchronization brackets every measurement.

| Measurement | Time |
| --- | ---: |
| Model + support preparation | 11.94 s |
| First view, all four classes + components | 4.43 s |
| Mean, all eight views | 3.87 s |
| Warm mean, views 2–8 | 3.79 s |
| p95 including first view | 4.21 s |

Frame inference includes prediction, full-resolution probability transfer/resize,
and connected-component conversion. CSVs separately record class inference,
loading and total time including rendering. Peak allocated GPU memory: 2.05 GiB.
This configuration exceeds a one-second-per-frame budget.

## Outputs and validation

`samples/SANSA/UAV/DJI_20261019155500_0001_S/one_shot_640/` contains:

- `detections.mp4`: eight-view preview, not the original 52-second video.
- `overlays/`: eight original-resolution labelled mask/box/centroid JPGs.
- `point_detections.json`: native UAV pixel observations with telemetry and mask RLE.
- `timings.csv`, `class_timings.csv`, `audit.json`.

All eight frames decode, all 189 observation masks/boxes/centroids validate, and
all 32 class timings exist. Observation IDs are local per view, not track IDs.
Scores are mean foreground probabilities, not calibrated detection confidence.
The inspected views show false positives, including fire-truck responses on
building regions; component counts are not accuracy or true-object counts.
No GT accuracy metrics, tracking, duplicate suppression, world projection, HTML
or API calls were produced. The updated UAV suite passes 67 tests. UGV bank
SHA256 remains unchanged (`97768b562b10aea2b5fbed280a00af86bc13783a028419efb6f76202095a473d`).

Reproduce CPU selection with the UAV CLI `select` command and inference with
`bash scripts/run_uav_sansa_test.sh` after selecting a fresh output path.
