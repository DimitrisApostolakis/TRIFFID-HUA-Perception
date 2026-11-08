# DE-ViT baseline on the PUI recording

The current offline baseline is [SINE](SINE_BASELINE.md). This report preserves the completed DE-ViT comparison.

The completed baseline uses the **released DE-ViT Small detector**, not a new model or a custom prototype classifier. No fine-tuning was performed. Its trained region-propagation detector consumes class representations extracted from our labelled supports using the authors' reference-mask pooling procedure.

## Completed run

Results: `samples/devit_baseline/full/`.

| Setting or measurement | Value |
|---|---|
| Input | First RGB frame in each recorded second; 65 selected frames |
| Classes / supports | All 28 currently available classes, 1 shot per class |
| Model | DE-ViT ViT-S/14, released few-shot checkpoint |
| Hardware / runtime | RTX 3060; PyTorch 1.13.1+cu117; Python 3.10 |
| Query resizing | Short edge 640, maximum edge 1024 |
| Region proposal budget | 100; official class shortlist of up to 10 per proposal |
| Filtering | Score 0.45, minimum box area 400 px², class-agnostic NMS IoU 0.3 |
| Tracking | BoT-SORT, appearance matching, camera-motion compensation, 1 Hz |
| Tracker updates | 65, including frames with no detections |
| Mean detector time | 0.545 seconds/frame |
| Mean complete frame time | 0.715 seconds, including image reading, tracking, registry and overlay writing |
| Mean after the first frame | 0.660 seconds/frame |
| Complete-frame 95th percentile | 0.695 seconds |
| First / slowest frame | 4.231 seconds; the only frame over one second |
| Peak allocated PyTorch CUDA memory | 2,639,292,416 bytes, approximately 2.46 GiB |
| Detections / tracked observations | 1,642 / 278 |
| Candidate object creation events | 48 distinct registry IDs, each emitted once |
| External API | Disabled; this runner has no API sender |

This is an **unpaced replay of frames sampled at 1 Hz**, processing every selected frame. The measured steady frame times support proceeding with a 1 Hz experiment on this desktop configuration, but do not establish a hard real-time deadline or Jetson performance. Model/support preparation and first-frame initialization must finish before a live deadline-sensitive stream starts. The run used a warm support cache; model preparation took 11.52 seconds and is reported separately.

## Detection quality assessment

The current one-shot baseline is **not suitable for deployment**. Manual inspection shows substantial false positives and wrong classes: the white van is labelled as fence/wall in early frames and as dry tree in another view. Correct vehicle detections also occur, but the confidence scores are not calibrated on this dataset.

The 48 registry identities are **not 48 verified physical objects**. They include incorrect detections, and physical duplicates or missed identities have not been measured against ground truth. The audit verifies only that no stable registry ID produces a second creation event. This distinction matters: suppressing duplicate event IDs cannot repair misclassification or guarantee physical object identity.

The first smoke run omitted the authors' demo post-processing and produced many overlapping class hypotheses. The final run uses the published demo's 0.45 threshold, 400 px² area cutoff, and class-agnostic NMS at 0.3. This can suppress overlapping real objects too; it is recorded as a baseline choice rather than an accuracy claim. Neither the model architecture nor the learned weights were modified.

## Inspect the artifacts

- `summary.json`: model, support hashes, configuration, latency and counts.
- `audit.json`: event-ID uniqueness check and manual quality assessment.
- `frames.jsonl`: per-frame detections, tracks, new events and timings.
- `selected_frames.json`: exact source images and recorded timestamps.
- `overlays/0000.jpg` through `overlays/0064.jpg`: grey detection boxes and green tracked boxes, class names, track IDs and registry IDs. `confirming` means no creation event yet.
- `new_objects/<UUID>.json`: one immutable creation event per registry object.
- `object_registry.json`: persistent identity records, latest boxes and appearance descriptors.
- `class_availability.json`: selected and unavailable classes; missing classes are not fabricated.

## Implementation

`models/devit.py` registers `backend=devit` with the common detector factory. It uses the official vendored Detectron2 and validates all learned checkpoint tensors. Only the checkpoint's COCO class-weight **data buffers** are replaced by the selected UGV support buffers. Predictions map contiguous detector indices back to the original TRIFFID IDs. Boxes use original-image xyxy coordinates; `mask=None` because this released Small baseline has its mask head disabled.

Supports are encoded once per unique image and cached under `ugv_models/devit_support_cache/`, keyed by the selected bank fingerprints, checkpoint hash, source revision and preprocessing specification. New predictions never become supports. Changing the selected classes/support images invalidates the cache. The model searches the selected class set on each frame without a separate complete support/query episode for each class.

`tracking/botsort.py` wraps Ultralytics 8.3.200 BoT-SORT. It reuses pooled DINOv2 query-region features for appearance matching, adding no second neural network. Association and duplicate removal are class-aware. The upstream buffer is scaled to retain lost tracks for ten seconds at 1 Hz. This wrapper is used by the offline baseline; the legacy ROS node's tracker is not replaced.

`discoveries.py` confirms candidates across two consecutive observed frames, assigns stable UUIDs, and emits creation events only for unseen identities. Known track IDs update existing records. After a track-ID change, same-class image overlap and appearance similarity can rebind the previous registry identity. Simultaneously visible tracks cannot both claim the same identity during that association step. Saved tracker-number aliases are not reused across process restarts. Atomic per-UUID event files allow missing local event files to be recovered from the registry without rewriting existing events.

This RGB-only registry requires overlap with the last image box during reidentification. Arbitrary camera motion, visually similar objects, class changes and long disappearances can still cause mistaken identity decisions. Reliable world-frame positions and capture-time robot pose would be needed to strengthen mapping identity. The existing fixed camera-calibration mismatch must be corrected before using the inherited geographic coordinates for this purpose. The baseline does not use that calibration.

Creation files are a **local outbox**, not acknowledged remote deliveries. The legacy ROS GeoJSON bridge/API upsert policy remains unchanged. A future API sender must consume creation events with acknowledgements/idempotency rather than repeatedly sending frame detections or accumulated snapshots.

## Reproduce locally

The prepared runtime uses `/home/oracle/miniforge3/envs/fewshot-dev/bin/python`. Extra packages live in `ugv_models/devit_runtime`; the shared environment was not modified. The vendored Detectron2 reuses a matching existing compiled `_C` extension through a local symlink. On another machine, use a separate Python 3.10 environment, install `src/triffid_ugv_perception/requirements-devit.txt`, and build/install the official DE-ViT fork of Detectron2. This old runtime is separate from the ROS/SANSA environment; live ROS deployment is not validated here.

```bash
DEVIT_PYTHON=/home/oracle/miniforge3/envs/fewshot-dev/bin/python \
  scripts/run_devit_baseline.sh \
  --output samples/devit_baseline/new_run
```

The output directory must be new. Defaults discover all bank classes ready for one shot. To select a subset, pass `--classes 'civilian vehicle,cone'`; at least two classes are required by this baseline adapter. `--shots 5` selects the currently complete five-shot classes when no class list is given. Explicit requested classes with missing supports fail validation. `--device cpu` is an explicit alternative; there is no silent backend/device fallback.

The same standalone detector adapter is registered for future integration, but its legacy dependencies are not installed into the Jazzy image. The successful test above uses the extracted RGB frames, not a live ROS DE-ViT node.

## Model provenance and checks

- Official source: https://github.com/mlzxy/devit, revision `9cab390fbe78b3bc8d62c224411dce1c71056cf9`.
- Official weights: https://rutgers.box.com/s/2lco6ab66pn3ufq6rh4gmyfzg9vfkm23.
- Files: `weights/trained/few-shot/vits_0089999.pth` and `weights/initial/background/background_prototypes.vits14.pth` from that release. Stored locally under `ugv_models/devit/weights/` with a size/hash/source manifest.
- Detector SHA256: `a26e742cee1a01f3bf40356a82900b574761753da50be09ed148c4b5ac4e6ed7`.
- Official support procedure: `demo/build_prototypes.ipynb`.
- Official display filtering: `demo/demo.py`.

The UGV suite has 26 passing tests, including the existing geometry/model/ROS-double checks and new tests for 1 Hz sampling, original class-ID mapping, empty-frame tracker updates, same-class distinct objects, class-aware tracking, one-time creation events, track reacquisition, and restart alias handling. Detection and physical identity accuracy still require manually labelled query data.
