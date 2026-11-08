# PUI data preparation

The working checkout is `refactor/ugv-sansa-modular`.

## Annotation bank

`ugv_supports/source_annotations` contains the supplied runner-compatible ZIP contents. Its semantic masks use the original 63 class IDs, with 255 reserved for void. The bank references these original images and indexed masks directly, using each class ID as `foreground_value`. No predictions are used as support annotations.

`scripts/import_pui_bank.py` checks the ontology and image/mask dimensions, deduplicates image content per class, prefers ground photographs before drone images, and fills up to five slots per class. It refuses to overwrite a bank. The generated `ugv_supports/import_report.json` records sources and hashes.

Result: 97 populated slots, 28 classes ready for one-shot inference, 10 ready for five-shot inference. The other 35 classes lack annotations in this ZIP; no examples were fabricated. The full 63-class bank therefore cannot pass five-shot validation. Selection of all 28 available classes at one shot and validation of the 10 complete classes both passed.

## Recordings and RGB dataset

The MCAP recording starting at `2026_10_20-16_13_20` contains UGV status/GPS; the SQLite recording starting at `2026_10_20-16_13_25` contains RGB/aligned depth. They overlap by approximately 63 seconds. Both should be replayed in one player or merged by recorded timestamps, rather than launching two independent player clocks.

The SQLite filenames contain download suffixes that differ from `metadata.yaml`. Local staging under `samples/pui_run/bags` corrects those references using symlinks to the original read-only files. Original recordings are not modified. The test container mounts the dataset directory at `/bags`; staging symlinks are intended for that container mount layout.

`scripts/extract_rosbag_rgb.py` exports all raw color frames, independently of inference and playback speed, converting YUY2 to BGR for PNG encoding. `samples/pui_rgb_frames` contains 1,190 PNGs at 1280×720, plus `frames.jsonl` with source bag/topic, original encoding, dimensions, frame ID and both bag and header timestamps. All 1,190 indexed files exist and filenames are unique. These images have no query annotations yet.

`scripts/merge_pui_bags.py` retains RGB, aligned depth, UGV status and `/fix`, merges both readers in recorded timestamp order, and writes `samples/pui_run/paired_bag`. Counts are recorded separately. `scripts/test_pui_replay.py` uses that merged bag when present, otherwise uses a single player with both inputs. It starts perception with available supports, records outputs, disables external position API and uploads, and saves command lines and logs. Environment overrides: `PUI_RUN`, `PUI_DEVICE`, `PUI_CLASSES`. Output directories must be new for recording. Run it in the prepared ROS container, with `/ws`, `/models` and `/bags` mounted and workspace setup sourced.

The custom `UnitreeStatus` and nested `GNSSPosition` definitions were recovered from the MCAP schema and built under `samples/pui_run/interfaces`; no approximate message layout is substituted. `scripts/pui_replay_qos.yaml` makes replayed `/fix` compatible with the existing bridge subscription.

## Validation limits

The UGV package builds, the recovered message packages build, and all 16 existing UGV tests pass. SANSA loads with the local checkpoints. These do not establish detector quality or throughput. Initial multi-input playback attempts appeared stalled; logs are retained. The merged-bag citizen CPU smoke test recorded 646 UGV status messages, all 64 GPS messages, and five completed Detection3DArray messages. All five arrays were empty. Their output intervals were approximately 16 seconds; this is an observation from this constrained CPU run, not a deployment hardware benchmark. Full 28-class GPU throughput remains unverified. Empty arrays do not establish positive detection quality.

Recorded calibration is `fx=910.3245849609375, fy=910.1997680664062, cx=645.7958984375, cy=366.5615539550781` at 1280×720. The branch uses different fixed 640×480 calibration. Correcting the node to consume recorded CameraInfo is required before trusting metric localization or geographic duplicate thresholds. This preparation does not change the inherited geometry implementation.

For the proposed efficient detection and creation-only delivery behavior, see `PUI_EFFICIENT_FEWSHOT.md`. That document is a design proposal; the production event registry and inference scheduler are not implemented by these data-preparation scripts.

## Completed positive smoke test

The one-shot `civilian vehicle` CPU test completed paired playback using the merged bag, with external APIs disabled. It produced a confirmed vehicle track (ID 3), a saved overlay visually locating the van, and a local GeoJSON snapshot. Results and per-frame counts are in `samples/pui_run_vehicle/results.json`; the overlay is `samples/pui_run_vehicle/overlays/detection_1792505667_836756348.png`. This is a positive pipeline smoke test, not an accuracy benchmark. Geographic output remains provisional because of the calibration mismatch above. The new-object-only event registry and efficient detector scheduling remain proposed work.
