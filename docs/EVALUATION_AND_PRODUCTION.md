# Frozen PUI evaluation and three-day delivery plan

The active scope is in `configs/evaluation_suite.json`: five UAV methods (FoRIS, INSID3, FROST, RSRM, FSSDINO) and eleven previously tested UGV adapters. Other requested UAV methods are deferred by the user's latest instruction. RSRM and FSSDINO use the available DINOv3-L weights with explicit variant labels, as authorized. This is not a reproduction of their released ViT-B configuration.

Completed comparison: [results](../samples/evaluation/REPORT.md). All five UAV and eleven UGV methods have successful results, with corrected RSRM/COSINE retries selected explicitly. Verification independently recomputed UAV mIoU from 60 saved binary masks, decoded all ten UAV video frames, checked the shared DINOv3 checkpoint hash and validated both query outputs for all eleven UGV models. Evidence: `samples/evaluation/verification.json`. CPU validation: **216 tests passed** with the documented compatible runtime and import mode.

## Data and scoring

**UAV:** the existing frozen `data/UAV-PUI/holdout_1_3/split.json` selects saved tasks 1 and 3, annotations 17 and 18. Each method uses the same one support for each of six classes. Query tasks and exact image-content duplicates are excluded. Primary mIoU pools class pixel counts within the union of saved brush annotations before averaging classes. Full-frame IoU assumes unlabeled pixels are background and is secondary. Window/furniture box F1 derives predicted boxes from connected mask components; it is not native instance AP. RSRM uses its 480-square default, FSSDINO its 512-square default, and FoRIS/INSID3/FROST 1024. All masks are restored to query dimensions. Resolutions and native decision rules differ; this is an operational comparison, not a controlled architecture ablation.

**UGV:** Label Studio project `ROSBAG-PUI` (7), saved tasks 49/50 and annotations 38/39. A consistent read-only database snapshot was exported under `data/ROSBAG-PUI/snapshot_20261107/`, with archive SHA256 `10a4713e9198f1cf73b1965ced0d410db21bd61b0209ff9eec37f2d4c13cca48`. It contains two original images, 37 boxes and 34 valid brush masks, plus one mask recovered from an RLE-only rectangle. One coordinate-only brush record is retained in source JSON and excluded from mask targets. No masks are invented from boxes.

`data/ROSBAG-PUI/evaluation_v2/` is a separate frozen support manifest and split. The active UGV bank is unchanged. Supports are checked against query decoded RGB pixels, including identical images with different encodings. No support was excluded in this snapshot. Other views of the same physical objects may remain. Preserve the fixed slot capacity: nonexistent reserved slots are never supports. The earlier `evaluation/` manifest had incorrect shortened slot arrays and is superseded by v2; corresponding failed logs remain for audit.

UGV metrics are COCO bbox AP at the adapters' existing fixed score thresholds, class-aware bbox precision/recall/F1 at IoU 0.5, and semantic union IoU within the saved brush region for methods returning masks. AP is truncated by the native detector acceptance thresholds and is labelled accordingly. Unsupported helmet annotations are reported separately; helmet is not silently counted as a supported class. Scores cannot be interpreted as exhaustive scene or independent-site accuracy. Human role IDs remain separate during detector scoring.

All methods use the same per-platform query data. The UGV policies preserve previously tested native supports (all available for PET/OPUS and comparison adapters; one-shot for DE-ViT/SINE/COSINE/SANSA). This is explicit in model metadata. Maskless detectors are not given synthetic rectangular masks. Isolated image evaluation does not test temporal association or re-ID.

## Run and artifacts

```bash
# Uses exactly the five active UAV methods, in separate processes, sequentially.
/home/oracle/miniforge3/envs/fewshot-dev/bin/python scripts/run_evaluation_suite.py \
  --platform uav --run NEW_UAV_RUN

# Uses all eleven previously tested UGV adapters on ROSBAG-PUI.
/home/oracle/miniforge3/envs/fewshot-dev/bin/python scripts/run_evaluation_suite.py \
  --platform ugv --run NEW_UGV_RUN

# Subset / fresh output. Existing completed and partial outputs are preserved.
bash scripts/run_uav_holdout.sh rsrm --output samples/evaluation/uav/NEW_RUN/rsrm
```

The suite stores status, complete stdout/stderr logs, environment freezes and per-model metrics under `samples/evaluation/{platform}/{run}/`. Models run in isolated prepared Python environments. New invocations share an advisory GPU lock; `--after-status` explicitly queues behind an older suite that predates the lock. Other GPU services must be considered when interpreting latency. No target hardware throughput claim follows from desktop RTX 3060 timings.

UAV outputs include restored masks, GT/prediction overlays, comparison video, per-class metrics and timings. New runs record input/checkpoint hashes, upstream source revision, local adapter hashes, Python/Torch/CUDA/GPU, seed and environment freeze. UGV comparison adapters already expose weight/support/revision metadata. Run names are immutable; failed attempts are retained and retried under a fresh name. `status=complete` requires successful exit plus metrics. Preparation, cold query and second query timings are separate; two frames do not support reliable p95 latency estimates.

RSRM's added inference dependencies are installed locally, not into a shared environment:

```bash
/home/oracle/miniforge3/envs/frost/bin/python -m pip install --no-deps \
  --target uav_models/runtime/dino fast-pytorch-kmeans==0.2.2 psutil==7.2.2
```

The frozen UGV snapshot also has `reproducibility_snapshot.json` and explicit environment snapshots captured after its initial run. These are labelled as post-run snapshots. The corrected COSINE environment keeps the complete installed Detectron2 package; only DE-ViT prepends its specialized fork. Its earlier missing-PointRend failure is retained in the initial suite log.

The original INSID3 metrics inherited a wrong refinement-description string from FROST. Its `metadata_corrections.json` records the correction to bilinear refinement and the original metrics hash. The model constructor and scores were correct; historical metrics were preserved. Future runs emit the corrected metadata.

CPU checks use the compatible Pillow/runtime path for the legacy Detectron2 tests:

```bash
PYTHONPATH=ugv_models/devit_runtime:src/triffid_uav_perception:src/triffid_ugv_perception \
  /home/oracle/miniforge3/envs/fewshot-dev/bin/python -m pytest -q --import-mode=importlib \
  src/triffid_uav_perception/tests src/triffid_ugv_perception/tests
```

Raw data, weights, vendor repositories, caches and rendered samples are ignored by Git. Preserve them with the source revision, manifests and environment records when transferring the experiment; a code-only checkout is not a complete runnable release.

## Three-day readiness plan

| Day | Deliverable | Evidence required before acceptance |
|---|---|---|
| 1 | Frozen five-model UAV / full UGV comparison and ranked accuracy/latency report | Identical splits, leakage checks, valid saved annotations, model/weight hashes, completed runs and examined failure cases |
| 2 | Selected model plus identity pipeline in the live ROS path | Explicit appearance/association configuration, synchronized RGB-D/pose/calibration, continuous annotated identity replay, HOTA/IDF1 and duplicate UUID report |
| 3 | Release candidate on production hardware | Reproducible container/assets, cold-start and sustained latency, memory/disk bounds, input drop/reconnect behavior, restart recovery and idempotent outbox delivery |

The live ROS node still selects legacy ByteTracker, while offline comparisons use MASA-R50. Resolving and validating that gap is a release requirement. See `docs/REID_SYSTEM.md` for the current implementation and research. The user has fixed the target at the current RTX 3060 system and **at least one processed frame per second**. Validate this on the complete sustained live pipeline after warmup, including association, projection and delivery, rather than using detector-only times. Accuracy and identity acceptance thresholds still need definition; the current two-image diagnostic cannot establish production readiness. The three-day date is a target, not a passed release gate.
