PUI detector setup
==================

The initial setup performed no image inference, support feature extraction, training or benchmarks.
The subsequent authorized five-model GPU run is documented in [PUI model tests](PUI_MODEL_TESTS_20261106.md).
Checkpoint compatibility checks construct/load models on CPU with model forward
calls blocked. This does not establish accuracy, latency or end-to-end adapter
correctness; those remain for the authorized comparison runs.

| Model | Source | Prepared variants |
|---|---|---|
| FSOD-VFM | [Official repository](https://github.com/Intellindust-AI-Lab/FSOD-VFM) | UPN-L + SAM2.1-L + DINOv2 ViT-L/14 |
| DINOv | [Official repository](https://github.com/UX-Decoder/DINOv) | Swin-T and Swin-L |
| YOLOE | [Official repository](https://github.com/THU-MIG/yoloe) | YOLOE-v8-S and YOLOE-v8-L segmentation checkpoints, visual prompts |

PUI supports are under `ugv_supports/PUI`: 200 labelled supports across 29
populated classes, up to 16 images per class. The bank manifest, masks, source
annotations and history moved together. Its SHA256 and every selected image/mask
fingerprint were unchanged by the directory move. The active bank has since been refreshed from UGV-PUI.zip; previous inputs remain in its history. Annotation-only model inputs are in `model_inputs/`:
the common manifest, native FSOD-VFM box supports and COCO mask annotations.
YOLOE receives tight support boxes derived from the annotated masks. DINOv is
the visual in-context prompting model, separate from the DINOv2 feature backbone.

Each model has an isolated environment under `ugv_models/`. Existing Torch
installations and compatible compiled MMCV CUDA kernels are shared to avoid
duplicating large packages. The Deformable-DETR extension API is connected to
MMCV through `ugv_models/msda_bridge`; numerical model execution remains untested
by request. SAM2's connected-components CUDA extension is built for this RTX
3060 (compute capability 8.6). The DINOv constructor's CUDA-only training indices
are held on CPU only during preflight. One obsolete loss-only checkpoint buffer,
`criterion.empty_weight`, is excluded; detector weights load strictly.

Import-only checks (no model inference):

```sh
bash scripts/model_python.sh FSOD-VFM
bash scripts/model_python.sh DINOv
bash scripts/model_python.sh YOLOE
```

Optional CPU checkpoint-structure checks also perform no forward passes:

```sh
CUDA_VISIBLE_DEVICES= bash scripts/model_python.sh DINOv scripts/check_detector_setup.py DINOv --load-cpu --variant large
```

`ugv_models/weights/assets.json` records downloaded checkpoint URLs, sizes and
SHA256 hashes. `ugv_models/setup_status/` records source revisions, runtime
versions and compatibility checks. FSOD-VFM reuses the already available DINOv2
weights in `D:\Models\Checkpoints`, through a link rather than another copy.

Output and data layout
----------------------

`configs/pui_models.json` is the comparison configuration. New experiment outputs:

```text
samples/<MODEL>/<DATASET>/<RUN>/
  detections.mp4
  tracking.mp4
  exports.mp4
  overlays/{detections,tracking,exports}/*.jpg
  timings.csv
  observations.jsonl
  detections.geojson
  audit.json
```

The audit retains reproducibility settings, support counts, thresholds, timing
summaries and limitations. It must not claim accuracy or ID-switch metrics
without ground-truth labels. Cold initialization and support preparation are
reported separately from warm inference. Raw tensor dumps and HTML reports are
excluded from this layout. No timing files or experiment results were generated
during setup.

Shared RGB frames and bags are under `data/PUI`. Earlier experiments are under
`archive/samples_legacy_20261106`; the move journal is `archive/PUI_MIGRATION.json`.
No previous data was deleted, so archiving alone does not free its disk space.
Legacy reproduction scripts reference the archive; they retain their old
verbose result formats and are not the entrypoints for the new comparison.

The previous combined videos were replaced by separate detection, tracking and
GeoJSON discovery views. The third video shows each newly confirmed UUID
once, matching detections.geojson. Its features have null geometry in the
RGB-only run. World-coordinate validation still applies separately to the API queue.
`observations.jsonl` retains only compact detections/IDs and diagnostic overlap
records, excluding masks and appearance tensors.
