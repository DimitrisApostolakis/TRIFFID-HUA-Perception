# PUI model tests — 2026-11-06

All five models processed the original rosbag at 1 Hz: 65 frames, 1280×720. Each model now has three separate 65-frame videos (detections, tracking and GeoJSON discovery exports), 195 stage-specific JPG overlays, timings.csv, observations.jsonl, detections.geojson and audit.json. The old combined video and flat overlay images were replaced after validating the replacements. Every video was decoded to verify its frame count. No accuracy evaluation, training, HTML, prediction tensor dumps or API publishing was performed.

The new UGV-PUI.zip export contains 40 annotated UGV images. The common bank selects 200 distinct image/class supports across 29 classes, at most 16 per class. Original per-class binary masks retain overlapping labels; boxes are not converted into masks. The previous bank is preserved in ugv_supports/PUI/history. Helmet is absent from the new export and is therefore not a selected class.

Archive SHA256: 8fe695c334bf (prefix; full value in the bank import report). Bank SHA256: 97768b562b10aea2b5fbed280a00af86bc13783a028419efb6f76202095a473d.

## Timings

RTX 3060 12 GiB; one model at a time. Warm detector timings exclude the first query frame and initialization/support preparation. MASA-R50 is shared across methods, with person-role grouping and the persistent discovery registry. Detector acceptance is 0.35; existing tracks may recover from 0.10. Native confidence scores are not calibrated or directly equivalent across models.

| Model | Warm detector mean | Warm detector p95 | First query | Warm tracker mean |
|---|---:|---:|---:|---:|
| PET-DINO | 251.4 ms | 270.6 ms | 3049.8 ms | 138.6 ms |
| OPUS | 119.6 ms | 132.4 ms | 748.5 ms | 113.7 ms |
| FSOD-VFM | 1924.4 ms | 1964.3 ms | 3108.9 ms | 157.7 ms |
| DINOv | 183.1 ms | 195.8 ms | 3661.5 ms | 162.9 ms |
| YOLOE | 56.5 ms | 66.1 ms | 3519.2 ms | 126.6 ms |

FSOD-VFM exceeds the 1-second budget before tracking in this configuration. Detector and tracker execute in separate compatible Python environments. The CSV staged_sum_s combines the two recorded processing stages, including duplicated image I/O and rendering; it is not a measured live end-to-end latency. All selected frames were processed unpaced, without dropping slow frames. Initialization, source revisions, checkpoint hashes, support counts, cache use and timing summaries are in each audit.

## Model configurations

- PET-DINO: trained Swin-T checkpoint, Visual-G, short-side target 800/max-side 1333, all selected supports. Its existing compatible huggingface-hub runtime is included explicitly.
- OPUS: released DINOv3 ConvNeXt-B checkpoint, native visual.G.16, 640 input policy, all selected supports.
- FSOD-VFM: DINOv2 ViT-L/14, UPN-L coarse proposals, SAM2.1-L, top 100 proposals, SAM batches of 8, 30 diffusion steps, alpha 0.3, decay exponent 0.5. Human support masks replace SAM-generated support masks. Area-weighted support pooling preserves tiny annotated objects that disappear under binary feature-grid sampling; a regression test covers this case. Query-mask pooling and graph diffusion use the native code.
- DINOv: trained Swin-T checkpoint, cached per-class mean native visual content tokens, joint 29-way head with class-isolated content attention, short side 640, class-aware NMS at 0.5. Query encoding is shared across all classes. Output class dimension is checked against the 29 support classes.
- YOLOE: trained v8-L segmentation checkpoint, visual box embeddings from all selected supports, normalized class means, 640 input, NMS 0.5. No text embeddings are supplied.

PET/OPUS use the largest connected support component for their native visual boxes; YOLOE uses the tight box enclosing the class mask; DINOv and FSOD-VFM consume class masks. These are native-modality adaptations, not a controlled equal-resolution/backbone ablation. Class masks can contain multiple same-class instances.

Outputs: samples/<MODEL>/original/ugv_pui_20261106/ contains detections.mp4, tracking.mp4, exports.mp4, overlays/{detections,tracking,exports}/, timings.csv, observations.jsonl, detections.geojson and audit.json. Intermediate predictions/mask previews/temporary registry databases are removed after successful runs; reusable support embeddings stay in ugv_models. The original raw frames stay in data/PUI/rgb_frames.

## Interpretation

No mAP, precision/recall, segmentation IoU, HOTA/IDF1 or physical duplicate metrics were calculated. new_objects counts registry UUID creation events, not verified unique physical objects. Query independence from site supports is not established; audit files record exact byte-level image overlap, which cannot detect differently encoded versions of the same view. Inspect videos qualitatively before choosing further tests.

## Repeat

```bash
bash scripts/run_pui_comparison.sh --run NEW_RUN_NAME
python scripts/verify_pui_comparison.py --run NEW_RUN_NAME
```

The dispatcher uses isolated prepared environments and processes each detector, then MASA. Use --models to select a subset; an existing complete run is skipped only when its bank and dataset hashes match. A failed model retains intermediate state for retry and produces a failed audit. Current-run regeneration is explicit with --force. The annotation helper was temporarily stopped to free GPU memory and restored after all runs; Label Studio stayed running.

Validation: all 115 package tests pass; artifact verification confirms 975 decoded video frames, 975 overlays and 325 timing rows across the five models, with matching bank/dataset hashes.

## Separate views and competing labels

- **Detections:** accepted model boxes, native confidence scores and available segmentation masks, before tracking. No tracker or discovery layer is drawn.
- **Tracking:** MASA outputs and persistent UUID state only. No raw detections or semantic mask layer is drawn. Recovered low-score tracks and ambiguous roles are labelled.
- **Exports:** newly confirmed registry UUIDs, shown once at first discovery and saved once in detections.geojson. Repeated sightings are excluded. Features have null geometry, with image boxes stored as bbox_px properties; they are valid unlocated GeoJSON features. The GeoJSON file export is separate from API delivery: all five runs still have zero API-ready events and zero sends. The API world-coordinate gate remains intact.

A diagnostic checks raw accepted predictions for different person-role labels with bbox IoU >= 0.5. These overlaps do not establish which label is correct, whether two nearby people are distinct, or model accuracy. Counts are frames containing at least one citizen/first-responder pair, out of 65 frames. The pairs in PET-DINO, OPUS, DINOv and YOLOE also have IoU >= 0.7, meeting the configured person-role merge gate. FSOD-VFM has three pairs between 0.5 and 0.7, below the merge gate, so they can remain as separate tracks. The tracking view shows the subsequent association rather than drawing raw hypotheses over it.

| Model | Frames with overlapping citizen/first-responder predictions |
|---|---:|
| PET-DINO | 21 |
| OPUS | 2 |
| FSOD-VFM | 3 |
| DINOv | 1 |
| YOLOE | 12 |

These counts are not an accuracy ranking: a detector missing people may have fewer competing labels. Details and raw-versus-tracked role disagreements are recorded in each audit and the compact observation log. The model weights, support selection, detection thresholds and tracker settings were unchanged for this replacement. Three diagnostic encodes now contribute to staged_sum_s; it is still not a live end-to-end benchmark.

| Model | Detections | Tracking | GeoJSON exports |
|---|---|---|---|
| PET-DINO | [Video](../samples/PET-DINO/original/ugv_pui_20261106/detections.mp4) | [Video](../samples/PET-DINO/original/ugv_pui_20261106/tracking.mp4) | [Video](../samples/PET-DINO/original/ugv_pui_20261106/exports.mp4) |
| OPUS | [Video](../samples/OPUS/original/ugv_pui_20261106/detections.mp4) | [Video](../samples/OPUS/original/ugv_pui_20261106/tracking.mp4) | [Video](../samples/OPUS/original/ugv_pui_20261106/exports.mp4) |
| FSOD-VFM | [Video](../samples/FSOD-VFM/original/ugv_pui_20261106/detections.mp4) | [Video](../samples/FSOD-VFM/original/ugv_pui_20261106/tracking.mp4) | [Video](../samples/FSOD-VFM/original/ugv_pui_20261106/exports.mp4) |
| DINOv | [Video](../samples/DINOv/original/ugv_pui_20261106/detections.mp4) | [Video](../samples/DINOv/original/ugv_pui_20261106/tracking.mp4) | [Video](../samples/DINOv/original/ugv_pui_20261106/exports.mp4) |
| YOLOE | [Video](../samples/YOLOE/original/ugv_pui_20261106/detections.mp4) | [Video](../samples/YOLOE/original/ugv_pui_20261106/tracking.mp4) | [Video](../samples/YOLOE/original/ugv_pui_20261106/exports.mp4) |

Service check: the annotation helper was restored after inference. Label Studio unexpectedly exited with code 137 during the final run and was restarted after model processes finished; its annotations were not edited by this task.

## GeoJSON discovery export

The third video was regenerated on CPU from saved observations; detection and tracking video hashes are unchanged. Each UUID is drawn only in its first-confirmation frame, so frames containing only previously known objects intentionally have no boxes. The GeoJSON contains the same first-confirmation boxes, classes, scores, tracker IDs, timestamps and UUIDs, with geometry set to null. Pixel boxes are properties, not geographic bounding boxes. API delivery remains disabled and its world-coordinate validation is unchanged.

| Model | GeoJSON features / unique UUIDs |
|---|---:|
| PET-DINO | 19 |
| OPUS | 15 |
| FSOD-VFM | 259 |
| DINOv | 74 |
| YOLOE | 26 |

Original detector/tracker timing measurements are retained. Additional CPU export rendering is recorded separately in each audit.
