# UAV PUI support bank

## Evaluation holdout: tasks 1 and 3

The active support bank excludes task IDs **1 and 3** and identical image-content
copies. The full portable dataset ZIP is preserved. The active bank now has
**47 usable supports across 11 populated classes**; dry grass has no remaining
support because its only masks belong to the two held-out frames.

The reproducible frozen evaluation split is `data/UAV-PUI/holdout_1_3/split.json`.
All four methods use one identical support per evaluated class: green tree,
window, building, dirt road, green grass and furniture. These are six independent
class episodes per frame. The two query images were never selected as supports;
other views of this site may contain the same physical objects.

Metrics use the union of annotated brush pixels as the primary evaluation region,
with full-frame IoU reported separately under an explicit background assumption.
Ground-truth coverage is approximately 90.2% and 95.0% of the two frames.
Point-box F1 uses connected components and the saved window/furniture boxes.
This is a two-frame diagnostic, not an independent-site accuracy benchmark.

The earlier full-bank inventory below describes the pre-holdout refresh; the
current counts are in `uav_supports/PUI/import_report.json`.

Updated from the current saved annotations in Label Studio project **UAV-PUI** (project 6), using a consistent read-only database snapshot.

Portable archive: `D:\Datasets\PUI\PUI-Dataset\UAV-PUI.zip`.
Archive SHA256: `59c69489df71d7390dbd87eafd2fd5e7b0f2e28dd3b39bb8beeb41d519673022`.

The archive follows UGV-PUI.zip: original images, native Label Studio JSON, COCO detection JSON, semantic JSON/RLE, per-class binary masks, uint16 single-label maps, detection/semantic/combined overlays and metadata/checksums. No panoptic export.

It contains **16 images, 16 saved annotations, 207 boxes and 184 brush regions**. One task-70 rectangle stored RLE instead of box coordinates; its bounding box was recovered from the stored mask, with the repair recorded in export_report.json. No segmentation was invented from a box.

The UAV ontology now contains **38 classes: 18 Point and 20 Polygon**. Furniture is Point, preserving original category ID **50**. Human roles retain separate labels/IDs (6, 23, 43).

Active bank: `uav_supports/PUI/bank.json` (`triffid.uav.support-bank.v1`).
It has **49 usable supports across 12 populated classes**, with five slots reserved per configured class. Empty slots are not supports. All selected image/mask pairs were loaded and validated.

| Class | Active supports | Available distinct annotated images |
| --- | ---: | ---: |
| green tree | 5 | 15 |
| smoke | 2 | 2 |
| first responder | 5 | 5 |
| window | 5 | 9 |
| building | 5 | 9 |
| dirt road | 5 | 15 |
| citizen | 5 | 8 |
| green grass | 5 | 9 |
| dry grass | 2 | 2 |
| door | 2 | 2 |
| fire truck | 3 | 3 |
| furniture | 5 | 9 |

The native bank selects the first five distinct annotated images per class. All remaining examples are retained in the portable export and its versioned bank inputs; missing examples are never duplicated.

Inputs: `uav_supports/PUI/portable_59c69489df71/`.
Previous manifest/report: `uav_supports/PUI/history/before_refresh_20261107T170218Z/`.
Previous masks and portable inputs are retained; the new mask directory is revision-specific. The UGV bank was not changed. Existing detector/tracker sample outputs were not rerun with this new bank.

Use `--support-bank /mnt/d/projects/triffid-hua-perception/uav_supports/PUI` when running inference. One-shot selection is available for all populated classes; five-shot selection requires a class with five active supports. Import provenance/counts are in `uav_supports/PUI/import_report.json`.
