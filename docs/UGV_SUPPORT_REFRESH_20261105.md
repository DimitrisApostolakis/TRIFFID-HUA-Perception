# UGV support-bank refresh, 5 November 2026

Source: `C:\Users\apost\Downloads\project-4-at-2026-11-05-03-31-b3d312cd.json`.
SHA256: `ea00a804e3fca0b4ebbf7a0a0e4289ed6ffb336c76694cc8c518f35d1de85e95`.

The supplied export contains 23 tasks, including all four tasks the user removed
in Label Studio. The explicit exclusion of task IDs **29, 31, 33 and 34** overrides
the export. Nineteen tasks contribute to the refreshed bank. Original images
and previous semantic/panoptic masks remain untouched.

The active `ugv_supports/bank.json` now references **88 filled slots**, **27 valid
one-shot classes** and **7 valid five-shot classes**. Five-shot classes are green
tree, green plant, building, dirt road, green grass, cone and furniture. Citizen
no longer has a UGV example. First responder, window and fire truck still have
one-shot examples but no longer have five distinct valid support images.

Label Studio brush alpha masks are decoded at original image resolution.
Independent binary masks per class preserve pixels where class annotations
overlap; the per-image semantic mask is only a preview with export-order
precedence. Bounding rectangles are not fabricated into segmentation masks.
Task 25 contains one dry-tree region (`5eeb22066888`) labelled as a brush but
holding only rectangle coordinates. It has no RLE pixels and is explicitly
skipped in the report. Other valid dry-tree regions remain usable.

Reproduction:

```bash
PYTHONPATH=src/triffid_ugv_perception \
  /home/oracle/miniforge3/envs/fewshot-dev/bin/python scripts/import_labelstudio_bank.py \
  /mnt/c/Users/apost/Downloads/project-4-at-2026-11-05-03-31-b3d312cd.json \
  --bank ugv_supports --exclude-task-ids 29 31 33 34
```

The importer refuses to overwrite an already imported revision. This revision
has already been applied. Its raw export, masks, manifest and report are under
`ugv_supports/labelstudio_ea00a804e3fc/`. The previous bank and import report are
archived under `ugv_supports/history/ea00a804e3fc/`. Historical model runs retain
their original support fingerprints; their results must not be presented as
using the refreshed bank. New support hashes invalidate model caches.

The importer validates every selected one-/five-shot episode before replacing
the active manifest. Tests exercise alpha decoding, invalid streams, explicit
task exclusion, overlapping classes and preservation of the old manifest.
