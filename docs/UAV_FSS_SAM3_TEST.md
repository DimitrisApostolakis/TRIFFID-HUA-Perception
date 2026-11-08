# UAV FSS-SAM3 baseline — 7 November 2026

The same eight angular views, four Point classes, one-shot examples and PUI bank
as [SANSA](UAV_SANSA_TEST.md) were used. First responder and citizen remain
separate labels. This is not a continuous full-video or tracking run.

Local source: `WongKinYiu/FSS-SAM3`, revision
`b82ba838678ae8c2f27f618d9500a50c171e223e`; local `sam3.pt` checkpoint SHA256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
No training, new weights or network downloads were required.

The adapter follows the repository's COCO one-shot collage settings: 1008x1008,
support at bottom (60%), query at top (40%), positive support box from the same
annotated support mask. The visual-only mode uses the native dummy text `visual`;
text-plus-visual uses the exact class name before adding the geometric prompt,
as the repository's default evaluator does. No negative mining was applied.
Model score filtering and mask thresholds are both 0.5. Returned masks are
unioned, query-cropped and nearest-neighbour resized to original image dimensions,
then passed through the same UAV connected-component exporter as SANSA.

## Results

RTX 3060, FP32, Torch 2.7.0+cu126; annotation helper stopped during measurement.
CUDA synchronization brackets all timings. Frame time includes inference,
query restoration and component conversion, and excludes loading/rendering.

| Mode | Preparation | Mean/view | Warm mean/view | p95 incl. cold view | Query output |
| --- | ---: | ---: | ---: | ---: | --- |
| Visual-only | 30.39 s | 3.83 s | 3.79 s | 4.02 s | none |
| Text + visual | 25.96 s | 4.48 s | 4.40 s | 4.82 s | 19 window-class components |

Text-plus-visual returned query masks in five views. It returned no first-responder,
citizen or fire-truck query masks. Native diagnostics show accepted support-side
masks; those are deliberately not exported as query detections. Components are
not ground-truth object counts, and no precision/recall or mAP was evaluated.
Neither mode meets a one-second budget. SANSA's warm mean was 3.79 s, using its
own native mask-prompting and resolution path; this is not a backbone-only ablation.

## Outputs and reproduction

Under `samples/FSS-SAM3/UAV/DJI_20261019155500_0001_S/`:

- `one_shot_1008_visual/`
- `one_shot_1008_text_visual/`

Each contains an eight-frame preview, eight overlays, native pixel-observation
JSON, frame/class CSV timings and audit. Both videos decode and all exported
mask RLE, boxes and centroids validate. Bank, selected-view manifest and support
fingerprints match the SANSA baseline. No HTML, API calls, tracking, geographic
projection or duplicate suppression was added. The annotation services were restored.

```bash
bash scripts/run_uav_fss_sam3_test.sh
bash scripts/run_uav_fss_sam3_test.sh fss-sam3-text
```

Use fresh output directories to rerun without replacing results. The modular
backend is `models/fss_sam3.py`; the existing UAV timed runner handles both methods.
Tests cover collage prompt coordinates, support/query separation and distinct
human-role text prompts. The updated UAV suite passes 70 tests.
