# Frame 0 polygon-class test

User-selected input: the first decoded frame of
`DJI_20261019155500_0001_S.MP4`. Original RGB is saved losslessly as
`data/UAV-PUI/DJI_20261019155500_0001_S/frame_000000/image.png`.
Its paired `image.srt` contains exactly the first original subtitle block:
00:00:00,000–00:00:00,033, FrameCnt 1 (zero-based video frame 0). Original
GPS, altitude, gimbal angles, focal-length field and capture time are preserved.
The SRT was parsed and checked against the source telemetry.

The camera pitch is -58.1 degrees. This is polygon-class segmentation on the
user's oblique camera image, with original pixel coordinates and no geographic
projection or orthorectification. No substitute nadir image was used.
The polygon export uses Unity's required `source_type: top_view_image` contract;
`input_origin: video_frame`, frame index and telemetry preserve the actual origin.
The export type does not change the camera angle or create an orthographic view.

FSS-SAM3 text-plus-visual, FP32, one-shot, native 1008 collage, same local model
and support bank as the eight-view test. Only supported Polygon classes are
requested; human-role Point classes are not part of this run.

| Class | Foreground pixels | Semantic components |
| --- | ---: | ---: |
| green tree | 1,701,632 | 74 |
| smoke | 0 | 0 |
| building | 62,044 | 1 |
| dirt road | 58,932 | 9 |
| green grass | 133,210 | 6 |
| dry grass | 13,820 | 1 |

Components are not individual tree/object counts. Per-class masks can overlap.
No ground-truth accuracy evaluation was performed.

Model/support preparation: **26.35 s**. All six class predictions plus component
conversion: **7.35 s**, including the cold first prediction. No warm-frame mean
is claimed from a single image. CUDA synchronization brackets measurements;
mask saving, overlays and checkpoint hashing are outside inference timing.

Outputs: `samples/FSS-SAM3/UAV/DJI_20261019155500_0001_S/frame_000000_polygons_text_visual/`:
combined overlay, six class overlays, six binary masks, native pixel polygon-mask
JSON, class timings and audit. Every binary mask was compared pixel-for-pixel
with reconstruction of its exported RLE. API, tracking and world projection
remain disabled. The annotation helper was restored after the run.
