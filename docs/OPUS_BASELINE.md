# OPUS baseline for PUI

The OPUS adapter runs joint K-way object detection from the existing annotated
support bank. No per-site training or class-specific fine tuning is performed.
It produces bounding boxes, not segmentation masks. The offline runner keeps API
delivery disabled and updates UDC on every sampled query, including empty frames.

## Model and prompts

- Official repository: <https://github.com/Megvii-Algo-Team/OpusDet>
- Source revision: `b4554bf3324d723267c66051874846f5af218846`.
- Full checkpoint: `opus_dinov3_convnext-b_text_visual_pretrain_all_amp_iter_260000-2c4dd683.pth`
  from <https://huggingface.co/Megvii-Algo-Team/OpusDet>.
- Backbone: DINOv3 ConvNeXt-B, widths 128/256/512/1024, depths 3/3/27/3.
  Three feature scales feed OPUS's hybrid encoder and six-layer decoder, with
  900 object queries. Mixed mode also uses the checkpoint's CLIP-B/32 text tower.
- The full checkpoint includes both backbone and text weights. The adapter
  constructs architectures from the upstream bundled local configs, then loads
  **every parameter strictly**. No gated backbone download is needed.
- Native preprocessing: aspect-preserving `FixScaleResize`, Pillow interpolation,
  640 square canvas padded with pixel value 114, BGR to RGB, ImageNet normalization.
- Support prompts use the largest connected component of each annotated mask as
  an instance bounding box. The upstream `visual.I.gen_embed` path extracts one
  256-dimensional vector per selected class/image annotation. Vectors are cached
  with checkpoint, source revision, bank/image/mask hashes, resolution, precision
  and preprocessing policy. Changed supports invalidate the cache.
- `visual.G.16` averages all available examples per class, up to the bank's cap of
  16, without padding or duplicating supports. All requested classes share one
  query-image backbone encoding. Query annotations and present-class filtering
  are never used.
- `text_visual.G.16` uses the same visual memory and the native text template
  `a photo of {class name}.`. Upstream inference averages visual and text vectors
  with equal weight. Visual-only mode does not use the text encoder.

The baseline uses all currently supported classes and examples. Counts and exact
example identities are exported in every run's `summary.json`.

## Isolated runtime

`ugv_models/opus_env` is a venv using the installed Python 3.12 / PyTorch 2.7.0
CUDA 12.6 environment as its base; its additional packages are isolated from
PET-DINO's older runtime. Versions are specified in
`src/triffid_ugv_perception/requirements-opus.txt`.

OpenMMLab dependencies come from the upstream pinned submodules: MMCV 2.2.0,
MMEngine 0.11.0rc2, MMDetection 3.3.0, LVIS API 0.5.3. There was no matching
prebuilt MMCV wheel for this installed PyTorch/CUDA pair. The workspace-local
CUDA 12.6 compiler lives in `ugv_models/opus_cuda`; build the operators using:

```bash
ugv_models/opus_env/bin/python scripts/prepare_opus_runtime.py
ugv_models/opus_env/bin/python scripts/build_opus_ops.py
```

The build script targets this RTX 3060's compute capability 8.6. It includes the
installed PyTorch CUDA library headers, and requires the toolkit's CUDA CCCL
headers. Adapt the architecture list before building for another GPU.

`prepare_opus_runtime.py` changes the pinned MMDetection MMCV upper import guard
from `<2.2.0` to `<2.3.0`, because OPUS itself pins MMCV 2.2.0. It does not spoof
versions or change model computation. The launcher deliberately imports this
patched source tree. Other detector environments are not patched.

## Reproduce inference

Choose new output directories for independent runs:

```bash
bash scripts/run_opus_baseline.sh \
  --output samples/opus_baseline/visual \
  --image-size 640 --precision float16 --confidence .35 \
  --use-all-supports --opus-prompt-mode visual \
  --tracker udc --udc-proof-threshold .35 --person-roles --save-tracking-cache

bash scripts/run_opus_baseline.sh \
  --output samples/opus_baseline/mixed \
  --image-size 640 --precision float16 --confidence .35 \
  --use-all-supports --opus-prompt-mode mixed \
  --tracker udc --udc-proof-threshold .35 --person-roles --save-tracking-cache
```

The shared runner selects the first camera image in each recorded second, then
processes every selected frame in order. Execution is unpaced; the source rate
is 1 Hz. It saves raw predictions at 0.35, ranked candidates, candidates at 0.1
for UDC recovery, observed tracks, local registry creation events, and overlays.
UDC initializes identities only after repeated observations. A raw first-frame
box can therefore have no track label yet. Raw labelled panels in the comparison
page make this difference explicit.

The registry context includes the OPUS prompt mode, support-memory fingerprint,
classes, query size, checkpoint, source dataset and tracker settings. Use a
separate persistent registry for each experimental configuration. Local UUID
creation events are emitted once per UUID; RGB association does not guarantee
one UUID per physical object after arbitrary motion or revisits. See
[persistence limits](PERSISTENT_DISCOVERIES.md).

## Evaluation limits

The current query frames do not have detection/identity ground truth. Detection
counts, overlapping boxes and UUID counts are diagnostic evidence, not AP,
precision/recall, IDF1, or a guarantee of better accuracy. Use the existing query
review pack to annotate evaluation frames before making those claims.

Compare warm inference and complete-frame latency separately, report cold start
and support preparation, and include frames exceeding one second. Native OPUS
640 and PET-DINO 800 differ in resolution/preprocessing as well as model; this
is a comparison of deployable configurations, not an isolated backbone ablation.

## Measured results, 2026-11-06

The study uses the same 65 one-Hz frames, 28 classes, all 141 class/image
supports from 25 photographs, confidence 0.35, UDC proof threshold 0.35, person
role grouping and low-confidence recovery down to 0.1. API delivery is disabled.
All full checkpoints loaded strictly. No query labels were supplied to inference.

[Open the raw/tracked comparison](../samples/opus_20261106/comparison.html) or
[read the exported measurements](../samples/opus_20261106/comparison_results.json).
The viewer shows labelled raw predictions separately from UDC/registry outputs,
so an unconfirmed track cannot be mistaken for a missing detector prediction.

| Configuration | Warm inference mean, s | Warm processing mean, s | Warm processing p95, s | Predictions at 0.35 |
|---|---:|---:|---:|---:|
| PET-DINO visual, native 800, FP16 | 0.271 | 0.423 | 0.479 | 318 |
| OPUS visual, native 640, FP16 | 0.145 | 0.278 | 0.302 | 243 |
| OPUS mixed, native 640, FP16 | 0.151 | 0.266 | 0.289 | 136 |
| OPUS visual, 960, FP16 | 0.153 | 0.253 | 0.270 | 14 |
| OPUS mixed, 960, FP16 | 0.167 | 0.265 | 0.279 | 22 |
| OPUS mixed, native 640, FP32 control | 0.153 | 0.265 | 0.279 | 137 |

Warm measurements exclude the first query. The per-frame processing timer
includes image decoding, detector and appearance descriptors, tracking,
registry/outbox updates, tracking-cache export and JPEG overlays. It ends before
the final frame JSONL logging and printing. Initial model/bank preparation is
also separate: 115.6 s for the fresh PET reference, 98.8/75.6 s for OPUS visual/
mixed at 640, and 100.5/76.7 s at 960; the FP32 control took 99.8 s including a
separate FP32 support cache. FP16 mixed mode reused the corresponding visual
support cache. Preparation includes Python imports and file validation/loading,
so these values are not pure GPU support-encoding times.

All six runs had zero timed query frames above one second, including their
first query. OPUS's first queries took 0.414–0.880 s; this does not include the
one-time preparation above. PET and OPUS use different PyTorch/CUDA runtimes and
native preprocessing. The 960 experiment changes both support and query scale;
it does not isolate query resolution or backbone size.

The qualitative result is weaker than the timing result. At native 640,
visual-only OPUS produced **zero civilian-vehicle predictions** at 0.35; mixed
mode recovered 48 vehicle observations but produced **zero first-responder
predictions**. Citizen observations were 15/11 for visual/mixed, versus 88 in the
PET reference. These are repeated frame-level predictions, not distinct objects
or recall scores. In inspected frame 45, mixed OPUS misses the visible group of
people. Increasing resolution to 960 did not solve this: it sharply reduced
predictions in both modes. Smaller output/registry counts therefore cannot be
treated as better deduplication or accuracy.

The native-640 mixed FP32 control produced 137 predictions versus FP16's 136,
including 48 vehicle observations, 10 citizen observations, and still **zero
first-responder predictions**. It did not resolve the missing people. This
control gives no evidence that reduced precision explains the observed gap.

**OPUS currently meets the measured processing budget, but this setup is not an
established accuracy upgrade over PET-DINO.** Keep native 640 as the OPUS
reference for further investigation. The inherited 0.35 threshold is not
calibrated across models, and the exact class-name prompts have not been tuned.
Ranked candidates are saved for later threshold evaluation. Independent query
labels are needed before claiming AP/recall gains or tuning thresholds against
an evaluation split.

All completed runs pass the saved-output audit: identical source frames, one
query encoding and tracker update per selected frame, no repeated creation UUID,
and consistent registry/outbox events. These checks do not validate physical
identity accuracy. The shared detector/tracker test suite passes 59 tests.
