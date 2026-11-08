# SANSA GPU benchmark

Measured the existing SANSA universal adapter on the RTX 3060 with the current
UGV-only support bank. No training, checkpoint changes, or API traffic.

Configuration: FP32, one support image per class, existing 640x640 input resize,
mask threshold 0.5, component confidence threshold 0.7 and minimum component area
400 pixels. SANSA internally resizes those inputs to its SAM2 image size of
1024x1024; 640 is not the internal backbone resolution. Uses the local SANSA
source revision `f562b5642940e08a57e3da2608b804b99af3763b`,
`adapter_generalist.pth` and `sam2_hiera_large.pt`.

Single-class first-responder test used all 65 comparison frames:

- Mean inference including semantic component conversion: 0.969 s.
- Inference p95: 0.986 s.
- Warm inference mean/p95: 0.960 / 0.984 s.
- First inference: 1.604 s; the only inference over one second.
- Mean loading + inference + overlay-writing time: 1.052 s.
- Initial model/support preparation: 11.06 s.
- Peak allocated CUDA memory: approximately 2.04 GiB.

The 27-class loop uses seven distributed frames from that same sequence, at
indices 0, 10, 20, 30, 40, 50 and 64. It is a timing sample, not a full 65-frame
accuracy comparison. Final timings are saved in
`samples/sansa_baseline/all_classes/summary.json`.

Completed 27-class result: mean inference **25.816 s**, p95 **25.894 s**,
warm mean **25.807 s**, mean including image loading and overlay writing
**25.901 s**. All seven frames exceeded one second. Initial model/support
preparation was 15.52 s and peak allocated CUDA memory approximately 2.04 GiB.
All seven logged source records match the stated subset of the common 65-frame
sequence, and all 27 supported classes were selected.

The existing adapter runs an independent support/query episode per class. CPU
support tensors are cached, but support and query backbone computation repeats
for every class and every incoming frame. This is a class loop, not native
joint K-way inference. Its timing does not establish the speed achievable by
a redesigned feature-caching adapter, mixed precision, or a smaller backbone.

Timing excludes tracking and registry work. GPU synchronization brackets every
measurement. Thus SANSA inference times must not be labelled full pipeline times
or compared as identical configurations to the other models. The measured
27-class configuration already greatly exceeds the one-second budget before
tracking. Output boxes come from semantic connected components, not native
instance predictions, and touching instances can merge.

Reproduction:

```bash
PYTHONPATH=src/triffid_ugv_perception OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  /home/oracle/miniforge3/envs/sansa/bin/python scripts/benchmark_sansa.py
```

Completed configurations are skipped when their summary exists. Frame logs and
overlays are under `samples/sansa_baseline/one_way/` and `all_classes/`. The
measurements assess throughput; detection counts are not accuracy measurements.
