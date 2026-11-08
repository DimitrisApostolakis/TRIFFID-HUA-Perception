# Pretrained COSINE offline baseline

The offline runner now defaults to released multi-scale COSINE, with joint
K-way visual/text prompts and instance masks. No training or fine-tuning was
performed. Support examples are encoded once into a fingerprinted cache; changing
the bank invalidates the cache. API delivery is disabled.

The refreshed UGV-only bank has 27 one-shot classes and seven complete five-shot
classes. See [support provenance](UGV_SUPPORT_REFRESH_20261105.md).

Measured on the desktop RTX 3060, with the same 65 timestamp-selected RGB frames:

| Run | Mean complete frame | Complete p95 | Frames over 1 s | Masks |
| --- | ---: | ---: | ---: | ---: |
| SINE, 27-way 1-shot, long edge 644 | 0.439 s | 0.489 s | 0 | 215 |
| COSINE, 27-way 1-shot, DINO 896 / CLIP 1024 | 1.196 s | 1.192 s | 65 | 346 |
| COSINE, 27-way 1-shot, DINO 644 / CLIP 736 | 0.588 s | 0.630 s | 0 | 277 |
| COSINE, 7-way 5-shot, DINO 644 / CLIP 736 | 0.579 s | 0.606 s | 0 | 153 |

The native COSINE run's first frame took 4.146 s. Its warm mean/p95 were
1.149/1.190 s. The smaller run's first frame took 0.787 s, and warm mean/p95 were
0.585/0.627 s. Smaller-run mean inference was 0.433 s, preparation 50.33 s and
peak allocated CUDA memory approximately 3.96 GiB. Preparation and cold support
encoding are outside the frame budget. Execution is unpaced replay, not a live
hard-deadline test. Jetson performance is untested.

Outputs:

- `samples/cosine_baseline/native_896_attempt5/`: completed native-resolution run.
- `samples/cosine_baseline/efficient_644/`: selected 27-way 1-shot baseline.
- `samples/cosine_baseline/five_shot_644/`: seven-class five-shot validation.
- Each completed run includes overlays, packed masks, frame logs, configuration,
  summary and an export audit. `validation_matrix.json` summarizes the suite.
- `samples/pui_registry/cosine_v1.json` and `cosine_v1_events/` hold persistent
  state and creation events for the selected baseline.

All three COSINE audits confirm the same 65 source frames, mask-derived boxes,
original ontology IDs, one DINO and one CLIP query encoding per frame, tracker
updates on every frame, and no repeated creation UUID. The smaller run emitted
16 candidate UUIDs. These are not 16 verified physical objects.

In an inspected early frame, COSINE correctly outlines and classifies the van
as a civilian vehicle. Native-resolution output also identifies an extinguisher.
There are still missed detections and inconsistent labels; more masks do not
establish higher accuracy. Query-frame mAP, mask AP, identity switches and physical
duplicate rate need labelled ground truth. The five-shot run has a different
class set and must not be interpreted as a direct shot-count accuracy comparison.

Validation: 37 UGV tests pass, all three GPU replays and export audits pass,
and `git diff --check` passes. Earlier incomplete attempt directories are
retained for debugging and are not results. No model accuracy claim follows
from the export-consistency tests.

Run:

```bash
bash scripts/run_cosine_baseline.sh \
  --output samples/cosine_baseline/new_run \
  --registry samples/pui_registry/cosine_v1.json
```

Use a fresh output directory. Omit `--registry` for an independent evaluation;
reuse it only for the same experiment/model. See [persistence limits and remaining
depth/pose requirements](PERSISTENT_DISCOVERIES.md). `--shots 5` selects the seven
complete classes. `--image-size 896` selects native resolution; the default 640
rounds to 644, with CLIP size 736 to align feature pyramids.

Checkpoint SHA256:
`8f85b892627eacdf0e63e0c1a147af5da681611ba9c32c9f6cf08af755d16fac`.
Size: 2,764,208,701 bytes. Downloaded from the authors' public ModelScope package
and checked against its published checksum. Source revision:
`ec7a66d1fefb037e43fdc04c7f2ee818d1aa639d`.

The adapter uses the published shared encoder, pixel decoder and MFormer, cached
visual/text support fusion, and FC-CLIP score fusion. The public FSOD package
references absent model variants, so its benchmark entrypoint cannot be used
directly. All model tensors load strictly. Six duplicate support-neck names are
restored from the exact shared `simfp_4` tensors; only training-loss
`criterion.empty_weight` is excluded.

Compatibility and efficiency changes, without editing third-party source:

- Use OpenCLIP's native text API to handle its version-specific token layout.
- Use the published LayerNorm2d for the older installed Detectron2.
- Read the Objects365 category constant without importing unrelated datasets.
- Share one CLIP forward across five feature levels; the public wrapper repeats it.
- Resize only top-ranked masks before instance filtering.
- Use MMCV's precompiled CUDA deformable-attention operator. Its maximum absolute
  difference from the authors' PyTorch reference was 5.96e-8 in the GPU smoke test.

Runtime: existing Python 3.10 / torch 1.13.1+cu117 / compiled Detectron2, with
project-local dependencies in `ugv_models/cosine_runtime`. The shared environment
was not modified. Dependency pins are in `requirements-cosine.txt`; mmcv-full
1.7.2 uses the cu117/torch1.13.0 wheel index. Inference uses FP16 autocast while
the published deformable pixel decoder and model weights remain FP32. This is
not a reproduction of the authors' original software environment or benchmark AP.
