# PET-DINO pretrained Visual-G baseline

Official Swin-T checkpoint, no training. All learned tensors, including BERT,
load strictly. Source revision `7830a462f9a320f34293cce0aabdda1256d9dc15`;
checkpoint SHA256 `beff20e86a46008b637edf6f519f100cb69981d1c116a03959663d276adc1736`.

Uses the same 27-way one-shot UGV bank and same 65 frames as COSINE. Visual-G
extracts support embeddings offline and averages examples per class. Query
frames supply no annotations or prompt boxes. Class names are present as API
metadata; the selected route uses visual exemplars. Citizen has no supports
and is not selected. This run cannot evaluate citizen detection.

PET-DINO outputs boxes, not masks. The existing BoT-SORT and local registry
remain active, with no API traffic. Appearance descriptors reuse backbone
features; physical deduplication is still not guaranteed.

Two runs distinguish support-box policy and confidence threshold:

- `samples/pet_baseline/full`: original union-of-class-mask support boxes,
  score threshold 0.45. Completed 65 frames, 109 detections, 7 candidate UUIDs.
  Warm mean 0.432 s, warm p95 0.452 s. First frame 3.804 s; the only frame over
  one second. Allocated GPU peak approximately 1.23 GiB. Export audit passed.
- `samples/pet_baseline/component_supports`: support box around the largest
  connected foreground region, score threshold 0.35. This avoids enclosing
  disconnected objects and the background between them. Changes to both prompt
  geometry and threshold mean the two runs do not isolate either factor.

The component-support run completed all 65 frames: mean complete processing
0.441 s, p95 0.481 s, maximum 0.636 s, no frames over one second. Mean inference
was 0.314 s. It exported 322 detections and 15 candidate UUIDs; its export audit
passed against the identical COSINE source-frame sequence. Warm mean/p95 were
0.438/0.477 s. These timings use the RTX 3060, not the Jetson.

Manual examples: `overlays/0001.jpg` detects the real cone and the distant car,
but misses the nearby van. `overlays/0056.jpg` detects both visible people and
the van. Both people receive the first-responder label; role correctness is
not verified, and this does not demonstrate civilian classification. The run
still contains missed objects and false positives. Overall accuracy superiority
over COSINE is not established. The offline default remains COSINE so the new
model can be reviewed explicitly before choosing a replacement.

Native inference setting: short edge 800, maximum edge 1333, FP32, RTX 3060.
Masks supply support-box geometry only; this is not mask-conditioned inference.
Fewer detections or more detections do not measure accuracy. No query ground-truth
AP or recall is available.

```bash
bash scripts/run_pet_baseline.sh \
  --output samples/pet_baseline/new_run \
  --image-size 800 --max-size 1333 --precision float32 --confidence 0.35
```

Dependencies are isolated in `ugv_models/pet_runtime`: mmcv 2.0.1 (precompiled
cu117/torch1.13.0 wheel), mmengine 0.10.7, transformers 4.38.2, tokenizers 0.15.2,
plus local runtime dependencies. The underlying fewshot-dev environment is
unchanged. Hugging Face BERT config/tokenizer files are local; BERT tensors come
from PET-DINO's full checkpoint. Training activation checkpointing and the
Objects365 training-memory bank are disabled during inference. No third-party
source or weights were edited. The wrapper uses the official embedding extraction
and joint multi-class prediction methods directly.

Validation: 38 UGV tests pass, including disconnected-instance prompt-box
selection. The baseline audit checks identical source frames, ontology IDs,
backbone-query counts, tracker updates and event UUID uniqueness. It does not
establish physical identity correctness or accuracy superiority.
