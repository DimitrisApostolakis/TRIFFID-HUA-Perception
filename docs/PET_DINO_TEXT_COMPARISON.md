# PET-DINO class-name text comparison, 5 November 2026

These saved runs use UDC's previous 50% proof gate. To reproduce them after the
default was lowered, add `--udc-proof-threshold 0.5`. See the
[35% gate comparison](UDC_THRESHOLD_035.md) for the current default.

The adapter now exposes `--pet-prompt-mode visual|text`. Visual-G remains the
default. Text mode activates the original PET-DINO text pathway with exact
ontology names. It does not combine visual and text scores, encode support
images, or train any weights. Native simultaneous visual/text fusion is not
claimed. The strict checkpoint load is unchanged.

## Controlled run

The same 28 classes, PET-DINO Swin-T checkpoint, 65 one-Hz source frames,
800/1333 input size, FP32 and 0.35 detector threshold are used. UDC settings,
person-role grouping, low-confidence recovery and registry confirmation match
the previous visual run. Each mode uses a separate registry. All APIs are off.
The report verifies frame, class, checkpoint and configuration equivalence.

Prompts are exactly:

```text
fence; green tree; smoke; first responder; fire hose; boot; green plant;
window; building; ladder; dirt road; dry tree; wall; civilian vehicle;
citizen; green grass; pole; dry grass; door; mud; bag; fire truck; cone;
dry plant; furniture; barrel; shovel; extinguisher
```

No synonyms or descriptive phrases were introduced. Text mode reports zero
visual supports and a null support cache in metadata. If `--classes` is omitted,
the runner selects the classes available in the bank for comparability; this
selection does not turn their visual supports into inputs to text detection.
An explicit `--classes` list avoids this bank-based class discovery.

## Results

In the first frame, vehicle detection scores change from 0.438/0.373 in visual
mode to 0.756/0.676 in text mode. Text detects the main orange cone at 0.638 and
an extinguisher at 0.772. It also labels a small bottle-like object as cone at
0.355, illustrating why stronger scores or more boxes do not prove higher
accuracy.

Across all frames, exact `first responder` text produces no accepted raw boxes
at 0.35, while visual mode produces competing responder/citizen hypotheses.
In inspected frame 56, text localizes both visible people but labels both
`citizen` (0.497/0.416). Role discrimination remains unresolved. None of the
text predictions for `green tree`, `dry tree`, `green grass` or `dry grass`
passes the operating threshold in this run.

Raw detection observations (repeated frame observations, not unique objects):

| Class | Visual-G | Exact class-name text |
| --- | ---: | ---: |
| civilian vehicle | 71 | 131 |
| cone | 5 | 66 |
| extinguisher | 4 | 47 |
| citizen | 87 | 75 |
| first responder | 88 | 0 |
| All classes | 316 | 634 |

No independent query ground truth is available; precision, recall and AP
remain unmeasured. The observations above are qualitative examples and output
counts, not a numeric accuracy comparison.

| Complete pipeline latency | Visual + UDC | Text + UDC |
| --- | ---: | ---: |
| Mean, excluding first frame | 471 ms | 540 ms |
| p95, excluding first frame | 591 ms | 687 ms |
| Mean, all 65 frames | 516 ms | 596 ms |
| First/max frame | 3.381 s | 4.146 s |
| Mean detector inference, including first | 410 ms | 430 ms |
| Preparation | 73.8 s | 54.6 s |

All subsequent text frames are below one second. These are desktop RTX 3060
unpaced offline measurements, not Jetson or live robot deadlines. The text run
also saves optional tracking candidates/features; total latency includes those
cache writes, while the referenced visual pipeline did not write that cache.
Treat the complete-time comparison with that export-cost difference in mind.

## Review and reproduction

[Interactive comparison](../samples/pet_text_20261105/comparison.html) has
labelled raw detections in its top row and UDC tracks in its bottom row. This
makes raw model output visible even when UDC suppresses a track. Raw labels
show detector scores; grey unlabelled boxes in tracked views are hypotheses.

[Machine-readable report](../samples/pet_text_20261105/comparison_results.json)
contains every class count, exact text prompts, controls and timing metadata.
The text run retains all ranked candidates for the existing COCO evaluator,
and a feature cache for future controlled tracker comparisons.

```bash
bash scripts/run_pet_baseline.sh \
  --output /mnt/d/projects/triffid-hua-perception/samples/pet_text_20261105/text_class_names \
  --pet-prompt-mode text --image-size 800 --max-size 1333 \
  --precision float32 --confidence 0.35 --tracker udc --person-roles \
  --save-tracking-cache
```

Use a fresh output directory for another run. Do not supply
`--use-all-supports` in text mode. Generate the comparison with
`scripts/report_pet_text_comparison.py` in the fewshot-dev environment with
`src/triffid_ugv_perception` on PYTHONPATH.

Validation: all 55 tests pass, including mode routing, absence of visual memory
in text inputs and accurate zero-support metadata. The 65-frame export audit
passes source-frame equivalence, encoder/update counts and UUID/outbox checks.
