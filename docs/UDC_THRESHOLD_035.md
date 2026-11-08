# UDC confirmation gate lowered to 35%

The UDC adapter now defaults to a 0.35 lifetime proof threshold, matching the
PET-DINO raw prediction threshold, instead of 0.5. Override it with
`--udc-proof-threshold VALUE` (0.35–1). This changes the tracker confirmation
gate; it does not invent detector outputs or change detector confidence.

The high-confidence association threshold remains 0.35, and the recovery floor
remains 0.1. Low-score boxes can sustain an existing track but cannot start one.
A repeated observation is still required before a UDC track is emitted, and
the registry still requires two consecutive emitted observations for a new UUID.
Person-role grouping and idempotent creation events remain enabled in this run.
A different proof threshold gets a separate registry context, preventing silent
reuse of a registry with different filtering settings.

## Controlled check

The 65 cached PET-DINO Visual-G frames use the same 28 classes and all 141
supports at 800/1333, FP32. The paired low-recovery replays have identical
candidate boxes, scores, features, timestamps, source hash, and UDC settings
except `proof_thr`. Replaying the cache avoids another GPU detector pass.
No API is enabled.

| Measure | Previous gate 0.50 | New gate 0.35 |
| --- | ---: | ---: |
| Emitted observations | 49 | 191 |
| Vehicle observations | 3 | 88 |
| Person observations | 42 | 43 |
| Registry UUIDs | 8 | 23 |
| High-overlap person pair occurrences | 0 | 0 |

In frame 0001 (the second source frame), the new gate emits both vehicles at
0.433 and 0.375; persistent UUIDs appear in frame 0002. Frame 0000 still shows
only raw hypotheses because repeated-observation confirmation is intentional.
The new tracker averages 3.24 ms per frame, p95 4.95 ms, excluding inference,
registry writes and rendering. No new end-to-end GPU latency measurement is
claimed for this threshold change.

Increased output coverage is not a measured accuracy improvement. More UUIDs
can include false detections and fragmentation. Lowering the proof gate can
admit persistent false positives. Zero overlap pairs does not establish zero
physical duplicates or false merges. Query ground truth remains unavailable.

[Compare old and new overlays](../samples/udc_threshold_035_20261105/threshold_comparison.html)
opens on the second frame. [Paired results](../samples/udc_threshold_035_20261105/threshold_comparison.json)
verify the shared cache and sole settings difference. The broader
[three-view replay](../samples/udc_threshold_035_20261105/comparison.html) also
compares grouping and low-score recovery under the new gate.

## Running

```bash
bash scripts/run_pet_baseline.sh \
  --output /mnt/d/projects/triffid-hua-perception/samples/udc_lower_gate_new_run \
  --use-all-supports --image-size 800 --max-size 1333 --precision float32 \
  --confidence 0.35 --tracker udc --person-roles --udc-proof-threshold 0.35
```

For cached replay, use `scripts/benchmark_trackers.py --udc-only
--udc-proof-threshold 0.35 --output NEW_DIRECTORY` in the configured
fewshot-dev environment. The paired report is generated with
`scripts/report_udc_threshold.py`.

Validation: 56 code tests pass, including that a 0.438-scored vehicle is
accepted after confirmation with the new default, that low scores cannot spawn
tracks, and that a configurable stricter gate retains its previous behavior.
All three replay consistency/event audits pass.
