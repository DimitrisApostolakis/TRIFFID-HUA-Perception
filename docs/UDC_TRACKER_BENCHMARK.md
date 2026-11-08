# PET-DINO with UDC tracking, 5 November 2026

This report records the original 50% proof gate. The default was subsequently
lowered to 35%; see [the threshold comparison](UDC_THRESHOLD_035.md). To reproduce
these historical settings, explicitly pass `--udc-proof-threshold 0.5` to the
runner and benchmark script.

UDC tracking is integrated as an optional detector-independent adapter. The
hospital detector, weights and role rules were not imported. BoT-SORT remains
the default. Both trackers can use physical-person identities with temporal
citizen/first-responder/military-personnel role evidence.

## Implementation

The tracking primitives derive from
[theodouk11/udc_tracker](https://github.com/theodouk11/udc_tracker), pinned to
`11188042c8b487374a75b609424998df3f041522`. The Apache-2.0 license and modification
notice are retained under `tracking/vendor/`. No new model is trained.

The adapter uses independent tracker pools for unrelated classes. Optional
`--person-roles` merges person-role hypotheses with box IoU >= 0.7 before
association, retaining alternatives as evidence. People share an internal
identity category; exported class IDs remain ontology IDs. Role scores accrue
over five elapsed seconds, with ambiguity reported when the leading two
cumulative scores differ by less than 20% of their total. The winning role
can change without replacing the persistent UUID or creating another event.
These shares are evidence proportions, not calibrated probabilities. There is
no permanent role lock, and an uncertain prediction is not thereby corrected.

UDC is configured for 1-Hz input: ten missed observations before active-track
expiry, a 60-second lost gallery, five-observation label/hit windows, one
successful post-birth match before track confirmation, and two observed
registry updates before a new UUID. A lifetime detection score >= 0.5 supplies
UDC's proof gate. High/low association thresholds are 0.35/0.1. Low-score boxes
can sustain an existing track but never start one. Histogram correlation must
reach 0.9 for gallery revival. These are provisional settings, not tuned
accuracy optima. The adapter advances missed one-second ticks when timestamps
jump, rather than treating a long gap as a single frame. Subsecond timing is
quantized for UDC motion/expiry; the role window uses timestamps directly.

Only tracks matched to a current observation are exported; predictions through
empty frames never create detections/events. Persistent registry aliases use
physical identity separately from role. Existing immutable creation events keep
the class known at creation; current registry records hold the latest role.
Changing a tracker/grouping policy requires a separate registry context. This
remains RGB-only tracking, without robot-pose/depth fusion or reliable identity
after arbitrary revisits. HSV recovery can confuse similarly coloured objects.

## Controlled replay

One PET-DINO Swin-T run used the updated 141-support, 28-class bank at
800/1333, FP32. It cached 3,383 candidate boxes with their detector feature
vectors at confidence >= 0.1 across the same 65 one-Hz frames. Of these, 316
were >= 0.35. The high-score detections are exactly equal to the prior
all-support 800 study, including boxes, scores and classes.

Four main comparisons consume the identical 316 detections/features in the
same order with fresh registries. A fifth UDC run additionally enables the
0.1–0.35 recovery candidates. All external APIs remain disabled. Timings below
measure tracker computation (including grouping and role evidence), excluding
detector inference, registry persistence and rendering.

| Tracker | Person grouping | Recovery floor | Mean tracker | p95 tracker | Observations | Person observations | Person UUIDs | Overlap person pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BoT-SORT | off | 0.35 | 25.35 ms | 38.75 ms | 156 | 104 | 12 | 52 |
| BoT-SORT | on | 0.35 | 22.46 ms | 39.45 ms | 104 | 52 | 6 | 0 |
| UDC | off | 0.35 | 2.61 ms | 5.73 ms | 69 | 66 | 12 | 33 |
| UDC | on | 0.35 | 2.13 ms | 3.63 ms | 36 | 33 | 6 | 0 |
| UDC | on | 0.10 | 2.95 ms | 5.36 ms | 49 | 42 | 7 | 0 |

An observation is one emitted track in one frame. An overlap pair is two
emitted person hypotheses in a frame with IoU >= 0.7; counts accumulate across
frames, not unique physical objects. This is a duplication proxy, not ground
truth. UDC's proof gate and confirmation differ from BoT-SORT, so fewer outputs
or UUIDs can mean filtering or fragmentation rather than improved accuracy.
The low-score run emitted ten low-confidence observations; none of the real
replays exercised appearance-gallery revival. Revival is covered synthetically.

All five runs passed checks for one update per frame, distinct current track
IDs and UUIDs, and exactly one local creation-event file per created UUID.
These checks do not establish physical identity accuracy. Person grouping cut
the measured overlap proxy to zero for both trackers, while UDC alone did not.
The clearest benefit here comes from separating person identity and role.

Inspecting frame 56 confirms a coverage trade-off: grouped BoT-SORT displays
both visible people and the van; grouped high-score UDC displays only the
standing person. Both still have ambiguous role evidence. Grouped UDC with
low-score recovery improves the observation count, but neither detector-role
accuracy nor tracking accuracy has been scored against ground truth.

The detector/cache-generation run averaged 0.533 s complete processing, with
0.532 s p95 and one 3.746 s first frame. Its startup preparation took 70.8 s.
It includes extra feature pooling and cache writes, and is not the final UDC
pipeline timing. Warm-up is required before promising a one-second deadline.
Replay estimates add detector inference to measured CPU replay; they are
explicitly estimates, not live end-to-end measurements.

The integrated PET-DINO + grouped UDC low-recovery run also completed all 65
frames. Its mean complete latency was 0.516 s, p95 0.600 s, and first/max frame
3.381 s, with preparation 73.8 s. Excluding the first frame, mean latency was
0.471 s and maximum 0.618 s. Its emitted track IDs, boxes, roles and ambiguity
flags exactly match the corresponding cached replay. The export audit passes.
This was an unpaced offline run, not a live robot deadline test.

The current UDC proof threshold has a material coverage cost: it emits no
observations in the first 38 frames, where foreground-van detections commonly
score about 0.43 and never clear 0.5. This is deliberate gating rather than a
missing detector output. Lowering that threshold needs an independent labelled
evaluation of false positives as well as recall. For now, grouped BoT-SORT is
the more useful baseline for coverage; UDC stays an optional experiment.

## Artifacts and reproduction

[Interactive comparison](../samples/udc_tracking_20261105/comparison/comparison.html)
shows all five views, with grey raw hypotheses and coloured observed tracks.
Orange marks ambiguous person roles. [Machine-readable results](../samples/udc_tracking_20261105/comparison/results.json)
include timings, settings, input hash and per-class counts. Outputs and support
assets remain local, ignored runtime data.

Create the feature cache:

```bash
bash scripts/run_pet_baseline.sh \
  --output /mnt/d/projects/triffid-hua-perception/samples/udc_tracking_20261105/detector_cache \
  --use-all-supports --image-size 800 --max-size 1333 --precision float32 \
  --confidence 0.35 --save-tracking-cache
```

Then run `scripts/benchmark_trackers.py` in the fewshot-dev environment with
`ugv_models/pet_runtime`, `ugv_models/devit_runtime` and
`src/triffid_ugv_perception` on PYTHONPATH, and OMP/MKL thread counts set to four.
Use a fresh output path; the script refuses to overwrite a comparison.

Run integrated UDC inference:

```bash
bash scripts/run_pet_baseline.sh \
  --output /mnt/d/projects/triffid-hua-perception/samples/udc_tracking_20261105/udc_pipeline \
  --use-all-supports --image-size 800 --max-size 1333 --precision float32 \
  --confidence 0.35 --tracker udc --person-roles
```

For grouped BoT-SORT use `--tracker botsort --person-roles`. BoT-SORT keeps the
existing high-confidence input policy. UDC uses PET candidates down to 0.1 for
recovery. Unrelated classes never share the person pool. Do not globally merge
all 28 classes: that would mix distinct co-located objects such as a person and
a bag. Even person-only overlap suppression can merge distinct people in heavy
occlusion; labelled tracking sequences are needed to quantify this risk.

Validation: 52 code tests pass, including role-change UUID continuity across
restart, empty frames, low-score recovery, independent tracker instances,
same-frame duplicate births, gap expiry, gallery revival and rejection of a
different-colour gallery candidate. Event and source-equivalence audits pass.
