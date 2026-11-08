# Persistent identity with revisable class evidence

The pre-change code, working diff, file hashes and original YOLOE results are in
`archive/backups/before_class_correction_20261106_213812/`. `code.zip` passed CRC
verification and includes the MASA vendor source. Existing results remain intact.

## Behaviour

1. MASA associates physical instances using its existing frozen appearance
   descriptors. A class mismatch can pass distinctive appearance and bounded
   movement/size/time gates. A lower-similarity recovery route requires overlap,
   a two-second gap limit and a stronger two-way appearance margin. Ambiguous
   associations wait instead of creating a duplicate. Near-identical hypotheses
   are grouped before matching and contribute at most one vote per class/frame.
2. The persistent UUID owns a bounded five-second class history. Predictions
   below 0.35 can recover tracks but do not vote. Votes use confidence and
   recency (0.8 decay/second). Changing class needs two of the last three reliable
   observations and a normalized evidence lead of at least 0.15.
3. Local GeoJSON has one feature per confirmed UUID. Later sightings update that
   feature's stable class, best actual observation for that class, last-seen time
   and class revision. Discovery identity/time remain fixed. `confidence` is the
   best native score for the selected class; `class_confidence` is its temporal
   weighted mean. The representative image, box and timestamp describe the same
   best observation. Native scores are not calibrated probabilities.
4. The third video shows NEW once and UPDATED on a class change or an improvement
   of at least 0.10 in best confidence. Ordinary sightings remain absent there.
   Tracking labels show stable class confidence and the current observation score.
5. Class state is persisted on discovery and meaningful changes; ordinary refreshes
   retain the existing flush interval. Class evidence is bounded by five entries
   and the per-object payload budget. Existing object/database/outbox/log limits
   remain enforced. Sent UUIDs are retained. API delivery remains disabled.

API creation payloads remain immutable. A local correction neither re-creates an
object nor PATCHes an already exported object. A future remote update contract
would be a separate change. World geometry remains null for this RGB-only run.

## Implementation and use

- `tracking/class_correction.py`: association gates, duplicate grouping, temporal
  evidence, and idempotent optional hooks for the pinned MASA source.
- `tracking/masa.py` and `tracking/factory.py`: reuse existing image features.
- `discovery/registry.py` and `discovery/store.py`: persistent label history and
  separate `created_ids` / `updated_ids`; incompatible policy databases are rejected.
- `scripts/pui_comparison_views.py`: local GeoJSON upserts and event overlays.

Enable the shared implementation with `--class-correction` in
`scripts/run_pui_comparison.sh`, using a fresh `--run` name. Existing comparison
runs default to their original policy for reproducibility. This change is
enabled for all five detectors in `samples/<MODEL>/original/class_correction_v1/`.
This does not deploy the offline experiment to the live ROS node.

## Verification on the original rosbag

65 sampled frames at 1 Hz, unchanged support bank and detector predictions.
Complete MASA candidate embeddings were extracted once on GPU, then both native
matching policies replayed on CPU. Baseline IDs agree with the GPU extraction on
65/65 frames. No training, API requests, held-out accuracy or tracking metrics.

The black bag keeps track 3 / UUID `266d3d9f782845fda0cd8a8f60a4ae56`: initially
chainsaw, corrected to bag at frame 13; best confidence improves to 0.6377 and
0.7753 at frames 14 and 15. It produces one discovery and subsequent local updates.
The neighbouring red helmet and real orange chainsaw retain different identities.

The corrected replay has 27 confirmed UUIDs, four class changes and 33 meaningful
local updates; the baseline has 26 UUIDs. Counts are not ground-truth object
counts and do not establish fewer duplicates or greater accuracy overall. Wrong
detector labels and fragmentation under occlusion/viewpoint changes can remain.
Thresholds were developed using this clip and require independent validation.

CPU association plus registry averages 5.47 ms baseline and 14.74 ms corrected
(approximately 9.27 ms added; corrected p95 45.71 ms). These exclude GPU image
feature extraction and rendering and are not live end-to-end measurements.

126 tests pass, including persistence across restart, competing neighbours,
low-confidence recovery, temporal hysteresis, immutable pending payloads and no
second creation after acknowledgement. Three 65-frame videos and 195 overlays
decode successfully; the GeoJSON rebuild matches the observation ledger exactly.

The run folder contains only the three videos, overlays, timings, observation
ledger, GeoJSON and audit. Frozen feature/replay work files remain under
`ugv_models/comparison_work/class_correction/YOLOE/` for reproducibility.

## Other-model rollout

PET-DINO, OPUS, FSOD-VFM and DINOv were rerun with `--class-correction --run
class_correction_v1`, retaining their old baselines. Each processed the same 65
frames with the same support bank and thresholds. Accepted raw predictions match
the original outputs on 65/65 frames for each model. The new association and
registry policy is shared with YOLOE; no detector weights or supports changed.

| Model | Confirmed UUIDs | Local updates | Class revisions | Mean compute/frame |
| --- | ---: | ---: | ---: | ---: |
| PET-DINO | 22 | 2 | 0 | 455 ms |
| OPUS | 15 | 2 | 0 | 274 ms |
| FSOD-VFM | 251 | 62 | 29 | 2247 ms |
| DINOv | 72 | 31 | 19 | 482 ms |

Local updates include class revisions and meaningful confidence improvements.
These are event counts, not accuracy or duplicate-rate metrics. Compute is the
sum of separately measured detector, GPU tracker and registry times, including
the cold first frame; it excludes rendering and is not measured live latency.
FSOD-VFM remains slower than the one-second live budget.

All 12 new videos decode to 65 frames each, all 780 new overlays are present,
and GeoJSON features exactly match ledger reconstruction with one feature per
confirmed UUID. All five runs have the same bank/dataset hashes and API delivery
disabled. The annotation helper was restored and Label Studio remains running.
The 22 relevant adapter, class-correction and view tests pass.
