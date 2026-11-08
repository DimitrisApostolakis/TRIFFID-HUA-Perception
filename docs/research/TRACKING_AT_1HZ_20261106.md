# Tracking for PET-DINO at 1 Hz: literature and implementation audit

Checked 2026-11-06. This review targets one received RGB image per second from a
moving UGV, 28 current classes with future class additions, and one creation
event per physical object. PET-DINO remains the few-shot detector. Research
claims below come from papers and official code; the recommendations are our
inference about suitability, not measured winners on PUI.

## What tracking does

Detection answers **what is visible, and where?** Tracking answers **which
previous physical instance does this detection belong to?** The persistent
registry answers **have we already reported that instance?** These are separate
tasks. Known objects should still be detected and shown with their IDs; only
their creation event is suppressed. Skipping a class after seeing it once would
also miss new objects of that class.

On each received frame, a tracking-by-detection system typically:

1. Receives boxes, scores, labels and optional instance appearance descriptors.
2. Predicts existing tracks' positions and uncertainty, usually with a motion
   filter. Camera-motion compensation can adjust the prediction for robot motion.
3. Scores detection/track pairs using geometry, appearance, time and compatible
   object types. A one-to-one association assigns detections to tracks and leaves
   uncertain pairs unmatched.
4. Updates matched tracks, retains briefly missing tracks, and confirms new
   candidates across observations. Weak detections can recover an established
   track without being allowed to create a new object by themselves.
5. Looks up a persistent object UUID. An existing UUID gets an observation
   update; a new confirmed UUID creates one local event. API delivery stays off.

At 1 Hz there are no intermediate observations for the tracker to inspect. A
50-pixel-wide person moving 100 pixels between samples has zero box overlap
even if the detector is perfect. Constant-velocity extrapolation also becomes
less reliable over the larger gap, particularly during camera turns or an
occlusion. Lowering a detector confidence threshold cannot solve this geometric
association failure.

Appearance helps, but two responders wearing similar uniforms, two cones, or
two neighbouring trees can look alike. A pooled PET-DINO feature is inexpensive
and useful, but is not equivalent to an instance-ReID model trained to separate
individuals. A colour histogram is an even weaker identity cue. Removing all
spatial checks and matching solely by similar appearance can create false merges.

Class prediction should be an attribute of an identity, with compatibility
rules. A person changing from a `citizen` prediction to `first responder` should
not automatically become a new physical person. Similar family rules could be
evaluated for green/dry trees. This does not mean freely merging unrelated
classes. The current implementation groups person roles only.

Not all support-bank labels define countable instances. Vehicles, people, cones,
tools and individual trees can have object identities; `dirt road`, `grass`,
`mud`, or changing smoke regions need a region/event or map-based definition.
A viewport-sized grass box is not a reliable unit for counting unique objects.
This review does not silently change which ontology classes are exported.

## Current implementation and verified repairs

UDC is our adapted version of
[theodouk11/udc_tracker](https://github.com/theodouk11/udc_tracker), based on
revision `11188042c8b487374a75b609424998df3f041522`. Its normal association uses
Kalman prediction and IoU, with high/low confidence stages. Later gallery
recovery uses HSV-histogram correlation, not PET-DINO features. There is no
camera-motion compensation in UDC. At 1 Hz, ten lost ticks represent about ten
seconds, and the gallery lasts sixty ticks. A candidate gets only one gallery
match attempt after enough updates; loss/re-entry during that interval can
still produce fragmentation. The registry adds PET-feature cosine similarity
plus image overlap, so it can also fail after a large viewpoint change.

Our BoT-SORT wrapper uses the installed Ultralytics implementation, class
constraints, sparse optical-flow camera compensation and supplied detector
features. It is **not** the published FastReID-based BoT-SORT-ReID configuration.
Its configured proximity threshold 0.5 masks appearance matches when IoU is
below 0.5. Thus enabling appearance does not by itself allow zero-overlap links.
Changing that gate needs controlled testing rather than an unrestricted
appearance-only merge rule.

Repairs made in this audit:

- **Gate before assignment.** UDC previously solved unconstrained Hungarian
  assignment and rejected inadmissible edges afterward. A rejected edge could
  prevent a valid alternative assignment. Invalid edges now receive a penalty
  inside the optimization, favouring admissible match cardinality and then IoU.
- **Preserve the assigned detection.** UDC and the role layer previously
  recovered metadata by finding the nearest detection to a filtered box. A
  lagged box near a crossing could take another person's role or appearance.
  The UDC core now records the exact selected detection index; both wrappers
  retain matched role evidence and appearance through to the role layer.
- **Pass recovery candidates to BoT-SORT.** The PET/OPUS runner previously sent
  candidates down to 0.1 only to UDC, despite BoT-SORT having a low-score recovery
  stage. Both now receive their backend's tracking candidates. Raw output still
  uses the requested detector threshold; weak candidates are not automatically
  accepted as new objects.
- **Preserve indices through confidence filtering.** The installed Ultralytics
  version numbers detections separately within high/low subsets. Treating that
  index as a row in the original input could assign the wrong metadata. The
  wrapper now carries original row indices through slicing and initialization.
- **Enforce compatibility in every stage.** Ultralytics' low-score stage bypasses
  the first-stage custom distance function. Separate class-compatible pools now
  prevent cross-class recovery, while one shared camera estimate avoids repeating
  the expensive motion extraction. Person roles still share the internal person
  pool. Track ID allocation is local to each wrapper, unique across its pools,
  and cannot be reset by creating another class pool or wrapper.
- **Age tracks across timestamp gaps.** BoT-SORT now advances bounded missing
  ticks without fabricated detections and estimates camera motion between actual
  received images only. Received-frame update counts stay separate from those
  missing ticks. Regression coverage checks expiry and timestamp ordering.

These fixes repair the offline tracking integration; they do not introduce a
new tracking algorithm or establish robustness after arbitrary revisits. The
first two regression tests failed before the repairs and passed afterward. A
cross-backend weak-recovery regression exposed the additional BoT-SORT index
problem. The final shared suite passes **66 tests**.

The UDC lifecycle still requires repeated observations: a raw first-frame box
can be unconfirmed, a local track appears after another observation, and the
registry normally requires two consecutive tracked observations to create its
UUID. That can mean the third observed frame for a new object's creation event.
This delay is confirmation policy, not evidence that the detector missed it.

## Suitable methods and availability

All candidates below can be considered with fixed external detector outputs.
None has been measured on our PUI bags except the existing UDC/BoT-SORT
wrappers. Published frame rates and benchmark scores are not transplanted to
our hardware, labels or 1-Hz stream. Using released pretrained weights is
different from training a tracker on our own site.

| Method | Primary source / code | Why relevant | Limitation for this task | Assessment |
|---|---|---|---|---|
| MASA / MASA-R50, CVPR 2024 | [Paper](https://arxiv.org/abs/2406.04221), [official code](https://github.com/siyuanliii/masa), [model zoo](https://github.com/siyuanliii/masa/blob/main/docs/model_zoo.md) | Learns general instance association from unlabelled images. R50 has its own backbone and accepts other detectors' boxes; released R50 weights are listed. Evaluation includes TAO, whose annotations are at 1 FPS. | Adds an image encoder/ROI embedding stage. Need a PET adapter and measured latency. Default demo distance/FPS gates must be configured for this stream; it is not a guaranteed long-term revisit solution. | **First new candidate for all supported object classes.** No per-site training required to try released weights. |
| MASA-OA, Information Sciences 2026 | [Paper](https://www.sciencedirect.com/science/article/pii/S0020025526004640), [official code](https://github.com/gonzalocarreteroh/MASA-OA) | Extends MASA with assignment optimization and occlusion-aware appearance/geometry fusion using a Kalman filter. Official implementation uses MASA-R50 embeddings. | Reported evaluations are MOT15/16/17, not this moving-camera 1-Hz setting. Extra motion heuristics may need retuning. | Evaluate after vanilla MASA, reusing its embeddings to isolate association changes. |
| Deep OC-SORT, 2023 | [Paper](https://arxiv.org/abs/2302.11813), [official code](https://github.com/GerardMaggiolino/Deep-OC-SORT) | Adds adaptive appearance/ReID and camera-motion compensation to observation-centric motion tracking. Useful control for occlusion and nonlinear motion. | Published model is multi-pedestrian tracking. Its standard ReID weights do not establish identity quality for trees/cones/tools; IoU-dependent matching must be audited at 1 Hz. | Strong practical control, especially for the person subtask. PET features versus pretrained ReID are separate variants. |
| BoT-SORT-ReID, 2022 | [Paper](https://arxiv.org/abs/2206.14651), [official code](https://github.com/NirAharon/BoT-SORT) | Motion, appearance and camera compensation in an established external-detector pipeline. | Proximity gating can reject the large displacements we care about. Dense pedestrian benchmarks do not prove 1-Hz all-class behaviour. | Retain a corrected, explicitly configured control. Do not confuse our PET-feature variant with published ReID. |
| Deep-EIoU, WACV workshop 2024 | [Paper](https://arxiv.org/abs/2306.13074), [official repository linked by authors](https://github.com/hsiangwei0903/Deep-EIoU) | Combines appearance with iteratively expanded box overlap, avoiding reliance on a conventional Kalman motion assumption for irregular motion. | Sports ReID is domain-specific; expanding overlap too far can confuse nearby similar people or objects. | Good geometry/appearance alternative when ordinary overlap matching breaks. |
| TrackTrack, CVPR 2025 | [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html), [official code](https://github.com/kamkyu94/TrackTrack) | Track-focused association and track-aware initialization target poor matches and redundant track births. | Published evidence mainly covers pedestrian/dance benchmarks rather than arbitrary classes sampled at 1 Hz. Features, initialization gates and timing need testing. | Useful modern comparison for fragmented/duplicate births, not an automatic fix. |
| BoostTrack / BoostTrack++, 2024; journal 2025 | [Paper](https://arxiv.org/abs/2408.13003), [official code](https://github.com/vukasin-stanojevic/BoostTrack) | Uses richer similarity and tracklet information to boost weak but plausible detection scores. | Can reinforce a persistent false detection. PET's grass-as-cone failure must not become a permanent boosted identity. | Secondary candidate for intermittent low confidence, with false-positive checks. |
| ARC-SORT, October 2026 preprint | [Paper](https://arxiv.org/abs/2610.22306), [official code](https://github.com/Athlitix/arcsort) | Uncertainty-adaptive expanded overlap, multiple motion models, camera compensation and appearance target fast motion under moving cameras. Public package/source are available. | Sports-focused weights; exact 1-Hz transfer untested. Its full reported pipeline includes interpolation and offline merging, which cannot be counted as causal online performance. | Promising recent experimental comparison; evaluate online-only and include actual ReID cost. |
| GLoMOT, AAAI 2026 | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/42500), [official code](https://github.com/sad123-yx/GLoMOT) | Explicitly designed for low-frame-rate online MOT. GNN associations combine position/appearance context and a node buffer for absent objects. | Different feature/domain distribution from PET and arbitrary classes. A graph checkpoint is present, but separate FastReID weights/embeddings are expected. Released gap-2/5/10 examples are not proof of performance at exactly 1 Hz. | **Most directly relevant recent LFR research candidate**, after a simpler baseline. |

Availability checks were made against official files, not just paper claims.
MASA's model zoo lists `masa_r50.pth`; the independent configuration is
[`masa_r50_plug_and_play.py`](https://github.com/siyuanliii/masa/blob/main/configs/masa-one/masa_r50_plug_and_play.py).
Its tracker exposes an FPS parameter and distance weighting, which need explicit
1-Hz settings. GLoMOT's repository contains
[`models/mot17_ablation.pth`](https://github.com/sad123-yx/GLoMOT/blob/main/models/mot17_ablation.pth),
about 1.16 MB; this is the graph checkpoint, not the full ReID pipeline.

## Relevant work that is not a first implementation choice

- **GLoMOT+ (2026):** the [official repository](https://github.com/sad123-yx/GLoMOT_plus)
  describes extended discriminative learning, but the inspected root contains
  only README, licence and gitignore. The code release is pending; do not treat
  README evaluation instructions as an available implementation.
- **SLAck (ECCV 2024):** [paper](https://arxiv.org/abs/2409.11235) learns semantic,
  spatial and appearance fusion for open-vocabulary tracking. However, the
  [official repository](https://github.com/siyuanliii/SLAck) currently contains
  README/docs, with the implementation described as forthcoming.
- **SORT-LFR (IEEE TMM 2026):** [primary publication](https://ieeexplore.ieee.org/abstract/document/11329201)
  proposes feature-first matching, acceleration-aware state estimation, adaptive
  appearance updates and trajectory-covariance threshold tuning. Its evaluation
  is vehicle/surveillance oriented. I could not verify a public author code and
  weights release in this check; it is a design reference rather than a ready
  drop-in candidate.
- **PuTR-CouT (September 2026):** [paper](https://arxiv.org/abs/2609.05038) is
  explicitly about roughly 1-FPS counting-by-tracking. Its synthetic training
  exploits static camera-trap backgrounds and short bursts; those assumptions
  differ from a continuously moving UGV. It is not our first replacement.
- Plain **SORT/ByteTrack/OC-SORT** remain inexpensive controls; another
  overlap/motion-only implementation does not directly supply general ReID.
  **StrongSORT** is an appearance baseline, but the extra offline components of
  **StrongSORT++** must be separated from online results. **SUSHI** provides a
  useful offline association comparison, not an immediate creation-event policy.
- **SAM2/Cutie/DEVA** mask propagation can maintain visible regions but requires
  masks/extra compute and does not by itself solve instance re-identification
  after sparse observations and revisits. It is not enough to replace a
  persistent registry with a video-mask ID.

## Recommendation and controlled test plan

**First compare PET-DINO + MASA-R50 against corrected UDC and an explicit
appearance/CMC control such as Deep OC-SORT or BoT-SORT.** MASA is the closest
design match for changing object classes; GLoMOT is the closest match to the
low-frame-rate research question. This is a prioritization judgment, not a
claim that either has already won on PUI. MASA-OA can then reuse MASA embeddings
to test its association changes without changing the detector or encoder.

1. Freeze the 65 frames, timestamps and PET-DINO 800/all-support predictions in
   `samples/opus_20261106/pet_reference`. Reuse its ranked candidates and cached
   appearance. A tracker comparison must not change detections, support examples
   or label mappings between methods.
2. Keep birth, recovery and display confidence policies explicit. Test motion
   plus PET features separately from additional pretrained appearance models.
   Do not interpret a lower output count as lower duplication.
3. Configure time windows in seconds, set actual 1-Hz inputs, and check
   appearance/spatial gates for zero-overlap cases. Assess camera compensation
   with large inter-frame displacement and reject unreliable warps. A 2D warp
   cannot fully remove depth-dependent parallax from UGV motion.
4. Annotate consistent physical instance IDs on representative query frames,
   including two similar objects, camera turns, role flips, missed detections,
   occlusions and revisits. Support labels alone are insufficient tracking GT.
   Report IDF1, HOTA/association accuracy, switches and fragmentation, plus
   duplicate creation events **and false merges**. Validate on held-out clips.
5. Measure ReID extraction and camera compensation as well as association. A
   cached-embedding CPU timing is not the full cost of a learned tracker. Report
   complete-frame mean/p95, frames exceeding one second and memory, with startup
   separate. The recent PET baseline was about 0.423 s per timed warm frame, so
   there is measured headroom, not a guarantee for a new model.
6. Keep a persistent UUID registry above the tracker. For static site revisits,
   integrate verified depth/pose/camera geometry into a world-coordinate map
   when available. People/vehicles additionally need dynamic identity reasoning.
   The current bags' missing CameraInfo/TF and unvalidated hardcoded intrinsics
   prevent claiming a calibrated spatial registry today. See
   [persistence limitations](../PERSISTENT_DISCOVERIES.md).

The tracker does not perform K-way N-shot recognition itself. PET-DINO handles
the support examples; the tracker associates its detections. General
class-agnostic instance matching allows class additions without learning new
per-class tracker heads, while semantic compatibility remains a configuration
choice. No existing tracker guarantees zero duplicate physical identities from
RGB alone in ambiguous scenes.

## Local correctness replay

The audit replays identical cached PET detections; no detector inference, bank
update, model training or API delivery is performed. Before/after runs use fresh
independent registries, so UUID strings cannot be compared as physical truth.

- UDC + person roles + low recovery: **189 to 190** observed track outputs,
  **23 to 23** registry objects, **7 to 7** person UUIDs, and zero same-family
  person overlap pairs in either run. Corrected tracker mean **3.18 ms**, p95
  **5.23 ms**; feature extraction is already cached.
- UDC without person grouping still has 33 overlapping person hypothesis pairs;
  grouping removes that particular proxy on these frames. This is not a measured
  duplicate-object rate, and zero overlapping boxes does not establish identity
  correctness.
- Corrected BoT-SORT with person grouping and low-score recovery has 118
  observations, six person UUIDs and zero overlapping person pairs. Association
  plus shared camera compensation averages **27.24 ms**, p95 **43.75 ms** with
  cached PET descriptors. UDC and BoT-SORT have different confirmation/association
  behaviour; 190 versus 118 observations is not an accuracy ranking.
- The two synthetic regressions isolate failures that aggregate counts can hide:
  lost admissible assignments, and a filtered box taking a neighbour's role and
  appearance. The cross-backend recovery test verifies that low-tier input
  slicing preserves the original detection identity.

Outputs are under `samples/tracker_audit_20261106/before` and `after`; corrected
BoT-SORT recovery controls are stored under `botsort_controls_v3`. Earlier
experiment outputs remain untouched. These results validate integration and
export invariants; tracking accuracy remains unlabelled.

[Open the raw/before/after viewer](../../samples/tracker_audit_20261106/comparison.html).
New learned tracker candidates in the literature table have not been installed
or benchmarked by this audit.
