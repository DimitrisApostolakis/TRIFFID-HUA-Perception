# Re-identification audit and deployment decision

Inspected 2026-11-07. The offline UGV comparison and the ROS entry point currently use different identity systems.

Update 2026-11-08: the dedicated [YOLOE/MASA live profile](UGV_LIVE_YOLOE_MASA.md) now
wires MASA-R50 and appearance-aware registry transport into the ROS path. The legacy
launch remains available. The audit below records the earlier implementation and selection rationale.

| Path | Association / appearance | Persistent identity |
|---|---|---|
| `ugv_node.py` | Legacy `tracker.ByteTracker`: Kalman boxes, high/low confidence assignment, IoU and auxiliary 3D distance. No learned appearance encoder. | Existing ROS delivery path; it does not automatically use the offline MASA configuration. |
| `run_pui_comparison.py` | `TrackingPipeline('masa', person_roles=True)`: pretrained MASA-R50 instance embeddings with external detections. | `DiscoveryService`, SQLite records and outbox, namespace and checkpoint-specific embedding space. |
| Older DE-ViT/SINE/COSINE baselines | BoT-SORT with pooled detector region features and camera motion compensation. | RGB overlap plus compatible appearance; detector features are not a separately trained re-ID model. |
| UDC alternative | HSV histogram gallery recovery. | A tracking alternative, not learned instance re-ID. |

The MASA wrapper sets a 60-second memory, confidence gates 0.35/0.10 and category compatibility. Person roles can be merged into one physical-person track with temporal role evidence. Optional class correction has stronger association gates; it is not enabled merely by selecting MASA. Persistent discovery adds one-to-one UUID reassociation and confirmation. `DiscoverySettings` defaults include appearance similarity 0.90, margin 0.05, overlap 0.15 and world radius 3 m. These are existing settings, not thresholds validated by this two-image evaluation.

## Research and choice

Use **MASA-R50 as the first general-object re-ID candidate to validate**. It is already integrated in offline comparisons and was designed for general instance association alongside arbitrary detectors/segmenters. That reduces new integration work during the three-day window. This is an engineering recommendation; it is not an assertion that MASA is accurate enough on PUI. [Official MASA implementation](https://github.com/siyuanliii/masa), [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Matching_Anything_by_Segmenting_Anything_CVPR_2024_paper.pdf).

For people, compare **OSNet through Torchreid** as an optional separate appearance provider. Its person re-ID objective is relevant to citizen/first-responder continuity, but it should not replace general-object embeddings for cones, hoses, equipment and vehicles. Keep physical person identity separate from role classification. [Official Torchreid/OSNet repository](https://github.com/KaiyangZhou/deep-person-reid).

**BoT-SORT + FastReID** is another established people-tracking baseline, combining appearance with motion and camera compensation. The existing BoT-SORT wrapper here instead receives detector descriptors; its `with_reid=True` flag does not mean that the original FastReID network is installed. [Official BoT-SORT](https://github.com/NirAharon/BoT-SORT), [FastReID](https://github.com/JDAI-CV/fast-reid).

Do not change the live default on the basis of isolated-image detector scores. ROSBAG-PUI tasks 49/50 contain semantic/instance annotations but no validated temporal identities. They cannot establish re-ID accuracy, ID switches, HOTA or IDF1. Use an identity-annotated continuous segment and [TrackEval](https://github.com/JonathonLuiten/TrackEval) for HOTA/IDF1 and identity-switch metrics. Include disappear/reappear, two nearby same-class objects, changing human role labels and camera motion; report physical duplicate UUID creations separately.

## Module boundaries for deployment

1. **Detector / segmenter:** consumes an image plus frozen supports; returns original-resolution boxes, native scores, class IDs and optional masks. No tracking IDs or API writes.
2. **Appearance provider:** owns its weights, preprocessing, descriptor dimensions and embedding-space fingerprint. A model change must not reuse incompatible saved galleries.
3. **Association:** consumes timestamped detections/embeddings and keeps short-term track IDs, with motion, compatible categories and one-to-one assignment.
4. **Discovery registry:** owns mission-scoped UUIDs, confirmation, bounded durable state, reacquisition and event deduplication.
5. **Projection and delivery:** validates synchronized calibration/pose, produces world coordinates, and manages an idempotent durable outbox.

These boundaries already exist partly in `models/`, `tracking/`, `discovery/`, `calibration.py` and delivery code. Appearance extraction is still coupled to MASA's wrapper. Separating that provider and wiring the chosen association into `ugv_node.py` remain deployment work. No live tracker switch or production deployment was performed by the evaluation task.
