# PUI few-shot perception and new-object delivery

The objective is to discover objects using labelled supports, maintain their identities across frames, and emit each newly discovered object once. SANSA is a baseline adapter, not a required model. API traffic stays disabled during development.

## Proposed execution

1. Decode timestamped RGB/depth/pose; pair depth and robot pose to each RGB capture timestamp, and keep a bounded latest-frame queue. Use capture-time pose when placing delayed inference results in the map. Never let inference backlog grow as the robot moves.
2. Encode support images/masks once when the bank or selected classes change. Prefer backends that can cache support representations and share one query backbone across classes. The present SANSA adapter runs an independent support/query episode for each selected class on every inference call; its CPU support tensors are cached, but model feature computation is repeated.
3. Detect on scheduled keyframes. Begin by benchmarking a configurable maximum detector rate, then trigger earlier when motion, scene change, unmatched foreground, or uncertain tracks justify it. Retain a maximum refresh interval so stationary or newly entering objects are still discovered. Suppressing repeated outbound events alone does not reduce detector compute. Choose rates from measurements on the deployment hardware.
4. Associate observations with tracks using class, image overlap, depth, and robot-motion compensation. Propagate tracks cheaply between detector calls. Use elapsed seconds for aging and confirmation when detector intervals vary. Advance tracks on empty observations too; the present early return skips that update.
5. Associate confirmed tracks with a session object registry in a fixed local world frame. Keep a stable object UUID across tracker loss/reacquisition. Use spatial uncertainty, appearance, class, and one-to-one assignment; radius alone can merge nearby people. Treat moving objects separately from stationary map assets. Without reliable pose or appearance, reidentification after losing a track is uncertain and must be measured.
6. Emit `object_created` only when a confirmed observation cannot be matched to a known object. Reobservations update internal state without another creation event. Maintain separate live perception and discovery event streams.

Example: frame A confirms a person and truck → two creation events. Frame B sees them again → zero. Frame C also confirms a second person → one. A temporary occlusion or improved confidence must not itself create another object.

## Delivery semantics

Log new-object events locally as JSONL while the API is disabled. Keep pending events until delivery succeeds; suppressing an event on observation is different from acknowledging its delivery. A depth-one queue that replaces pending deltas would lose discoveries, so retain a pending event map or durable outbox instead. Use the stable UUID as an idempotency key when supported by the destination. Persist the registry/outbox if identity must survive process restarts. A client alone cannot guarantee exactly-once creation after an ambiguous network failure when the API offers neither idempotency nor a safe identity lookup.

Current code already tracks observations and accumulates geographic features, but publishes frame GeoJSON repeatedly and periodically considers the whole accumulated collection for API upsert, including PATCH on confidence improvements. Those behaviors are different from the requested creation-only event stream. Existing geographic merging uses a fixed 3 m radius; it is a baseline, not a sufficient identity model.

## Evaluation on paired recordings

Use one ROS player with both original experiment bags so their recorded timestamps share one timeline. Measure end-to-end latency, processed versus received frames, inference calls per second, peak GPU memory, unique discoveries, duplicate creation events, identity switches and missed objects. Evaluate reappearance and nearby same-class objects explicitly. Frame extraction exports every RGB frame independently of inference speed.

The extracted frames are unlabelled query data; the annotation ZIP supplies supports. Verify that query frames and support images do not contain the same source frames before claiming independent-query accuracy. Use a manually labelled subset and identity annotations for quantitative accuracy and duplicate rates. ROS regression tests and successful playback alone do not measure detection quality.

Implementation order: verify the paired-bag baseline; add the local creation-event registry/outbox and identity tests; add bounded inference scheduling and time-based tracking; compare interchangeable few-shot backends under identical support/query splits and compute budgets. Do not choose a backend on segmentation accuracy alone.
