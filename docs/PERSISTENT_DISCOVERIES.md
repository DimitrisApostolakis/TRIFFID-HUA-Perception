# Persistent discoveries at one frame per second

This document describes the archived offline JSON registry. For the current live
ROS SQLite outbox and delivery policy, see [Persistent creation-only discoveries](PERSISTENT_DISCOVERY_DELIVERY.md).

The current offline pipeline prevents repeated creation events for a known
registry UUID. It does **not** guarantee that every real-world object receives
only one UUID. Changing the detector does not remove this distinction.

Implemented protections:

- Mask overlap suppression removes duplicate instance hypotheses within a frame.
- Class-aware BoT-SORT updates on every sampled frame, including empty frames,
  and retains lost tracks for ten seconds at 1 Hz.
- Two consecutive tracked observations are required before creating a UUID.
- Lost identities can rebind using same-class appearance plus image overlap.
  Appearance descriptors reuse detector features; there is no extra ReID network.
- Rebinding retires the previous tracker alias, preventing a stale alias from
  assigning one saved UUID to two current tracks.
- Registry and creation-event JSON files are written atomically. Repeated
  observations do not create another event for the same UUID. Missing outbox
  files can be recovered from the registry after a crash.
- `--registry PATH` shares persistent state across process restarts. By default,
  its outbox is also persistent, beside that registry. `--outbox PATH` overrides
  the location. Omitting `--registry` creates an independent evaluation registry.
- Registry context binds it to the dataset-index fingerprint, detector backend
  and checkpoint. Incompatible experiment/model reuse fails instead of silently
  comparing descriptors from different embedding spaces. Tracker aliases are
  deliberately not trusted across restarts.

Example:

```bash
bash scripts/run_cosine_baseline.sh \
  --output samples/cosine_baseline/new_run \
  --registry samples/pui_registry/cosine_v1.json
```

The same experiment/model may reuse its registry; SINE and COSINE need separate
registries unless an explicit cross-model identity migration is implemented.
Replaying earlier images against a registry last updated at the end of a run
is not an exactly-once physical-object test: image-overlap matching depends on
viewpoint and previous observations. Use fresh registries for independent
algorithm comparisons and evaluate restart/revisit behavior separately.

Remaining work for robust physical deduplication:

1. Synchronize calibrated aligned depth, RGB, robot pose and heading at capture
   time. Transform masked depth points into a stable odometry/map frame using
   verified camera intrinsics and camera-to-robot extrinsics.
2. Associate static objects using world position, uncertainty, class and
   appearance. A single GPS-radius rule can merge neighboring cones or people;
   GPS precision and different object motion require explicit treatment.
3. Track moving people and vehicles with motion and appearance. A stationary
   world-position lookup cannot establish their identity after arbitrary motion.
4. Measure duplicates, false merges, missed objects and identity switches on
   labelled sequences with occlusion and revisits. UUID uniqueness alone does
   not measure these failures.

The current paired bag contains RGB, aligned depth, `/fix` and `/b2/ugv_status`.
It does not contain CameraInfo or TF topics. The existing node's hardcoded
intrinsics are not validated for these 1280x720 images; blindly treating them as
correct would produce misleading map associations. The offline baseline does
not currently extract or fuse depth/pose for its registry.

API traffic remains disabled. A future sender needs acknowledgement state and
server-side idempotency using the persistent object UUID. An atomic local outbox
alone does not guarantee exactly-once delivery across network retries.
