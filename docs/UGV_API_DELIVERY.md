# UGV API delivery

The YOLOE-v8-L + MASA live profile supports persistent-object PUT/PATCH delivery.
Automatic HTTP delivery remains disabled (`publish_to_api=false`). The legacy
profile remains creation-only unless `discovery_updates_enabled` is enabled.

## Identity and publishing

MASA's session-scoped track ID resolves to the registry's persistent client UUID.
The UUID is sent as Feature `id` and `properties.id`, `object_id` and
`client_object_id` during creation. The acknowledged API POI ID is stored against
that UUID. Re-identification changes the track association, not the client UUID
or the mapped POI. Ambiguous matches remain local. Disappearance does not DELETE
the POI.

The background worker uses:

| Operation | Endpoint | Body | Required acknowledgement |
|---|---|---|---|
| New confirmed, geographically located object | `PUT /wp-json/map-manager/v1/features` | Complete GeoJSON Feature with persistent client ID | HTTP 200/201 and nonempty top-level `id` |
| Changed known object | `PATCH /wp-json/map-manager/v1/features/{remote_id}` | Complete GeoJSON Feature; top-level `id` addresses the remote POI, properties retain client identity | HTTP 200 and `{"success":true}` |

PATCH matches the Unity client's full-Feature format and the manually successful
2026-11-08 altitude update. This verifies PATCH acceptance, not PUT retry
deduplication. Automatic delivery still requires the existing explicit
`api_uuid_uniqueness_verified` deployment assertion. The status endpoint
`/observer-sync/v1/status` is separate and is not modified by this sender.

## Live defaults

| Setting | Value | Effect |
|---|---|---|
| `discovery_updates_enabled` | `true` in live profile | Allows updates of already acknowledged POIs |
| `discovery_update_interval_s` | 2 seconds | Minimum interval after an acknowledgement before the next PATCH for that object; retries are also spaced at least this far apart |
| `discovery_update_distance_m` | 1 metre | Horizontal motion of a Point or corresponding LineString vertex triggers an update |
| `discovery_update_altitude_m` | 1 metre | Altitude, height, or geometry elevation changes trigger an update |
| `discovery_update_confidence_delta` | 0.1 | Confidence changes of at least 0.1 trigger an update |
| `discovery_flush_interval_s` | 1 second | Saves the latest known observation to durable state |
| `publish_to_api` | `false` | No automatic network requests |

Class, category, style and other stable property changes also trigger updates.
Small motion/measurement fluctuations and observation timestamps do not. Compare
changes to the last acknowledged Feature, so gradual movement accumulates rather
than being lost by comparing only consecutive frames. These are configurable
initial thresholds and still need validation against real geographic noise.

New eligible objects enter the queue immediately after confirmation. The worker
polls every 0.25 seconds. Known-object updates include up to one second of local
flush latency, the per-object interval and network time; this is not a promise
of exactly one HTTP request every two seconds. Static unchanged objects generate
no repeated API request. Local detections and GeoJSON retain frame-driven output.

## Durability and bounded state

SQLite stores the latest desired Feature, the last acknowledged Feature, the
remote ID, and at most one queued request per object. Pending unattempted updates
coalesce to the latest observation; if the object returns to its acknowledged
state before sending, the pending update is canceled. Creation requests and
attempted updates remain immutable. A newer observation arriving during a
request is retained and considered after acknowledgement.

Retries reuse the same payload and operation key. Creation keys derive from
namespace/client UUID; each PATCH has its own persisted operation UUID. Expired
leases survive restarts, and stale acknowledgements cannot remove a newer
lease. Permanent HTTP failures retain the queued event for review. A missing POI
does not silently create a replacement. Requests run outside the registry lock.

The existing queue count, payload size and SQLite/journal byte limits apply to
both operations. When capacity is occupied, the latest desired state remains on
the object for a later attempt to queue it. Additive schema migration preserves
existing identities, mapped POIs and queued creation payloads. Databases
are migrated to schema v2; older binaries reject that version rather than
mistaking queued updates for creations. Use a database backup when rolling back
to an older binary. Existing sent objects
without a saved Feature acquire their update baseline from a new valid sighting
and successful PATCH. `pending_updates` is exposed in discovery status.

## Validation

`tests/test_discovery_updates.py` covers coalescing, static jitter, rate limiting,
re-ID to the original POI, observations during creation, concurrent newer
observations, restart/lease recovery, immutable retries, shared capacity,
permanent errors, schema migration, and the successful PATCH response format.
Its API is fake; tests do not contact production.

Run the delivery tests:

```bash
PYTHONPATH=src/triffid_ugv_perception \
  /home/oracle/miniforge3/envs/fewshot-dev/bin/python -m pytest -q \
  --import-mode=importlib src/triffid_ugv_perception/tests/test_discovery_updates.py \
  src/triffid_ugv_perception/tests/test_persistent_discovery.py
```
