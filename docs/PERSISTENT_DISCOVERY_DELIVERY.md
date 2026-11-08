# Persistent creation-only discoveries

This document describes the legacy creation-only mode. The selected live profile
now also supports [durable PUT/PATCH updates](UGV_API_DELIVERY.md), including
coalescing and a two-second per-object limit. Automatic uploads remain disabled.

Optional all-class temporal correction and its YOLOE verification are documented
in [CLASS_CORRECTION.md](CLASS_CORRECTION.md). It revises local GeoJSON features
without creating another API event.

Implemented on `refactor/ugv-sansa-modular`. API delivery stays disabled. No
external API was called and no detector evaluation was run for this change.

## Runtime flow

`tracked observation → confirmation → persistent object UUID → valid-location gate → SQLite outbox → background create → acknowledgement`

Two consecutive observations above 0.35 confidence, at most 2.5 seconds apart,
confirm a new object. Empty, low-confidence and out-of-order frames cannot
advance confirmation. The UUID and any eligible creation payload commit together
before the object is reported as newly discovered. Existing UUIDs update local
class, confidence and identity evidence; they never produce another creation or
API update. Person role changes do not create another physical-person identity.
The live per-frame GeoJSON topic still includes current observations, with
`object_id` and `discovery_state`; it is not the API creation queue.

Confirmed objects without a usable world location stay local. Upload eligibility
requires valid GPS, the selected heading topic, receipt age at most three seconds,
and GPS header time within three seconds of the RGB observation where a header
exists. Seed/API-only position is ineligible. Float32 heading has no acquisition
stamp, so only receipt freshness can be checked. This gate does not establish
survey accuracy or correct magnetic declination. Calibration, synchronized depth
and world-pose accuracy still determine map accuracy. Per-frame GeoJSON has
`geometry: null` while the location gate fails.

## Modules

| File under `triffid_ugv_perception/` | Responsibility |
|---|---|
| `discovery/settings.py` | Confirmation, association, retry and resource limits |
| `discovery/registry.py` | Bounded candidates/aliases, confirmation, UUID association, local updates |
| `discovery/features.py` | Compact encoding, physical class groups, identity and location checks |
| `discovery/store.py` | SQLite transactions, immutable outbox, leases and acknowledgements |
| `discovery/transport.py` | Creation-only HTTP transport and response validation |
| `discovery/outbox.py` | Background network worker with backoff; no network under the database lock |
| `discovery/service.py` | Application boundary, bounded logs and status |
| `geojson_bridge.py` | ROS adapter, position gate, status topic and optional snapshots |
| `storage.py` | Bounded debug-image retention and atomic size-limited GeoJSON snapshots |

No new model or ReID network is added. Rich track dictionaries may include
`bbox` and normalized `appearance` evidence from an existing tracker. Reassociation
requires strong compatible appearance plus image overlap or nearby valid world
position. Similar alternatives and position-only matches are held as uncertain.
Each UUID may be assigned to at most one track per frame. Tracker aliases expire
and are deliberately not trusted after a restart. The embedding-space identifier
and mission/camera namespace must match the database on reopening.

**Identity limitation:** the existing ROS `Detection3DArray` carries tracker IDs
and 3D boxes, but no image boxes or appearance descriptors. Stable current tracker
IDs are supported directly. Ambiguous reappearances cannot safely recover a
physical identity from this message alone and may be withheld. A changed ID with
sufficiently different position may still be treated as a new object. Full
physical-object deduplication across arbitrary motion, occlusions and revisits
is not guaranteed. The rich registry interface supports visual evidence when the
tracking integration supplies it; this change does not replace the live tracker.

## Durable delivery contract

The outbox has one immutable payload per UUID. An acknowledged event is removed
from the queue; its compact `sent` record and remote ID remain for the mission.
Failures use exponential backoff from one to 60 seconds. An in-flight lease expires
after 30 seconds, with a unique lease token so an old response cannot acknowledge
a newer attempt. Permanent 4xx failures are retained as blocked entries for review.

The transport uses the existing API's `PUT` creation verb. It sends
`Idempotency-Key: <namespace>:<UUID>` and the same `namespace` and
`client_object_id` properties on every retry. There are no proximity GET, PATCH
or update requests. A validated nonempty response `id` is required to acknowledge.

**Before enabling delivery, the server must enforce unique
`(namespace, client_object_id)` and return the original remote ID on replay.**
A request header alone cannot make an API idempotent. The current backend has not
been verified or modified. `publish_to_api:=true` fails unless
`api_uuid_uniqueness_verified:=true` is also explicitly set; that second flag is
an operator assertion of the server contract, not an automatic server check.
A lost response after a server commit otherwise permits duplicate remote rows.

## Default storage and operation

Compose mounts `./state:/ws/state`. Use a distinct persistent directory/database
and namespace for each mission/camera. Preserve that directory across restarts;
deleting it also deletes duplicate-suppression history. Do not share a database
between different embeddings or independent model experiments. A nonblocking owner
lock rejects a second service opening the same database.

| Resource | Default limit/behavior |
|---|---|
| Objects including all sent IDs | 10,000; no automatic eviction |
| Unconfirmed candidates | 512 in memory |
| Pending/in-flight/blocked creations | 1,000 and 8 MiB encoded payloads |
| One encoded feature | 16 KiB |
| SQLite plus rollback journal | 64 MiB budget; conservative page cap reserves journal/header overhead |
| Appearance | At most 512 float32 dimensions per object |
| Observation refresh | One latest state per object, flushed every five seconds |
| Discovery logs | 1 MiB × 3 files including backups |
| Live frames/video | No files by default |
| Optional RGB / thermal debug | Each directory: 100 PNGs and 64 MiB; oldest managed files rotate |
| Optional GeoJSON snapshots | Disabled; one current snapshot plus optional fixed-stamp snapshot, each at most 4 MiB |

The database can fill before the object-count limit. Capacity exhaustion preserves
sent identities and pending payloads, pauses new registrations/queue admissions,
and leaves tracking/status active. Waiting objects can enter the outbox after
space becomes available. Payloads are deleted only following acknowledgement.
Database pages are reused and incrementally reclaimed; there is no growing WAL
or per-frame history. Never evict sent identities just to make room: doing so
would allow the same object to be created again.

`/ugv/detections/front/discovery_status` reports object, sent, pending, waiting,
blocked and byte counts plus capacity pressure. Managed-file limits do not cover
raw bags, model caches, historical samples or unrelated system/ROS logs. No
historical artifacts are deleted. The archived offline runners retain their old
JSON registry; this SQLite service is wired into the live ROS bridge and does not
automatically import old JSON identities or historical remote rows.

## Validation and measured overhead

All 102 package tests passed. Unit/integration checks use synthetic observations, ROS doubles and fake servers,
including a server that commits a creation then loses its response. Checks cover
confirmation, repeats, role changes, restart association, ambiguity, conflicting
IDs, immutable retries, stale leases, permanent failures, queue/object/page caps,
exclusive ownership, API-off behavior, network concurrency, position freshness,
null invalid geometry and bounded optional artifacts.

A synthetic run on the project's mounted drive processed 1,200 frames, with 20
objects per frame and one logical second between observations. API was disabled;
no detector or tracker inference was measured. There were exactly 20 object rows,
20 queued events and 53,248 bytes of managed files after the run.

| Operation | Median | 95th percentile |
|---|---:|---:|
| Ordinary observation frames (960) | 1.205 ms | 1.431 ms |
| Periodic durable refresh (239) | 21.237 ms | 23.921 ms |
| Initial durable confirmation of 20 objects (one frame) | 226.057 ms | one sample |

Disk commits occur on the observation path; network requests run separately.
These measurements are local overhead estimates, not live robot latency or
physical-identity accuracy results. New-object bursts and slower storage can cost
more than ordinary frames.

Repeat the synthetic timing run from the repository root:

```bash
PYTHONPATH=src/triffid_ugv_perception python scripts/benchmark_discovery.py --work-dir /path/to/project
```

The benchmark removes its temporary database on exit. Its work directory selects
the filesystem being measured; omit it to measure the system temporary directory.
