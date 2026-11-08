# Repository contents and external assets

This source snapshot includes UGV/UAV inference adapters, tracking, calibration,
support-bank tooling, durable UGV PUT/PATCH delivery, evaluation runners, tests,
container configuration, example schemas and documentation. API publishing is
disabled in the selected UGV live profile.

## Included in Git

| Content | Location |
|---|---|
| Application code and tests | `src/` |
| Evaluation, import, verification and preparation tools | `scripts/` |
| Configuration, selected model revisions and checkpoint hashes | `configs/`, UGV `config/` |
| Docker build/deployment definitions | Root Dockerfiles and Compose files |
| Architecture, protocols, recorded result summaries and example schemas | `docs/`, module READMEs, `INTERFACE.md` |
| Adapted UDC tracking primitives and attribution | `tracking/vendor/udc_core.py`, `UDC_LICENSE`, `UDC_NOTICE.md` |

Research/model adapters reference external implementations. The adapted UDC
primitives retain their accompanying license and modification notice. Other
model source checkouts are provisioned externally; this snapshot does not copy
their complete repositories or redistribute their weights.

## Kept outside this push

| Content | Local locations/examples |
|---|---|
| Model weights, vendor checkouts, compiled extensions and environments | `ugv_models/`, `uav_models/`, `.venv/`, checkpoint files |
| Real support images, masks and annotations | `ugv_supports/`, `uav_supports/`, `data/` |
| Drone video, SRT and ROS recordings | `uav_videos/`, `uav_data/`, `*.mp4`, `*.srt`, bags |
| Generated previews, full metrics, exports and debug images | `samples/`, `uav_samples/`, detection/thermal frame directories |
| SQLite identity/outbox databases and runtime logs | `state/`, `log/` |
| Build artifacts and caches | `build/`, `install/`, Python/test caches |
| Local backups and transfer/install archives | `archive/` |
| Credentials and private keys | Environment/credential files; never source literals |
| Separate Unity projects | `UnityGaussianSplattingAutoWatcher`, `Unity-AR-TRIFFID` are separate repositories |

These exclusions preserve the local files. They are not deleted when the source
snapshot is committed. Selected compact result summaries can be committed as
documentation; complete result/media packages need separate explicit handling.

`best.pt` is already tracked as a Git LFS pointer in repository history. It is
unchanged in this snapshot. Adding checkpoint patterns to `.gitignore` does not
remove an existing tracked file or its history.

## Reproducing this workstation snapshot

Use the model setup documents, independent UAV/UGV support banks, frozen split
manifests, source revisions and checkpoint/support hashes. Obtain external
assets through their authorized sources. Store credentials in environment
variables or local ignored files.

Some scripts and site configs contain explicit workstation paths such as
`/mnt/d/...` and `/home/oracle/miniforge3/...`. These are reproducibility records
for this installation, not portable defaults. Change them for another machine.
`configs/uav_pui.json` also records the older UAV baseline commit/branch; it is
provenance, not a declaration of the current repository HEAD.

The selected UGV live profile uses a prepared model runtime mounted into the
ROS container. It is not a self-contained model image. The UAV Compose service
starts a prepared SANSA shell; other evaluation backends have separate runtimes.
Original datasets and generated artifacts referenced in documentation are
intentionally absent from a clean clone.

Validation for this snapshot: the combined UGV/UAV package suite passed 244
tests locally. Optional vendor/runtime-dependent tests may skip in a clean
environment. Real camera/geographic accuracy and cross-flight re-ID acceptance
remain separate from source publication.
