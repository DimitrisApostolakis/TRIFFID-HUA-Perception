# TRIFFID UGV/UAV Perception Module (T4.1)

**Harokopio University of Athens, TRIFFID Research Project**

**Contact: amichailidou@hua.gr**

## Current modules

| Module | Input | Output |
|---|---|---|
| UGV | Live RGB-D, CameraInfo/TF and robot GPS/heading | ROS detections, masks, debug images, persistent identities and geographic GeoJSON; optional durable PUT/PATCH delivery |
| UAV | Recorded drone video with DJI SRT, selected frames, or a supplied image | Versioned pixel-space masks/point observations; optional offline MASA association; geographic projection in a separate Unity project |

Automatic API publishing is disabled by default. The UAV CLI is a recorded-media
pipeline; continuous UAV live inference is not yet integrated.

## UGV

The selected live profile uses **YOLOE-v8-L + MASA-R50**, an independent visual
support bank, calibrated RGB-D localization and persistent re-ID evidence. The
legacy launch retains SANSA defaults. Fence/wall export as LineStrings; other
UGV classes export as Points.

On the prepared RTX 3060 workstation:

```bash
docker compose -p triffid-ugv-live \
  -f docker-compose.yml -f docker-compose.ugv-live.yml up -d perception
```

The profile mounts an externally prepared model runtime and weights. Follow the
setup guide before using it on a new machine.

- [Architecture and code structure](docs/UGV_ARCHITECTURE.md)
- [Live deployment and verification](docs/UGV_LIVE_YOLOE_MASA.md)
- [Persistent PUT/PATCH delivery](docs/UGV_API_DELIVERY.md)
- [Module setup and parameters](src/triffid_ugv_perception/README.md)

## UAV

Python selects views, runs support-conditioned segmentation, extracts connected
components and exports masks at original image resolution. Optional MASA
tracking retains one representative observation per confirmed identity. Unity
uses capture telemetry, camera calibration and the scene mesh to produce
geographic GeoJSON.

```bash
PYTHONPATH=src/triffid_uav_perception \
  python -m triffid_uav_perception.uav_node --help
```

The main CLI currently selects SANSA. FoRIS, INSID3, FROST, RSRM, FSSDINO and
FSS-SAM3 adapters are exercised through dedicated runners. RSRM/FSSDINO tests
use the available DINOv3-L replacement. UAV identity persistence is scoped to one
tracking run; it does not yet share the UGV durable API publisher.

- [Architecture, models and Unity boundary](docs/UAV_ARCHITECTURE.md)
- [CLI commands and output schema](src/triffid_uav_perception/README.md)
- [FSS-SAM3 experiment](docs/UAV_FSS_SAM3_TEST.md)

## Evaluation and reproducibility

- [Dataset preparation and split policy](docs/PUI_DATA_PREPARATION.md)
- [Model assets and prepared runtimes](docs/MODEL_SETUP.md)
- [Evaluation and production plan](docs/EVALUATION_AND_PRODUCTION.md)
- [Interface specification](INTERFACE.md)
- [What is included in Git and what remains external](docs/REPOSITORY_CONTENTS.md)

Code, tests, scripts, configuration and documentation belong in this repository.
Datasets, supports, model weights, environments, generated videos/exports and
SQLite mission state remain external. Some research runners and site configs
record workstation-specific paths; adjust them before using another machine.
The existing `best.pt` Git LFS pointer is historical and unchanged.

On the prepared development environment, run the package regression suite with:

```bash
PYTHONPATH=ugv_models/devit_runtime:src/triffid_uav_perception:src/triffid_ugv_perception \
  /home/oracle/miniforge3/envs/fewshot-dev/bin/python -m pytest -q --import-mode=importlib \
  src/triffid_ugv_perception/tests src/triffid_uav_perception/tests
```

## Repository layout

```text
configs/                         Site/model/suite configuration
src/triffid_ugv_perception/       ROS 2 Jazzy application and tests
src/triffid_uav_perception/       UAV Python application and tests
src/triffid_telesto/              Existing REST helper package
scripts/                         Preparation, evaluation and verification tools
docs/                            Architecture, setup, protocols and result summaries
Dockerfile, docker-compose*.yml   UGV build and deployment profiles
Dockerfile.uav                    UAV SANSA runtime definition
```
