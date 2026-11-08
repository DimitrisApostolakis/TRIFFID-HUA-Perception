# PET-DINO updated bank and controlled study, 5 November 2026

The new export `project-4-at-2026-11-05-14-38-2dc40cb7.json` is imported.
SHA256: `b0adb1231ce44f169fd598bcb41a5d07c039e5abe4eaf995d66158d38285565b`.
The active bank contains 141 populated class/image slots from 25 images,
covering 28 classes. It includes three citizen images, four first-responder
images, three civilian-vehicle images and six cone images. Counts refer to
distinct support images per class, not the number of annotated instances.

Tasks 29, 31, 33 and 34 remain explicitly excluded as instructed. The export
still contains them. Six new originals were found in
`D:\Datasets\PUI\PUI - Raw_data\generic_capturing` and copied into the portable
bank revision. Original raw data was not modified. All support masks are decoded
from annotations, not generated predictions. Three brush-labelled regions lack
pixel RLE and are recorded as skipped (tasks 25/dry tree, 43/dry grass,
44/green tree). Each class is capped at 16 supports; dirt road has 18 distinct
annotated candidates, of which 16 are selected. Existing bank/report revisions
are archived under `ugv_supports/history/`.

Current supports:

| Class | Images | Class | Images |
| --- | ---: | --- | ---: |
| fence | 2 | green tree | 16 |
| smoke | 1 | first responder | 4 |
| fire hose | 3 | boot | 4 |
| green plant | 8 | window | 4 |
| building | 7 | ladder | 3 |
| dirt road | 16 | dry tree | 3 |
| wall | 4 | civilian vehicle | 3 |
| citizen | 3 | green grass | 13 |
| pole | 2 | dry grass | 5 |
| door | 4 | mud | 1 |
| bag | 3 | fire truck | 5 |
| cone | 6 | dry plant | 5 |
| furniture | 11 | barrel | 1 |
| shovel | 2 | extinguisher | 2 |

## Available-backbone constraint

The authors supply Swin-T and Swin-L configurations, but their public model
repository currently lists only the trained PET-DINO Swin-T checkpoint. Local
checkpoint search found no PET-DINO Swin-L weights, and the user confirmed none
are available. Therefore a trained backbone-size accuracy/timing ablation
cannot be completed with available assets. Swapping in an independently
pretrained Swin-L or loading Swin-T weights non-strictly would not produce a
valid PET-DINO Large comparison. No such substitution was performed.

Checkpoint availability was checked at
`https://huggingface.co/api/models/fuweifu/PET-DINO` and model search at
`https://huggingface.co/api/models?search=PET-DINO&limit=50`.
The adapter now accepts `--pet-backbone swin-l` only with an explicit checkpoint,
and strict tensor loading remains mandatory. Large inference is not validated.

## Controlled available-model study

All runs use the same pretrained Swin-T checkpoint, 28 classes, 65 original
1-Hz query frames, FP32, score threshold 0.35, identical tracking settings,
independent registries and disabled external APIs. Support prompts use the box
around the largest connected annotated region. Raw ranked candidates are saved
before the operating threshold, enabling later COCO evaluation without another
model run.

| Setting | Supports | Nominal short / max edge |
| --- | --- | --- |
| `swin_t_800_1shot` | First image per class (28 total) | 800 / 1333 |
| `swin_t_480_all` | All valid bank supports (141 total) | 480 / 800 |
| `swin_t_640_all` | All valid bank supports (141 total) | 640 / 1066 |
| `swin_t_800_all` | All valid bank supports (141 total) | 800 / 1333 |

The first and last rows isolate support count at fixed resolution. The three
all-support rows isolate resolution with identical support identities. Changing
resolution also changes the resized support representations. This is an
input-resolution/support-count study, not a backbone-size study. Aspect ratio
is preserved; the maximum-edge cap may reduce the actual short edge below the
nominal target. Cache construction and checkpoint loading are measured
separately from frame processing. Runtime measurements are desktop RTX 3060,
not Jetson results or a live hard-deadline test.

Measured complete frame-processing latency (including tracking and export):

| Setting | Mean | p95 | Maximum | Preparation |
| --- | ---: | ---: | ---: | ---: |
| 800, 1 shot/class | 441 ms | 473 ms | 720 ms | 60.6 s |
| 480, all supports | 308 ms | 327 ms | 488 ms | 88.6 s |
| 640, all supports | 362 ms | 376 ms | 561 ms | 93.2 s |
| 800, all supports | 445 ms | 473 ms | 627 ms | 100.1 s |

All 65 frames in each run finished below one second. At 800, the all-support
mean was about 1% above one-shot; most extra support work happened during
preparation. Preparation is excluded from frame latency and must complete
before accepting a live stream. These measurements establish timing feasibility
on this GPU, not an accuracy winner.

Artifacts: [interactive frame comparison](../samples/pet_ablation_20261105_v2/comparison.html),
[timing plot](../samples/pet_ablation_20261105_v2/timing_plot.png), and
[machine-readable results](../samples/pet_ablation_20261105_v2/study_results.json).

Reproduce the completed matrix (finished rows are skipped):

```bash
/home/oracle/miniforge3/envs/fewshot-dev/bin/python scripts/run_pet_ablation.py
```

Individual run:

```bash
bash scripts/run_pet_baseline.sh --output samples/pet_baseline/new_run \
  --use-all-supports --image-size 800 --max-size 1333 --precision float32 --confidence 0.35
```

`--use-all-supports` selects each class's valid populated prefix, up to 16.
This is a variable-shot episode, not a claim that every class has 16 examples.
Support counts and identities are stored in every summary and cache key.
Fixed-shot selection remains available with `--shots N`.

## Accuracy status and review

The supplied JSON annotates support photographs, not the extracted rosbag query
frames. No independent query ground truth was supplied for this study. Numeric
accuracy, AP or recall is therefore **pending**, not zero and not estimated from
detection counts. More supports need not improve accuracy: the completed
480/640 all-support runs produce no cones above the fixed threshold, despite
the one-shot run producing cone hypotheses. Neither count establishes truth.
People can receive competing citizen/first-responder hypotheses. Role labels
and persistent UUIDs still require validation.

Visual inspection of the all-support 800 overlays confirms this limitation:
frame 56 has overlapping citizen/first-responder boxes on the same people.
Frame 1 localizes the foreground van, while the visible cone has no labelled
accepted detection. This inspection does not establish aggregate precision or
recall. Class-gated tracking can retain separate IDs for competing class
hypotheses, so creation-event uniqueness alone does not guarantee one physical
object per registry entry.

Four model views are available in the local `comparison.html` artifact. An
unlabelled review set of 14 distributed query images is under
`samples/pet_ablation_20261105_v2/review/`. Its empty COCO template is explicitly
marked unlabelled; the evaluator refuses to score it as negative ground truth.
Label all objects of the evaluation categories, including empty frames where
appropriate, keeping citizen and first responder distinct. Avoid copying model
predictions into ground truth without independent review.

Once completed COCO query annotations exist:

```bash
PYTHONPATH=ugv_models/pet_runtime:src/triffid_ugv_perception \
  /home/oracle/miniforge3/envs/fewshot-dev/bin/python scripts/evaluate_pet_ablation.py \
  samples/pet_ablation_20261105_v2/swin_t_800_all QUERY_GROUND_TRUTH.json
```

The evaluator matches original frame filenames (including Label Studio upload
prefixes), computes COCO box AP/AP50 and per-class precision/recall at IoU 0.5
and score 0.35, and writes `accuracy.json`. Precision/recall treats non-crowd
boxes; COCO AP uses the standard crowd semantics. Query annotations must be
complete for their stated evaluation categories. Categories without GT are
excluded from COCO AP by its standard convention.

All code tests currently pass (43), including new-image import, variable-shot
selection and accuracy evaluator tests for duplicate/wrong-class false
positives. Each run also checks source-frame equivalence, ontology IDs,
backbone call counts, tracker updates and UUID creation-event uniqueness.
Those export audits do not verify physical deduplication or classification.
