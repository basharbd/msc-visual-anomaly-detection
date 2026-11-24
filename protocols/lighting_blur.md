# Lighting and blur robustness

## Question and design

How does one-shot AnomalyDINO respond to uniform digital darkening and mild Gaussian blur of the test image? Use the official MVTec AD splits, all normal and defective test images, the frozen baseline model, and the [three fixed one-shot reference selections](adino_mvtec_1shot_448_references.json). Preserve reference augmentation, masks, scoring and nearest-neighbour settings. Compare each condition separately with its matched clean baseline.

| Condition | Test input |
| --- | --- |
| original | Same resized lossless study input path with no perturbation |
| brightness_085, brightness_070 | Multiply 8-bit RGB channel values by 0.85 or 0.70 and round to nearest, ties to even |
| blur_075, blur_150 | Pillow GaussianBlur with radius 0.75 or 1.50 pixels |

Apply one intervention at a time after the baseline bicubic resize to short edge 448, before tensor conversion, normalization and patch-grid cropping. The pilot stores temporary lossless RGB PNGs at the resized dimensions. The upstream resize then receives an image already at its target size. Original-resolution ground truth remains the spatial reference for evaluation. Digital darkening is a controlled sensitivity test. It does not model a complete change in camera exposure or illumination.

## Pilot and reporting

Use Bottle and the first reference. Require all 83 original-image scores and all 83 raw maps to match the archived baseline by filename, map shape and absolute tolerance 1e-6, with zero relative tolerance. Check input pairing and labels. Inspect transform strengths on normal training images before viewing perturbed test metrics. Fix strengths before comparative inference and record any protocol amendments. Proceed to perturbations only after identity verification. Reconstruct maps at original dimensions and use the baseline evaluation code/settings for subsequent metric comparisons.

For the full study, pair by category and reference. Report 100 times (perturbed metric minus original metric) in percentage points, first averaging references within category, then categories equally. Primary endpoints are image AUROC and pixel AUPRO (FPR limit 0.30). Also report image AP, image F1-max, pixel AUROC and pixel F1-max. F1-max is threshold-optimized per condition. Report missing runs and retain per-category results. The four perturbations yield 180 planned category-reference runs. Three reference choices share test images and do not constitute independent test datasets.

## DTU HPC identity pilot

From the repository root with the archived baseline raw maps and predictions available:

```bash
mkdir -p logs
bsub < jobs/lighting_blur_identity.lsf
```

The verification record is written to `results/lighting_blur/original/bottle_1shot_seed0/run_complete.json` only after successful comparisons. The job refuses to overwrite an existing output. Keep failed runs for diagnosis.

## Full paired study on DTU

From the repository root, run `bash jobs/submit_lighting_blur.sh` once. The inference array covers the four perturbations for each of the 45 category/reference pairs. Completed inference records are reused. The CPU array reconstructs maps in temporary WORK3 directories and evaluates against the original masks. LSF holds the summary until both arrays succeed. Inspect failed tasks in `logs/lighting_blur_*` before retrying. Partial inference directories are preserved for diagnosis.

The summary reads the matched clean metrics from `reports/baseline_reproduction/per_run.csv`. The output under `reports/lighting_blur/` contains the 180 matched runs, category-level differences averaged across the three references, and the equally weighted macro change in percentage points. The existing executed notebook presents the bottle pilot.

The [bottle seed-0 pilot table](../reports/lighting_blur/bottle_seed0_pilot.csv) transcribes the rounded results displayed by the executed HPC localization cells. The independently recomputed clean AUPRO was 0.964145085146, with zero observed difference from the archived baseline. Only the 1.50-pixel blur lowered both primary metrics in this pilot. The table covers one reference and one category.
