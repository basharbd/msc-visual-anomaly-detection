# Controlled lighting and blur — complete results

180 paired evaluations (15 categories, 3 fixed normal references, 4 perturbations), matched to the 45 clean runs. Category means are equally weighted. Brightness scales resized RGB intensities; blur uses Pillow Gaussian radius in resized-image pixels. Reference images remain clean. Full protocol: [lighting_blur.md](../../protocols/lighting_blur.md).

| Condition | Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel F1-max | Pixel AUPRO |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 96.52% | 98.14% | 95.96% | 96.26% | 57.94% | 91.72% |
| Brightness ×0.85 | 96.62% | 98.23% | 96.04% | 96.27% | 57.98% | 91.72% |
| Brightness ×0.70 | 96.46% | 98.12% | 95.88% | 96.24% | 57.88% | 91.60% |
| Blur 0.75 px | 95.86% | 97.73% | 95.47% | 96.08% | 57.29% | 91.24% |
| Blur 1.50 px | 93.44% | 96.17% | 93.68% | 95.47% | 55.79% | 90.19% |

## Data files

- [Per-run metrics](per_run.csv)
- [Per-category paired changes](per_category.csv)
- [Macro changes in percentage points](macro_change_pp.csv)
- [Selected paired score examples](selected_blur_cases.csv)
- [All 83 paired Bottle scores](bottle_paired_scores.csv), reference index 0, original versus strong blur ([source hashes](bottle_paired_scores_provenance.json))
- [Original bottle pilot](bottle_seed0_pilot.csv) (historical validation)

Selected cases are extreme examples, not a representative sample. Anomaly scores are not calibrated probabilities; no decision threshold is inferred from these particular examples. The separate [normal-only calibration study](../threshold_calibration/summary.json) reports fixed-threshold results. Synthetic darkening and blur are controlled stress tests, not evidence of deployment performance.
