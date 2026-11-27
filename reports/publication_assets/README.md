# Final scientific figures and tables

Verified result package, 27 November 2026. Data source: HPC night run 30402340; final summary job 30404718. All 900 ablation evaluations and 225 threshold comparisons are included. Figures are vector PDF and preview PNG. Eight figures are generated directly by the analysis notebooks using scripts/visualize.py. Tables are generated as CSV and LaTeX (booktabs required). Regenerate with Python, pandas, NumPy and Matplotlib: python scripts/visualize.py --output outputs/publication_assets (from the repository root). Use an unused destination; the verified files here remain unchanged.

Ablation uses one true normal reference per run, 15 categories, 3 fixed reference choices, 4 banks, 5 image conditions. Each bank contains 8192 patch embeddings. A paired gain compares against rotation-only at the same condition; a robustness gap compares against the same variant on clean images. FPR and FNR use the baseline only and a frozen normal-only threshold. F1-max is threshold-optimized separately, not fixed-threshold performance.

The strongest blur augmentation result is +1.5401 percentage points in image AUROC and +0.1571 pp in pixel AUPRO under blur radius 1.50. Clean AUPRO decreases by 0.1039 pp. Category differences are heterogeneous; reference choices share test images, so do not treat them as independent replicates or confidence intervals. No claim of independent external validation, significance, or reduced fixed-threshold false alarms for the augmented method. The empirical scope is one-shot detection, as described in the thesis. Zero-shot methods are discussed as related work, not evaluated experimentally. Archived baseline audit records two localized Screw strong-blur map differences; image-level metrics identical and largest per-run AUPRO difference 0.001012 pp. See ../final_scientific_analysis/audit.json.

Figures 01–02: all four variants across five conditions; 03–04: category-level paired strong-blur effects; 05: frozen threshold; 06: descriptive reference-choice sensitivity; 07: clean baseline category comparison; 08: macro changes under controlled perturbations. Figure 01/02 use a restricted y-axis to resolve small differences; labels report absolute metrics. Table 01 includes all six metrics and 20 variant-condition cells, table 02 paired effects and clean-relative gaps, table 03 category effects, table 04 threshold results, table 05 reference sensitivity. Original source CSV files are unchanged.

The canonical input files are in the sibling report directories. No duplicate data copies are required.

## Notebook-generated figures and thesis mapping

Run the numbered notebooks in order. Each analysis notebook calls generate_publication_assets with its study name and saves a matching PDF and PNG in a fresh output directory. The checked-in files below are copied from those successful notebook executions, not independently redrawn in LaTeX. The PDF files included in the thesis have been verified against these exports using SHA-256; the PNG previews displayed in the notebooks match the committed images byte for byte. Figures 06 and the robustness heatmap are supplementary diagnostics; the heatmap is generated within Notebook 03.

| Figure stem | Source notebook | Canonical data | Thesis figure label |
|---|---|---|---|
| 01_ablation_image_auroc | 04 | reference_augmentation/macro.csv | fig:ablation_image |
| 02_ablation_pixel_aupro | 04 | reference_augmentation/macro.csv | fig:ablation_pixel |
| 03_strong_blur_category_image | 04 | reference_augmentation/per_category.csv | fig:ablation_category_image |
| 04_strong_blur_category_pixel | 04 | reference_augmentation/per_category.csv | fig:ablation_category_pixel |
| 05_threshold_calibration | 05 | threshold_calibration/macro.csv | fig:threshold_calibration |
| 06_reference_sensitivity | 04 (reused in 05) | reference_augmentation/per_run.csv | Supplementary |
| 07_baseline_category_comparison | 02 | baseline_reproduction/per_category.csv | fig:baseline_category_comparison |
| 08_robustness_macro_change | 03 | lighting_blur/macro_change_pp.csv | fig:robust_subset_change |

The thesis uses the seven corresponding PDF exports in its Backmatter folder with a unified_ filename prefix. The numerical source reports and original table assets remain unchanged. The Bottle paired-score plot and selected anomaly-map panel are not yet unified: their raw per-image predictions and maps are held on DTU HPC, not in this checkout. Do not substitute rounded plot coordinates or reconstructed images for the original records. No new GPU experiment is needed to retrieve those archived outputs.
