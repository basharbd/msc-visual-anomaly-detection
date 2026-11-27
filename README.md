![Visual anomaly detection for industrial inspection](assets/project_banner.svg)

# Vision Foundation Models for Industrial Anomaly Detection

*Robustness and Reference Augmentation in One-Shot Learning*

MSc thesis in Human-Centered Artificial Intelligence · Technical University of Denmark (DTU)

[**Defense presentation (PDF)**](presentation/Defense_Presentation.pdf) · [Notebooks](#explore-the-project) · [Results](#key-results) · [Reproducibility](REPRODUCIBILITY.md)

## Overview

Can an inspection system detect and locate defects using just one normal reference image, even when lighting or sharpness changes?

This research reproduces **AnomalyDINO** using frozen **DINOv2** features on all 15 categories of **MVTec AD**, studies sensitivity to brightness and Gaussian blur, and evaluates reference augmentation at a fixed memory-bank size.

**Python · PyTorch · DINOv2 · FAISS · OpenCV · DTU HPC**

## Key results

| Research question | Main finding |
| --- | --- |
| **RQ1 — Baseline** | 96.52% image AUROC and 91.72% pixel AUPRO on original images. |
| **RQ2 — Robustness** | Strong blur reduces image AUROC by 3.08 percentage points; baseline false-positive rate rises from 8.32% to 20.34% at frozen thresholds. |
| **RQ3 — Improvement** | Blur-aware references raise strong-blur image AUROC from 93.44% to **94.98%** at the same 8,192-patch budget. |

<p align="center">
  <img src="reports/publication_assets/figures/01_ablation_image_auroc.png" width="720" alt="Measured image AUROC across the four reference banks and five imaging conditions">
</p>

Blur augmentation improves image ranking in 13 of 15 categories. Localization gains are smaller (+0.16 pp under strong blur), and clean pixel AUPRO decreases by 0.10 pp.

**Scope:** 45 clean runs, 180 perturbed evaluations, 900 fixed-budget ablations and 225 frozen-threshold comparisons. Calibration uses additional normal training images. The experiments evaluate one-shot detection, not zero-shot methods or real production deployment; augmented-bank false-alarm reduction remains unverified.

[Full results](reports/) · [Anomaly maps](reports/publication_assets/figures/09_selected_blur_cases.png) · [Robustness plot](reports/publication_assets/figures/08_robustness_macro_change.pdf) · [All figures](reports/publication_assets/README.md)

## Explore the project

| Notebook | Purpose |
| --- | --- |
| [01 · Dataset exploration](notebooks/01_dataset_exploration.ipynb) | Data, masks and references |
| [02 · Baseline reproduction](notebooks/02_baseline_reproduction.ipynb) | Reproduce AnomalyDINO |
| [03 · Robustness analysis](notebooks/03_robustness_analysis.ipynb) | Brightness and blur |
| [04 · Proposed improvement](notebooks/04_proposed_improvement.ipynb) | Fixed-budget augmentation |
| [05 · Final results](notebooks/05_final_results.ipynb) | Calibration and error analysis |

**Code:** [Run experiments](scripts/run_experiment.py) · [Evaluate](scripts/evaluate.py) · [Visualize](scripts/visualize.py)

**Documentation:** [Protocols](protocols/) · [Research logs](research_logs/) · [Reproduction guide](REPRODUCIBILITY.md)

## Reproduce the analysis

From a fresh clone, using Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/visualize.py --output outputs/publication_assets
jupyter notebook
```

Notebooks 02–05 analyze recorded results on CPU. Dataset exploration and selected anomaly-map reconstruction require external data and archived inputs. Use a fresh output directory for each run. Analysis and archive-backed reconstruction have been verified; the reorganized inference workflow has not been independently replayed on GPU.

[Full instructions and verification record](REPRODUCIBILITY.md)

## Defense and references

[**Presentation PDF**](presentation/Defense_Presentation.pdf) · [Editable LaTeX source](presentation/Defense_Presentation.tex)

18 main slides and 5 backup slides, using the same nine experimental figure PDFs as the thesis.

Core references: [AnomalyDINO (WACV 2025)](https://openaccess.thecvf.com/content/WACV2025/html/Damm_AnomalyDINO_Boosting_Patch-Based_Few-Shot_Anomaly_Detection_with_DINOv2_WACV_2025_paper.html) · [DINOv2](https://arxiv.org/abs/2304.07193) · [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).

MVTec AD images and mask derivatives are used for non-commercial research under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Raw dataset images and prediction maps remain external; upstream code and weights retain their own licenses.

---

**Bashar Khaled Bdewi** · DTU Compute · Supervisor: Jeppe Revall Frisvad

[GitHub profile](https://github.com/basharbd)
