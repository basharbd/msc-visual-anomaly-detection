# Visual anomaly detection for industrial inspection

DTU MSc thesis in Human-Centered Artificial Intelligence. This project studies how vision foundation models detect defective products and localize defects when few normal reference images are available.

## Explore the notebooks

| Notebook | What to look for |
| --- | --- |
| [Dataset Exploration](notebooks/dataset_exploration.ipynb) | Verified splits, mask pairing, and a normal reference alongside a defective test image and its mask. |
| [Baseline Setup and Smoke Test](notebooks/baseline_setup_and_smoke_test.ipynb) | The first bottle run, recomputed image AUROC, and the normal/defective pairs that overlap. |
| [Baseline Reproduction](notebooks/baseline_reproduction.ipynb) | All 45 clean runs, category variation, and the six macro means compared with the matched published setting. |
| [Lighting and Blur Robustness](notebooks/lighting_and_blur_robustness.ipynb) | Executed image-level bottle pilot. See the [pilot localization results](reports/lighting_blur/bottle_seed0_pilot.csv) and [full-study protocol](protocols/lighting_blur.md). |

## Baseline results

The baseline uses frozen DINOv2 ViT-S/14 features and one normal reference per run. Each of the 15 categories was evaluated with three fixed reference choices, giving 45 runs. The table shows equal-weight category averages across those choices.

| Image AUROC | Image AP | Image F1-max | Pixel AUROC | Pixel F1-max | Pixel AUPRO |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 96.52% | 98.14% | 95.96% | 96.26% | 57.94% | 91.72% |

[Image AUROC by category](reports/baseline_reproduction/figures/image_auroc.svg) · [Pixel F1-max by category](reports/baseline_reproduction/figures/pixel_f1max.svg) · [Evaluation provenance](reports/baseline_reproduction/provenance.json)

To regenerate both figures from the archived CSV, run the following command from a checkout with Python and Matplotlib installed.

```bash
python scripts/plot_baseline.py
```

## Repository guide

| Directory | Contents |
| --- | --- |
| [protocols](protocols/) | Fixed reference filenames and hashes, plus the 43-run manifest used after the two baseline pilots |
| [scripts](scripts/) | Inference, evaluation, verification and plotting entry points |
| [jobs](jobs/) | DTU HPC batch jobs for data preparation, inference and evaluation |
| [configs](configs/) | Recorded Python environment for DTU HPC |
| [reports](reports/) | Baseline metrics, provenance and figures |
| [results](results/) | Archived Bottle smoke-test predictions and metrics |
| [research_logs](research_logs/) | Historical run and dataset records |
| [third_party](third_party/) | Pinned upstream AnomalyDINO implementation |

Start with the [reference protocol](protocols/adino_mvtec_1shot_448_references.json), [baseline inference script](scripts/run_baseline.py) and [category evaluator](scripts/evaluate_baseline_category.py) to trace a run. The [GPU array](jobs/baseline_inference_array.lsf) and [CPU array](jobs/baseline_evaluation_array.lsf) use the same category and reference manifest. These jobs expect the DTU HPC dataset, checkpoints and environment paths defined in the scripts. Initialize the upstream submodule with `git submodule update --init --recursive` before inference.

Study names describe the research question. Category and seed identify individual runs within a study. Archived machine-generated records retain their original identifiers and hashes. The baseline and official Bottle pilot scripts recognize existing HPC output directories. Fresh Bottle smoke-test runs write under `outputs/` so the archived predictions remain available.
