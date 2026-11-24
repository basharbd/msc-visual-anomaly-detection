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

| Directory | What it contains and why it is here |
| --- | --- |
| [notebooks](notebooks/) | Executed, illustrated analyses for the dataset, smoke test, clean baseline, and robustness pilot. Start here to read the work without a DTU account. Re-executing the dataset and robustness notebooks requires access to the underlying data or run outputs. |
| [configs](configs/) | The Python package versions recorded for the DTU experiment environment. This snapshot documents the environment used for the runs. |
| [protocols](protocols/) | The 45 fixed one-shot reference selections with file hashes, the 43 remaining clean runs after the two initial runs, and the lighting/blur study design. These files define what gets compared. |
| [scripts](scripts/) | Shared Python code for inference, evaluation, path resolution and plotting. The bottle-specific scripts also document the earlier checks on features, maps and predictions. |
| [jobs](jobs/) | DTU LSF submission files for dataset preparation, GPU and CPU checks, pilot runs and full experiments. The lighting/blur launcher submits the dependent inference, evaluation and summary jobs. |
| [results](results/) | The small, archived smoke-test inputs to the analysis notebook: per-image predictions and a run record. The full inference outputs and raw anomaly maps reside on DTU HPC. |
| [reports](reports/) | Analysis tables and figures. The clean baseline includes per-run, per-category and macro summaries with provenance. The lighting/blur directory contains the measured bottle pilot table. |
| [research_logs](research_logs/) | Dated records of the original dataset verification and smoke test. Their follow-up notes describe what was pending at the time. |
| [third_party](third_party/) | The official AnomalyDINO implementation, pinned as a Git submodule so each experiment can identify the upstream revision. |

On DTU, `data/`, `logs/`, and `outputs/` hold local inputs, scheduler logs and generated files. Temporary Git transfer bundles are also local. The notebook and report links above point to the artifacts published on GitHub.

Start with the [reference protocol](protocols/adino_mvtec_1shot_448_references.json), [baseline inference script](scripts/run_baseline.py) and [category evaluator](scripts/evaluate_baseline_category.py) to trace a run. The [GPU array](jobs/baseline_inference_array.lsf) and [CPU array](jobs/baseline_evaluation_array.lsf) use the same category and reference manifest. These jobs expect the DTU HPC dataset, checkpoints and environment paths defined in the scripts. Initialize the upstream submodule with `git submodule update --init --recursive` before inference.

Study names describe the research question. Category and seed identify individual runs within a study. Archived machine-generated records retain their original identifiers and hashes. The baseline and official Bottle pilot scripts recognize existing HPC output directories. Fresh Bottle smoke-test runs write under `outputs/` so the archived predictions remain available.
