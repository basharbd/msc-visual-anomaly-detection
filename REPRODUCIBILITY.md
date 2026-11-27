# Reproducing the research

This guide distinguishes **reproducing the analysis from recorded results**, **reconstructing selected maps from archived predictions**, and **re-running model inference**. The first two were checked independently in a fresh GitHub checkout on 27 November 2026. A new full GPU replay of the reorganized workflow was not performed. The original experimental commits, manifests and archived output records remain authoritative.

## 1. Reproduce the reported analysis (CPU only)

Use Python 3.11 in a separate environment. The following commands do not run inference or modify the published reports:

```bash
git clone https://github.com/basharbd/msc-visual-anomaly-detection.git
cd msc-visual-anomaly-detection
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/visualize.py --output outputs/publication_assets
jupyter notebook
```

Open the numbered notebooks in the repository root or notebooks directory and use a kernel from the same Python environment. Choose a new output directory for each CLI regeneration; the generator refuses existing destinations and paths overlapping protected research records. The standard CLI generates figures 01–08 (eight PDF and PNG pairs) and five tables in CSV and LaTeX. Notebook 05 additionally generates figures 09 and 10; it is not necessary to run the GPU pipeline to generate these from recorded inputs.

| Notebook | Inputs and outputs |
|---|---|
| [01 Dataset exploration](notebooks/01_dataset_exploration.ipynb) | Official extracted MVTec AD data; split counts, masks and reference checks. Set MVTEC_ROOT. |
| [02 Baseline reproduction](notebooks/02_baseline_reproduction.ipynb) | Historical smoke-test evidence and 45 archived clean-run metric rows; Figure 6.1. |
| [03 Robustness analysis](notebooks/03_robustness_analysis.ipynb) | 180 matched perturbed-run metric rows; Figure 7.1 and supplementary heatmap. |
| [04 Proposed improvement](notebooks/04_proposed_improvement.ipynb) | 900 matched fixed-budget evaluations; Figures 8.1–8.4 and supplementary reference sensitivity. |
| [05 Final results](notebooks/05_final_results.ipynb) | 225 threshold comparisons; Figure 7.2, Figure 9.2 from the full-precision Bottle score export; optionally Figure 9.1 from original images and archived maps. |

Without the official dataset and archive paths, Notebook 05 explicitly skips only the selected-case reconstruction cell. It does not generate replacement images or maps. Notebook 01 requires the full dataset. Notebooks 02–05 otherwise work from checked-in result files without DTU access. See the [complete figure mapping](reports/publication_assets/README.md).

### Integrity and numerical checks

On macOS:

```bash
(cd reports && shasum -a 256 -c SHA256SUMS)
```

On Linux:

```bash
(cd reports && sha256sum -c SHA256SUMS)
```

The [recorded Overleaf source-file hashes](reports/publication_assets/overleaf_figure_hashes.json) match the nine PDF exports actually included in the thesis. Figure 06 (reference-choice sensitivity) is supplementary. The standard CLI and notebooks share plotting functions, but rendering can differ across library/backend versions; the checked-in notebook-export PDFs are the canonical thesis assets. Compare numerical values before drawing conclusions about visual encoding differences.

## 2. Dataset, configuration and source identity

Use the [official MVTec AD archive](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) under CC BY-NC-SA 4.0. The official copy verified on 28 October 2026 has 15 categories, 3,629 normal training images, 1,725 test images (467 normal, 1,258 anomalous) and 1,258 ground-truth masks. The earlier Bottle smoke test used a separate dataset copy and is not the final official baseline.

| Setting | Recorded value |
|---|---|
| Backbone | Frozen DINOv2 ViT-S/14, 384-dimensional patch embeddings |
| Input | Bicubic resize to short edge 448, crop bottom/right to multiples of 14, ImageNet normalization |
| Reference | One normal image per run; three fixed choices per category, indices 0–2 |
| Baseline bank | Upstream agnostic preprocessing, rotation policy, category-dependent foreground masks, no reference masking |
| Retrieval | Exact 1-NN normalized L2 with CPU FAISS; distance divided by two (cosine distance) |
| Image score | Mean of largest floor(1%) patch scores, minimum one |
| Map reconstruction | Original-resolution OpenCV linear interpolation, Gaussian smoothing sigma 4 |
| Test perturbations | Brightness 0.85/0.70 and Pillow Gaussian blur radius 0.75/1.50 after resize |
| Fixed-budget ablation | 8,192 stored patches for every bank; reference-side brightness 0.85 and blur 0.75 |
| Calibration | 40 additional normal train images per category, excluding the three references; 39th smallest score; freeze thresholds before test scoring |
| Primary metrics | Image AUROC and pixel AUPRO (pixel FPR integration limit 0.30) |

Authoritative specifications: [reference manifest](protocols/adino_mvtec_1shot_448_references.json), [lighting/blur](protocols/lighting_blur.md), [reference augmentation](protocols/reference_augmentation.md), [calibration protocol](protocols/threshold_calibration.md), [calibration manifest](protocols/threshold_calibration_manifest.json), [official dataset verification](research_logs/MVTec_AD_dataset.md).

- Pinned AnomalyDINO: `b9d1c2648e3a5247437d4d953d907a8f3d994457`.
- DINOv2 checkpoint SHA-256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.
- DINOv2 Python source fingerprint: `fabaa1cb43240673152198284e8cf911268e7d3a0b2214ede94527774e15f5dc`.

The baseline runner checks the checkpoint at `torch.hub.get_dir()/checkpoints/dinov2_vits14_pretrain.pth` and the Python files in `torch.hub.get_dir()/facebookresearch_dinov2_main` before inference. The source fingerprint is the SHA-256 over sorted relative Python file paths and file contents, each separated by NUL, excluding __pycache__. A fresh download of an unpinned main branch is not guaranteed to match. The [historical environment freeze](configs/hpc-py311-freeze.txt) records Python package versions; the original GPU environment was Python 3.11.13, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, FAISS CPU 1.11.0 and A100-PCIE-40GB hardware for the ablation.

## 3. Reconstruct the selected error-analysis figures (CPU, archived inputs)

Figure 9.2 uses [all 83 paired Bottle scores](reports/lighting_blur/bottle_paired_scores.csv) at reference index 0, with [archive source hashes](reports/lighting_blur/bottle_paired_scores_provenance.json). It is reproducible from GitHub without the raw archive.

Figure 9.1 needs the four original test images, three defect masks, four reference images and eight raw patch maps, plus the paired inference/evaluation records and prediction CSVs. The [source-file manifest](reports/final_scientific_analysis/selected_error_case_provenance.json) identifies the exact relative paths and SHA-256 hashes. The generator verifies the reference identities, archived scores, run metadata and recomputed top-1% scores, then reconstructs maps using the pinned upstream code. The resulting [mask-region measurements](reports/final_scientific_analysis/selected_error_case_measurements.csv) are saved without rounding.

In the same CPU environment:

```bash
git submodule update --init third_party/AnomalyDINO
export MVTEC_ROOT=/path/to/official/extracted/mvtec_ad
export VAD_RESULTS_ROOT=/path/to/archived/results
jupyter notebook notebooks/05_final_results.ipynb
```

Use the official dataset folder containing the 15 categories. The results root contains the original clean archive (EXP-003, or baseline_reproduction) and lighting_blur/blur_150. The generator accepts a subset containing the exact required source files; Notebook 01 still needs the complete dataset. Do not copy raw images or raw maps into the public repository. Figure 09 and its notebook image output are attributed MVTec AD derivatives shared under CC BY-NC-SA 4.0 for non-commercial research. All analysis outputs go into new directories.

On DTU, the recorded paths are:

| Archive | Recorded location |
|---|---|
| Official dataset | /work3/s183356/datasets/mvtec_official |
| Clean and perturbed predictions | $HOME/msc-visual-anomaly-detection/results/EXP-003 and results/lighting_blur (relative to that checkout) |
| Matched ablation | /work3/s183356/experiments/reference_augmentation_night_30402340 |
| Frozen calibration | /work3/s183356/experiments/threshold_calibration_night_30402340 |

The WORK3 archive paths are historical storage locations, not persistent public downloads. The dataset verification record notes that scratch storage has no backup. Keep independent authorized backups of raw predictions, maps, run records, calibration thresholds, checkpoint and DINOv2 source before scratch cleanup. This guide does not claim that such a backup has already been made.

## 4. Historical experiment provenance (GPU/HPC)

Use the original Git commits and original experiment outputs for exact provenance. Create a separate checkout/worktree; do not run archived job/cancellation scripts against current jobs or modify the original experiment directory. Adapting machine-specific paths changes source hashes. A full bitwise replay across different hardware, drivers and package versions is not guaranteed.

| Study | Authoritative source |
|---|---|
| Clean baseline | [Run-level provenance](reports/baseline_reproduction/provenance.json) and the [historical source-hash mapping](research_logs/reproducibility_audit_20261127.json) |
| Lighting/blur | [05d2c08](https://github.com/basharbd/msc-visual-anomaly-detection/tree/05d2c08243bf059698238e6df12a1cb80a918802) |
| Initial ablation (444 reused evaluations) | [cd056cb](https://github.com/basharbd/msc-visual-anomaly-detection/tree/cd056cb32ca8d75a671592852ef2987cc84a3c39) |
| Recovery ablation (456 evaluations) | [d66cb70](https://github.com/basharbd/msc-visual-anomaly-detection/tree/d66cb70c39100cdb7f921d7cc76aaeac711ca360) |
| Calibration | [5215309](https://github.com/basharbd/msc-visual-anomaly-detection/tree/5215309058593a05327eb0685db151656c0b555c) |

The baseline provenance records six source-file hashes. All six were matched to the historical Git blobs, including the final original CPU-array definition at [68a9961](https://github.com/basharbd/msc-visual-anomaly-detection/blob/68a9961f54d31500fce93b16fce737a898299ceb/jobs/exp003_cpu_array.lsf). The three baseline inference-record commits and five additional historical commits are accessible as ancestors. Historical scheduler copies in the present layout are not a runnable launch kit or replacements for these exact Git blobs.

The [night-run archive](research_logs/night_run_30402340/) retains the 900-row verified inference/evaluation manifest, job IDs, execution plan, original worker and scheduler commands, and frozen threshold verification. The [recovery notes](research_logs/reference_augmentation_recovery.md) and [parity audit](reports/final_scientific_analysis/audit.json) record the bounded recovery and two localized Screw strong-blur map differences. The original clean runs, 180 perturbed evaluations, 900 matched ablations and 225 threshold comparisons were completed; the reorganized inference entry point was not independently GPU-replayed.

## 5. Public workflow for new runs

The current repository exposes three public scripts. The first two dispatch to retained experimental implementation modules; visualize.py regenerates analysis assets. For inference/evaluation, configure a separate compatible GPU/CPU environment with the historical dependency versions and the pinned model cache. The CPU analysis requirements alone do not install PyTorch/CUDA or FAISS. At DTU the original jobs loaded python3/3.11.13 and activated $HOME/.venvs/vad-py311. Use an appropriately allocated LSF GPU node for inference, not the login node.

Verify the commands before launching anything:

```bash
python scripts/run_experiment.py --dry-run baseline --category bottle --seed 0
python scripts/run_experiment.py --dry-run robustness 1
python scripts/run_experiment.py --dry-run ablation --category bottle --output /path/to/NEW_ABLATION_OUTPUT
python scripts/evaluate.py --dry-run baseline --category bottle --seed 0
python scripts/evaluate.py --dry-run robustness 1
python scripts/evaluate.py --dry-run robustness-summary --dest /path/to/NEW_ROBUSTNESS_REPORT
python scripts/evaluate.py --dry-run ablation --start 1 --count 10 --output /path/to/NEW_ABLATION_OUTPUT
python scripts/evaluate.py --dry-run ablation-summary --output /path/to/NEW_ABLATION_OUTPUT --dest /path/to/NEW_REPORT
python scripts/run_experiment.py --dry-run calibration --category bottle --output /path/to/NEW_CALIBRATION
python scripts/evaluate.py --dry-run calibration-summary --output /path/to/FROZEN_CALIBRATION --archive /path/to/ARCHIVED_RESULTS --dest-root /path/to/NEW_REPORT_ROOT
```

Use --dataset-root before the study name, or set MVTEC_ROOT, to choose the extracted dataset. The baseline uses checkout-relative results/baseline_reproduction (or an existing legacy EXP-003 directory); the perturbation study uses results/lighting_blur. Robustness requires the completed clean baseline and a successful 83-image Bottle identity check; it indexes 180 tasks by sorted category, reference index and perturbation condition. It is not a standalone fresh-run command without these prerequisites. VAD_EVAL_WORK_ROOT may select an isolated evaluation workspace; its parent must exist. Ablation and calibration use explicit output roots. Summary commands require new destinations. Perform new experiments only in an isolated reproduction checkout and never overwrite published reports. Keep failed partial runs for diagnosis. Inspect the protocol and original recorded jobs before executing a full study.

Dry runs confirm command construction only, not GPU execution. The original job/environment paths are machine-specific; no claim of fully portable GPU execution is made.

## 6. Verification record (27 November 2026)

The [machine-readable audit](research_logs/reproducibility_audit_20261127.json) records the exact checked revision, CPU package versions, verified source hashes and test outcomes. The fresh clone used the already-installed Python 3.11.14 CPU analysis environment; this was not a fresh dependency installation.

| Check | Result |
|---|---|
| Published checksum entries | 58/58 |
| Standard figure pairs (CLI) | 8/8 PDF and 8/8 PNG byte-identical in this environment |
| Regenerated numerical / LaTeX tables | 5/5 CSV and 5/5 TeX byte-identical |
| Notebooks 02–05 without archives | 20 code cells completed; selected-map cell explicitly skipped |
| Notebook 05 with actual archived inputs | 7 code cells completed; both final error-analysis figures generated |
| Notebook figure previews | All 10 canonical PNGs matched across the tested notebook executions |
| Reconstructed map statistics and input provenance | Both files byte-identical to the verified committed exports |
| Previously verified Overleaf PDF hashes | 9/9 matched against the fresh clone |
| Public CLI dry runs | 10/10 |
| Historical baseline source hashes | 6/6 |
| Historical commits | 8/8 accessible as ancestors |
| Notebook 01 | Schema verified locally; previously executed on HPC with the full official dataset |
| New GPU inference | Not performed |

Earlier verification: all five original notebooks (21 code cells at that revision) executed on HPC on 27 November; the reorganized CPU summary commands independently rebuilt all 900 ablation and 225 calibration comparisons from archived outputs, with seven aggregate/per-run CSVs agreeing within 1e-12. The earlier CLI figure check exhibited minor encoding/layout differences for three category plots; the fresh-clone check above matched all eight standard figure pairs. These environment-specific results do not guarantee identical rendering everywhere.

## 7. Interpretation and limits

Equal-weight category macro averaging uses three fixed reference choices per category; the choices share test images and are not independent statistical replicates. The fixed-size augmentation study is exploratory and controls cardinality rather than composition. F1-max is threshold-optimized separately. Normal-only calibration targets a nominal 5% tail, not guaranteed test FPR; fixed-threshold results concern the baseline only. Digital darkening and Gaussian blur are controlled stress tests, not evidence of deployment performance. Zero-shot methods are discussed as related work rather than experimentally evaluated.
