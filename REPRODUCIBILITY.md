# Reproducing the research

## 1. Reproduce the published analysis (CPU only)

Install the analysis dependencies from requirements.txt. Run the five notebooks in the repository root or notebooks directory. The dataset notebook requires the separately downloaded MVTec AD data; set MVTEC_ROOT. The other four notebooks use the checked-in result files. Run python scripts/visualize.py --output outputs/publication_assets to regenerate all six publication figures and five tables in a new directory. The importable generate_publication_assets(output_dir) function uses the same implementation. The destination must not exist and must be separate from protected research records. The publication tables are generated from the unrounded numerical reports; F1-max is threshold-optimized separately, not fixed-threshold F1.

All five notebooks were re-executed on DTU HPC on 27 November 2026, 21 code cells total, with no errors. The regenerated tables were byte-identical to the verified publication assets. The reorganized CLI independently rebuilt all 900 ablation and 225 calibration comparisons in a separate HPC validation directory; all seven aggregate/per-run CSV tables matched numerically within 1e-12. No new GPU experiment was launched in the reorganization.

## 2. Dataset, configuration and source identity

Use the official MVTec AD archive according to the dataset terms. The verified official copy has 3,629 normal training images and 1,725 test images (467 normal and 1,258 anomalous), all 1,258 anomaly masks checked. Every baseline/augmentation run uses one normal reference, one of three deterministic choices, frozen DINOv2 ViT-S/14, 448-pixel short edge, upstream agnostic preprocessing, rotation, category-dependent masks, and 1-NN normalized L2 with CPU FAISS. All ablation banks contain 8,192 patches. The 40 extra normal images per category are used only for the separate threshold calibration.

- [Reference identities and hashes](protocols/adino_mvtec_1shot_448_references.json)
- [Lighting/blur protocol](protocols/lighting_blur.md)
- [Reference augmentation protocol](protocols/reference_augmentation.md)
- [Calibration protocol and frozen manifest](protocols/threshold_calibration.md)
- [Exact GPU environment](configs/hpc-py311-freeze.txt)

Upstream AnomalyDINO submodule: b9d1c2648e3a5247437d4d953d907a8f3d994457. Frozen DINOv2 checkpoint SHA-256: b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9. DINOv2 Python source fingerprint: fabaa1cb43240673152198284e8cf911268e7d3a0b2214ede94527774e15f5dc.

## 3. Exact historical replay

The original experiment runners are preserved as ancestors of the final repository. Do not edit the recorded commits or run old scheduler/cancellation scripts against current jobs. Use a separate worktree, initialize the pinned submodule, configure the recorded dataset/cache/environment, and allocate appropriate compute resources through DTU LSF. Paths in the original code are DTU-specific; adapting them changes source hashes. Raw result archives and dataset files are not included in this repository.

| Study | Authoritative provenance |
|---|---|
| Clean baseline | [Recorded commits, runner hashes and environment](reports/baseline_reproduction/provenance.json); original EXP-003 runners in those commits |
| Lighting/blur | [Original full notebook and protocol at 05d2c08](https://github.com/basharbd/msc-visual-anomaly-detection/tree/05d2c08243bf059698238e6df12a1cb80a918802) |
| Initial ablation | [cd056cb](https://github.com/basharbd/msc-visual-anomaly-detection/tree/cd056cb32ca8d75a671592852ef2987cc84a3c39) — 444 reused evaluations |
| Recovery ablation | [d66cb70](https://github.com/basharbd/msc-visual-anomaly-detection/tree/d66cb70c39100cdb7f921d7cc76aaeac711ca360) — 456 evaluations |
| Calibration | [5215309](https://github.com/basharbd/msc-visual-anomaly-detection/tree/5215309058593a05327eb0685db151656c0b555c) |

The [night-run archive](research_logs/night_run_30402340/) preserves the execution plan, job IDs, exact inference/evaluation hashes and actual orchestration scripts. Historical jobs are retained under jobs/historical; use the original worktrees and recorded job definitions for exact replay. The recovery amendment and original failures are recorded in [recovery notes](research_logs/reference_augmentation_recovery.md). The independently verified [parity audit](reports/final_scientific_analysis/audit.json) reports the localized raw-map differences and metric-level agreement.

## 4. Simplified public workflow

The three public entry points dispatch to the retained implementation modules. Use dry-run before launching any job. Example commands (the dry-run flag prints the command without allocating resources):

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

The public inference and evaluation scripts accept --dataset-root before the study name, or the MVTEC_ROOT environment variable. The original DTU dataset location remains the default; VAD_EVAL_WORK_ROOT optionally selects the evaluation workspace base (its parent must exist). Baseline and robustness still expect the recorded checkout-relative archive layout, including the clean baseline and identity check. The robustness and ablation summary commands require new destinations and refuse existing directories. The calibration entry points require explicit calibration and output locations; the summary also requires the archived baseline and robustness predictions. They preserve the original frozen manifests and threshold algorithm. Perform all new inference/evaluation in an isolated reproduction checkout with dedicated output directories. Do not replace the published verified report CSVs. The earlier reorganization validated all dispatch routes, five inference/evaluation help commands, and both ablation and calibration CPU summary routes against archived reports. After the path and overwrite-safety changes, all ten dry-run routes and seven rejection tests passed locally on 27 November 2026. All five regenerated CSV tables matched the verified tables exactly, and all five LaTeX tables were byte-identical. Six figures were generated in PDF and PNG format. These tests do not establish GPU replay of the revised entry points. The historical commits remain authoritative for exact source hashes and experiment provenance.

## 5. Interpretation

Equal-weight category macro averaging uses three fixed reference choices per category. The choices share test images, so they are not independent statistical replicates. The fixed-size augmentation study is exploratory and controls cardinality rather than exact composition. Calibration targets a nominal 5% normal tail, not guaranteed test FPR. Fixed-threshold results apply only to the baseline. No zero-shot results or external deployment claims are made.
