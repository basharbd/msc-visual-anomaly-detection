# Completed HPC night run — 26–27 November 2026

**Historical evidence, not a runnable launch kit.** Do not execute the archived scripts: some reference original job IDs and cancellation commands.

- 900 inference records and 900 evaluation records verified. The gate manifest records each inference SHA-256; evaluation_manifest.csv adds each evaluation SHA-256 and exact runner/evaluator hashes.
- 444 matched records reused from commit cd056cb32ca8d75a671592852ef2987cc84a3c39; 456 produced with recovery commit d66cb70c39100cdb7f921d7cc76aaeac711ca360. All 900 ablation inferences used A100-PCIE-40GB.
- The calibration runner was pinned at 5215309058593a05327eb0685db151656c0b555c. All 45 frozen thresholds verified and 225 comparisons summarized.
- GPU stage 30402340 finished 00:17 CET. Calibration summary 30402557 completed. CPU evaluation acceleration moved 12 pending batches into 30404717 without stopping running jobs. Final summary 30404718 succeeded at 01:24 CET. The 12 cancelled original pending tasks were superseded, not failed evaluations.
- The exact group-selection and reuse plan, original gate and worker commands, and acceleration decision are retained here. The archived results remain in their original WORK3 directories. No GPU inference was rerun for the GitHub organization. On 27 November, the five notebooks and publication-table generator were re-executed, and the two reorganized CPU summary commands rebuilt all 900 ablation and 225 calibration comparisons in an isolated validation checkout. Seven report CSVs matched the archived numerical results within 1e-12.

See [scientific audit](../../reports/final_scientific_analysis/audit.json), [reference-augmentation report](../../reports/reference_augmentation/), and [threshold report](../../reports/threshold_calibration/).
