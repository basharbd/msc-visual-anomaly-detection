# Frozen normal-only threshold calibration — completed

45 frozen thresholds based on 40 additional clean training images per category; 225 matched test comparisons. See [protocol](../../protocols/threshold_calibration.md), [summary](summary.json), [macro](macro.csv), [category results](per_category.csv), and [per-run counts](per_run.csv).

Strong blur increases the baseline macro false-positive rate from 8.32% to 20.34%. This is additional normal calibration data, not strictly one-shot total data use; no calibrated augmentation result is claimed.
