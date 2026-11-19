# msc-visual-anomaly-detection
MSc thesis at DTU investigating zero- and few-shot industrial visual anomaly detection using vision foundation models.

## Current baseline

I use frozen DINOv2 features to compare an inspection image with one defect-free reference. I evaluated the baseline in 45 runs across 15 MVTec AD categories, using three fixed references per category and one reference per run. The evaluation covers both image-level defect detection and pixel-level localization.

## Ongoing work

I am investigating how lighting changes and blur affect detection and localization.
