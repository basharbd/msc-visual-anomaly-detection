# HPC execution

The completed run used historical DTU LSF submissions and worktrees. The files in [historical](historical/) are archival copies from the earlier repository layout, **not launch instructions for this checkout**. They contain machine-specific paths and job IDs and may use renamed paths rather than the byte-identical original historical sources. The exact original baseline CPU-array source hash matches commit 68a9961f54d31500fce93b16fce737a898299ceb. To reproduce an original run exactly, check out its recorded commit into a separate directory and use that commit's original jobs and scripts layout. See [reproduction guide](../REPRODUCIBILITY.md).

For new runs, use the three public entry points from a correctly configured DTU compute allocation. Keep output paths separate from published artifacts. Do not execute historical cancellation/recovery scripts against current jobs.
