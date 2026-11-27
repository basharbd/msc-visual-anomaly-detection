#!/bin/bash
set -euo pipefail
d="$HOME/msc-vad-recovery/logs/night_run_30402340"
module load python3/3.11.13
source "$HOME/.venvs/vad-py311/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
case "$1" in
eval) cd "$HOME/msc-vad-recovery"; python -u "$d/eval.py";;
ablation_summary) cd "$HOME/msc-vad-recovery"; python -u scripts/analyze_improvement.py summarize --output /work3/s183356/experiments/reference_augmentation_night_30402340 --dest "$d/reports/reference_augmentation";;
calibration_summary) cd "$HOME/msc-vad-calibration"; python -u -c 'from pathlib import Path; import sys; sys.path.insert(0,"scripts"); import threshold_calibration as c; c.OUT=Path("/work3/s183356/experiments/threshold_calibration_night_30402340"); c.ROOT=Path(sys.argv[1]); c.summarize()' "$d";;
*) exit 2;;
esac
