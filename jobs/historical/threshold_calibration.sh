#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/msc-vad-calibration"
module load python3/3.11.13
source "$HOME/.venvs/vad-py311/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
case "$1" in
infer)
 c=$(printf '%s\n' bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper | sed -n "$LSB_JOBINDEX p")
 test -n "$c"
 python -u scripts/threshold_calibration.py infer --category "$c";;
summarize) python -u scripts/threshold_calibration.py summarize;;
*) exit 2;;
esac
