#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/msc-vad-recovery"
module load python3/3.11.13
source "$HOME/.venvs/vad-py311/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
test "${LSB_JOBID:-}" = 30402340
out=/work3/s183356/experiments/reference_augmentation_night_30402340
cal=/work3/s183356/experiments/threshold_calibration_night_30402340
d="$HOME/msc-vad-recovery/logs/night_run_30402340"
# Run the pinned recovery method, reusing only full matched groups.
for c in bottle carpet grid tile toothbrush transistor wood zipper; do
 echo "START_ABLATION $c $(date)"
 python -u scripts/proposed_improvement.py --category "$c" --output "$out"
 echo "DONE_ABLATION $c $(date)"
done
cd "$HOME/msc-vad-calibration"
for c in cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
 echo "START_CALIBRATION $c $(date)"
 python -u -c 'from pathlib import Path; import sys; sys.path.insert(0,"scripts"); import threshold_calibration as c; c.OUT=Path(sys.argv[1]); c.infer(sys.argv[2])' "$cal" "$c"
 echo "DONE_CALIBRATION $c $(date)"
done
printf 'GPU_STAGE_COMPLETE %s\n' "$(date)" | tee "$d/gpu_complete.txt"
