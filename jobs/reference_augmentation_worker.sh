#!/usr/bin/env bash
set -e
cd "$HOME/msc-vad-development"
module load python3/3.11.13
source "$HOME/.venvs/vad-py311/bin/activate"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
output=/work3/s183356/experiments/reference_augmentation
case "$1" in
  smoke_gpu)
    cats=(bottle screw)
    python -u scripts/proposed_improvement.py --category "${cats[$((LSB_JOBINDEX-1))]}" --smoke --output "${output}_smoke"
    ;;
  smoke_cpu)
    python -u scripts/analyze_improvement.py evaluate --smoke --start "$LSB_JOBINDEX" --output "${output}_smoke"
    ;;
  smoke_gate)
    python -u scripts/analyze_improvement.py summarize --smoke --output "${output}_smoke" --dest reports/reference_augmentation_smoke
    python - <<'PY'
import json
from pathlib import Path
p=Path("/work3/s183356/experiments/reference_augmentation_smoke/rotation_only")
for c in ("original","brightness_085"):
    r=json.loads((p/c/"bottle_1shot_seed0/run_complete.json").read_text())
    parity=r["archive_parity"]
    assert parity is not None and parity["strict"] and max(parity["max_score_error"],parity["max_map_error"])<=1e-6
print("GPU parity, paired metrics and fixed bank-size gate PASSED")
PY
    ;;
  infer)
    cats=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
    python -u scripts/proposed_improvement.py --category "${cats[$((LSB_JOBINDEX-1))]}"
    ;;
  evaluate)
    python -u scripts/analyze_improvement.py evaluate --start "$(( (LSB_JOBINDEX-1)*10+1 ))" --count 10
    ;;
  summarize)
    python -u scripts/analyze_improvement.py summarize
    ;;
  *) exit 2;;
esac
