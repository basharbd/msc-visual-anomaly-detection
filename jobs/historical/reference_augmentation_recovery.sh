#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/msc-vad-recovery"
module load python3/3.11.13
source "$HOME/.venvs/vad-py311/bin/activate"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
case "$1" in
infer)
  catname=$(printf '%s\n' bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper | sed -n "$LSB_JOBINDEX p")
  missing=$(python - "$catname" <<'PY'
from pathlib import Path
import sys
cat=sys.argv[1]
root=Path('/work3/s183356/experiments/reference_augmentation')
archive=root.with_name(root.name+'_recovery_preserved_20261126')
variants=['rotation_only','brightness','blur','combined']
conditions=['original','brightness_085','brightness_070','blur_075','blur_150']
missing=0
for cond in conditions:
    paths=[root/v/cond/f'{cat}_1shot_seed{s}' for s in range(3) for v in variants]
    if all((p/'run_complete.json').is_file() for p in paths):
        continue
    missing+=1
    for p in paths:
        for f in (p,p.with_name(p.name+'.partial')):
            if f.exists():
                dest=archive/f.relative_to(root)
                assert not dest.exists(),dest
                dest.parent.mkdir(parents=True,exist_ok=True)
                f.rename(dest)
                print('PRESERVED',f,'->',dest,file=sys.stderr,flush=True)
print(missing)
PY
)
  if [ "$missing" -gt 0 ]; then
    python -u scripts/proposed_improvement.py --category "$catname"
  else
    echo "$catname already complete; no recomputation"
  fi
  ;;
evaluate)
  python -u scripts/analyze_improvement.py evaluate --start "$(( (LSB_JOBINDEX-1)*10+1 ))" --count 10
  ;;
summarize)
  python -u scripts/analyze_improvement.py summarize
  ;;
*) exit 2;;
esac
