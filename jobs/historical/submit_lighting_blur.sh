#!/usr/bin/env bash
# Submit from the thesis repository root on DTU HPC.
set -euo pipefail
mkdir -p logs
common=(-n 4 -R 'span[hosts=1]' -W 02:00)
run='module load python3/3.11.13 && source "$HOME/.venvs/vad-py311/bin/activate" && python -u scripts/lighting_blur_study.py'
submit() {
    local receipt
    receipt=$(bsub "$@")
    echo "$receipt" >&2
    [[ "$receipt" =~ Job[[:space:]]+\<([0-9]+)\> ]] || return 1
    echo "${BASH_REMATCH[1]}"
}
gpu=$(submit -J 'lighting_blur_gpu[1-180]%2' -q gpua100 "${common[@]}" \
    -gpu 'num=1:mode=exclusive_process' -R 'rusage[mem=4GB]' \
    -o 'logs/lighting_blur_gpu_%J_%I.log' "bash -lc '$run infer \"\$LSB_JOBINDEX\"'")
cpu=$(submit -J 'lighting_blur_cpu[1-180]%1' -q hpc "${common[@]}" \
    -w "done($gpu)" -R 'rusage[mem=16GB]' \
    -o 'logs/lighting_blur_cpu_%J_%I.log' "bash -lc '$run evaluate \"\$LSB_JOBINDEX\"'")
submit -J lighting_blur_summary -q hpc -W 00:15 -w "done($cpu)" \
    -o 'logs/lighting_blur_summary_%J.log' "bash -lc '$run summarize'"
