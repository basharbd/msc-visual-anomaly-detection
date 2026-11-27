#!/bin/bash
set -euo pipefail
d="$HOME/msc-vad-recovery/logs/night_run_30402340"
python3 "$d/verify.py" "$1"
# Only cancel superseded jobs that are still entirely pending, after full verification.
if [ "$1" = ablation ]; then ids="30402005 30402003 30377351 30391837 30391839 30391840 30391841 30391842 30391843"; else ids="30402084 30391846"; fi
for id in $ids; do
 states=$(bjobs -noheader -o stat "$id" 2>/dev/null || true)
 if [ -n "$states" ] && ! printf '%s\n' "$states" | grep -qvE '^[[:space:]]*PEND[[:space:]]*$'; then
   bkill "$id" || true
   echo "Superseded pending job $id"
 else
   echo "Preserved non-pending or completed job $id"
 fi
done
