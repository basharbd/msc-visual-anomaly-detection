"""Public entry point for the thesis experiment workflow."""
import argparse
import os
import shlex
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
ROUTES = {'baseline': ('run_baseline.py', []), 'robustness': ('lighting_blur_study.py', ['infer']), 'ablation': ('proposed_improvement.py', []), 'calibration': ('threshold_calibration.py', ['infer'])}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it")
    parser.add_argument("--dataset-root", type=Path, help="Official extracted MVTec AD directory (or set MVTEC_ROOT)")
    parser.add_argument("study", choices=ROUTES)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    file, prefix = ROUTES[args.study]
    if args.study == "calibration":
        options = argparse.ArgumentParser(prog="run_experiment.py calibration")
        options.add_argument("--category", required=True)
        options.add_argument("--output", type=Path, required=True, help="Dedicated calibration archive")
        c = options.parse_args(args.args)
        entry = "from pathlib import Path; import sys; sys.path.insert(0,sys.argv[1]); import os; import threshold_calibration as c; c.DATA=Path(os.environ.get(\"MVTEC_ROOT\",str(c.DATA))).expanduser().resolve(); c.OUT=Path(sys.argv[2]); c.infer(sys.argv[3])"
        cmd = [sys.executable, "-c", entry, str(ROOT/"implementation"), str(c.output.resolve()), c.category]
    else:
        if args.study == "ablation" and "--help" not in args.args and not any(a=="--output" or a.startswith("--output=") for a in args.args):
            parser.error("Ablation requires an explicit --output directory.")
        cmd = [sys.executable, str(ROOT/"implementation"/file), *prefix, *args.args]
    env = os.environ.copy()
    if args.dataset_root:
        env["MVTEC_ROOT"] = str(args.dataset_root.expanduser().resolve())
    if args.dry_run:
        print(("MVTEC_ROOT=" + shlex.quote(env["MVTEC_ROOT"]) + " ") if "MVTEC_ROOT" in env else "", end="")
        print(shlex.join(cmd))
        return
    if args.dataset_root and not Path(env["MVTEC_ROOT"]).is_dir():
        parser.error("Dataset directory does not exist.")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
if __name__ == "__main__":
    main()
