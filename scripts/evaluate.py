"""Public entry point for the thesis experiment workflow."""
import argparse
import os
import shlex
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
ROUTES = {'baseline': ('evaluate_baseline_category.py', []), 'robustness': ('lighting_blur_study.py', ['evaluate']), 'robustness-summary': ('lighting_blur_study.py', ['summarize']), 'ablation': ('analyze_improvement.py', ['evaluate']), 'ablation-summary': ('analyze_improvement.py', ['summarize']), 'calibration-summary': ('threshold_calibration.py', ['summarize'])}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing it")
    parser.add_argument("--dataset-root", type=Path, help="Official extracted MVTec AD directory (or set MVTEC_ROOT)")
    parser.add_argument("study", choices=ROUTES)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    file, prefix = ROUTES[args.study]
    if args.study == "calibration-summary":
        options = argparse.ArgumentParser(prog="evaluate.py calibration-summary")
        options.add_argument("--output", type=Path, required=True, help="Frozen calibration archive")
        options.add_argument("--archive", type=Path, required=True, help="Archived baseline and lighting/blur results root")
        options.add_argument("--dest-root", type=Path, required=True, help="New root for reports/threshold_calibration")
        c = options.parse_args(args.args)
        if (c.dest_root/"reports/threshold_calibration").exists():
            options.error("Destination report already exists; choose a fresh root.")
        entry = "from pathlib import Path; import sys; sys.path.insert(0,sys.argv[1]); import threshold_calibration as c; c.OUT=Path(sys.argv[2]); c.ARCHIVE=Path(sys.argv[3]); c.ROOT=Path(sys.argv[4]); c.summarize()"
        cmd = [sys.executable, "-c", entry, str(ROOT/"implementation"), str(c.output.resolve()), str(c.archive.resolve()), str(c.dest_root.resolve())]
    else:
        if args.study in {"robustness-summary", "ablation-summary"} and "--help" not in args.args:
            summary = argparse.ArgumentParser(prog=f"evaluate.py {args.study}", add_help=False)
            summary.add_argument("--dest", type=Path, required=True)
            dest, _ = summary.parse_known_args(args.args)
            if dest.dest.expanduser().exists():
                summary.error("Choose a new summary directory; destination already exists.")
        if args.study in {"ablation","ablation-summary"} and "--help" not in args.args:
            required = ["--output"] + (["--dest"] if args.study=="ablation-summary" else [])
            for flag in required:
                if not any(a==flag or a.startswith(flag+"=") for a in args.args):
                    parser.error(f"{args.study} requires an explicit {flag}.")
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
