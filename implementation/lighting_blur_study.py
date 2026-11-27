"""Run one indexed LSF task or summarize the complete paired study."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from lighting_blur import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/lighting_blur"
CLEAN = ROOT / "reports/baseline_reproduction/per_run.csv"
METRICS = ["image_auroc", "image_ap", "image_f1max", "pixel_auroc", "pixel_f1max", "pixel_aupro"]


def read_record(path):
    return json.loads(path.read_text())


def verify_inference(folder, base, condition):
    info = read_record(folder / "run_complete.json")
    expected = dict(experiment="lighting_blur", category=base.category, seed=base.seed,
                    condition=condition, status="INFERENCE_COMPLETED", reference=base.reference,
                    reference_sha256=base.reference_sha256)
    if any(info.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Inference metadata mismatch: {folder}")
    if not (folder / "predictions.csv").is_file() or info["test_images"] != info["npy_maps"]:
        raise ValueError(f"Incomplete predictions: {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("infer", "evaluate", "summarize"))
    parser.add_argument("index", type=int, nargs="?")
    parser.add_argument("--dest", type=Path, help="New directory for summary CSVs")
    args = parser.parse_args()
    if args.stage == "summarize":
        if args.dest is None:
            parser.error("Summarize requires --dest.")
        if args.dest.exists():
            parser.error("Choose a new summary directory; destination already exists.")
    baseline = pd.read_csv(CLEAN).sort_values(["category", "seed"])
    manifest = read_record(ROOT / "protocols/adino_mvtec_1shot_448_references.json")
    assert set(zip(baseline.category, baseline.seed)) == {
        (cat, seed) for cat in manifest["references"] for seed in range(3)}
    tasks = [(base, cond) for base in baseline.itertuples() for cond in CONDITIONS[1:]]
    assert len(tasks) == 180
    pilot = read_record(DATA / "original/bottle_1shot_seed0/run_complete.json")
    assert pilot["identity_check"] == "passed" and pilot["maps_compared"] == 83
    if args.stage != "summarize":
        if args.index is None or not 1 <= args.index <= len(tasks):
            parser.error("Choose an array index from 1 to 180.")
        tasks = [tasks[args.index - 1]]

    rows = []
    for base, condition in tasks:
        folder = DATA / condition / f"{base.category}_1shot_seed{base.seed}"
        common = ["--category", base.category, "--seed", str(base.seed), "--condition", condition]
        if args.stage == "infer" and not (folder / "run_complete.json").exists():
            subprocess.run([sys.executable, ROOT / "implementation/run_baseline.py", *common], check=True)
        verify_inference(folder, base, condition)
        if args.stage == "infer":
            return
        metrics_file = folder / "evaluation_metrics.json"
        if args.stage == "evaluate" and not metrics_file.exists():
            subprocess.run([sys.executable, ROOT / "implementation/evaluate_baseline_category.py", *common], check=True)
        result = read_record(metrics_file)
        digest = hashlib.sha256((folder / "run_complete.json").read_bytes()).hexdigest()
        assert result["inference_record_sha256"] == digest and result["pro_fpr_limit"] == 0.3
        assert (result["category"], result["seed"], result["condition"]) == (base.category, base.seed, condition)
        row = dict(category=base.category, seed=base.seed, condition=condition, reference=base.reference)
        for metric in METRICS:
            value = result["metrics"][metric]
            assert -1e-12 <= value <= 1 + 1e-12
            clean = getattr(base, metric)
            row.update({metric: value, metric + "_original": clean, metric + "_change_pp": 100 * (value - clean)})
        rows.append(row)

    if args.stage == "summarize":
        dest = args.dest.resolve()
        dest.mkdir(exist_ok=False, parents=True)
        runs = pd.DataFrame(rows)
        columns = [m + "_change_pp" for m in METRICS]
        category = runs.groupby(["category", "condition"])[columns].mean()
        macro = category.groupby("condition").mean()
        runs.to_csv(dest / "per_run.csv", index=False)
        category.to_csv(dest / "per_category.csv")
        macro.to_csv(dest / "macro_change_pp.csv")
        print(macro.to_string())


if __name__ == "__main__":
    main()
