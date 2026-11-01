import json
import os
import resource
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/work3/s183356/datasets/mvtec_official")

NEW = ROOT / "results/EXP-003/bottle_1shot_seed0"
OLD = ROOT / "results/EXP-002/official_bottle_1shot_seed0"

WORK = ROOT / "results/EXP-003_eval_workspaces/bottle_1shot_seed0"
MAPS = WORK / "anomaly_maps/seed=0"
RESULT = NEW / "evaluation_metrics.json"

sys.path.insert(0, str(ROOT / "third_party/AnomalyDINO"))

from src.utils import dists2map
from src.post_eval import (
    parse_dataset_files,
    eval_classification,
    eval_segmentation,
)


def main():
    assert (NEW / "run_complete.json").is_file(), (
        "EXP-003 inference has not completed"
    )
    assert not RESULT.exists(), (
        "EXP-003 evaluation result already exists"
    )
    assert not WORK.exists(), (
        "Evaluation workspace already exists; refusing overwrite"
    )

    inference = json.loads(
        (NEW / "run_complete.json").read_text()
    )
    assert inference["experiment"] == "EXP-003"
    assert inference["category"] == "bottle"
    assert inference["seed"] == 0
    assert inference["test_images"] == 83
    assert inference["npy_maps"] == 83
    assert inference["tiff_maps"] == 0

    source = NEW / "anomaly_maps/seed=0/bottle/test"
    files = sorted(source.glob("*/*.npy"))
    assert len(files) == 83

    expected = {
        f"{p.parent.name}/{p.stem}"
        for p in (DATA / "bottle/test").glob("*/*.png")
    }
    actual = {
        p.relative_to(source).with_suffix("").as_posix()
        for p in files
    }
    assert actual == expected, "Prediction pairing mismatch"

    print("=== EXP-003 CPU EVALUATION ===", flush=True)
    print("Original inference:", NEW, flush=True)
    print("Isolated workspace:", WORK, flush=True)
    print("Predictions:", len(files), flush=True)

    reconstructed_count = 0

    for src in files:
        relative = src.relative_to(source)
        target_dir = MAPS / "bottle/test" / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        # Symlink only: never copy or modify original NPY.
        linked_npy = target_dir / relative.name
        linked_npy.symlink_to(src.resolve())

        image_path = (
            DATA / "bottle/test"
            / relative.parent
            / (src.stem + ".png")
        )

        with Image.open(image_path) as image:
            width, height = image.size

        raw = np.load(src, allow_pickle=False)
        assert raw.ndim == 2
        assert np.isfinite(raw).all()

        reconstructed = dists2map(raw, (height, width))
        assert reconstructed.shape == (height, width)
        assert np.isfinite(reconstructed).all()

        tifffile.imwrite(
            target_dir / (src.stem + ".tiff"),
            reconstructed
        )
        reconstructed_count += 1

    assert reconstructed_count == 83
    print("TIFF maps reconstructed:", reconstructed_count, flush=True)

    gt_files, pred_stems = parse_dataset_files(
        object_name="bottle",
        dataset_base_dir=str(DATA),
        anomaly_maps_dir=str(MAPS),
        dataset="MVTec",
    )

    assert len(gt_files) == len(pred_stems) == 83
    assert len(set(pred_stems)) == 83

    for gt, stem in zip(gt_files, pred_stems):
        assert Path(stem + ".npy").is_file()
        assert Path(stem + ".tiff").is_file()
        if gt is not None:
            assert Path(gt).is_file()

    print("Prediction and ground-truth pairing: PASSED", flush=True)

    image_auc, image_ap, image_f1 = eval_classification(
        gt_files,
        pred_stems,
        aggregation_statistics="meantop1p",
    )

    pixel_aupro, pixel_auc, pixel_f1 = eval_segmentation(
        gt_files,
        pred_stems,
        pro_integration_limit=0.3,
        delete_tiff_files=False,
    )

    measured = {
        "image_auroc": float(image_auc),
        "image_ap": float(image_ap),
        "image_f1max": float(image_f1),
        "pixel_auroc": float(pixel_auc),
        "pixel_f1max": float(pixel_f1),
        "pixel_aupro": float(pixel_aupro),
    }

    old_metrics = json.loads(
        (OLD / "evaluation_metrics.json").read_text()
    )

    print("\n=== EXP-002 vs EXP-003 METRICS ===", flush=True)

    for name, value in measured.items():
        previous = float(old_metrics[name])
        difference = abs(value - previous)

        print(
            f"{name}: EXP-002={previous:.12f}, "
            f"EXP-003={value:.12f}, "
            f"difference={difference:.3e}",
            flush=True,
        )

        assert np.isclose(
            value, previous, rtol=0, atol=1e-10
        ), f"Metric mismatch: {name}"

    assert np.isclose(
        measured["image_auroc"],
        inference["image_auroc"],
        rtol=0,
        atol=1e-10,
    )

    peak_ram_mib = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )

    result = {
        "experiment": "EXP-003",
        "category": "bottle",
        "seed": 0,
        "test_images": 83,
        "metrics": measured,
        "pro_fpr_limit": 0.3,
        "comparison": "EXP-002",
        "comparison_tolerance": 1e-10,
        "comparison_passed": True,
        "reconstructed_tiff_maps": reconstructed_count,
        "evaluation_workspace": str(WORK),
        "original_inference_preserved": True,
        "peak_python_rss_mib": peak_ram_mib,
    }

    with RESULT.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print("\nALL SIX METRICS: MATCH EXP-002", flush=True)
    print("EXP-003 CPU EVALUATION: PASSED", flush=True)
    print("Peak Python RSS:", peak_ram_mib, "MiB", flush=True)
    print("Original EXP-003 predictions preserved.", flush=True)


if __name__ == "__main__":
    main()
