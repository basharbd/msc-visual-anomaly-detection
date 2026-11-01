import argparse
import hashlib
import json
import resource
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/work3/s183356/datasets/mvtec_official")
WORK_ROOT = Path("/work3/s183356/exp003_eval_workspaces")
UPSTREAM = ROOT / "third_party/AnomalyDINO"

sys.path.insert(0, str(UPSTREAM))

from src.utils import dists2map
from src.post_eval import (
    parse_dataset_files,
    eval_classification,
    eval_segmentation,
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    assert args.seed in (0, 1, 2)

    category = args.category
    seed = args.seed

    output = (
        ROOT / "results/EXP-003"
        / f"{category}_1shot_seed{seed}"
    )
    source = (
        output / "anomaly_maps"
        / f"seed={seed}" / category / "test"
    )
    work = WORK_ROOT / f"{category}_1shot_seed{seed}"
    maps = work / "anomaly_maps" / f"seed={seed}"
    result = output / "evaluation_metrics.json"

    assert not work.exists(), "Workspace already exists"
    assert not result.exists(), "Metrics already exist"

    inference_path = output / "run_complete.json"
    assert inference_path.is_file()

    inference = json.loads(inference_path.read_text())

    assert inference["experiment"] == "EXP-003"
    assert inference["status"] == "INFERENCE_COMPLETED"
    assert inference["category"] == category
    assert inference["seed"] == seed
    assert inference["protocol"] == "ADINO-MVTEC-P01"
    assert inference["tiff_maps"] == 0

    test_paths = sorted(
        (DATA / category / "test").glob("*/*.png")
    )
    expected = {
        f"{p.parent.name}/{p.stem}"
        for p in test_paths
    }

    files = sorted(source.glob("*/*.npy"))
    actual = {
        p.relative_to(source).with_suffix("").as_posix()
        for p in files
    }

    assert actual == expected, "Prediction pairing mismatch"
    assert len(files) == inference["test_images"]
    assert len(files) == inference["npy_maps"]
    assert not list(source.glob("*/*.tiff"))

    normal_count = sum(
        p.parent.name == "good" for p in test_paths
    )
    anomalous_count = len(test_paths) - normal_count

    assert normal_count == inference["normal_images"]
    assert anomalous_count == inference["anomalous_images"]

    print("=== EXP-003 CPU EVALUATION ===", flush=True)
    print("Category:", category, flush=True)
    print("Seed:", seed, flush=True)
    print("Images:", len(files), flush=True)
    print("Workspace:", work, flush=True)

    # Create an isolated workspace on WORK3.
    work.mkdir(parents=True, exist_ok=False)

    for index, src in enumerate(files, 1):
        relative = src.relative_to(source)
        target = maps / category / "test" / relative.parent
        target.mkdir(parents=True, exist_ok=True)

        # Read-only link to the original prediction.
        (target / relative.name).symlink_to(src.resolve())

        image_path = (
            DATA / category / "test"
            / relative.parent / (src.stem + ".png")
        )

        with Image.open(image_path) as image:
            width, height = image.size

        raw = np.load(src, allow_pickle=False)
        assert raw.ndim == 2 and raw.size > 0
        assert np.isfinite(raw).all()

        reconstructed = dists2map(raw, (height, width))
        assert reconstructed.shape == (height, width)
        assert np.isfinite(reconstructed).all()

        tifffile.imwrite(
            target / (src.stem + ".tiff"),
            reconstructed
        )

        if index % 20 == 0 or index == len(files):
            print(
                f"Reconstructed {index}/{len(files)}",
                flush=True
            )

    gt_files, pred_stems = parse_dataset_files(
        object_name=category,
        dataset_base_dir=str(DATA),
        anomaly_maps_dir=str(maps),
        dataset="MVTec",
    )

    assert len(gt_files) == len(pred_stems) == len(files)
    assert len(set(pred_stems)) == len(files)

    for gt, stem in zip(gt_files, pred_stems):
        assert Path(stem + ".npy").is_file()
        assert Path(stem + ".tiff").is_file()
        if gt is not None:
            assert Path(gt).is_file()

    print("Prediction pairing: PASSED", flush=True)

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

    metrics = {
        "image_auroc": float(image_auc),
        "image_ap": float(image_ap),
        "image_f1max": float(image_f1),
        "pixel_auroc": float(pixel_auc),
        "pixel_f1max": float(pixel_f1),
        "pixel_aupro": float(pixel_aupro),
    }

    assert all(np.isfinite(v) for v in metrics.values())
    assert np.isclose(
        metrics["image_auroc"],
        inference["image_auroc"],
        rtol=0,
        atol=1e-10,
    ), "Inference/evaluation AUROC mismatch"

    peak_rss_mib = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024
    )

    record = {
        "experiment": "EXP-003",
        "protocol": "ADINO-MVTEC-P01",
        "category": category,
        "seed": seed,
        "test_images": len(files),
        "normal_images": normal_count,
        "anomalous_images": anomalous_count,
        "metrics": metrics,
        "pro_fpr_limit": 0.3,
        "reconstructed_tiff_maps": len(files),
        "evaluation_workspace": str(work),
        "inference_predictions_preserved": True,
        "evaluation_script_sha256": sha256(Path(__file__)),
        "peak_python_rss_mib": peak_rss_mib,
    }

    # Save only after all checks and metrics succeed.
    with result.open("x") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print("=== FINAL EVALUATION METRICS ===", flush=True)
    for name, value in metrics.items():
        print(f"{name}: {value:.12f}", flush=True)

    print("Peak Python RSS MiB:", peak_rss_mib, flush=True)
    print("EXP-003 CATEGORY EVALUATION: PASSED", flush=True)


if __name__ == "__main__":
    main()
