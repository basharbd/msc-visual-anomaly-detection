import json
import resource
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/work3/s183356/datasets/mvtec_official")
OUTPUT = ROOT / "results/EXP-002/official_bottle_1shot_seed0"
MAPS = OUTPUT / "anomaly_maps/seed=0"
RESULT = OUTPUT / "evaluation_metrics.json"

sys.path.insert(0, str(ROOT / "third_party/AnomalyDINO"))

from src.post_eval import (
    parse_dataset_files,
    eval_classification,
    eval_segmentation,
)


def main():
    assert OUTPUT.is_dir(), "EXP-002 inference outputs not found"
    assert not RESULT.exists(), "Evaluation results already exist"

    gt_files, pred_stems = parse_dataset_files(
        object_name="bottle",
        dataset_base_dir=str(DATA),
        anomaly_maps_dir=str(MAPS),
        dataset="MVTec",
    )

    assert len(gt_files) == len(pred_stems) == 83
    assert len(set(pred_stems)) == 83

    normal_count = sum(gt is None for gt in gt_files)
    anomaly_count = len(gt_files) - normal_count

    assert normal_count > 0
    assert anomaly_count > 0

    for gt, stem in zip(gt_files, pred_stems):
        assert Path(stem + ".npy").is_file(), stem
        assert Path(stem + ".tiff").is_file(), stem

        if gt is not None:
            assert Path(gt).is_file(), gt

    print("EXP-002 prediction pairing: PASSED", flush=True)
    print("Test images:", len(pred_stems), flush=True)
    print("Normal:", normal_count, flush=True)
    print("Anomalous:", anomaly_count, flush=True)

    image_auroc, image_ap, image_f1 = eval_classification(
        gt_files,
        pred_stems,
        aggregation_statistics="meantop1p",
    )

    pixel_aupro, pixel_auroc, pixel_f1 = eval_segmentation(
        gt_files,
        pred_stems,
        pro_integration_limit=0.3,
        delete_tiff_files=False,
    )

    inference = json.loads((OUTPUT / "metrics.json").read_text())

    assert np.isclose(
        image_auroc,
        inference["image_auroc"],
        rtol=0,
        atol=1e-10,
    ), "Image AUROC mismatch between inference and evaluation"

    peak_ram_mib = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )

    metrics = {
        "experiment": "EXP-002",
        "category": "bottle",
        "seed": 0,
        "test_images": len(pred_stems),
        "normal_images": normal_count,
        "anomalous_images": anomaly_count,
        "image_auroc": float(image_auroc),
        "image_ap": float(image_ap),
        "image_f1max": float(image_f1),
        "pixel_auroc": float(pixel_auroc),
        "pixel_f1max": float(pixel_f1),
        "pixel_aupro": float(pixel_aupro),
        "pro_fpr_limit": 0.3,
        "tiff_files_preserved": True,
        "peak_python_rss_mib": peak_ram_mib,
    }

    with RESULT.open("x") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")

    print("EXP-002 EVALUATION COMPLETED", flush=True)
    print("Metrics saved:", RESULT, flush=True)
    print(f"Peak Python RSS: {peak_ram_mib:.1f} MiB")


if __name__ == "__main__":
    main()
