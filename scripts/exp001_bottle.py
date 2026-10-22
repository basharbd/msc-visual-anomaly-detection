import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mvtec_anomaly_detection"
OUTPUT = ROOT / "results/EXP-001/bottle_1shot_seed0"

sys.path.insert(0, str(ROOT / "third_party/AnomalyDINO"))

from src.backbones import get_model
from src.detection import run_anomaly_detection
from src.utils import get_dataset_info

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    assert torch.cuda.is_available(), "CUDA GPU unavailable"

    objects, anomalies, masking, rotation = get_dataset_info(
        "MVTec", "agnostic", data_path=str(DATA)
    )

    reference = sorted((DATA / "bottle/train/good").glob("*.png"))[0]

    print("Experiment: EXP-001", flush=True)
    print("Category: bottle", flush=True)
    print("Reference:", reference.name, flush=True)

    model = get_model(
        "dinov2_vits14", "cuda", smaller_edge_size=448
    )

    scores, memory_time, inference_times = run_anomaly_detection(
        model=model,
        object_name="bottle",
        data_root=str(DATA),
        n_ref_samples=1,
        object_anomalies=anomalies,
        plots_dir=str(OUTPUT),
        save_examples=False,
        masking=masking["bottle"],
        mask_ref_images=False,
        rotation=rotation["bottle"],
        knn_metric="L2_normalized",
        knn_neighbors=1,
        faiss_on_cpu=True,
        seed=0,
        save_patch_dists=False,
        save_tiffs=False,
    )

    paths = sorted(scores)
    labels = [0 if p.startswith("good/") else 1 for p in paths]
    values = [float(scores[p]) for p in paths]

    assert len(paths) == 83, f"Expected 83 test images, got {len(paths)}"
    assert len(set(labels)) == 2, "Both classes are required"

    auc = float(roc_auc_score(labels, values))

    with (OUTPUT / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "anomaly_score"])
        for path, label, score in zip(paths, labels, values):
            writer.writerow([path, label, score])

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints/dinov2_vits14_pretrain.pth"
    )

    result = {
        "experiment": "EXP-001",
        "category": "bottle",
        "model": "dinov2_vits14",
        "resolution": 448,
        "shots": 1,
        "seed": 0,
        "preprocess": "agnostic",
        "rotation": rotation["bottle"],
        "masking": masking["bottle"],
        "faiss_on_cpu": True,
        "reference_image": reference.name,
        "dataset_revision": "c75b39616f84db43677bcc8228caaafaf5096d7f",
        "checkpoint_sha256": hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest(),
        "test_images": len(paths),
        "normal_images": labels.count(0),
        "anomalous_images": labels.count(1),
        "image_auroc": auc,
        "memory_bank_seconds": memory_time,
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print("Image-level AUROC:", auc, flush=True)
    print("Results saved to:", OUTPUT, flush=True)
    print("EXP-001 FINISHED", flush=True)

if __name__ == "__main__":
    main()
