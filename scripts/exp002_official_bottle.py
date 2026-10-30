import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/work3/s183356/datasets/mvtec_official")
OUTPUT = ROOT / "results/EXP-002/official_bottle_1shot_seed0"
MANIFEST = ROOT / "protocols/adino_mvtec_1shot_448_references.json"

sys.path.insert(0, str(ROOT / "third_party/AnomalyDINO"))

from src.backbones import get_model
from src.detection import run_anomaly_detection
from src.utils import get_dataset_info


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    assert not OUTPUT.exists(), "Output already exists; refusing overwrite."

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["protocol_id"] == "ADINO-MVTEC-P01"

    ref_entry = manifest["references"]["bottle"]["0"]
    reference = DATA / ref_entry["path"]

    assert reference.is_file()
    assert sha256(reference) == ref_entry["sha256"]
    assert reference.name == "000.png"
    assert sorted(os.listdir(reference.parent))[0] == reference.name

    upstream_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "third_party/AnomalyDINO"),
         "rev-parse", "HEAD"], text=True
    ).strip()

    assert upstream_commit == manifest["upstream_commit"]

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints/dinov2_vits14_pretrain.pth"
    )

    assert sha256(checkpoint) == (
        "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
    ), "Checkpoint fingerprint mismatch"

    source = (
        Path(torch.hub.get_dir())
        / "facebookresearch_dinov2_main"
    )

    digest = hashlib.sha256()
    files = sorted(
        p for p in source.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    assert digest.hexdigest() == (
        "fabaa1cb43240673152198284e8cf911268e7d3a0b2214ede94527774e15f5dc"
    ), "DINOv2 source fingerprint mismatch"

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    assert torch.cuda.is_available(), "CUDA GPU unavailable"

    objects, anomalies, masking, rotation = get_dataset_info(
        "MVTec", "agnostic", data_path=str(DATA)
    )
    assert len(objects) == 15

    model = get_model(
        "dinov2_vits14", "cuda", smaller_edge_size=448
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)

    print("Experiment: EXP-002", flush=True)
    print("Dataset:", DATA, flush=True)
    print("Reference:", reference, flush=True)

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
        save_patch_dists=True,
        save_tiffs=True,
    )

    paths = sorted(scores)
    labels = [0 if p.startswith("good/") else 1 for p in paths]
    values = [float(scores[p]) for p in paths]

    assert len(paths) == 83, "Unexpected test image count"
    assert len(set(labels)) == 2

    auc = float(roc_auc_score(labels, values))

    maps_dir = OUTPUT / "anomaly_maps/seed=0/bottle/test"
    npy_count = len(list(maps_dir.rglob("*.npy")))
    tiff_count = len(list(maps_dir.rglob("*.tiff")))

    assert npy_count == 83, f"Expected 83 NPY maps, got {npy_count}"
    assert tiff_count == 83, f"Expected 83 TIFF maps, got {tiff_count}"

    with (OUTPUT / "predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "anomaly_score"])
        for path, label, score in zip(paths, labels, values):
            writer.writerow([path, label, format(score, ".17g")])

    result = {
        "experiment": "EXP-002",
        "protocol": "ADINO-MVTEC-P01",
        "category": "bottle",
        "dataset_root": str(DATA),
        "model": "dinov2_vits14",
        "resolution": 448,
        "shots": 1,
        "seed": 0,
        "preprocess": "agnostic",
        "rotation": rotation["bottle"],
        "masking": masking["bottle"],
        "faiss_on_cpu": True,
        "reference": str(reference.relative_to(DATA)),
        "reference_sha256": sha256(reference),
        "checkpoint_sha256": sha256(checkpoint),
        "dinov2_source_fingerprint": digest.hexdigest(),
        "upstream_commit": upstream_commit,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "test_images": len(paths),
        "normal_images": labels.count(0),
        "anomalous_images": labels.count(1),
        "npy_maps": npy_count,
        "tiff_maps": tiff_count,
        "image_auroc": auc,
        "memory_bank_seconds": memory_time,
        "pixel_metrics": "NOT YET EVALUATED"
    }

    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    print("Image AUROC:", auc, flush=True)
    print("NPY maps:", npy_count, flush=True)
    print("TIFF maps:", tiff_count, flush=True)
    print("Results:", OUTPUT, flush=True)
    print("EXP-002 INFERENCE COMPLETED", flush=True)


if __name__ == "__main__":
    main()
