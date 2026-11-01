import argparse
import csv
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/work3/s183356/datasets/mvtec_official")
MANIFEST = ROOT / "protocols/adino_mvtec_1shot_448_references.json"
UPSTREAM = ROOT / "third_party/AnomalyDINO"

EXPECTED_COUNTS = {
    "bottle": 83, "cable": 150, "capsule": 132,
    "carpet": 117, "grid": 78, "hazelnut": 110,
    "leather": 124, "metal_nut": 115, "pill": 167,
    "screw": 160, "tile": 117, "toothbrush": 42,
    "transistor": 100, "wood": 79, "zipper": 151,
}

CHECKPOINT_SHA = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
SOURCE_SHA = (
    "fabaa1cb43240673152198284e8cf911268e7d3a0b2214ede94527774e15f5dc"
)

sys.path.insert(0, str(UPSTREAM))
from src.backbones import get_model
from src.detection import run_anomaly_detection
from src.utils import get_dataset_info


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(source):
    digest = hashlib.sha256()
    files = sorted(
        p for p in source.rglob("*.py")
        if "__pycache__" not in p.parts
    )
    assert files, "DINOv2 source files not found"

    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category", required=True,
        choices=sorted(EXPECTED_COUNTS)
    )
    parser.add_argument(
        "--seed", required=True, type=int,
        choices=(0, 1, 2)
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    category = args.category
    seed = args.seed

    output = (
        ROOT / "results/EXP-003"
        / f"{category}_1shot_seed{seed}"
    )

    assert not output.exists(), (
        f"Output already exists; refusing overwrite: {output}"
    )

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["protocol_id"] == "ADINO-MVTEC-P01"
    assert set(manifest["references"]) == set(EXPECTED_COUNTS)

    entry = manifest["references"][category][str(seed)]
    reference = DATA / entry["path"]

    assert reference.is_file()
    assert sha256(reference) == entry["sha256"]

    candidates = sorted(os.listdir(
        DATA / category / "train/good"
    ))
    assert reference.name == candidates[seed], (
        "Reference differs from upstream deterministic selection"
    )

    test_root = DATA / category / "test"
    test_paths = sorted(test_root.glob("*/*.png"))
    assert len(test_paths) == EXPECTED_COUNTS[category]

    expected_keys = {
        f"{p.parent.name}/{p.name}" for p in test_paths
    }
    assert any(k.startswith("good/") for k in expected_keys)
    assert any(not k.startswith("good/") for k in expected_keys)

    for folder in test_root.iterdir():
        if folder.is_dir():
            assert all(
                p.is_file() and p.suffix == ".png"
                for p in folder.iterdir()
            ), f"Unexpected test directory contents: {folder}"

    upstream_commit = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"],
        text=True
    ).strip()
    assert upstream_commit == manifest["upstream_commit"]

    checkpoint = (
        Path(torch.hub.get_dir())
        / "checkpoints/dinov2_vits14_pretrain.pth"
    )
    assert checkpoint.is_file()
    assert sha256(checkpoint) == CHECKPOINT_SHA

    source = (
        Path(torch.hub.get_dir())
        / "facebookresearch_dinov2_main"
    )
    assert source_fingerprint(source) == SOURCE_SHA

    objects, anomalies, masking, rotation = get_dataset_info(
        "MVTec", "agnostic", data_path=str(DATA)
    )
    assert set(objects) == set(EXPECTED_COUNTS)

    print("Experiment: EXP-003", flush=True)
    print("Category:", category, flush=True)
    print("Seed:", seed, flush=True)
    print("Reference:", reference, flush=True)
    print("Test images:", len(test_paths), flush=True)
    print("Output:", output, flush=True)

    if args.preflight:
        print("EXP-003 PREFLIGHT: PASSED", flush=True)
        return

    assert torch.cuda.is_available(), "CUDA GPU unavailable"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = get_model(
        "dinov2_vits14", "cuda", smaller_edge_size=448
    )

    output.mkdir(parents=True, exist_ok=False)

    # Provide a fresh anomaly-type dictionary to avoid
    # modifying upstream configuration across calls.
    fresh_anomalies = {
        name: list(types)
        for name, types in anomalies.items()
    }

    scores, memory_time, inference_times = run_anomaly_detection(
        model=model,
        object_name=category,
        data_root=str(DATA),
        n_ref_samples=1,
        object_anomalies=fresh_anomalies,
        plots_dir=str(output),
        save_examples=False,
        masking=masking[category],
        mask_ref_images=False,
        rotation=rotation[category],
        knn_metric="L2_normalized",
        knn_neighbors=1,
        faiss_on_cpu=True,
        seed=seed,
        save_patch_dists=True,
        save_tiffs=False,
    )

    assert set(scores) == expected_keys
    assert set(inference_times) == expected_keys

    maps_dir = (
        output / "anomaly_maps"
        / f"seed={seed}" / category / "test"
    )
    npy_files = sorted(maps_dir.glob("*/*.npy"))
    tiff_files = list(maps_dir.glob("*/*.tiff"))

    expected_maps = {
        f"{p.parent.name}/{p.stem}" for p in test_paths
    }
    actual_maps = {
        p.relative_to(maps_dir).with_suffix("").as_posix()
        for p in npy_files
    }

    assert actual_maps == expected_maps
    assert not tiff_files, "Unexpected TIFF predictions"

    for path in npy_files:
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        assert raw.ndim == 2 and raw.size > 0
        assert np.isfinite(raw).all(), path

    paths = sorted(scores)
    labels = [int(not p.startswith("good/")) for p in paths]
    values = [float(scores[p]) for p in paths]

    assert np.isfinite(values).all()
    assert set(labels) == {0, 1}

    image_auc = float(roc_auc_score(labels, values))

    with (output / "predictions.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "label", "anomaly_score"])
        for path, label, value in zip(paths, labels, values):
            writer.writerow([path, label, format(value, ".17g")])

    repo_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True
    ).strip()

    result = {
        "experiment": "EXP-003",
        "protocol": "ADINO-MVTEC-P01",
        "status": "INFERENCE_COMPLETED",
        "category": category,
        "seed": seed,
        "shots": 1,
        "resolution": 448,
        "model": "dinov2_vits14",
        "preprocessing": "agnostic",
        "rotation": rotation[category],
        "masking": masking[category],
        "mask_ref_images": False,
        "knn_metric": "L2_normalized",
        "knn_neighbors": 1,
        "faiss_on_cpu": True,
        "reference": entry["path"],
        "reference_sha256": entry["sha256"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "dinov2_source_fingerprint": SOURCE_SHA,
        "upstream_commit": upstream_commit,
        "thesis_commit": repo_commit,
        "runner_sha256": sha256(Path(__file__)),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "hostname": socket.gethostname(),
        "lsf_job_id": os.environ.get("LSB_JOBID"),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "test_images": len(paths),
        "normal_images": labels.count(0),
        "anomalous_images": labels.count(1),
        "npy_maps": len(npy_files),
        "tiff_maps": 0,
        "image_auroc": image_auc,
        "memory_bank_seconds": float(memory_time),
        "pixel_metrics": "PENDING_CPU_EVALUATION",
    }

    temporary = output / "run_complete.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output / "run_complete.json")

    print("Image AUROC:", image_auc, flush=True)
    print("NPY maps:", len(npy_files), flush=True)
    print("TIFF maps:", len(tiff_files), flush=True)
    print("EXP-003 INFERENCE COMPLETED", flush=True)


if __name__ == "__main__":
    main()
