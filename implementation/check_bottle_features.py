from pathlib import Path
import sys

import cv2
import faiss
import numpy as np
import torch

ROOT = Path.home() / "msc-visual-anomaly-detection"
DATA = ROOT / "data/mvtec_anomaly_detection/bottle"

sys.path.insert(0, str(ROOT / "third_party/AnomalyDINO"))
from src.backbones import get_model

reference = sorted((DATA / "train/good").glob("*.png"))[0]
normal = sorted((DATA / "test/good").glob("*.png"))[0]
anomaly = sorted((DATA / "test/contamination").glob("*.png"))[0]

def read_rgb(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("CUDA:", torch.cuda.is_available(), flush=True)
assert torch.cuda.is_available()

model = get_model("dinov2_vits14", "cuda", smaller_edge_size=448)

ref_tensor, _ = model.prepare_image(read_rgb(reference))
ref_features = np.ascontiguousarray(
    model.extract_features(ref_tensor), dtype=np.float32
)
faiss.normalize_L2(ref_features)

index = faiss.IndexFlatL2(ref_features.shape[1])
index.add(ref_features)

def anomaly_score(path):
    tensor, _ = model.prepare_image(read_rgb(path))
    features = np.ascontiguousarray(
        model.extract_features(tensor), dtype=np.float32
    )
    faiss.normalize_L2(features)
    distances, _ = index.search(features, 1)
    distances = distances.ravel() / 2.0
    k = max(1, int(len(distances) * 0.01))
    return float(np.mean(np.sort(distances)[-k:]))

print("Reference:", reference.name, flush=True)
print("Normal image:", normal.name, flush=True)
print("Normal score:", anomaly_score(normal), flush=True)
print("Defective image:", anomaly.name, flush=True)
print("Defective score:", anomaly_score(anomaly), flush=True)
print("BOTTLE SMOKE TEST FINISHED", flush=True)
