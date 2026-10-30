import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/work3/s183356/datasets/mvtec_official")
MAPS = (
    ROOT / "results/EXP-002/official_bottle_1shot_seed0/"
    "anomaly_maps/seed=0/bottle/test"
)

sys.path.insert(0, str(ROOT / "third_party/AnomalyDINO"))

from src.utils import dists2map


def main():
    npy_files = sorted(MAPS.glob("*/*.npy"))
    tiff_files = sorted(MAPS.glob("*/*.tiff"))

    assert len(npy_files) == 83
    assert len(tiff_files) == 83

    npy_names = {p.relative_to(MAPS).with_suffix("") for p in npy_files}
    tiff_names = {p.relative_to(MAPS).with_suffix("") for p in tiff_files}
    assert npy_names == tiff_names, "Prediction file pairing mismatch"

    checked = 0
    largest_error = 0.0

    for file in npy_files:
        image_path = (
            DATA / "bottle/test"
            / file.parent.name / (file.stem + ".png")
        )

        with Image.open(image_path) as image:
            width, height = image.size

        raw = np.load(file, allow_pickle=False)
        reconstructed = dists2map(raw, (height, width))
        original = tifffile.imread(file.with_suffix(".tiff"))

        assert reconstructed.shape == original.shape, file
        assert reconstructed.dtype == original.dtype, file

        difference = float(
            np.max(np.abs(reconstructed - original))
        )

        largest_error = max(largest_error, difference)

        assert np.array_equal(reconstructed, original), (
            f"Reconstruction mismatch: {file}"
        )

        checked += 1

        if checked % 10 == 0 or checked == 83:
            print(f"Verified {checked}/83 maps", flush=True)

    print("\n=== FINAL RECONSTRUCTION RESULT ===")
    print("Maps verified:", checked)
    print("Maximum absolute difference:", largest_error)
    print("Original files modified: NO")
    print("ALL 83 RECONSTRUCTIONS: PASSED")


if __name__ == "__main__":
    main()
