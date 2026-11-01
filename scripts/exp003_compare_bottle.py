import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

OLD = ROOT / "results/EXP-002/official_bottle_1shot_seed0"
NEW = ROOT / "results/EXP-003/bottle_1shot_seed0"

TOLERANCE = 1e-6


def read_scores(folder):
    with (folder / "predictions.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 83
    result = {
        row["image"]: (
            int(row["label"]),
            float(row["anomaly_score"])
        )
        for row in rows
    }
    assert len(result) == 83
    return result


def read_maps(folder):
    path = folder / "anomaly_maps/seed=0/bottle/test"
    files = sorted(path.glob("*/*.npy"))
    assert len(files) == 83
    return {
        file.relative_to(path).as_posix(): file
        for file in files
    }


def main():
    assert (OLD / "predictions.csv").is_file()
    assert (NEW / "run_complete.json").is_file(), (
        "EXP-003 inference has not completed"
    )

    old_scores = read_scores(OLD)
    new_scores = read_scores(NEW)

    assert old_scores.keys() == new_scores.keys()

    score_errors = []

    for key in sorted(old_scores):
        old_label, old_score = old_scores[key]
        new_label, new_score = new_scores[key]

        assert old_label == new_label
        assert np.isfinite([old_score, new_score]).all()
        score_errors.append(abs(old_score - new_score))

    old_maps = read_maps(OLD)
    new_maps = read_maps(NEW)

    assert old_maps.keys() == new_maps.keys()

    exact_maps = 0
    close_maps = 0
    maximum_error = 0.0

    for name in sorted(old_maps):
        a = np.load(old_maps[name], allow_pickle=False)
        b = np.load(new_maps[name], allow_pickle=False)

        assert a.shape == b.shape, name
        assert a.dtype == b.dtype, name
        assert np.isfinite(a).all() and np.isfinite(b).all()

        error = float(np.max(np.abs(a - b)))
        maximum_error = max(maximum_error, error)

        if np.array_equal(a, b):
            exact_maps += 1
        if np.allclose(a, b, rtol=0, atol=TOLERANCE):
            close_maps += 1

    max_score_error = max(score_errors)

    print("=== EXP-002 vs EXP-003 ===")
    print("Image scores compared:", len(score_errors))
    print("Maximum score difference:", max_score_error)
    print("Maps compared:", len(old_maps))
    print("Exactly identical maps:", exact_maps)
    print("Maps within tolerance:", close_maps)
    print("Maximum map difference:", maximum_error)
    print("Tolerance:", TOLERANCE)

    assert max_score_error <= TOLERANCE
    assert close_maps == 83
    print("PREDICTION COMPARISON: PASSED")


if __name__ == "__main__":
    main()
