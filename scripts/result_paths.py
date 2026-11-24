"""Locate existing HPC run outputs when opening an older checkout."""
import sys
from pathlib import Path

LEGACY_DIRECTORIES = {
    "official_bottle_pilot": "EXP-002",
    "baseline_reproduction": "EXP-003",
}


def result_directory(root, study):
    results = root / "results"
    current = results / study
    previous = results / LEGACY_DIRECTORIES[study]
    if previous.exists():
        if current.exists() and not current.samefile(previous):
            raise RuntimeError(f"Two output directories exist for {study}: {current}, {previous}")
        return previous
    return current


if __name__ == "__main__":
    print(result_directory(Path(__file__).resolve().parents[1], sys.argv[1]))
