"""Generate baseline figures from the archived category means."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

REPORT = Path(__file__).resolve().parents[1] / "reports" / "baseline_reproduction"


def main():
    with (REPORT / "per_category.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    destination = REPORT / "figures"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "mvtec-baseline"})
    for metric, label in [("image_auroc", "Image AUROC"), ("pixel_f1max", "Pixel F1-max")]:
        values = [float(row[f"{metric}_mean"]) for row in rows]
        categories = [row["category"].replace("_", " ").capitalize() for row in rows]
        fig, ax = plt.subplots(figsize=(7, 5.6), layout="constrained")
        ax.barh(categories, values, height=0.62, color="#286576")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel(f"{label} (mean across three one-shot reference selections)")
        ax.set_title(f"{label} by MVTec AD category", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(destination / f"{metric}.svg", metadata={"Date": None})
        plt.close(fig)


if __name__ == "__main__":
    main()
