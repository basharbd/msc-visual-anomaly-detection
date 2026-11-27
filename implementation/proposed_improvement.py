"""Fixed-budget reference augmentation; separate from the pinned upstream."""
import argparse
import csv
import hashlib
import json
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from run_baseline import (ROOT, DATA, MANIFEST, UPSTREAM, EXPECTED_COUNTS,
                          CHECKPOINT_SHA, SOURCE_SHA, sha256, source_fingerprint,
                          get_model, get_dataset_info)
from lighting_blur import CONDITIONS, perturb
from src.utils import augment_image
from src.post_eval import mean_top1p

VARIANTS = {"rotation_only": (), "brightness": ("brightness_085",),
            "blur": ("blur_075",), "combined": ("brightness_085", "blur_075")}
DEFAULT_OUTPUT = Path("/work3/s183356/experiments/reference_augmentation")
ARCHIVE = Path.home() / "msc-visual-anomaly-detection/results"


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def rgb(path):
    array = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f"Cannot read {path}")
    return cv2.cvtColor(array, cv2.COLOR_BGR2RGB)


def fixed_budget(pool, count, seed):
    assert len(pool) >= count > 0
    if len(pool) == count:
        return np.ascontiguousarray(pool, dtype=np.float32)
    ids = np.random.default_rng(seed).choice(len(pool), count, replace=False)
    return np.ascontiguousarray(pool[np.sort(ids)], dtype=np.float32)


def validate_environment(manifest):
    assert torch.cuda.is_available(), "CUDA GPU unavailable"
    assert manifest["protocol_id"] == "ADINO-MVTEC-P01"
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    assert commit == manifest["upstream_commit"]
    hub = Path(torch.hub.get_dir())
    assert sha256(hub / "checkpoints/dinov2_vits14_pretrain.pth") == CHECKPOINT_SHA
    assert source_fingerprint(hub / "facebookresearch_dinov2_main") == SOURCE_SHA
    return commit


def features(model, image):
    tensor, grid = model.prepare_image(image)
    array = np.ascontiguousarray(model.extract_features(tensor), dtype=np.float32)
    assert array.ndim == 2 and len(array) == grid[0] * grid[1]
    assert np.isfinite(array).all()
    return array, grid


def banks(model, category, seed, reference, rotation, variants):
    raw = rgb(reference)
    views = augment_image(raw) if rotation else [raw]
    chunks = {v: [] for v in variants}
    start = time.perf_counter()
    for j, view in enumerate(views):
        original, _ = features(model, view)
        altered = {}
        for condition in {c for v in variants for c in VARIANTS[v]}:
            resized = model.transform.transforms[0](Image.fromarray(view))
            altered[condition], _ = features(model, perturb(resized, condition))
            assert altered[condition].shape == original.shape
        for variant in variants:
            pool = np.concatenate([original, *(altered[c] for c in VARIANTS[variant])], axis=0)
            chunks[variant].append(fixed_budget(pool, len(original), stable_seed(category, seed, j, "bank")))
    indexes = {}
    counts = {}
    for variant, arrays in chunks.items():
        bank = np.ascontiguousarray(np.concatenate(arrays), dtype=np.float32)
        faiss.normalize_L2(bank)
        index = faiss.IndexFlatL2(bank.shape[1])
        index.add(bank)
        indexes[variant] = index
        counts[variant] = index.ntotal
    assert len(set(counts.values())) == 1
    return indexes, counts, time.perf_counter() - start


def run_category(model, manifest, upstream_commit, category, seeds, conditions, variants, output):
    _, _, masking, rotation = get_dataset_info("MVTec", "agnostic", data_path=str(DATA))
    paths = sorted((DATA / category / "test").glob("*/*.png"))
    assert len(paths) == EXPECTED_COUNTS[category]
    labels = [int(p.parent.name != "good") for p in paths]
    keys = [f"{p.parent.name}/{p.name}" for p in paths]
    references, indexes, counts, bank_times = {}, {}, {}, {}
    for seed in seeds:
        entry = manifest["references"][category][str(seed)]
        reference = DATA / entry["path"]
        assert sha256(reference) == entry["sha256"]
        references[seed] = entry
        indexes[seed], counts[seed], bank_times[seed] = banks(model, category, seed, reference, rotation[category], variants)
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    for condition in conditions:
        pending = {}
        for seed in seeds:
            for variant in variants:
                final = output / variant / condition / f"{category}_1shot_seed{seed}"
                if (final / "run_complete.json").is_file():
                    record = json.loads((final / "run_complete.json").read_text())
                    assert record["variant"] == variant and record["reference_sha256"] == references[seed]["sha256"]
                    assert record["runner_sha256"] in {sha256(Path(__file__)), "5c9eb15bcbb1dceb44755ef53d52e90def0c32d07f618b1e02f22c1b26d00d1f", "55fb45bd4dd1ab33db906f863253366f02983f6fc0d77fe9fa79a2295d06b37f"}
                    continue
                assert not final.exists(), f"Partial final output: {final}"
                temp = final.with_name(final.name + ".partial")
                temp.mkdir(parents=True, exist_ok=False)
                pending[seed,variant] = (final,temp,{},0.0)
        if not pending:
            continue
        shared_time = 0.0
        for path,key in zip(paths,keys):
            t0 = time.perf_counter()
            image = rgb(path)
            if condition != "original":
                image = perturb(model.transform.transforms[0](Image.fromarray(image)), condition)
            array, grid = features(model, image)
            if masking[category]:
                # Shared, reproducible mask; deliberately independent of reference seed.
                np.random.seed(stable_seed(category, condition, key, "mask"))
                mask = model.compute_background_mask(array, grid, threshold=10, masking_type=masking[category])
            else:
                mask = np.ones(len(array), dtype=bool)
            assert mask.any()
            query = np.ascontiguousarray(array[mask], dtype=np.float32)
            faiss.normalize_L2(query)
            shared_time += time.perf_counter() - t0
            for (seed,variant),(final,temp,scores,_) in pending.items():
                t1 = time.perf_counter()
                distances,_ = indexes[seed][variant].search(query, 1)
                raw = np.zeros(len(mask), dtype=float)
                raw[mask] = (distances / 2).squeeze()
                scores[key] = float(mean_top1p(raw))
                map_path = temp / "anomaly_maps" / f"seed={seed}" / category / "test" / path.parent.name / (path.stem + ".npy")
                map_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(map_path, raw.reshape(grid))
                pending[seed,variant] = (final,temp,scores,pending[seed,variant][3]+time.perf_counter()-t1)
        for (seed,variant),(final,temp,scores,search_time) in pending.items():
            with (temp/"predictions.csv").open("w",newline="") as f:
                writer = csv.writer(f); writer.writerow(["image","label","anomaly_score"])
                writer.writerows((k,l,format(scores[k],".17g")) for k,l in zip(keys,labels))
            # The archive lookup uses the final category name but reads the temporary maps.
            parity = None
            if variant == "rotation_only":
                # Temporarily use the final-name component for archive matching via explicit path.
                archive_folder = ARCHIVE / ("EXP-003" if condition=="original" else f"lighting_blur/{condition}") / final.name
                if (archive_folder/"run_complete.json").is_file():
                    with (archive_folder/"predictions.csv").open() as f:
                        old = {r["image"]:float(r["anomaly_score"]) for r in csv.DictReader(f)}
                    assert set(old)==set(scores)
                    se=max(abs(old[k]-scores[k]) for k in scores)
                    me=0.0
                    for p in (temp/"anomaly_maps").rglob("*.npy"):
                        a=np.load(p,allow_pickle=False); b=np.load(archive_folder/p.relative_to(temp),allow_pickle=False)
                        assert a.shape==b.shape
                        me=max(me,float(np.max(np.abs(a-b))))
                    parity={"max_score_error":se,"max_map_error":me,"strict":not bool(masking[category])}
                    if not masking[category]:
                        # Recovery guard: original smoke gate remains unchanged.
                        parity["original_tolerance"] = 1e-6
                        parity["original_tolerance_passed"] = max(se, me) <= 1e-6
                        parity["recovery_score_tolerance"] = 1e-6
                        parity["recovery_map_tolerance"] = 5e-6
                        assert se <= 1e-6 and me <= 5e-6, (category,condition,se,me)
            record = dict(experiment="reference_augmentation", protocol="ADINO-MVTEC-P01",
                status="INFERENCE_COMPLETED", category=category, seed=seed, variant=variant,
                condition=condition, shots=1, resolution=448, model="dinov2_vits14",
                preprocessing="agnostic", rotation=rotation[category], masking=masking[category],
                mask_ref_images=False, knn_metric="L2_normalized", knn_neighbors=1, faiss_on_cpu=True,
                reference=references[seed]["path"], reference_sha256=references[seed]["sha256"],
                checkpoint_sha256=CHECKPOINT_SHA, dinov2_source_fingerprint=SOURCE_SHA,
                upstream_commit=upstream_commit, thesis_commit=commit, runner_sha256=sha256(Path(__file__)),
                transform_sha256=sha256(ROOT/"implementation/lighting_blur.py"),
                python_version=sys.version.split()[0], torch_version=torch.__version__,
                gpu=torch.cuda.get_device_name(0), hostname=socket.gethostname(), lsf_job_id=os.environ.get("LSB_JOBID"),
                completed_utc=datetime.now(timezone.utc).isoformat(), test_images=len(paths),
                normal_images=labels.count(0), anomalous_images=labels.count(1), npy_maps=len(paths), tiff_maps=0,
                bank_patches=counts[seed][variant], shared_bank_build_seconds=bank_times[seed],
                shared_test_features_seconds=shared_time, search_and_save_seconds=search_time,
                archive_parity=parity, image_auroc=float(roc_auc_score(labels,[scores[k] for k in keys])))
            (temp/"run_complete.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
            temp.rename(final)
            print("COMPLETE",category,seed,variant,condition,"patches",counts[seed][variant],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category",choices=sorted(EXPECTED_COUNTS),required=True)
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text())
    upstream_commit=validate_environment(manifest)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    faiss.omp_set_num_threads(4)
    torch.set_num_threads(4)
    model=get_model("dinov2_vits14","cuda",smaller_edge_size=448)
    run_category(model,manifest,upstream_commit,args.category,[0] if args.smoke else range(3),
                 ("original","brightness_085") if args.smoke else CONDITIONS,tuple(VARIANTS),args.output)


if __name__=="__main__":
    main()

