"""Evaluate fixed-budget reference augmentation and produce paired reports."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from proposed_improvement import DEFAULT_OUTPUT, VARIANTS, CONDITIONS, ROOT, EXPECTED_COUNTS, sha256
from evaluate_baseline_category import evaluate, WORK_ROOT

METRICS = ["image_auroc","image_ap","image_f1max","pixel_auroc","pixel_f1max","pixel_aupro"]


def tasks(smoke=False):
    categories = ("bottle","screw") if smoke else sorted(EXPECTED_COUNTS)
    return [(cat,seed,variant,condition) for cat in categories
            for seed in ([0] if smoke else range(3)) for variant in VARIANTS
            for condition in (("original","brightness_085") if smoke else CONDITIONS)]


def checked_record(output, task):
    cat,seed,variant,condition = task
    folder=output/variant/condition/f"{cat}_1shot_seed{seed}"
    inference=json.loads((folder/"run_complete.json").read_text())
    assert (inference["category"],inference["seed"],inference["variant"],inference["condition"])==task
    assert inference["status"]=="INFERENCE_COMPLETED" and inference["test_images"]==EXPECTED_COUNTS[cat]
    return folder,inference


def evaluate_task(output, task):
    cat,seed,variant,condition=task
    folder,record=checked_record(output,task)
    if not (folder/"evaluation_metrics.json").is_file():
        with TemporaryDirectory(dir=WORK_ROOT.parent,prefix="reference_augmentation_eval_") as tmp:
            evaluate(SimpleNamespace(category=cat,seed=seed,variant=variant,condition=condition),
                     Path(tmp),output_override=folder)
    metrics=json.loads((folder/"evaluation_metrics.json").read_text())
    assert metrics["inference_record_sha256"]==sha256(folder/"run_complete.json")
    assert (metrics["category"],metrics["seed"],metrics["variant"],metrics["condition"])==task
    assert metrics["pro_fpr_limit"]==0.3
    assert all(-1e-12<=metrics["metrics"][m]<=1+1e-12 for m in METRICS)
    return dict(category=cat,seed=seed,variant=variant,condition=condition,
                reference_sha256=record["reference_sha256"],bank_patches=record["bank_patches"],
                shared_bank_build_seconds=record["shared_bank_build_seconds"],
                shared_test_features_seconds=record["shared_test_features_seconds"],
                search_and_save_seconds=record["search_and_save_seconds"],
                archive_max_map_error=(record.get("archive_parity") or {}).get("max_map_error"),
                **metrics["metrics"])


def summarize(output,dest,smoke=False):
    rows=[]
    for task in tasks(smoke):
        folder,_=checked_record(output,task)
        assert (folder/"evaluation_metrics.json").is_file(), f"Missing evaluation: {folder}"
        rows.append(evaluate_task(output,task))
    runs=pd.DataFrame(rows)
    paired=["category","seed","condition"]
    assert not runs.duplicated(paired+["variant"]).any()
    assert (runs.groupby(paired).bank_patches.nunique()==1).all()
    assert (runs.groupby(paired).reference_sha256.nunique()==1).all()
    assert len(runs)==(16 if smoke else 900)
    base=runs[runs.variant=="rotation_only"][paired+METRICS].rename(columns={m:m+"_baseline" for m in METRICS})
    runs=runs.merge(base,on=paired,validate="many_to_one")
    clean=runs[runs.condition=="original"][["category","seed","variant"]+METRICS].rename(columns={m:m+"_clean" for m in METRICS})
    runs=runs.merge(clean,on=["category","seed","variant"],validate="many_to_one")
    for m in METRICS:
        runs[m+"_delta_pp"]=100*(runs[m]-runs[m+"_baseline"])
        runs[m+"_robustness_gap_pp"]=100*(runs[m]-runs[m+"_clean"])
    cols=METRICS+[m+s for m in METRICS for s in ("_delta_pp","_robustness_gap_pp")]
    cat=runs.groupby(["category","variant","condition"],as_index=False)[cols].mean()
    macro=cat.groupby(["variant","condition"],as_index=False)[cols].mean()
    # Factorial interaction: combined - brightness - blur + rotation-only (percentage points).
    pivot=cat.pivot(index=["category","condition"],columns="variant",values=METRICS)
    interaction=pd.DataFrame({m:100*(pivot[m]["combined"]-pivot[m]["brightness"]-pivot[m]["blur"]+pivot[m]["rotation_only"]) for m in METRICS})
    dest.mkdir(parents=True,exist_ok=False)
    runs.to_csv(dest/"per_run.csv",index=False)
    cat.to_csv(dest/"per_category.csv",index=False)
    macro.to_csv(dest/"macro.csv",index=False)
    interaction.to_csv(dest/"factorial_interaction_pp.csv")
    print("Validated",len(runs),"paired evaluations; wrote",dest)
    print(macro[["variant","condition","image_auroc","pixel_aupro","image_auroc_delta_pp","pixel_aupro_delta_pp"]].to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("evaluate","summarize"))
    p.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    p.add_argument("--dest",type=Path,default=ROOT/"reports/reference_augmentation")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--start",type=int,default=1)
    p.add_argument("--count",type=int,default=1)
    a=p.parse_args()
    if a.stage=="summarize":
        summarize(a.output,a.dest,a.smoke)
    else:
        all_tasks=tasks(a.smoke)
        assert 1<=a.start<=len(all_tasks) and a.count>0
        for task in all_tasks[a.start-1:a.start-1+a.count]:
            print("EVALUATE",task,flush=True)
            evaluate_task(a.output,task)


if __name__=="__main__":
    main()

