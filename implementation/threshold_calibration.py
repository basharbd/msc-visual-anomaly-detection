"""Separate normal-only threshold calibration; no modification of baseline outputs."""
import argparse, csv, hashlib, json, math, os, random, socket, statistics, subprocess, sys
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
DATA=Path("/work3/s183356/datasets/mvtec_official")
ARCHIVE=Path.home()/"msc-visual-anomaly-detection/results"
OUT=Path("/work3/s183356/experiments/threshold_calibration_v1")
MANIFEST=ROOT/"protocols/threshold_calibration_manifest.json"
REF=ROOT/"protocols/adino_mvtec_1shot_448_references.json"
CONDITIONS=("original","brightness_085","brightness_070","blur_075","blur_150")
N=40
RANK=math.ceil((N+1)*.95)
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def atomic_json(p,x):
    assert not p.exists(),p
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(x,indent=2)+"\n")
    tmp.replace(p)
def manifest():
    refs=json.loads(REF.read_text())
    categories={}
    for c,entries in sorted(refs["references"].items()):
        excluded={Path(e["path"]).name for e in entries.values()}
        pool=[p for p in (DATA/c/"train/good").glob("*.png") if p.name not in excluded]
        assert len(pool)>=N and len(excluded)==3,c
        pool.sort(key=lambda p:hashlib.sha256(("CAL-v1|"+c+"|"+p.name).encode()).hexdigest())
        paths=[str(p.relative_to(DATA)) for p in pool[:N]]
        assert len(set(paths))==N
        categories[c]={"references":entries,"calibration_paths":paths}
    m={"protocol":"ADINO-CAL-P01","n":N,"rank_1based":RANK,"nominal_tail":.05,"selection":"First 40 by SHA-256 of CAL-v1|category|filename, excluding all three references", "categories":categories}
    atomic_json(MANIFEST,m)
    print("Frozen manifest",len(categories),"categories",N,"normals each; rank",RANK)
def infer(category):
    from run_baseline import (get_model, run_anomaly_detection, get_dataset_info,
        CHECKPOINT_SHA,SOURCE_SHA,source_fingerprint,UPSTREAM)
    import numpy as np
    import torch
    import faiss
    m=json.loads(MANIFEST.read_text()); entry=m["categories"][category]
    assert m["n"]==N and m["rank_1based"]==RANK
    assert torch.cuda.is_available()
    assert subprocess.check_output(["git","-C",str(UPSTREAM),"rev-parse","HEAD"],text=True).strip()==json.loads(REF.read_text())["upstream_commit"]
    cache=Path(torch.hub.get_dir())
    assert digest(cache/"checkpoints/dinov2_vits14_pretrain.pth")==CHECKPOINT_SHA
    assert source_fingerprint(cache/"facebookresearch_dinov2_main")==SOURCE_SHA
    ref_hashes=set()
    for e in entry["references"].values():
        assert digest(DATA/e["path"])==e["sha256"]
        ref_hashes.add(e["sha256"])
    normals={p:digest(DATA/p) for p in entry["calibration_paths"]}
    assert len(normals)==N and not (set(normals.values())&ref_hashes)
    _,_,masking,rotation=get_dataset_info("MVTec","agnostic",data_path=str(DATA))
    torch.set_num_threads(4);faiss.omp_set_num_threads(4)
    model=get_model("dinov2_vits14","cuda",smaller_edge_size=448)
    with TemporaryDirectory(prefix="adino_cal_") as t:
        root=Path(t); train=root/category/"train/good"; test=root/category/"test/good"
        train.mkdir(parents=True);test.mkdir(parents=True)
        for e in entry["references"].values():
            p=DATA/e["path"]; (train/p.name).symlink_to(p)
        for rel in normals:
            p=DATA/rel;(test/p.name).symlink_to(p)
        assert not set(p.name for p in train.iterdir()) & set(p.name for p in test.iterdir())
        for seed in range(3):
            out=OUT/category/f"ref{seed}.json"
            if out.exists():
                old=json.loads(out.read_text())
                assert old["manifest_sha256"]==digest(MANIFEST) and old["runner_sha256"]==digest(Path(__file__)) and old["calibration_sha256"]==normals
                print("Verified existing record",category,seed,flush=True);continue
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            scores,_,_=run_anomaly_detection(model=model,object_name=category,data_root=str(root),
               n_ref_samples=1,object_anomalies={category:[]},plots_dir=str(root),
               save_examples=False,masking=masking[category],mask_ref_images=False,rotation=rotation[category],
               knn_metric="L2_normalized",knn_neighbors=1,faiss_on_cpu=True,seed=seed,save_patch_dists=False,save_tiffs=False)
            assert set(scores)=={"good/"+Path(p).name for p in normals}
            vals={k:float(v) for k,v in scores.items()}
            assert all(math.isfinite(v) for v in vals.values())
            tau=sorted(vals.values())[RANK-1]
            atomic_json(out,{"category":category,"seed":seed,"protocol":m["protocol"],"reference":entry["references"][str(seed)],"calibration_sha256":normals,"scores":vals,"threshold":tau,"rank":RANK,"manifest_sha256":digest(MANIFEST),"runner_sha256":digest(Path(__file__)),"commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"checkpoint_sha256":CHECKPOINT_SHA,"source_fingerprint":SOURCE_SHA,"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"python":sys.version.split()[0],"host":socket.gethostname(),"job_id":os.environ.get("LSB_JOBID"),"completed_utc":now()})
            print("CALIBRATION_COMPLETE",category,seed,tau,flush=True)
def summarize():
    m=json.loads(MANIFEST.read_text())
    keys=[(c,s) for c in sorted(m["categories"]) for s in range(3)]
    records={}
    # All thresholds are frozen before any test score or test label is read.
    for c,s in keys:
        p=OUT/c/f"ref{s}.json";x=json.loads(p.read_text())
        assert x["category"]==c and x["seed"]==s and x["manifest_sha256"]==digest(MANIFEST) and x["runner_sha256"]==digest(Path(__file__))
        assert len(x["scores"])==N and x["threshold"]==sorted(x["scores"].values())[RANK-1]
        assert set(x["calibration_sha256"])==set(m["categories"][c]["calibration_paths"])
        records[c,s]=(x,digest(p))
    rows=[]
    for c,s in keys:
        x,h=records[c,s];tau=x["threshold"]; clean=None
        for t in CONDITIONS:
            parent=ARCHIVE/"EXP-003" if t=="original" else ARCHIVE/"lighting_blur"/t
            p=parent/f"{c}_1shot_seed{s}"
            ir=json.loads((p/"run_complete.json").read_text())
            assert ir["reference_sha256"]==x["reference"]["sha256"] and ir["reference"]==x["reference"]["path"] and ir["checkpoint_sha256"]==x["checkpoint_sha256"]
            f=p/"predictions.csv";rr=list(csv.DictReader(f.open()))
            ids={r["image"]:int(r["label"]) for r in rr}
            assert len(rr)==len(ids)==ir["test_images"] and set(ids.values())=={0,1}
            if clean is None: clean=ids
            else: assert ids==clean
            assert all(math.isfinite(float(r["anomaly_score"])) for r in rr)
            fp=sum(int(r["label"])==0 and float(r["anomaly_score"])>tau for r in rr)
            fn=sum(int(r["label"])==1 and float(r["anomaly_score"])<=tau for r in rr)
            n0=sum(v==0 for v in ids.values());n1=len(ids)-n0
            rows.append(dict(category=c,seed=s,condition=t,threshold=tau,normal=n0,anomalous=n1,fp=fp,fn=fn,fpr=fp/n0,fnr=fn/n1,tpr=1-fn/n1,calibration_record_sha256=h,predictions_sha256=digest(f)))
    assert len(rows)==225
    report=ROOT/"reports/threshold_calibration"
    assert not report.exists()
    report.mkdir(parents=True)
    def write(name,rs):
        with (report/name).open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    write("per_run.csv",rows)
    macro=[]
    for t in CONDITIONS:
        a=[x for x in rows if x["condition"]==t];assert len(a)==45
        z={"condition":t,**{k:statistics.mean(x[k] for x in a) for k in ["fpr","fnr","tpr"]}}
        if not macro: base=z.copy()
        z.update({k+"_change_pp":100*(z[k]-base[k]) for k in ["fpr","fnr","tpr"]})
        macro.append(z)
    write("macro.csv",macro)
    write("per_category.csv",[{"category":c,"condition":t,**{k:statistics.mean(x[k] for x in rows if x["category"]==c and x["condition"]==t) for k in ["fpr","fnr","tpr"]}} for c in sorted(m["categories"]) for t in CONDITIONS])
    atomic_json(report/"summary.json",{"protocol":m["protocol"],"calibration_images_per_category":N,"rank":RANK,"manifest_sha256":digest(MANIFEST),"runs":len(rows),"macro":macro,"completed_utc":now()})
    print("COMPLETE",json.dumps(macro,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["manifest","infer","summarize"]);p.add_argument("--category")
    a=p.parse_args()
    if a.mode=="manifest":manifest()
    elif a.mode=="infer":infer(a.category)
    else:summarize()
