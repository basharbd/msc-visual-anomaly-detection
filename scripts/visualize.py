"""Generate publication assets from verified CSV reports without modifying them."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/publication_assets"


def generate_publication_assets(output_dir=DEFAULT_OUTPUT, study="all"):
    """Generate study figures (and, for all studies, five tables) from verified reports."""
    # Notebook and CLI use the same plotting defaults without changing the caller style.
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        return _generate_publication_assets(output_dir, study)


def _generate_publication_assets(output_dir, study):
    if study not in {"all", "baseline", "robustness", "ablation", "calibration"}:
        raise ValueError(f"Unknown study: {study}")
    data = ROOT / "reports"
    output = Path(output_dir).expanduser().resolve()
    for protected in (data, ROOT / "results", ROOT / "research_logs"):
        protected = protected.resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError(f"Output overlaps protected research records: {output}")
    output.mkdir(parents=True, exist_ok=False)
    figs, tables = output / "figures", output / "tables"
    figs.mkdir()
    tables.mkdir()
    m=pd.read_csv(data/'reference_augmentation/macro.csv')
    c=pd.read_csv(data/'reference_augmentation/per_category.csv')
    r=pd.read_csv(data/'reference_augmentation/per_run.csv')
    cal=pd.read_csv(data/'threshold_calibration/macro.csv')
    audit=json.loads((data/'final_scientific_analysis/audit.json').read_text())
    metrics=['image_auroc','image_ap','image_f1max','pixel_auroc','pixel_f1max','pixel_aupro']
    variants=['rotation_only','brightness','blur','combined']
    names=['Rotation only','Brightness','Blur','Combined']
    conditions=['original','brightness_085','brightness_070','blur_075','blur_150']
    labels=['Original','Brightness\n0.85','Brightness\n0.70','Blur\n0.75','Blur\n1.50']
    assert len(r)==900 and len(m)==20 and len(c)==300 and len(cal)==5
    assert r.bank_patches.eq(8192).all()
    assert np.allclose(r.groupby(['variant','condition'])[metrics].mean().sort_index(),m.set_index(['variant','condition'])[metrics].sort_index(),atol=1e-12,rtol=0)
    assert len(pd.read_csv(data/'threshold_calibration/per_run.csv'))==225
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':10,'axes.titlesize':11,'axes.titleweight':'semibold','pdf.fonttype':42,'savefig.dpi':220,'axes.axisbelow':True})
    colors=['#475569','#C78526','#007C83','#6855A0']
    markers=['o','s','D','^']
    def save(fig,stem, fixed_layout=False):
        bbox = None if fixed_layout else 'tight'
        fig.savefig(figs/(stem+'.pdf'),bbox_inches=bbox,pad_inches=.12,
                    metadata={'CreationDate': None, 'ModDate': None})
        fig.savefig(figs/(stem+'.png'),bbox_inches=bbox,pad_inches=.12)
        plt.close(fig)
    def profiles(metric,title,ylab,stem):
        fig,ax=plt.subplots(figsize=(7.1,3.55),layout='constrained')
        for v,n,col,mk in zip(variants,names,colors,markers):
            vals=m[m.variant==v].set_index('condition').loc[conditions,metric]*100
            ax.plot(range(5),vals,marker=mk,color=col,label=n,lw=1.6,ms=5)
        ax.set_xticks(range(5),labels);ax.set_ylabel(ylab);ax.set_title(title,loc='left')
        ax.grid(axis='y',alpha=.22);ax.legend(loc='lower left',ncol=2,frameon=False,fontsize=8)
        ax.set_ylim(min(m[metric])*100-.65,max(m[metric])*100+.3)
        save(fig,stem)
    if study in {"all", "ablation"}:
        profiles('image_auroc','Image-level detection across conditions','Image AUROC (%)','01_ablation_image_auroc')
        profiles('pixel_aupro','Pixel-level localization across conditions','Pixel AUPRO (%)','02_ablation_pixel_aupro')
        for metric,title,stem in [('image_auroc','Strong blur: image-level effect of blur augmentation','03_strong_blur_category_image'),('pixel_aupro','Strong blur: localization effect of blur augmentation','04_strong_blur_category_pixel')]:
            x=c[(c.variant=='blur')&(c.condition=='blur_150')].sort_values(metric+'_delta_pp')
            vals=x[metric+'_delta_pp']
            fig,ax=plt.subplots(figsize=(7.1,4.6))
            fig.subplots_adjust(left=.18, right=.97, bottom=.13, top=.91)
            ax.barh(x.category.str.replace('_',' '),vals,color=['#007C83' if v>=0 else '#B64B45' for v in vals],height=.62)
            ax.axvline(0,color='#475569',lw=.8);ax.grid(axis='x',alpha=.22)
            ax.set_xlabel('Difference vs rotation-only (percentage points)')
            ax.set_title(title,loc='left')
            ax.margins(x=.12)
            save(fig,stem, fixed_layout=True)
    if study in {"all", "calibration"}:
        fig,ax=plt.subplots(figsize=(7.1,3.55),layout='constrained')
        d=cal.set_index('condition').loc[conditions]
        x=np.arange(5)
        ax.bar(x-.18,d.fpr*100,.36,label='False positive rate',color='#007C83')
        ax.bar(x+.18,d.fnr*100,.36,label='False negative rate',color='#C78526')
        ax.set_xticks(x,labels);ax.set_ylabel('Rate (%)');ax.set_title('Frozen normal-only threshold under imaging shifts',loc='left')
        ax.legend(frameon=False,loc='upper left',ncol=2,fontsize=8)
        ax.set_ylim(0,27);ax.grid(axis='y',alpha=.22)
        for container in ax.containers: ax.bar_label(container,fmt='%.1f',padding=2,fontsize=7.5)
        save(fig,'05_threshold_calibration')
    z=r[(r.variant=='blur')&(r.condition=='blur_150')].groupby('seed')[['image_auroc_delta_pp','pixel_aupro_delta_pp']].mean()
    if study in {'all', 'ablation'}:
        # Reference sensitivity: three paired reference-choice macro means; no confidence interval claim.
        fig,ax=plt.subplots(figsize=(7.1,3.55),layout='constrained')
        ax.bar(np.arange(3)-.18,z.image_auroc_delta_pp,.36,label='Image AUROC',color='#007C83')
        ax.bar(np.arange(3)+.18,z.pixel_aupro_delta_pp,.36,label='Pixel AUPRO',color='#6855A0')
        ax.set_xticks(range(3),['Reference index 0','Reference index 1','Reference index 2'])
        ax.set_ylabel('Paired gain (percentage points)');ax.set_title('Strong blur: effect across reference choices',loc='left')
        ax.legend(frameon=False,loc='upper right',fontsize=8);ax.set_ylim(0,2.1);ax.grid(axis='y',alpha=.22)
        for container in ax.containers: ax.bar_label(container,fmt='%.2f',padding=2,fontsize=8)
        save(fig,'06_reference_sensitivity')
    if study in {"all", "baseline"}:
        b = pd.read_csv(data / "baseline_reproduction/per_category.csv").sort_values("category")
        assert len(b) == 15 and b.category.is_unique
        y = np.arange(len(b))
        image = 100 * b.image_auroc_mean
        pixel = 100 * b.pixel_aupro_mean
        fig, ax = plt.subplots(figsize=(7.1, 4.6))
        fig.subplots_adjust(left=.18, right=.96, bottom=.19, top=.96)
        ax.hlines(y, np.minimum(image, pixel), np.maximum(image, pixel), color="#a6a6a6", lw=1.1)
        ax.scatter(image, y, color="#183c5b", s=32, label="Image AUROC", zorder=3)
        ax.scatter(pixel, y, color="#c83b40", s=32, marker="s", label="Pixel AUPRO", zorder=3)
        ax.set_yticks(y, b.category.str.replace("_", " ").str.capitalize())
        ax.invert_yaxis()
        ax.set_xlim(60, 101)
        ax.set_xticks(np.arange(60, 101, 10))
        ax.set_xlabel("Score (%)")
        ax.grid(axis="x", alpha=.22)
        ax.legend(loc="lower center", bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
        save(fig, "07_baseline_category_comparison", fixed_layout=True)
    if study in {"all", "robustness"}:
        delta = pd.read_csv(data / "lighting_blur/macro_change_pp.csv").set_index("condition")
        order = ["brightness_085", "brightness_070", "blur_075", "blur_150"]
        delta = delta.loc[order]
        fig, ax = plt.subplots(figsize=(7.1, 3.35), layout="constrained")
        x = np.arange(4)
        ax.bar(x-.18, delta.image_auroc_change_pp, .36, label="Image AUROC", color="#183c5b")
        ax.bar(x+.18, delta.pixel_aupro_change_pp, .36, label="Pixel AUPRO", color="#c83b40")
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set_xticks(x, ["Brightness 0.85", "Brightness 0.70", "Blur 0.75", "Blur 1.50"])
        ax.set_ylabel("Change (percentage points)")
        ax.grid(axis="y", alpha=.22)
        ax.legend(loc="lower left", frameon=False)
        save(fig, "08_robustness_macro_change")
    if study == 'all':
        # Numeric tables: percentages only for absolute scores; pp for deltas.
        names_map=dict(zip(variants,names)); cond_map=dict(zip(conditions,['Original','Brightness 0.85','Brightness 0.70','Blur 0.75','Blur 1.50']))
        idx=pd.MultiIndex.from_product([conditions,variants],names=['condition','variant'])
        a=m.set_index(['condition','variant']).loc[idx].reset_index()
        a[metrics]=a[metrics]*100
        a['condition']=a.condition.map(cond_map);a['variant']=a.variant.map(names_map)
        short=['Image AUROC','Image AP','Image F1-max','Pixel AUROC','Pixel F1-max','Pixel AUPRO']
        def table(df,stem,caption,label,fmt='%.2f'):
            df.to_csv(tables/(stem+'.csv'),index=False,float_format='%.10f')
            tex=df.to_latex(index=False,escape=True,float_format=lambda x: fmt%x,caption=caption,label=label,position='htbp')
            (tables/(stem+'.tex')).write_text(tex,encoding='utf-8')
        table(a[['condition','variant']+metrics].rename(columns=dict(zip(['condition','variant']+metrics,['Condition','Bank']+short))),
              '01_ablation_all_metrics','Macro-averaged ablation scores (percent). Equal weight per category; three fixed reference choices.','tab:ablation_all_metrics')
        primary=m.set_index(['condition','variant']).loc[idx].reset_index()
        primary=primary[primary.variant!='rotation_only'].copy()
        primary['condition']=primary.condition.map(cond_map);primary['variant']=primary.variant.map(names_map)
        table(primary[['condition','variant','image_auroc_delta_pp','pixel_aupro_delta_pp','image_auroc_robustness_gap_pp','pixel_aupro_robustness_gap_pp']].rename(columns={'condition':'Condition','variant':'Bank','image_auroc_delta_pp':'AUROC gain','pixel_aupro_delta_pp':'AUPRO gain','image_auroc_robustness_gap_pp':'AUROC gap','pixel_aupro_robustness_gap_pp':'AUPRO gap'}),
              '02_paired_primary_effects','Paired gain versus rotation-only and degradation versus the same bank on original images (percentage points).','tab:paired_primary')
        category=c[(c.variant=='blur')&(c.condition=='blur_150')][['category','image_auroc_delta_pp','pixel_aupro_delta_pp']].sort_values('image_auroc_delta_pp',ascending=False)
        table(category.rename(columns={'category':'Category','image_auroc_delta_pp':'Image AUROC gain','pixel_aupro_delta_pp':'Pixel AUPRO gain'}),'03_strong_blur_category_effects','Paired category-level effect of blur augmentation under strong blur (percentage points).','tab:strong_blur_categories')
        d=cal.set_index('condition').loc[conditions].reset_index()
        for col in ['fpr','fnr','tpr']: d[col]*=100
        d['condition']=d.condition.map(cond_map)
        table(d[['condition','fpr','fnr','tpr','fpr_change_pp','fnr_change_pp']].rename(columns={'condition':'Condition','fpr':'FPR (%)','fnr':'FNR (%)','tpr':'TPR (%)','fpr_change_pp':'FPR change (pp)','fnr_change_pp':'FNR change (pp)'}),'04_threshold_calibration','Frozen-threshold macro rates and changes relative to original images. Calibration uses clean normal training images.','tab:threshold_calibration')
        table(z.reset_index().rename(columns={'seed':'Reference index','image_auroc_delta_pp':'Image AUROC gain','pixel_aupro_delta_pp':'Pixel AUPRO gain'}),'05_reference_sensitivity','Strong-blur paired macro gains by reference choice (percentage points).','tab:reference_sensitivity')
    return {'figures': figs, 'tables': tables}




def generate_error_case_panel(dataset_root, results_root, output_dir):
    """Reconstruct selected cases from the original dataset and archived patch maps.

    No model inference is performed. Output files are new, and no archived
    predictions or dataset files are modified. Requires the pinned upstream
    submodule and the optional CPU map-analysis dependencies.
    """
    import hashlib
    import importlib.util
    import subprocess
    from PIL import Image, ImageFilter

    dataset_root = Path(dataset_root).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (dataset_root, results_root, ROOT / "reports",
                 ROOT / "results", ROOT / "research_logs"):
        path = path.resolve()
        if output_dir == path or path in output_dir.parents or output_dir in path.parents:
            raise ValueError(f"Output overlaps protected inputs: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(output_dir)

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    manifest = json.loads((ROOT / "protocols/adino_mvtec_1shot_448_references.json").read_text())
    upstream = ROOT / "third_party/AnomalyDINO"
    utils = upstream / "src/utils.py"
    if not utils.is_file():
        raise FileNotFoundError("Initialize the pinned AnomalyDINO submodule first.")
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != manifest["upstream_commit"]:
        raise ValueError("AnomalyDINO submodule does not match the reference manifest.")
    if subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--", "src/utils.py"],
        text=True,
    ).strip():
        raise ValueError("The pinned map reconstruction source has local changes.")
    spec = importlib.util.spec_from_file_location("pinned_anomalydino_utils", utils)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dists2map = module.dists2map

    candidates = [results_root / name for name in ("EXP-003", "baseline_reproduction")
                  if (results_root / name).is_dir()]
    if not candidates:
        raise FileNotFoundError("Archived clean baseline not found in VAD_RESULTS_ROOT.")
    if len(candidates) == 2 and not candidates[0].samefile(candidates[1]):
        raise ValueError("Ambiguous clean baseline archives; choose a root with one archive.")
    clean = candidates[0]
    cases_path = ROOT / "reports/lighting_blur/selected_blur_cases.csv"
    cases = pd.read_csv(cases_path)
    baseline = pd.read_csv(ROOT / "reports/baseline_reproduction/per_run.csv")
    if len(cases) != 4 or cases.duplicated(["category", "reference_index", "test_image"]).any():
        raise ValueError("Unexpected selected-case list.")
    panels, provenance, measurements = [], [], []

    for case in cases.itertuples(index=False):
        category, seed, rel = case.category, int(case.reference_index), Path(case.test_image)
        if rel.is_absolute() or len(rel.parts) != 2 or ".." in rel.parts or rel.suffix != ".png":
            raise ValueError(f"Invalid test image path: {rel}")
        reference = manifest["references"][category][str(seed)]
        reference_path = dataset_root / reference["path"]
        if digest(reference_path) != reference["sha256"]:
            raise ValueError(f"Reference hash mismatch: {reference_path}")
        image_path = dataset_root / category / "test" / rel
        with Image.open(image_path) as source:
            original = source.convert("RGB")
        width, height = original.size
        # Matches the pinned torchvision Resize(int, BICUBIC) on PIL input.
        size = (448, int(448 * height / width)) if width <= height else (int(448 * width / height), 448)
        resized = original.resize(size, Image.Resampling.BICUBIC)
        blurred = resized.filter(ImageFilter.GaussianBlur(radius=1.50))
        normal = rel.parent.name == "good"
        mask_path = None if normal else (
            dataset_root / category / "ground_truth" / rel.parent / (rel.stem + "_mask.png")
        )
        if normal:
            mask = np.zeros((height, width), dtype=bool)
        else:
            with Image.open(mask_path) as source:
                mask = np.asarray(source.convert("L")) > 0
            if mask.shape != (height, width) or not mask.any():
                raise ValueError(f"Invalid ground-truth mask: {mask_path}")

        base_row = baseline[(baseline.category == category) & (baseline.seed == seed)]
        if len(base_row) != 1:
            raise ValueError("Missing or duplicate archived clean run.")
        base_row = base_row.iloc[0]
        run_name = f"{category}_1shot_seed{seed}"
        maps, scores = [], []
        input_files = {"test_image": image_path, "reference_image": reference_path}
        if mask_path is not None:
            input_files["ground_truth"] = mask_path
        for condition, archive, expected_score in (
            ("original", clean, case.original_score),
            ("blur_150", results_root / "lighting_blur" / "blur_150", case.blur_150_score),
        ):
            folder = archive / run_name
            record_path = folder / "run_complete.json"
            evaluation_path = folder / "evaluation_metrics.json"
            predictions_path = folder / "predictions.csv"
            record = json.loads(record_path.read_text())
            evaluation = json.loads(evaluation_path.read_text())
            expected = {"category": category, "seed": seed, "protocol": manifest["protocol_id"],
                        "status": "INFERENCE_COMPLETED", "reference": reference["path"],
                        "reference_sha256": reference["sha256"], "upstream_commit": commit,
                        "resolution": 448, "shots": 1}
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Archived run metadata mismatch: {folder}")
            if condition == "original":
                if digest(record_path) != base_row.inference_record_sha256 or digest(evaluation_path) != base_row.evaluation_record_sha256:
                    raise ValueError(f"Clean archive hash mismatch: {folder}")
            else:
                if record.get("experiment") != "lighting_blur" or record.get("condition") != condition:
                    raise ValueError(f"Wrong perturbation archive: {folder}")
                if evaluation.get("inference_record_sha256") != digest(record_path):
                    raise ValueError(f"Perturbed evaluation provenance mismatch: {folder}")
            predictions = pd.read_csv(predictions_path)
            match = predictions[predictions.image == rel.as_posix()]
            if len(match) != 1 or int(match.iloc[0].label) != int(not normal):
                raise ValueError(f"Prediction pairing failed: {folder}/{rel}")
            score = float(match.iloc[0].anomaly_score)
            if not np.isclose(score, expected_score, atol=1e-12, rtol=0):
                raise ValueError(f"Selected-case score differs from verified CSV: {folder}/{rel}")
            raw_path = folder / "anomaly_maps" / f"seed={seed}" / category / "test" / rel.with_suffix(".npy")
            raw = np.load(raw_path, allow_pickle=False)
            if raw.ndim != 2 or not raw.size or not np.isfinite(raw).all():
                raise ValueError(f"Invalid raw anomaly map: {raw_path}")
            # Upstream image score: mean of the largest floor(1%) patch scores, minimum one.
            flat = raw.ravel()
            k = max(1, int(flat.size * .01))
            if not np.isclose(np.mean(sorted(flat, reverse=True)[:k]), score, atol=1e-12, rtol=0):
                raise ValueError(f"Raw map does not reproduce the archived image score: {raw_path}")
            reconstructed = dists2map(raw, (height, width))
            if reconstructed.shape != (height, width) or not np.isfinite(reconstructed).all():
                raise ValueError(f"Invalid reconstructed map: {raw_path}")
            maps.append(reconstructed)
            scores.append(score)
            input_files.update({f"{condition}_map": raw_path, f"{condition}_predictions": predictions_path,
                                f"{condition}_inference": record_path, f"{condition}_evaluation": evaluation_path})
        if not np.isclose(scores[1] - scores[0], case.change, atol=1e-12, rtol=0):
            raise ValueError(f"Paired change mismatch: {category}/{rel}")
        # A shared full-range scale avoids concealing the effect of blur.
        vmax = max(float(np.max(maps[0])), float(np.max(maps[1])), np.finfo(float).eps)
        panels.append(dict(category=category, rel=rel.as_posix(), seed=seed, original=resized,
                           blurred=blurred, mask=mask, maps=maps, scores=scores, vmax=vmax))
        measurements.append(dict(category=category, reference_index=seed, test_image=rel.as_posix(),
                                 original_score=scores[0], blur_150_score=scores[1], change=scores[1]-scores[0],
                                 original_inside_mask=float(maps[0][mask].mean()) if mask.any() else None,
                                 blurred_inside_mask=float(maps[1][mask].mean()) if mask.any() else None,
                                 original_outside_mask=float(maps[0][~mask].mean()) if (~mask).any() else None,
                                 blurred_outside_mask=float(maps[1][~mask].mean()) if (~mask).any() else None))
        provenance.append({"category": category, "reference_index": seed, "test_image": rel.as_posix(),
                           "inputs": {name: {"path": str(path.relative_to(dataset_root) if path.is_relative_to(dataset_root)
                                                         else path.relative_to(results_root)),
                                             "root": "dataset" if path.is_relative_to(dataset_root) else "results",
                                             "sha256": digest(path)} for name, path in input_files.items()}})

    # Complete validation before creating any output files.
    output_dir.mkdir(parents=True, exist_ok=False)
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        _render_error_case_panel(panels, output_dir)
    pd.DataFrame(measurements).to_csv(output_dir / "selected_error_case_measurements.csv", index=False)
    (output_dir / "selected_error_case_provenance.json").write_text(
        json.dumps({"upstream_commit": commit, "reconstruction_source_sha256": digest(utils),
                    "selected_cases_sha256": digest(cases_path), "cases": provenance}, indent=2) + "\n"
    )
    return output_dir


def _render_error_case_panel(panels, output_dir):
    """Render validated inputs, maintaining a common color scale within each row."""
    from matplotlib.colors import Normalize
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.1, 7.7))
    grid = fig.add_gridspec(len(panels), 6, width_ratios=[1, 1, 1, 1, 1, .075],
                            left=.015, right=.95, bottom=.055, top=.93, hspace=.42, wspace=.06)
    headers = ("Original input", "Blur 1.50 px", "Ground truth", "Original map", "Blurred map")
    for row, case in enumerate(panels):
        axes = [fig.add_subplot(grid[row, j]) for j in range(5)]
        for ax in axes:
            ax.set_axis_off()
        axes[0].imshow(case["original"], interpolation="none")
        axes[1].imshow(case["blurred"], interpolation="none")
        mask = case["mask"]
        axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        # The map is reconstructed at original resolution exactly as in evaluation.
        # All image/map panels retain aspect ratio; contours are in the same coordinates.
        for ax, values in zip(axes[3:], case["maps"]):
            im = ax.imshow(values, cmap="magma", norm=Normalize(0, case["vmax"]), interpolation="nearest")
            if mask.any() and not mask.all():
                ax.contour(mask, levels=[.5], colors=["cyan"], linewidths=.6)
        cax = fig.add_subplot(grid[row, 5])
        bar = fig.colorbar(im, cax=cax, ticks=[0, case["vmax"]])
        bar.ax.set_yticklabels(["0", f'{case["vmax"]:.2f}'])
        bar.ax.tick_params(labelsize=7, length=2)
        for ax, score in zip(axes[3:], case["scores"]):
            ax.text(.5, -.06, f"Score {score:.3f}", ha="center", va="top", transform=ax.transAxes)
        if row == 0:
            for ax, header in zip(axes, headers):
                ax.text(.5, 1.27, header, ha="center", va="bottom", transform=ax.transAxes)
        fig.text(.015, axes[0].get_subplotspec().get_position(fig).y1 + .014,
                 f'{case["category"].capitalize()}  {case["rel"]}  |  Ref. {case["seed"]}  |  '
                 f'Delta {case["scores"][1]-case["scores"][0]:+.3f}', weight="semibold")
    fig.text(.015, .011, "Image and mask source: MVTec AD (MVTec Software GmbH), CC BY-NC-SA 4.0; resized/blurred.", fontsize=6.6)
    fig.savefig(output_dir / "09_selected_blur_cases.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output_dir / "09_selected_blur_cases.png", dpi=220)
    plt.close(fig)



def generate_bottle_paired_scatter(output_dir):
    """Plot all 83 measured paired Bottle scores from the verified archive export."""
    import hashlib
    from sklearn.metrics import roc_auc_score
    output_dir = Path(output_dir).expanduser().resolve()
    for path in (ROOT / "reports", ROOT / "results", ROOT / "research_logs"):
        path = path.resolve()
        if output_dir == path or path in output_dir.parents or output_dir in path.parents:
            raise ValueError(f"Output overlaps protected inputs: {output_dir}")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source = ROOT / "reports/lighting_blur/bottle_paired_scores.csv"
    provenance = json.loads(source.with_name("bottle_paired_scores_provenance.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance["paired_scores_sha256"]:
        raise ValueError("Paired-score export hash mismatch.")
    d = pd.read_csv(source, float_precision="round_trip")
    assert len(d) == 83 and d.image.is_unique and d.label.value_counts().to_dict() == {1: 63, 0: 20}
    assert d.category.eq("bottle").all() and d.reference_index.eq(0).all()
    assert np.allclose(d.blur_150_score-d.original_score, d.change, rtol=0, atol=1e-12)
    base = pd.read_csv(ROOT / "reports/baseline_reproduction/per_run.csv").query("category == 'bottle' and seed == 0").iloc[0]
    blur = pd.read_csv(ROOT / "reports/lighting_blur/per_run.csv").query("category == 'bottle' and seed == 0 and condition == 'blur_150'").iloc[0]
    assert np.isclose(roc_auc_score(d.label,d.original_score),base.image_auroc,rtol=0,atol=1e-12)
    assert np.isclose(roc_auc_score(d.label,d.blur_150_score),blur.image_auroc,rtol=0,atol=1e-12)
    output_dir.mkdir(parents=True, exist_ok=False)
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42})
        fig, ax = plt.subplots(figsize=(6.1,5.1))
        fig.subplots_adjust(left=.16,right=.97,bottom=.22,top=.96)
        ax.plot([.1,.8],[.1,.8],ls="--",lw=1,color="#777777",label="No change",zorder=1)
        for label,name,color in [(0,"Normal (n=20)","#183c5b"),(1,"Anomalous (n=63)","#c83b40")]:
            x=d[d.label==label]
            ax.scatter(x.original_score,x.blur_150_score,s=27,alpha=.72,color=color,label=name,edgecolors="none",zorder=3)
        ax.set(xlim=(.1,.8),ylim=(.1,.8),xlabel="Original image anomaly score",ylabel="Blurred image anomaly score")
        ax.set_xticks(np.arange(.1,.81,.1));ax.set_yticks(np.arange(.1,.81,.1))
        ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.18)
        handles,labels=ax.get_legend_handles_labels()
        ax.legend([handles[i] for i in [1,2,0]],[labels[i] for i in [1,2,0]],loc="upper center",bbox_to_anchor=(.5,-.17),ncol=3,frameon=False,fontsize=8)
        fig.savefig(output_dir/"10_bottle_paired_scores.pdf",metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(output_dir/"10_bottle_paired_scores.png",dpi=220)
        plt.close(fig)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New directory for generated figures and tables")
    parser.add_argument("--study", choices=("all", "baseline", "robustness", "ablation", "calibration"), default="all")
    args = parser.parse_args()
    matplotlib.use("Agg")
    paths = generate_publication_assets(args.output, study=args.study)
    print("Generated figures and tables:", paths)


if __name__ == "__main__":
    main()
