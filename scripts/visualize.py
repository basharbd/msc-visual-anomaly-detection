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
