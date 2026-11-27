"""Regenerate portfolio SVGs from the verified published metric files."""
from pathlib import Path
import csv,json,statistics,html,re
R=Path(__file__).resolve().parents[1]; A=R/'assets'; A.mkdir(exist_ok=True)
P=R/'reports/lighting_blur'
rows=list(csv.DictReader((P/'per_run.csv').open()))
ms=['image_auroc','image_ap','image_f1max','pixel_auroc','pixel_f1max','pixel_aupro']
cs=['brightness_085','brightness_070','blur_075','blur_150']; labels=['Brightness ×0.85','Brightness ×0.70','Blur 0.75 px','Blur 1.50 px']
cats=sorted({r['category'] for r in rows})
assert len(rows)==180 and len(cats)==15 and len({(r['category'],r['seed'],r['condition']) for r in rows})==180
base=json.loads((R/'reports/baseline_reproduction/summary.json').read_text())
b0={m:100*base['metrics'][m]['macro_mean'] for m in ms}
values={}
for c in cs:
 rr=[r for r in rows if r['condition']==c]
 assert len(rr)==45 and {(r['category'],r['seed']) for r in rr}=={(k,str(s)) for k in cats for s in range(3)}
 values[c]={m:100*statistics.mean(float(r[m]) for r in rr) for m in ms}
 for m in ms: assert abs(100*statistics.mean(float(r[m+'_original']) for r in rr)-b0[m])<1e-8
case=[]
with (P/'selected_blur_cases.csv').open() as f:
 for r in csv.DictReader(f):
  case.append(dict(category=r['category'],seed=int(r['reference_index']),image=r['test_image'],clean=float(r['original_score']),blur=float(r['blur_150_score'])))
assert len(case)==4
def svg(name,w,h,body):
 (A/name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" rx="16" fill="#101d2d"/><g font-family="Arial, Helvetica, sans-serif">{body}</g></svg>\n')
def t(x,y,s,size=18,color='#e8f0fa',weight=400,anchor='start'):
 return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(s))}</text>'
def rect(x,y,w,h,fill,rx=8):return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}"/>'
def line(x,y,x2,y2,c='#566d88',sw=1):return f'<line x1="{x}" y1="{y}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{sw}"/>'
b=t(36,38,'DTU MSc THESIS  /  COMPUTER VISION & APPLIED AI',15,'#7fcbd6',700)
b+=t(36,91,'Visual anomaly detection',39,weight=700)+t(36,135,'for industrial inspection',36,weight=700)+t(36,173,'Vision foundation models • One-shot learning • Robustness',17,'#b4c6dc')
for i,(v,l) in enumerate([(f"{b0['image_auroc']:.2f}%",'IMAGE AUROC'),(f"{b0['pixel_aupro']:.2f}%",'PIXEL AUPRO'),('15','MVTec AD CATEGORIES')]):
 x=36+i*205;b+=rect(x,204,192,87,'#1b3148')+t(x+13,242,v,28,'#7ee2c0',700)+t(x+13,269,l,11,'#b8cce1',700)
b+=line(678,24,678,306,'#2b415a')
for i,(l,sub) in enumerate([('01  NORMAL REFERENCE','DINOv2 patch embeddings'),('02  TEST IMAGE','Nearest-neighbor distance'),('03  ANOMALY MAP','Image score + localization')]):
 y=35+i*91;b+=rect(708,y,360,68,'#1b3148')+t(728,y+25,l,15,'#7fcbd6',700)+t(728,y+48,sub,16)
 if i<2:b+=line(888,y+68,888,y+89,'#7fcbd6',2)
b+=t(36,324,'Measured clean-baseline macro averages • one reference per run • research prototype',13,'#a5bad1')
svg('project_banner.svg',1104,346,b)
b=t(28,36,'How the inspection pipeline works',25,weight=700)+t(28,61,'Method schematic — frozen DINOv2, patch-level nearest-neighbor search',15,'#b4c6dc')
for y,first,second,third in [(93,'Normal reference','Rotated views + ViT','Reference patch bank'),(200,'Test image','Frozen DINOv2','Test patch embeddings')]:
 for x,label in [(28,first),(252,second),(476,third)]:b+=rect(x,y,192,64,'#1b3148')+t(x+96,y+37,label,15,anchor='middle',weight=700)
 for x in [220,444]:b+=line(x,y+32,x+31,y+32,'#7fcbd6',2)+f'<path d="M{x+25} {y+27} l7 5 -7 5" fill="none" stroke="#7fcbd6" stroke-width="2"/>'
b+=line(668,125,714,125,'#7fcbd6',2)+line(668,232,714,232,'#7fcbd6',2)+line(714,125,714,232,'#7fcbd6',2)+line(714,178,750,178,'#7fcbd6',2)
b+=rect(750,140,146,77,'#27445a')+t(823,168,'FAISS 1-NN',16,weight=700,anchor='middle')+t(823,191,'Cosine distance',13,'#b4c6dc',anchor='middle')
b+=line(896,178,920,178,'#7fcbd6',2)+rect(920,140,156,77,'#24483f')+t(998,167,'Anomaly map',16,weight=700,anchor='middle')+t(998,191,'Image score',14,'#bfe9db',anchor='middle')
b+=t(28,304,'One normal reference per run; rotation-only baseline; no encoder fine-tuning.',15,'#b4c6dc')
svg('method_overview.svg',1104,326,b)
b=t(28,38,'Robustness: change from clean baseline',25,weight=700)+t(28,63,'15 categories × 3 reference choices × 4 test-only perturbations',15,'#b4c6dc')
b+=rect(28,83,14,14,'#71c6d9',2)+t(48,95,'Image AUROC',14)+rect(184,83,14,14,'#ffb56e',2)+t(204,95,'Pixel AUPRO',14)
zero=828; scale=150
for tick in [-3,-2,-1,0,1]:
 x=zero+tick*scale;b+=line(x,119,x,331,'#39516b')+t(x,355,str(tick),13,'#b4c6dc',anchor='middle')
for i,(c,l) in enumerate(zip(cs,labels)):
 y=126+i*52;b+=t(28,y+20,l,16,weight=700)
 for j,(m,color) in enumerate([('image_auroc','#71c6d9'),('pixel_aupro','#ffb56e')]):
  val=values[c][m]-b0[m];xx=zero+min(val,0)*scale;ww=max(abs(val)*scale,1)
  b+=rect(xx,y+j*21,ww,14,color,2)+t(xx-6 if val<0 else xx+ww+6,y+j*21+12,f'{val:+.2f}',13,color,700,'end' if val<0 else 'start')
b+=t(548,382,'Percentage-point change (higher is better)',14,'#b4c6dc',anchor='middle')
svg('robustness_results.svg',1104,406,b)
b=t(28,38,'What stronger blur changes',25,weight=700)+t(28,63,'Selected extreme paired cases • measured anomaly scores, not probabilities',15,'#b4c6dc')
b+=rect(28,83,14,14,'#71c6d9',2)+t(48,95,'Original',14)+rect(138,83,14,14,'#ffb56e',2)+t(158,95,'Blur 1.50 px',14)
for i,r in enumerate(case):
 y=122+i*65;b+=t(28,y+17,r['category'].capitalize()+' / '+r['image'],16,weight=700)+t(28,y+37,'Reference '+str(r['seed']),12,'#b4c6dc')
 for j,(k,color) in enumerate([('clean','#71c6d9'),('blur','#ffb56e')]):
  v=r[k];w=v/.7*570;b+=rect(425,y+j*22,w,15,color,2)+t(433+w,y+j*22+13,f'{v:.3f}',14,color,700)
b+=t(28,400,'Selected examples illustrate failure modes; they do not estimate false-positive or false-negative rates.',14,'#b4c6dc')
svg('error_analysis.svg',1104,424,b)
