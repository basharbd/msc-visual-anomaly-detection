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
# The portfolio banner is a conceptual illustration, not an experimental plot.
# Its design deliberately omits performance values; measured metrics live in the README.
def banner():
 w,h=1440,570
 b=['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="570" viewBox="0 0 1440 570" role="img" aria-labelledby="title desc">',
    '<title id="title">Visual anomaly detection for industrial inspection</title>',
    '<desc id="desc">Conceptual schematic showing a normal reference and test image, frozen DINOv2, patch matching, and anomaly localization. Not measured results.</desc>',
    '<defs><linearGradient id="header" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#102b43"/><stop offset="1" stop-color="#215b79"/></linearGradient><linearGradient id="bottle" x1="0" x2="1"><stop stop-color="#809aa7"/><stop offset=".45" stop-color="#e8f4f8"/><stop offset="1" stop-color="#8ba7b5"/></linearGradient><radialGradient id="hot"><stop stop-color="#fff5a8"/><stop offset=".2" stop-color="#f8be45"/><stop offset=".44" stop-color="#e75a55"/><stop offset=".72" stop-color="#934a9b"/><stop offset="1" stop-color="#423174" stop-opacity="0"/></radialGradient></defs>',
    '<rect width="1440" height="570" rx="22" fill="#f3f7fa"/>',
    '<path d="M22 0H1418Q1440 0 1440 22V190Q1170 154 930 178T0 183V22Q0 0 22 0Z" fill="url(#header)"/>',
    '<g font-family="Arial, Helvetica, sans-serif">']
 b += [t(44,40,'DTU COMPUTE  /  MSc THESIS',18,'#8bd2d5',700),t(44,98,'Visual anomaly detection',53,'#ffffff',700),t(44,151,'for industrial inspection',43,'#e1f3fc',700),
       t(1392,55,'VISION FOUNDATION MODELS',17,'#b7dbe9',700,'end'),t(1392,85,'One-shot learning',25,'#ffffff',700,'end'),
       t(1392,119,'Robustness & reference augmentation',18,'#c7e4f0',400,'end')]
 def add(x):b.append(x)
 def bottle(cx,cy,defective=False,heat=False):
  # All shapes are illustrative; no experimental image is synthesized or implied.
  add(f'<g transform="translate({cx},{cy})"><rect x="-17" y="-62" width="34" height="18" rx="4" fill="{"#603c95" if heat else "#9cb4bf"}"/><path d="M-19 -45H19L29 -30Q35 -24 35 -15V45Q35 55 25 55H-25Q-35 55 -35 45V-15Q-35 -24 -29 -30Z" fill="{"#43377b" if heat else "url(#bottle)"}" stroke="{"#7963b5" if heat else "#a7bdc8"}" stroke-width="2"/>')
  if defective: add('<path d="M-8 9L4 -1 1 11 14 2" fill="none" stroke="#ad5262" stroke-width="3" stroke-linecap="round"/>')
  if heat:add('<ellipse cx="5" cy="8" rx="28" ry="33" transform="rotate(32 5 8)" fill="url(#hot)"/>')
  add('</g>')
 xs=[38,390,742,1094]
 labels=['Normal reference &amp; test','Frozen DINOv2','Patch matching','Anomaly localization']
 sub=['Single normal reference','Visual feature extraction','Nearest normal features','Image score and pixel map']
 for i,x in enumerate(xs):
  add(f'<rect x="{x}" y="218" width="308" height="260" rx="16" fill="white" stroke="#cfdee7" stroke-width="2"/>')
  add(t(x+154,444,html.unescape(labels[i]),22,'#173653',700,'middle'))
  add(t(x+154,465,sub[i],15,'#526a7d',400,'middle'))
  if i<3:
   add(f'<path d="M{x+318} 330h22m-8 -8 8 8 -8 8" fill="none" stroke="#007e80" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>')
 # Image inputs
 for cx,label,defect in [(120,'Normal',False),(264,'Test',True)]:
  add(rect(cx-61,239,122,170,'#18394f',12))
  bottle(cx,319,defective=defect)
  add(t(cx,394,label,16,'#e1f3fc',700,'middle'))
 # Frozen feature extractor: stacked translucent feature planes
 for j in range(3):
  x=454+j*42;y=253+j*15
  add(f'<path d="M{x} {y+21}l76 -26v100l-76 26Z" fill="#8bbecb" fill-opacity=".72" stroke="#3b7c96" stroke-width="2"/>')
  for k in range(1,4):
   add(f'<path d="M{x+k*19} {y+21-k*6.5}v100 M{x} {y+21+k*25}l76 -26" stroke="#eff9fb" stroke-opacity=".62" stroke-width="1.5"/>')
 # Matching illustration: patch grids and comparison symbol
 for j,gx in enumerate([766,938]):
  for r in range(5):
   for c in range(5):
    color=['#c0dce2','#76afba','#417e91','#21566c'][(r*3+c*5+j)%4]
    if j and r==2 and c in (2,3):color='#e5a060'
    add(rect(gx+c*19,275+r*19,17,17,color,2))
 add(t(894,333,'↔',31,'#007e80',700,'middle'))
 # Localization
 add(rect(1151,239,194,170,'#171c3a',12))
 bottle(1248,323,heat=True)
 add(t(1248,397,'Illustrative heatmap',15,'#e1f3fc',400,'middle'))
 add(line(44,506,1396,506,'#c7d7e1',1.5))
 add(t(44,544,'INDUSTRIAL INSPECTION   •   FROZEN VISUAL FEATURES   •   REFERENCE AUGMENTATION',18,'#215b79',700))
 add(t(1396,544,'Conceptual illustration · not measured output',15,'#526a7d',400,'end'))
 add('</g></svg>')
 (A/'project_banner.svg').write_text(''.join(b)+'\n')
banner()
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
