from pathlib import Path
import hashlib,json,math,sys,csv
home=Path.home(); d=Path(__file__).parent
plan=json.loads((d/'plan.json').read_text())
root=Path('/work3/s183356/experiments')
p=root/'reference_augmentation_night_30402340'
croot=root/'threshold_calibration_night_30402340'
digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
cats='bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()
variants=['rotation_only','brightness','blur','combined']; conditions=['original','brightness_085','brightness_070','blur_075','blur_150']
if sys.argv[1]=='ablation':
 m=json.loads((home/'msc-vad-recovery/protocols/adino_mvtec_1shot_448_references.json').read_text())
 runner=digest(home/'msc-vad-recovery/scripts/proposed_improvement.py')
 for rec in plan['reused']:
  f=p/rec['path']
  assert digest(f/'run_complete.json')==rec['record'] and digest(f/'evaluation_metrics.json')==rec['evaluation']
 rows=[]
 for c in cats:
  for t in conditions:
   for s in range(3):
    group=[]
    for v in variants:
     f=p/v/t/f'{c}_1shot_seed{s}'; r=json.loads((f/'run_complete.json').read_text())
     assert (r['category'],r['seed'],r['variant'],r['condition'])==(c,s,v,t)
     assert r['status']=='INFERENCE_COMPLETED' and r['reference_sha256']==m['references'][c][str(s)]['sha256']
     assert r['upstream_commit']==m['upstream_commit'] and r['runner_sha256'] in {runner,'5c9eb15bcbb1dceb44755ef53d52e90def0c32d07f618b1e02f22c1b26d00d1f'}
     assert 'A100' in r['gpu']
     assert r['npy_maps']==r['test_images'] and not f.with_name(f.name+'.partial').exists()
     group.append((r['reference_sha256'],r['bank_patches']))
     rows.append(dict(path=str(f.relative_to(p)),sha256=digest(f/'run_complete.json')))
    assert len(set(group))==1
 assert len(rows)==900
else:
 base=home/'msc-vad-calibration'; mp=base/'protocols/threshold_calibration_manifest.json'
 m=json.loads(mp.read_text()); runner=digest(base/'scripts/threshold_calibration.py')
 rows=[]
 for c in cats:
  for s in range(3):
   f=croot/c/f'ref{s}.json'; r=json.loads(f.read_text())
   assert (r['category'],r['seed'])==(c,s) and r['reference']==m['categories'][c]['references'][str(s)]
   assert r['manifest_sha256']==digest(mp) and r['runner_sha256']==runner
   assert len(r['scores'])==40 and all(math.isfinite(x) for x in r['scores'].values())
   assert set(r['calibration_sha256'])==set(m['categories'][c]['calibration_paths'])
   assert r['rank']==39 and r['threshold']==sorted(r['scores'].values())[38] and 'A100' in r['gpu']
   rows.append(dict(path=str(f.relative_to(croot)),sha256=digest(f)))
 assert len(rows)==45
(d/(sys.argv[1]+'_verified.json')).write_text(json.dumps(rows,indent=2)+'\n')
print('VERIFIED',sys.argv[1],len(rows),flush=True)
