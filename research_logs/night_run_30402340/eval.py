import os,sys,json
from pathlib import Path
d=Path(__file__).parent
sys.path.insert(0,str(Path.home()/'msc-vad-recovery/scripts'))
import analyze_improvement as a
p=Path('/work3/s183356/experiments/reference_augmentation_night_30402340')
groups={tuple(x) for x in json.loads((d/'plan.json').read_text())['pending_groups']}
tasks=[t for t in a.tasks() if (t[0],t[3]) in groups]
assert len(tasks)==456
idx=int(os.environ['LSB_JOBINDEX'])-1
for t in tasks[idx*10:(idx+1)*10]:
 print('EVALUATE',t,flush=True)
 a.evaluate_task(p,t)
