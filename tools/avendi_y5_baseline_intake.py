import hashlib,json,sys
from pathlib import Path
from zipfile import ZipFile
import torch
from experiments.avendi_y5.composition import panel,measure
from experiments.avendi_y4.models import Y4TypedCompositionModel,AvendiY4ByteModel,get_y4_byte_spec,parse_bytes_with_model
from tools.avendi_y4_frozen_bridge import load_model,historical_parity,make_families
from tools.avendi_y4_frozen_donor import run as donor_run

torch.set_num_threads(2)
p=Path(sys.argv[1])
h=hashlib.sha256()
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
out={'schema':'avendi-y5-baseline-intake-v1','source':{'file':p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()},'device':'cpu','seed':7,'execution':'bounded frozen inference; exploratory not confirmatory','executors':{},'donor':{},'reader':{}}
with ZipFile(p) as z, torch.inference_mode():
 for arm in ('endpoint_only','true_step','true_step_iit'):
  job=f'C-categorical_{arm}-s7'
  model=Y4TypedCompositionModel('categorical',128,seed=7)
  digest=load_model(z,job,model)
  receipt=json.loads(z.read(f'jobs/{job}/receipt.json'))
  record={'job_id':job,'manifest_sha256':digest,'owner_receipt_checkpoint_sha256':receipt['result']['train']['checkpoint_sha256'],'owner_receipt_completed_updates':receipt['completed_updates'],'panels':{}}
  for d in (3,4,5,6):
   m=measure(model,panel(32,5000+d if d<5 else 6000+d,d),'cpu')
   record['panels'][str(d)]={'correct':m['correct'],'total':m['total']}
  out['executors'][arm]=record
  if arm in ('true_step','true_step_iit'):
   out['donor'][arm]={'fixture_seed':98271,'pairs_per_depth':256,'patch_step':1,'depths':{str(d):donor_run(model,d) for d in (2,3,4)}}
 byte=AvendiY4ByteModel(get_y4_byte_spec('S3',cpu_canary=False),seed=7,use_aux=True)
 reader_digest=load_model(z,'A-pointer-s7',byte)
 rr={'job_id':'A-pointer-s7','manifest_sha256':reader_digest,'owner_receipt_checkpoint_sha256':json.loads(z.read('jobs/A-pointer-s7/receipt.json'))['result']['train']['checkpoint_sha256'],'historical_reference_parity':historical_parity(z,byte,7),'panels':{}}
 families=make_families(1)
 for name in ('native','brackets'):
  correct=total=0
  for fam in families:
   for cell in fam['cells']:
    c=dict(cell)
    if name=='brackets':
     body=' '.join(f'[{k}=>{v}]' for k,v in c['rows'])
     c['public_prompt']=f"ROWS: {body} || STEP={c['depth']} Q:{c['query']}"
    parsed=parse_bytes_with_model(byte,dict(c,gold=''))
    correct+=int(parsed['exact_parse']);total+=1
  rr['panels'][name]={'exact_parse':correct,'total':total}
 out['reader']=rr
Path('research/avendi/Y5_BASELINE_RECEIPT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
