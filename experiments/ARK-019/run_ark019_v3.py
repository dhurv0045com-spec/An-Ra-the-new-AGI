from __future__ import annotations

import argparse, copy, hashlib, itertools, json, math, os, random, sys, time, traceback, zipfile
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent; REPO=HERE.parents[1]
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent/'ARK-018'))
import ark019_v3_core as C
from ark018_v3_common import Ark018GPT, EXPECTED_SCIENCE_SHA, EXPECTED_BIRTH_SHA, CONTEXT, autocast_ctx, load_tokenizer, memmap_u16, model_state_hash, make_scaler, eval_buffer
from ark018_v3_binding_fast import select_binding_tokens

ARK018_ROOT=Path('/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1')
OUT=Path('/content/drive/MyDrive/genisis-arkenstone/ARK019_GUARDIAN_V3')
EXPECTED_HORIZON=8000

def hbytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def hjson(x:Any)->str:return hbytes(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode())
def hfile(p:Path)->str:
    h=hashlib.sha256(); f=p.open('rb')
    with f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def savej(p:Path,x:Mapping[str,Any])->None:
    p.parent.mkdir(parents=True,exist_ok=True); y=dict(x); y.pop('receipt_sha256',None); y['receipt_sha256']=hjson(y)
    q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(y,indent=2,sort_keys=True,default=str)+'\n'); q.replace(p)

def setup():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32=False
    try: torch.use_deterministic_algorithms(True,warn_only=False)
    except Exception: torch.use_deterministic_algorithms(True,warn_only=True)
    if torch.cuda.is_available():
        try: torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
        except Exception: pass

def device():
    if not torch.cuda.is_available():raise RuntimeError('ARK-019 V3 requires Colab CUDA/T4')
    return torch.device('cuda')
def opt_for(m,lr):return torch.optim.AdamW(m.parameters(),lr=lr,betas=(.9,.95),eps=1e-8,weight_decay=.1)
def opt_to(o,d):
    for s in o.state.values():
        for k,v in list(s.items()):
            if torch.is_tensor(v):s[k]=v.to(d)

def state_hash(st:Mapping[str,torch.Tensor])->str:
    h=hashlib.sha256()
    for n,t in sorted(st.items()):h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def snap(m,o,s):return {'model':cpu_state(m),'optimizer':copy.deepcopy(o.state_dict()),'scaler':copy.deepcopy(s.state_dict()),'cpu_rng':torch.get_rng_state().cpu(),'cuda_rng':[x.cpu() for x in torch.cuda.get_rng_state_all()]}
def restore(x,d,lr=C.HIGH_LR):
    m=Ark018GPT().to(d);m.load_state_dict(x['model']);o=opt_for(m,lr);o.load_state_dict(x['optimizer']);opt_to(o,d)
    for g in o.param_groups:g['lr']=lr
    s=make_scaler(d);s.load_state_dict(x.get('scaler',{}));torch.set_rng_state(x['cpu_rng']);torch.cuda.set_rng_state_all(x['cuda_rng']);return m,o,s

def load_substrate():
    pd=ARK018_ROOT/'prepared'; rp=pd/'ARK-018_PREPARED_RECEIPT.json'
    if not rp.exists():raise FileNotFoundError(rp)
    r=json.loads(rp.read_text())
    if r.get('science_sha256')!=EXPECTED_SCIENCE_SHA or r.get('birth_sha256')!=EXPECTED_BIRTH_SHA or int(r.get('horizon_updates',-1))!=EXPECTED_HORIZON:raise RuntimeError('ARK-018 prepared identity mismatch')
    if hfile(pd/'tokenizer.json')!=r.get('tokenizer_sha256'):raise RuntimeError('tokenizer hash mismatch')
    tok=load_tokenizer(pd/'tokenizer.json'); bufs={k:memmap_u16(pd/f'{k}.bin') for k in ('train','control','sealed')}; counts=np.load(pd/'token_counts.npy')
    for k in bufs:
        ex=r.get('cache_sha256',{}).get(f'{k}.bin')
        if ex and hfile(pd/f'{k}.bin')!=ex:raise RuntimeError(f'{k} cache hash mismatch')
    return r,tok,bufs,counts

def ckpt_path(seed):return ARK018_ROOT/'checkpoints'/f'seed_{seed}'/'SCIENCE_ONLY.pt'
def ckpt_receipt(seed,prep):
    p=ckpt_path(seed)
    if not p.exists():raise FileNotFoundError(p)
    x=torch.load(p,map_location='cpu',weights_only=False)
    if int(x.get('step',-1))!=int(prep['horizon_updates']) or x.get('arm')!='SCIENCE_ONLY' or x.get('science_sha256')!=EXPECTED_SCIENCE_SHA or x.get('tokenizer_sha256')!=prep['tokenizer_sha256']:raise RuntimeError(f'seed {seed} checkpoint identity mismatch')
    return {'seed':seed,'path':str(p),'sha256':hfile(p),'step':int(x['step']),'model_sha256':state_hash(x['model'])}

def factsets(keys,vals,seed):
    x=[tuple(zip(kt,vt)) for kt in itertools.combinations(keys,3) for vt in itertools.permutations(vals,3)];random.Random(seed).shuffle(x);return x[:400],x[400:450],x[450:500]
def sem(fs):return [(f,q,a) for f in fs for q,a in f]
def tpls(tok):return {'A':{'p':tok.encode('Facts: ').ids,'m':tok.encode(' means ').ids,'s':tok.encode('; ').ids,'q':tok.encode('Query: ').ids,'t':tok.encode(' means').ids},'B':{'p':tok.encode('Map: ').ids,'m':tok.encode(' -> ').ids,'s':tok.encode(' | ').ids,'q':tok.encode('Requested ').ids,'t':tok.encode(' =>').ids}}
def render(t,f,q,order):
    ids=list(t['p'])
    for i in order:
        k,v=f[i];ids+=[int(k)]+list(t['m'])+[int(v)]+list(t['s'])
    ids+=list(t['q'])+[int(q)]+list(t['t']);return ids[-(CONTEXT-1):]
def rows(t,s,idx,mode,seed,step):
    out=[]
    for i in idx:
        f,q,a=s[int(i)]
        order=(0,1,2) if mode=='canonical' else (2,1,0) if mode=='reversed' else C.query_order_perm(f,q) if mode=='query_order' else C.augmented_perm(seed,step,int(i)) if mode=='augmented' else C.nonidentity_perm(seed,step,int(i))
        out.append((render(t,f,q,order),int(a)))
    return out
def blogits(m,rs,d):
    w=max(len(p) for p,_ in rs);x=torch.zeros((len(rs),w),dtype=torch.long,device=d);a=torch.tensor([a for _,a in rs],dtype=torch.long,device=d)
    for i,(p,_) in enumerate(rs):x[i,-len(p):]=torch.tensor(p,dtype=torch.long,device=d)
    return m(x)[:,-1,:],a
def bloss(m,rs,d):
    z,a=blogits(m,rs,d);return F.cross_entropy(z.float(),a,reduction='none')
@torch.no_grad()
def bmode(m,t,s,d,mode):
    hit=n=0
    for st in range(0,len(s),64):
        idx=list(range(st,min(st+64,len(s))));z,a=blogits(m,rows(t,s,idx,mode,0,0),d);hit+=int((z.argmax(-1)==a).sum());n+=len(a)
    return hit/max(1,n)
@torch.no_grad()
def bmetrics(m,t,s,d):
    c=bmode(m,t,s,d,'canonical');o=bmode(m,t,s,d,'reversed');q=bmode(m,t,s,d,'query_order');return {'canonical':c,'order_only':o,'query_order':q,'qualified':c>=.90 and o>=.85 and q>=.85,'healthy_margin':c>=.95 and o>=.90 and q>=.90}

def real_batch(buf,n,seed,step,d):
    xs=[];ys=[];starts=[];need=CONTEXT+1
    for slot in range(n):
        z=int.from_bytes(hashlib.sha256(f'ark019-real:{seed}:{step}:{slot}'.encode()).digest()[:8],'big')%(len(buf)-need+1);r=np.asarray(buf[z:z+need],dtype=np.int64);xs.append(r[:-1]);ys.append(r[1:]);starts.append(z)
    return torch.tensor(np.stack(xs),dtype=torch.long,device=d),torch.tensor(np.stack(ys),dtype=torch.long,device=d),starts
def pnames(m):
    a={'tok.weight','blocks.0.attn.qkv.weight','blocks.4.mlp.2.weight','blocks.9.attn.qkv.weight','ln_f.weight'};return [n for n,_ in m.named_parameters() if n in a]
def psnap(m,names):
    p=dict(m.named_parameters());return {n:p[n].detach().float().clone() for n in names}
def pdelta(x,m):
    p=dict(m.named_parameters());return math.sqrt(sum(float(((p[n].detach().float()-v)**2).sum()) for n,v in x.items()))
def backup(m):return [p.detach().clone() for p in m.parameters()]
def fulldelta(b,m):return math.sqrt(sum(float(((p.detach().float()-x.float())**2).sum()) for x,p in zip(b,m.parameters())))
def cap_project(b,m,raw,cap):
    if raw<=cap or raw<=0:return raw
    s=cap/raw
    with torch.no_grad():
        for x,p in zip(b,m.parameters()):p.copy_(x+(p-x)*s)
    return cap
def displacement(m,parent):return math.sqrt(sum(float(((v.detach().float().cpu()-parent[k].float())**2).sum()) for k,v in m.state_dict().items()))

def mixed(m,o,sc,train,tA,sA,tB,sB,stream,step,replay,cap,d,names,full=False):
    dorep=replay==32 or (replay==64 and step%2==0);nr=C.REAL_SLOTS-int(dorep);x,y,starts=real_batch(train,nr,stream,step,d);bi=C.deterministic_indices(stream,step,C.SKILL_B_SLOTS,len(sB),'skill-b');br=rows(tB,sB,bi,'augmented',stream,step)
    o.zero_grad(set_to_none=True)
    with autocast_ctx(d):
        z=m(x);tl=F.cross_entropy(z.float().reshape(-1,z.size(-1)),y.reshape(-1),reduction='none').view(nr,-1).mean(1);pieces=[tl,bloss(m,br,d)];ri=None
        if dorep:
            ri=C.deterministic_indices(stream,step,1,len(sA),f'replay-{replay}')[0];pieces.append(bloss(m,rows(tA,sA,[ri],'nonidentity',stream+replay,step),d))
        loss=torch.cat(pieces).sum()/C.BATCH_SLOTS
    sc.scale(loss).backward();sc.unscale_(o);gn=float(torch.nn.utils.clip_grad_norm_(m.parameters(),1.0));pb=psnap(m,names);fb=backup(m) if cap is not None or full else None;sc.step(o);sc.update();pd=pdelta(pb,m);raw=ap=None;capped=False
    if fb is not None:
        raw=fulldelta(fb,m);ap=raw
        if cap is not None and raw>cap:ap=cap_project(fb,m,raw,cap);capped=True
    return {'loss':float(loss.detach()),'grad':gn,'projected_delta':pd,'raw_full_delta':raw,'applied_full_delta':ap,'capped':capped,'replay':dorep,'replay_level':replay if dorep else 0,'replay_index':ri,'real_slots':nr,'real_starts_sha256':hjson(starts)}

def parent_dir(seed):return OUT/'parents'/f'seed_{seed}'
def acquire_parent(seed,prep,tok,tA,fa,d):
    pd=parent_dir(seed);pd.mkdir(parents=True,exist_ok=True);rp=pd/'PARENT_RESULT.json';sp=pd/'PARENT_STATE.pt'
    if rp.exists() and sp.exists():
        r=json.loads(rp.read_text())
        if r.get('status')=='QUALIFIED' and r.get('source_checkpoint_sha256')==hfile(ckpt_path(seed)):return r
        raise RuntimeError(f'incompatible parent cache {seed}')
    tr,co,se=map(sem,fa);base=torch.load(ckpt_path(seed),map_location='cpu',weights_only=False);m=Ark018GPT().to(d);m.load_state_dict(base['model']);o=opt_for(m,C.HIGH_LR);sc=make_scaler(d);traj=[];streak=0;qstep=None
    for step in range(1,1501):
        idx=C.deterministic_indices(seed,step,64,len(tr),'skill-a-acq');o.zero_grad(set_to_none=True)
        with autocast_ctx(d):loss=bloss(m,rows(tA,tr,idx,'augmented',seed,step),d).mean()
        sc.scale(loss).backward();sc.unscale_(o);torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);sc.step(o);sc.update()
        if step%100==0:
            mm=bmetrics(m,tA,co,d);traj.append({'step':step,'loss':float(loss.detach()),**mm});streak=streak+1 if mm['qualified'] else 0;print('PARENT',seed,step,mm,flush=True)
            if streak>=3:qstep=step;break
    if qstep is None:savej(rp,{'status':'FAILED_TO_QUALIFY','seed':seed,'trajectory':traj});return json.loads(rp.read_text())
    sm=bmetrics(m,tA,se,d)
    if not sm['qualified']:savej(rp,{'status':'SEALED_NOT_QUALIFIED','seed':seed,'sealed':sm,'trajectory':traj});return json.loads(rp.read_text())
    torch.save(snap(m,o,sc),sp);r={'schema':'arkenstone-ark019-v3-parent/v1','status':'QUALIFIED','seed':seed,'qualification_confirmation_step':qstep,'sealed':sm,'trajectory':traj,'source_checkpoint_sha256':hfile(ckpt_path(seed)),'parent_model_sha256':model_state_hash(m),'parent_state_sha256':hfile(sp),'tokenizer_sha256':prep['tokenizer_sha256']};savej(rp,r);del m,o;torch.cuda.empty_cache();return r

def load_parent(seed):return torch.load(parent_dir(seed)/'PARENT_STATE.pt',map_location='cpu',weights_only=False)
def science(m,bufs,d):return {'control':eval_buffer(m,bufs['control'],d,61901,sequences=24),'sealed':eval_buffer(m,bufs['sealed'],d,61902,sequences=24)}

def capcal(parent_seed,bseed,ps,train,tA,sA,tB,sB,d):
    p=OUT/'matched_sets'/f'p{parent_seed}_b{bseed}'/'CAP_CALIBRATION.json'
    if p.exists():return json.loads(p.read_text())
    m,o,sc=restore(ps,d,C.LOW_LR);names=pnames(m);ds=[]
    for step in range(1,C.CAP_SHADOW_STEPS+1):ds.append(float(mixed(m,o,sc,train,tA,sA,tB,sB,bseed,step,0,None,d,names,True)['applied_full_delta']))
    med=float(np.median(ds));r={'schema':'arkenstone-ark019-v3-capcal/v1','parent_seed':parent_seed,'skill_b_seed':bseed,'shadow_lr':C.LOW_LR,'steps':C.CAP_SHADOW_STEPS,'full_delta_norms':ds,'median_low_delta':med,'cap16x':16*med,'parent_model_sha256':state_hash(ps['model'])};savej(p,r);del m,o;torch.cuda.empty_cache();return r

def save_ckpt(p,m,o,sc,step,ctrl,cnt,ps,bs,arm,cap):
    x={'schema':'arkenstone-ark019-v3-ckpt/v1','parent_seed':ps,'skill_b_seed':bs,'arm':arm,'step':step,'model':cpu_state(m),'optimizer':o.state_dict(),'scaler':sc.state_dict(),'cpu_rng':torch.get_rng_state().cpu(),'cuda_rng':[x.cpu() for x in torch.cuda.get_rng_state_all()],'controller':copy.deepcopy(ctrl),'counters':copy.deepcopy(cnt),'cap16x':cap};q=p.with_suffix('.tmp');torch.save(x,q);q.replace(p)
def armdir(ps,bs,arm):return OUT/'matched_sets'/f'p{ps}_b{bs}'/arm

def run_arm(ps,bs,arm,parent,cap16,bufs,tA,sA,tB,sB,d,deadline):
    od=armdir(ps,bs,arm);od.mkdir(parents=True,exist_ok=True);rp=od/'RESULT.json';cp=od/'CHECKPOINT.pt'
    if rp.exists():
        r=json.loads(rp.read_text())
        if r.get('status')=='COMPLETE' and int(r.get('step',-1))==C.HORIZON:return r
        raise RuntimeError(f'incompatible arm result {rp}')
    m,o,sc=restore(parent,d,C.HIGH_LR);ctrl=C.initial_controller(arm);cnt={'replay_slots':0,'real_slots':0,'skill_b_slots':0,'capped_steps':0,'protection_updates':0,'projected_path':0.0,'displaced_real_slots':0};traj=[];start=0
    if cp.exists():
        x=torch.load(cp,map_location='cpu',weights_only=False)
        if x.get('parent_seed')!=ps or x.get('skill_b_seed')!=bs or x.get('arm')!=arm or abs(float(x.get('cap16x'))-float(cap16))>1e-12:raise RuntimeError('resume identity mismatch')
        m.load_state_dict(x['model']);o.load_state_dict(x['optimizer']);opt_to(o,d);sc.load_state_dict(x['scaler']);torch.set_rng_state(x['cpu_rng']);torch.cuda.set_rng_state_all(x['cuda_rng']);ctrl=x['controller'];cnt=x['counters'];start=int(x['step']);pp=od/'PARTIAL.json';traj=json.loads(pp.read_text()).get('trajectory',[]) if pp.exists() else []
    pe=Ark018GPT().to(d);pe.load_state_dict(parent['model']);baseline=science(pe,bufs,d);del pe;torch.cuda.empty_cache();names=pnames(m)
    if start==0:traj.append({'step':0,'a_sealed':bmetrics(m,tA,sA['sealed'],d),'b_sealed':bmetrics(m,tB,sB['sealed'],d),'science':baseline,'full_displacement':0.0,'controller':copy.deepcopy(ctrl)})
    for step in range(start+1,C.HORIZON+1):
        if time.monotonic()>=deadline:raise RuntimeError(f'wall reached {ps}/{bs}/{arm}/{step}')
        replay,cap=C.treatment(arm,ctrl,cap16);cnt['protection_updates']+=int(bool(replay or cap));rec=mixed(m,o,sc,bufs['train'],tA,sA['train'],tB,sB['train'],bs,step,replay,cap,d,names);cnt['replay_slots']+=int(rec['replay']);cnt['displaced_real_slots']+=int(rec['replay']);cnt['real_slots']+=int(rec['real_slots']);cnt['skill_b_slots']+=C.SKILL_B_SLOTS;cnt['capped_steps']+=int(rec['capped']);cnt['projected_path']+=float(rec['projected_delta'])
        if step%C.EVAL_EVERY==0:
            ac=bmetrics(m,tA,sA['control'],d);bc=bmetrics(m,tB,sB['control'],d);prev=ctrl.get('skill_b_confirmation_step');C.update_controller(arm,ctrl,step,ac,bc)
            if prev is None and ctrl.get('skill_b_confirmation_step') is not None:ctrl['skill_b_confirm_counters']=copy.deepcopy(cnt)
            row={'step':step,'loss':rec['loss'],'a_control':ac,'a_sealed':bmetrics(m,tA,sA['sealed'],d),'b_control':bc,'b_sealed':bmetrics(m,tB,sB['sealed'],d),'science':science(m,bufs,d),'controller':copy.deepcopy(ctrl),'counters':copy.deepcopy(cnt),'full_displacement':displacement(m,parent['model']),'last_projected_delta':rec['projected_delta'],'last_raw_full_delta':rec['raw_full_delta'],'last_applied_full_delta':rec['applied_full_delta']};traj.append(row);savej(od/'PARTIAL.json',{'schema':'arkenstone-ark019-v3-partial/v1','status':'PARTIAL' if step<C.HORIZON else 'COMPLETE','parent_seed':ps,'skill_b_seed':bs,'arm':arm,'step':step,'trajectory':traj,'controller':ctrl,'counters':cnt});print(f'[{ps}/{bs}/{arm}] {step}/{C.HORIZON} A={ac["qualified"]} B={bc["qualified"]} state={ctrl["state"]}',flush=True)
        if step%C.CHECKPOINT_EVERY==0 or step==C.HORIZON:save_ckpt(cp,m,o,sc,step,ctrl,cnt,ps,bs,arm,cap16)
    f=traj[-1];bn=float(baseline['sealed']['nll']);fn=float(f['science']['sealed']['nll']);r={'schema':'arkenstone-ark019-v3-arm/v1','status':'COMPLETE','step':C.HORIZON,'parent_seed':ps,'skill_b_seed':bs,'arm':arm,'cap16x':cap16,'trajectory':traj,'controller':ctrl,'counters':cnt,'skill_b_qualification_step':ctrl.get('skill_b_qualification_step'),'skill_b_confirmation_step':ctrl.get('skill_b_confirmation_step'),'science_sealed_nll_relative_change_from_parent':(fn-bn)/max(bn,1e-12),'final_a_sealed':f['a_sealed'],'final_b_sealed':f['b_sealed'],'final_model_sha256':model_state_hash(m)};savej(rp,r);del m,o;torch.cuda.empty_cache();return r

def smoke(parent,bufs,tA,sA,tB,sB,d):
    def go(m,o,sc,start,n):
        names=pnames(m);tr=[]
        for st in range(start,start+n):
            r=mixed(m,o,sc,bufs['train'],tA,sA,tB,sB,919001,st,64,None,d,names);tr.append((st,r['replay'],round(r['loss'],12)))
        return tr
    a,b,c=restore(parent,d);go(a,b,c,1,3);ss=snap(a,b,c);t1=go(a,b,c,4,10);h1=model_state_hash(a);x,y,z=restore(ss,d);t2=go(x,y,z,4,10);h2=model_state_hash(x);ok=h1==h2 and t1==t2;r={'schema':'arkenstone-ark019-v3-smoke/v1','status':'PASS' if ok else 'FAIL','hash_uninterrupted':h1,'hash_resumed':h2,'telemetry_identical':t1==t2}
    if not ok:raise RuntimeError('exact-resume smoke failed')
    del a,b,x,y;torch.cuda.empty_cache();return r

def calibrate(parent,bufs,tA,sA,tB,sB,d):
    m,o,sc=restore(parent,d);names=pnames(m)
    for st in (1,2):mixed(m,o,sc,bufs['train'],tA,sA,tB,sB,929001,st,64,None,d,names)
    torch.cuda.synchronize();t=time.monotonic();n=5
    for st in range(3,3+n):mixed(m,o,sc,bufs['train'],tA,sA,tB,sB,929001,st,64,None,d,names)
    torch.cuda.synchronize();up=(time.monotonic()-t)/n;torch.cuda.synchronize();t=time.monotonic();bmetrics(m,tA,sA[:150],d);bmetrics(m,tB,sB[:150],d);torch.cuda.synchronize();ev=time.monotonic()-t
    updates=len(C.PRETRAIN_SEEDS)*len(C.SKILL_B_SEEDS)*len(C.ARMS)*C.HORIZON;events=len(C.PRETRAIN_SEEDS)*len(C.SKILL_B_SEEDS)*len(C.ARMS)*(C.HORIZON//C.EVAL_EVERY);proj=(updates*up+events*ev*3+(2*600+4*C.CAP_SHADOW_STEPS*1.5+20)*up+C.PACKAGING_RESERVE_MINUTES*60)*C.RUNTIME_SAFETY_FACTOR;r={'schema':'arkenstone-ark019-v3-runtime/v1','update_seconds':up,'small_eval_seconds':ev,'projected_updates':updates,'projected_eval_events':events,'projected_total_seconds':proj,'wall_seconds':C.WALL_MINUTES*60,'safety_factor':C.RUNTIME_SAFETY_FACTOR,'fits':proj<=C.WALL_MINUTES*60};del m,o;torch.cuda.empty_cache();return r

def package():
    man={str(p.relative_to(OUT)):hfile(p) for p in sorted(OUT.rglob('*.json')) if p.name!='ZIP_MANIFEST.json'};savej(OUT/'ZIP_MANIFEST.json',{'schema':'arkenstone-ark019-v3-manifest/v1','members':man});z=OUT/'ARKENSTONE_ARK019_V3_GUARDIAN_RESULTS.zip'
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as f:
        for p in sorted(OUT.rglob('*.json')):f.write(p,str(p.relative_to(OUT)))
    (OUT/(z.name+'.sha256')).write_text(hfile(z)+'  '+z.name+'\n');return z

def build_world(tok,counts):
    ids=select_binding_tokens(tok,counts)
    if len(ids)<24 or len(set(ids[:24]))!=24:raise RuntimeError('need 24 distinct binding tokens')
    fa=factsets(ids[:6],ids[6:12],424218);fb=factsets(ids[12:18],ids[18:24],424219);tt=tpls(tok);return ids[:24],tt,{'train':sem(fa[0]),'control':sem(fa[1]),'sealed':sem(fa[2])},{'train':sem(fb[0]),'control':sem(fb[1]),'sealed':sem(fb[2])},fa

def run_all():
    setup();d=device();OUT.mkdir(parents=True,exist_ok=True);started=time.monotonic();deadline=started+(C.WALL_MINUTES-C.PACKAGING_RESERVE_MINUTES)*60;prep,tok,bufs,counts=load_substrate();sources=[ckpt_receipt(s,prep) for s in C.PRETRAIN_SEEDS];ids,tt,sA,sB,fa=build_world(tok,counts);entry={'schema':'arkenstone-ark019-v3-entry/v1','r2_bundle_sha256':C.R2_BUNDLE_SHA256,'r2_verdict':C.R2_VERDICT,'prepared_receipt_sha256':hfile(ARK018_ROOT/'prepared/ARK-018_PREPARED_RECEIPT.json'),'tokenizer_sha256':prep['tokenizer_sha256'],'token_counts_sha256':hfile(ARK018_ROOT/'prepared/token_counts.npy'),'science_sha256':EXPECTED_SCIENCE_SHA,'sources':sources,'selected_token_ids':ids,'skill_a_ids':ids[:12],'skill_b_ids':ids[12:],'overlap':sorted(set(ids[:12])&set(ids[12:])),'arms':list(C.ARMS),'horizon':C.HORIZON}
    if entry['overlap']:raise RuntimeError('A/B token overlap')
    savej(OUT/'ARK-019_V3_ENTRY_RECEIPT.json',entry);parents={}
    for s in C.PRETRAIN_SEEDS:
        r=acquire_parent(s,prep,tok,tt['A'],fa,d);parents[str(s)]=r
        if r.get('status')!='QUALIFIED':raise RuntimeError(f'parent {s} not qualified')
    p0=load_parent(C.PRETRAIN_SEEDS[0]);sm=smoke(p0,bufs,tt['A'],sA['train'],tt['B'],sB['train'],d);savej(OUT/'ARK-019_V3_SMOKE.json',sm);cal=calibrate(p0,bufs,tt['A'],sA['train'],tt['B'],sB['train'],d);savej(OUT/'ARK-019_V3_RUNTIME_CALIBRATION.json',cal)
    if not cal['fits']:raise RuntimeError(f'full R3 does not fit 175-minute wall: {cal}')
    ars={};caps={}
    for ps in C.PRETRAIN_SEEDS:
        parent=load_parent(ps)
        for bs in C.SKILL_B_SEEDS:
            cr=capcal(ps,bs,parent,bufs['train'],tt['A'],sA['train'],tt['B'],sB['train'],d);caps[f'p{ps}_b{bs}']=cr
            for arm in C.ARMS:ars[f'p{ps}_b{bs}_{arm}']=run_arm(ps,bs,arm,parent,float(cr['cap16x']),bufs,tt['A'],sA,tt['B'],sB,d,deadline)
    dec=C.decide(ars);savej(OUT/'CAPABILITY_STATE_TRACE.json',{'schema':'arkenstone-ark019-v3-traces/v1','traces':{k:v['trajectory'] for k,v in ars.items()}});savej(OUT/'GUARDIAN_POLICY_SPEC.json',{'schema':'arkenstone-ark019-v3-policy/v1','status':'CANDIDATE' if dec.get('authorized') else 'NOT_PROMOTED','states':['PLASTIC','SPARSE64','REPLAY32','EMERGENCY_CAP16X','CONSOLIDATE'],'control':'SKILL_A CONTROL only','sealed':'measurement only','warning':'ORDER_ONLY or QUERY_ORDER <0.90 twice','formal_failure':'canonical<0.90 or order/query<0.85','healthy_exit':'3 evals canonical>=.95 and order/query>=.90','decision':dec});res={'schema':'arkenstone-ark019-v3-result/v1','status':'COMPLETE','decision':dec,'parents':parents,'cap_calibrations':caps,'wall_seconds':time.monotonic()-started,'claim_boundary':'real-text proxy Guardian only'};savej(OUT/'ARK-019_V3_RESULT.json',res);z=package();res['bundle_path']=str(z);res['bundle_sha256']=hfile(z);return res

def smoke_mode():
    setup();d=device();OUT.mkdir(parents=True,exist_ok=True);prep,tok,bufs,counts=load_substrate();[ckpt_receipt(s,prep) for s in C.PRETRAIN_SEEDS];ids,tt,sA,sB,fa=build_world(tok,counts);p=acquire_parent(C.PRETRAIN_SEEDS[0],prep,tok,tt['A'],fa,d)
    if p.get('status')!='QUALIFIED':raise RuntimeError('smoke parent failed')
    parent=load_parent(C.PRETRAIN_SEEDS[0]);sm=smoke(parent,bufs,tt['A'],sA['train'],tt['B'],sB['train'],d);savej(OUT/'ARK-019_V3_SMOKE.json',sm);cal=calibrate(parent,bufs,tt['A'],sA['train'],tt['B'],sB['train'],d);savej(OUT/'ARK-019_V3_RUNTIME_CALIBRATION.json',cal);print(json.dumps({'smoke':sm,'runtime':cal},indent=2));return {'status':'PASS' if sm['status']=='PASS' and cal['fits'] else 'BLOCKED'}

def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['smoke','all'],default='all');x=a.parse_args()
    try:r=smoke_mode() if x.mode=='smoke' else run_all();print('ARK-019 V3 STATUS',r.get('status'));print('VERDICT',r.get('decision',{}).get('verdict'));return 0
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);savej(OUT/'ARK-019_V3_FAILURE.json',{'schema':'arkenstone-ark019-v3-failure/v1','status':'FAILED','exception':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});traceback.print_exc()
        try:package()
        except Exception:pass
        return 1
if __name__=='__main__':raise SystemExit(main())
