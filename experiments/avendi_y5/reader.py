"""Y5 controlled reader, corrected byte learning, and qualified preservation forks.

All stochastic conditions derive from a seed; evaluation corpora never train. These
are experiments, not claims that any endpoint has qualified.
"""
from __future__ import annotations
import random
import math
from hashlib import sha256
import re
import torch
import torch.nn.functional as F
from anra_v5.elendril.representation import UTF8, SpecialToken
from experiments.avendi_y4.models import (AvendiY4ByteModel, get_y4_byte_spec, apply_radial_norm_cap,
    encode_y4_typed_batch, IDX_TO_ENT, Y4TypedCompositionModel)
from experiments.avendi_y4.tasks import build_functional_graph_families, oracle_walk_public_bytes

SEEDS = (7, 19, 41)
OBJECTIVES = ('ce', 'pointer', 'sham')


def jobs():
    out = []
    for seed in SEEDS:
        for task in ('reader', 'answer'):
            for fmt in ('fixed', 'varied'):
                for obj in (('ce',) if task == 'reader' else OBJECTIVES):
                    out.append(dict(id=f'AB-{task}-{fmt}-{obj}-s{seed}', stage='AB', arm=f'{task}-{fmt}-{obj}', seed=seed, depends=[], updates=1024))
        base = f'AB-answer-fixed-ce-s{seed}'
        for arm in ('new', 'new_cap', 'old_control'):
            out.append(dict(id=f'F-{arm}-s{seed}', stage='F', arm=arm, seed=seed, depends=[base], updates=512))
    return out


def _render(c, style):
    rows = c['rows']
    if style == 0:
        body = ' ; '.join(f'{a}->{b}' for a,b in rows)
    elif style == 1:
        body = ' ; '.join(f'{a} => {b}' for a,b in rows)
    elif style == 2:
        body = ' ; '.join(f'[{a}={b}]' for a,b in rows)
    else:
        body = ' ; '.join(f'({a} : {b})' for a,b in rows)
    return f'ROWS: {body} || STEP={c["depth"]} Q:{c["query"]}'


def _target(c, task):
    if task == 'reader':
        return c['gold']
    if task == 'new':
        return c['gold'][::-1]
    return c['gold']


def _make_items(seed, count, task, fmt, panel='train'):
    # Identical underlying graph streams across matched arms and formatting.
    offset = {'train': 0, 'dev': 100003, 'confirm': 200003, 'unseen': 300007}[panel]
    fams = build_functional_graph_families(panel, count, seed + offset, depths=(1,2,3,4))
    items = []
    for fi, fam in enumerate(fams):
        for ci, c in enumerate(fam['cells']):
            style = int(fmt[-1]) if fmt.startswith('style') else ((fi + ci) % 3 if fmt == 'varied' and panel == 'train' else (3 if fmt == 'unseen' or panel == 'unseen' else 0))
            prompt = _render(c, style)
            # Distinguish the old/new tasks by public instruction; base and old
            # forks share identical instruction and task definition.
            prompt = ('REVERSE ANSWER ' if task == 'new' else 'ANSWER ' if task == 'answer' else '') + prompt
            x = dict(c, public_prompt=prompt, gold=_target(c, task), walk_gold=c['gold'], style=style)
            items.append(x)
    return items


def _span_labels(item, length, sham=False):
    # Preserve class counts while independently randomizing positions for sham.
    p = item['public_prompt']; y = torch.zeros(length, dtype=torch.long)
    pos = p.index('ROWS:') + len('ROWS:')
    for a,b in item['rows']:
        ia = p.index(a, pos); ib = p.index(b, ia + len(a))
        for i in range(ia, ia+len(a)): y[i+1] = 1
        for i in range(ib, ib+len(b)): y[i+1] = 2
        pos = ib+len(b)
    q = p.rindex('Q:') + 2
    for i in range(q, q + len(item['query'])): y[i+1] = 3
    if sham:
        end = len(p)+1
        true = y[:end].tolist()
        rng = random.Random(int.from_bytes(sha256(item['world_id'].encode()).digest()[:8],'big'))
        # Preserve exact class counts but independently reassign positions, rejecting
        # any fixed relabeling of the real roles (including identity).
        for _ in range(1000):
            candidate=true.copy(); rng.shuffle(candidate)
            maps={}
            for a,b in zip(true,candidate): maps.setdefault(a,set()).add(b)
            if any(len(v)>1 for v in maps.values()): break
        else: raise ValueError('unable to construct independent sham')
        y[:end]=torch.tensor(candidate)
    return y


def loss(model, batch, objective='ce'):
    device = next(model.parameters()).device
    frames = [UTF8.frame(x['public_prompt'], x['gold']) for x in batch]
    if any(len(f.ids) > model.spec.context_length for f in frames):
        raise ValueError('full answer does not fit context')
    n = max(len(f.ids) for f in frames)
    ids = torch.full((len(batch), n), UTF8.pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    spans = torch.full_like(ids, -100)
    for i,(f,item) in enumerate(zip(frames,batch)):
        ids[i,:len(f.ids)] = torch.tensor(f.ids, device=device)
        for j in range(1,len(f.ids)):
            if f.loss_mask[j]: labels[i,j-1] = f.ids[j]
        sp = _span_labels(item, len(f.ids), objective == 'sham')
        # Supervise only prompt tokens, no padding or answer leakage.
        end = f.ids.index(UTF8.special(SpecialToken.ANSWER))
        spans[i,:end] = sp[:end].to(device)
    logits, hidden, _ = model.forward_tokens(ids)
    result = F.cross_entropy(logits.reshape(-1,logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    span_loss=F.cross_entropy(model.span_head(hidden).reshape(-1,4), spans.reshape(-1), ignore_index=-100)
    result=result + (.25 if objective in ('pointer','sham') else 0.0)*span_loss
    return result


def _batches(n, updates, seed, size=16):
    # Full shuffled passes, carrying across pass boundaries; every cell appears.
    rng = random.Random(seed); order=[]; result=[]
    for _ in range(updates):
        batch=[]
        while len(batch)<size:
            if not order:
                order=list(range(n)); rng.shuffle(order)
            take=min(size-len(batch),len(order)); batch.extend(order[:take]); del order[:take]
        result.append(batch)
    return result


def _decode_batch(model, items, limit=8):
    """Autoregressive decode with a preregistered limit independent of gold."""
    dev=next(model.parameters()).device
    sequences=[]
    for item in items:
        ids=list(UTF8.frame(item['public_prompt'],'').ids)
        ids=ids[:ids.index(UTF8.special(SpecialToken.ANSWER))+1]
        if len(ids)+limit>model.spec.context_length:
            raise ValueError('prompt plus fixed generation budget exceeds context')
        sequences.append(ids)
    raw=[[] for _ in sequences]; done=[False]*len(items)
    with torch.no_grad():
        for _ in range(limit):
            n=max(map(len,sequences))
            batch=torch.full((len(items),n),UTF8.pad_id,dtype=torch.long,device=dev)
            for i,ids in enumerate(sequences): batch[i,:len(ids)]=torch.tensor(ids,device=dev)
            logits,_,_=model.forward_tokens(batch)
            for i,ids in enumerate(sequences):
                if done[i]: continue
                t=int(logits[i,len(ids)-1].argmax()); raw[i].append(t); ids.append(t)
                done[i]=t==UTF8.eos_id
            if all(done): break
    result=[]
    for tokens,eos in zip(raw,done):
        content=tokens[:-1] if eos else tokens
        try: value=UTF8.decode(content)
        except (UnicodeError,ValueError): value=None
        result.append((value,tokens,eos))
    return result


def parse_reader_output(text):
    """Reject partial, ambiguous, out-of-vocabulary, and malformed parses."""
    if not isinstance(text,str): return None
    match=re.fullmatch(r'([A-H][0-7]>[A-H][0-7](?:;[A-H][0-7]>[A-H][0-7])*)\|([A-H][0-7])\|([1-4])',text)
    if not match: return None
    pairs=[tuple(x.split('>')) for x in match.group(1).split(';')]
    if len({a for a,b in pairs})!=len(pairs) or any(a==b for a,b in pairs): return None
    return {'rows':pairs,'query':match.group(2),'depth':int(match.group(3))}


def _execute(parsed, executor):
    if parsed is None: return None
    graph=dict(parsed['rows']); node=parsed['query']
    for _ in range(parsed['depth']):
        if node not in graph: return None
        node=graph[node]
    if executor is None: return node
    device=next(executor.parameters()).device
    cell={**parsed,'gold':node}
    rk,rv,q,mask,_=encode_y4_typed_batch([cell],device)
    with torch.no_grad():
        idx=int(executor.forward_trace(rk,rv,q,mask,steps=parsed['depth'])['final_logits'].argmax(-1)[0])
    return IDX_TO_ENT[idx]


def _span_parse(item,ids,labels):
    groups={1:[],2:[],3:[]}
    current=[]; previous=0
    for token,label in zip(ids+[0],labels+[0]):
        if label!=previous:
            if previous in groups and current:
                try: groups[previous].append(UTF8.decode(current))
                except (ValueError,UnicodeError): return None,labels
            current=[]
        if label in groups and token>=4: current.append(token)
        previous=label
    if not groups[1] or len(groups[1])!=len(groups[2]) or len(groups[3])!=1: return None,labels
    depth=re.search(r'STEP=([1-4])(?:\s|$)',item['public_prompt'])
    if not depth: return None,labels
    serialized=';'.join(a+'>'+b for a,b in zip(groups[1],groups[2]))+'|'+groups[3][0]+'|'+depth.group(1)
    return parse_reader_output(serialized),labels


def evaluate(model, items, task, executor=None):
    model.eval(); rows=[]; families={}; chain_families={}; oracle_families={}
    decoded=[]
    for start in range(0,len(items),16):
        batch=items[start:start+16]
        if task=='reader':
            ids=[]
            for item in batch:
                frame=list(UTF8.frame(item['public_prompt'],'').ids)
                ids.append(frame[:frame.index(UTF8.special(SpecialToken.ANSWER))])
            if max(map(len,ids))>model.spec.context_length: raise ValueError('reader prompt exceeds context')
            padded=torch.full((len(ids),max(map(len,ids))),UTF8.pad_id,dtype=torch.long,device=next(model.parameters()).device)
            for i,x in enumerate(ids): padded[i,:len(x)]=torch.tensor(x,device=padded.device)
            with torch.no_grad():
                _,hidden,_=model.forward_tokens(padded)
                predictions=model.span_head(hidden).argmax(-1).tolist()
            decoded.extend(_span_parse(x,t,p[:len(t)]) for x,t,p in zip(batch,ids,predictions))
        else: decoded.extend(_decode_batch(model,batch))
    for item,dec in zip(items,decoded):
        if task=='reader':
            parsed,tokens=dec
            prediction=parsed
            eos=True
            correct = parsed is not None and parsed['rows']==item['rows'] and parsed['query']==item['query'] and parsed['depth']==item['depth']
        else:
            prediction,tokens,eos = dec
            correct = prediction == item['gold'] and eos
        row={'id':item['world_id'],'family_id':item['family_id'],'public_prompt':item['public_prompt'],'gold_rows':item['rows'],'gold_query':item['query'],'gold_depth':item['depth'],'prediction':prediction,'predicted_ids':tokens,'gold':item['gold'],'gold_ids':list(UTF8.encode(item['gold']))+[UTF8.eos_id] if task!='reader' else None,'eos':eos if task!='reader' else None,'correct':correct,'format':item['style'],'depth':item['depth'],'assignment':item['assign'],'reverse_order':item['reverse_order']}
        if task == 'reader':
            # Exact reader output is the connected input contract. The independent
            # reference walk is measured, never substituted for model output.
            # Parsed solely from predicted span labels and public depth.
            row['predicted_span_class_ids']=tokens
            row['prompt_ids']=list(UTF8.frame(item['public_prompt'],'').ids)[:-2]
            row['gold_span_class_ids']=_span_labels(item,len(row['prompt_ids'])).tolist()
            row['reference_walk'] = item['walk_gold']
            row['parsed'] = parsed is not None
            row['reference_chain_prediction'] = _execute(parsed,None)
            row['chain_prediction'] = _execute(parsed,executor) if executor is not None else None
            row['chain_correct'] = row['chain_prediction'] == item['walk_gold'] if executor is not None else False
            oracle_graph={'rows':item['rows'],'query':item['query'],'depth':item['depth']}
            row['oracle_executor_prediction'] = _execute(oracle_graph,executor) if executor is not None else None
        rows.append(row); families.setdefault(item['family_id'],[]).append(correct)
        if task=='reader' and executor is not None:
            chain_families.setdefault(item['family_id'],[]).append(row['chain_correct'])
            oracle_families.setdefault(item['family_id'],[]).append(row['oracle_executor_prediction']==item['walk_gold'])
    model.train()
    return {'predictions':rows,'correct':sum(r['correct'] for r in rows),'denominator':len(rows),
            'chain_correct':sum(r.get('chain_correct',False) for r in rows) if task=='reader' and executor is not None else None,
            'chain_family_correct':sum(all(v) for v in chain_families.values()) if task=='reader' and executor is not None else None,
            'chain_family_denominator':len(chain_families) if task=='reader' and executor is not None else None,
            'chain_family_rate':sum(all(v) for v in chain_families.values())/max(1,len(chain_families)) if task=='reader' and executor is not None else None,
            'usable_parse_count':sum(r.get('reference_chain_prediction') is not None for r in rows) if task=='reader' else None,
            'usable_parse_denominator':len(rows) if task=='reader' else None,
            'usable_parse_rate':sum(r.get('reference_chain_prediction') is not None for r in rows)/max(1,len(rows)) if task=='reader' else None,
            'oracle_family_correct':sum(all(v) for v in oracle_families.values()) if task=='reader' and executor is not None else None,
            'oracle_family_denominator':len(oracle_families) if task=='reader' and executor is not None else None,
            'oracle_family_rate':sum(all(v) for v in oracle_families.values())/max(1,len(oracle_families)) if task=='reader' and executor is not None else None,
            'chain_executor':'frozen_y4' if executor is not None else 'BLOCKED: missing qualified checkpoint',
            'oracle_executor_correct':sum(r.get('oracle_executor_prediction')==r.get('reference_walk') for r in rows) if task=='reader' and executor is not None else None,
            'family_correct':sum(all(v) for v in families.values()),'families':len(families),
            'family_rate':sum(all(v) for v in families.values())/max(1,len(families))}


def preflight():
    fixed=_make_items(7,4,'answer','fixed'); varied=_make_items(7,4,'answer','varied')
    assert len(fixed)==32 and {x['assign'] for x in fixed}=={0,1}
    for a,b in zip(fixed,varied):
        assert a['rows']==b['rows'] and a['gold']==b['gold']
        assert oracle_walk_public_bytes(UTF8.frame(_render(a,0),a['walk_gold']).ids)==a['walk_gold']
        assert not torch.equal(_span_labels(a,len(a['public_prompt'])+1),_span_labels(a,len(a['public_prompt'])+1,True))
    for task in ('reader','answer','new'):
        for style in range(4):
            for c in _make_items(7,4,task,f'style{style}','dev'):
                frame=UTF8.frame(c['public_prompt'],c['gold'])
                assert len(frame.ids) <= get_y4_byte_spec('S3',cpu_canary=True).context_length
                assert len(frame.ids)-len(UTF8.encode(c['gold']))-1+8 <= get_y4_byte_spec('S3',cpu_canary=True).context_length
    batches=_batches(len(fixed),4,7,8)
    assert set(sum(batches,[])) == set(range(len(fixed)))
    return {'ok':True,'eight_cell_exposure':True,'matched_graphs':True,'sham_differs_every_cell':True,'independent_walk':True}


def run(job,ctx):
    seed=int(job['seed']); stage=job['stage']; arm=job['arm']; canary=ctx.canary
    if stage == 'AB':
        task,fmt,obj=arm.split('-')
    else:
        task,fmt,obj='answer','fixed','ce'
        if arm in ('new','new_cap'): task='new'
    model=AvendiY4ByteModel(get_y4_byte_spec('S3',cpu_canary=canary), seed=seed, use_aux=True).to(ctx.device)
    old_task='answer'; source={'kind':'AB_answer','id':job['depends'][0]} if stage=='F' else None
    if stage=='F':
        prev=ctx.previous(job['depends'][0])
        if prev.get('qualified',False) or canary:
            ctx.load_parent(job['depends'][0],model)
        else:
            # Use a historical reader only after independently measuring the
            # original capability on a held-out Y5 dev panel. Never select on confirm.
            if ctx.full is None:
                return {'status':'GATED','reason':'AB parent unqualified and Y4 archive unavailable','qualified':False}
            from zipfile import ZipFile
            from experiments.avendi_y5.composition import load_model
            try:
                with ZipFile(ctx.full) as archive:
                    digest=load_model(archive,f'A-pointer-s{seed}',model)
            except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
                return {'status':'GATED','reason':'qualified old source unavailable: '+str(exc),'qualified':False}
            baseline=evaluate(model,_make_items(seed,16,'reader','fixed','dev'),'reader')
            if baseline['family_rate'] < .8:
                return {'status':'GATED','reason':'Y4 reader failed independent old-capability qualification','baseline':baseline,'qualified':False}
            old_task='reader'
            source={'kind':'Y4_pointer_reader','checkpoint_sha256':digest,'baseline':baseline}
            if arm=='old_control': task='reader'
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    parent_new_dev=None; parent_new_split=None
    if stage=='F':
        dev_new=_make_items(seed,1 if canary else 16,'new','fixed','dev')
        parent_new_dev=evaluate(model,dev_new,'new')
        parent_new_split=parent_new_dev if ctx.split=='dev' else evaluate(model,_make_items(seed,1 if canary else 16,'new','fixed',ctx.split),'new')
        if not canary and parent_new_dev['family_rate'] >= .8:
            return {'status':'GATED','reason':'new capability already qualified at parent; no acquisition headroom','parent_new_dev':parent_new_dev,'qualified':False}
    n=4 if canary else 128
    items=_make_items(seed,n,task,fmt,'train')
    updates=2 if canary else int(job['updates'])
    batches=_batches(len(items),updates,seed,size=16)
    # Canary exercises each cell as well, without substantive local training.
    if canary: batches=_batches(len(items),2,seed,size=16)
    split=ctx.split
    eval_items=_make_items(seed,1 if canary else 16,task,fmt,split)
    old_items=_make_items(seed,1 if canary else 16,old_task,'fixed',split)
    unseen=_make_items(seed,1 if canary else 16,task,'unseen',split)
    executor=None; bridge={'status':'BLOCKED','reason':'Y4 checkpoint archive unavailable'}
    if task=='reader' and ctx.full is not None:
        from zipfile import ZipFile
        from experiments.avendi_y5.composition import load_model
        executor_seed = seed if seed in (7,19) else 41
        try:
            executor=Y4TypedCompositionModel('categorical',128,seed=executor_seed).to(ctx.device)
            with ZipFile(ctx.full) as archive:
                digest=load_model(archive,f'C-categorical_true_step-s{executor_seed}',executor)
            executor.eval()
            for parameter in executor.parameters(): parameter.requires_grad_(False)
            bridge={'status':'AVAILABLE','checkpoint_sha256':digest,'executor_seed':executor_seed}
        except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
            executor=None; bridge={'status':'BLOCKED','reason':str(exc)}
    milestones={}
    def on_step(step,m):
        milestones[int(step)]={'update':step,'evaluation':evaluate(m,eval_items,task,executor)}
        if stage=='F':
            with torch.no_grad():
                params=dict(m.named_parameters())
                delta=math.sqrt(sum(float((p.detach()-initial[n].to(p.device)).float().square().sum()) for n,p in params.items()))
                base=math.sqrt(sum(float(initial[n].float().square().sum()) for n in params))
            milestones[int(step)]['relative_parameter_delta']=delta/max(base,1e-12)
            milestones[int(step)]['old_endpoint']=evaluate(m,old_items,old_task)
            milestones[int(step)]['new_endpoint']=evaluate(m,_make_items(seed,1 if canary else 16,'new','fixed',split),'new')
        return milestones[int(step)]
    cap_stats=[]
    def post_step(step,m):
        if arm=='new_cap': cap_stats.append(apply_radial_norm_cap(m,initial,max_rel_norm=.001))
    trained=ctx.train(model,items,batches,lambda m,b:loss(m,b,'pointer' if task=='reader' else obj),lr=3e-4,on_step=on_step if split=='dev' else None,post_step=post_step if arm=='new_cap' else None)
    primary=evaluate(model,eval_items,task,executor)
    result={'status':'COMPLETE','training':trained,'primary':primary,'milestones':trained.get('milestones',[milestones[k] for k in sorted(milestones)]),
            'qualified':primary['family_rate']>=.8 and not canary,'engineering_canary':bool(canary),
            'design':{'objective':obj,'format':fmt,'task':task,'training_cells':len(items),'updates':updates,
                      'distinct_cells_exposed':len(set(sum(batches,[]))),'batch_size':16,
                      'cell_exposure_counts':{str(i):sum(b.count(i) for b in batches) for i in range(len(items))},
                      'training_framed_tokens':sum(len(UTF8.frame(items[i]['public_prompt'],items[i]['gold']).ids) for b in batches for i in b),
                      'generation_token_limit':8}}
    if stage=='AB':
        result['formats']={str(style):evaluate(model,_make_items(seed,1 if canary else 16,task,f'style{style}',split),task,executor) for style in range(4)}
        result['unseen_format']=result['formats']['3']
        if task=='reader': result['bridge']=bridge
    if stage=='F':
        result['old_endpoint']=evaluate(model,old_items,old_task)
        result['old_source']=source
        result['new_endpoint']=evaluate(model,_make_items(seed,1 if canary else 16,'new','fixed',split),'new')
        result['old_qualified']=result['old_endpoint']['family_rate']>=.8
        result['new_qualified']=result['new_endpoint']['family_rate']>=.8
        result['parent_new_dev']=parent_new_dev
        result['parent_new_endpoint']=parent_new_split
        result['new_family_gain']=result['new_endpoint']['family_rate']-parent_new_split['family_rate']
        result['new_acquired']=parent_new_dev['family_rate'] < .8 and result['new_family_gain'] >= .15
        result['qualified']=bool(not canary and result['old_qualified'] and result['new_qualified'] and result['new_acquired'])
        result['radial_cap_last']=cap_stats[-1] if cap_stats else None
        result['radial_cap_trajectory_since_resume']=cap_stats
        result['qualification_threshold_family_rate']=.8
        result['parent_qualified']=bool(prev.get('qualified',False))
    return result
