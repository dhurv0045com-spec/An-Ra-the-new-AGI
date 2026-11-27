import torch
import pytest
from experiments.avendi_y5 import reader
from anra_v5.elendril.representation import UTF8


def test_eight_cells_and_matched_streams_and_sham():
    assert reader.preflight()['ok']
    a=reader._make_items(7,4,'answer','fixed')
    b=reader._make_items(7,4,'answer','varied')
    assert len(a)==32
    assert all(x['rows']==y['rows'] and x['gold']==y['gold'] for x,y in zip(a,b))
    assert all(not torch.equal(reader._span_labels(x,len(x['public_prompt'])+1),reader._span_labels(x,len(x['public_prompt'])+1,True)) for x in a)
    assert {x['style'] for x in b} == {0,1,2}
    assert {x['style'] for x in reader._make_items(7,1,'answer','varied','unseen')} == {3}


def test_exposure_and_job_dependencies():
    batches=reader._batches(32,4,7,size=8)
    assert set(sum(batches,[])) == set(range(32))
    jobs=reader.jobs(); ids={j['id'] for j in jobs}
    assert len(jobs)==len(ids)==33
    assert all(set(j['depends']) <= ids for j in jobs)
    assert all(j['depends'] for j in jobs if j['stage']=='F')


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.bias=torch.nn.Parameter(torch.zeros(UTF8.vocabulary_size))
        self.spec=type('Spec',(),{'context_length':512})()
        self.span_head=torch.nn.Identity()
    def forward_tokens(self, x):
        logits=self.bias.view(1,1,-1).expand(x.shape[0],x.shape[1],-1)
        return logits, torch.zeros(*x.shape,4), []


def test_full_sequence_and_eos_loss():
    item=reader._make_items(7,1,'answer','fixed')[0]
    model=UniformModel(); reader.loss(model,[item]).backward()
    ids=list(UTF8.encode(item['gold']))+[UTF8.eos_id]
    # Each target byte and EOS receives a negative gradient under uniform logits.
    assert len(ids)==3
    assert all(model.bias.grad[t] < 0 for t in ids)


def test_defect_detection_rejects_overlong_answer():
    item=reader._make_items(7,1,'answer','fixed')[0]
    item['gold']='X'*512
    with pytest.raises(ValueError, match='full answer'):
        reader.loss(UniformModel(),[item])

def test_reader_parse_and_chain_fail_closed():
    c=reader._make_items(7,1,'reader','fixed')[0]
    serialized=';'.join(a+'>'+b for a,b in c['rows'])+'|'+c['query']+'|'+str(c['depth'])
    parsed=reader.parse_reader_output(serialized)
    assert reader._execute(parsed,None)==c['walk_gold']
    assert reader.parse_reader_output(serialized+'garbage') is None
    assert reader.parse_reader_output('A0>B0;A0>C0|A0|1') is None
    assert reader._execute(reader.parse_reader_output('A0>B0|C0|1'),None) is None


def test_sham_has_same_counts_and_not_fixed_relabeling():
    c=reader._make_items(7,1,'answer','fixed')[0]
    true=reader._span_labels(c,len(c['public_prompt'])+1).tolist()
    sham=reader._span_labels(c,len(c['public_prompt'])+1,True).tolist()
    assert sorted(true)==sorted(sham)
    assert any(len({b for a,b in zip(true,sham) if a==label})>1 for label in set(true))

class ScriptedModel(torch.nn.Module):
    def __init__(self, sequence):
        super().__init__(); self.dummy=torch.nn.Parameter(torch.zeros(1)); self.sequence=sequence
        self.spec=type('Spec',(),{'context_length':512})(); self.calls=0
    def forward_tokens(self,x):
        logits=torch.full((*x.shape,UTF8.vocabulary_size),-100.0)
        logits[:,:,self.sequence[min(self.calls,len(self.sequence)-1)]]=100.0
        self.calls+=1
        return logits,torch.zeros(*x.shape,4),[]


def test_decode_fixed_budget_and_raw_eos():
    items=reader._make_items(7,1,'answer','fixed')[:2]
    ids=list(UTF8.encode('A0'))+[UTF8.eos_id]
    result=reader._decode_batch(ScriptedModel(ids),items)
    assert all(x==('A0',ids,True) for x in result)
    items[0]['gold']='a different long answer'
    result=reader._decode_batch(ScriptedModel([UTF8.encode('Z')[0]]),items)
    assert all(len(tokens)==8 and not eos for _,tokens,eos in result)


def test_matched_format_panels_share_graphs():
    panels=[reader._make_items(19,4,'reader',f'style{x}','dev') for x in range(4)]
    assert all([x['rows'] for x in p]==[x['rows'] for x in panels[0]] for p in panels)
    assert all({x['style'] for x in p}=={i} for i,p in enumerate(panels))


def test_canonical_split_disjointness_and_renamed_duplicates():
    splits = {sp: reader._make_items(7, 4, 'answer', 'fixed', sp) for sp in ('train', 'dev', 'confirm', 'transfer')}
    res = reader.verify_canonical_split_disjointness(splits)
    assert res['passed'] and res['renamed_duplicate_caught']
    # Confirm and transfer graphs must be completely disjoint (separate panels + unseen style 3 on transfer).
    assert not ({x['canon_sig'] for x in splits['confirm']} & {x['canon_sig'] for x in splits['transfer']})
    # Inject a renamed duplicate (swapped prefix + reversed rows + style 3) into confirm -> must be caught.
    orig = splits['train'][0]
    tampered = dict(splits)
    dup = dict(orig, world_id='confirm-x', family_id='confirm-x', rows=list(reversed(orig['rows'])),
               public_prompt='ANSWER ' + reader._render({**orig, 'rows': list(reversed(orig['rows']))}, 3))
    tampered['confirm'] = [*splits['confirm'], dup]
    assert not reader.verify_canonical_split_disjointness(tampered)['passed']


def test_blind_baselines_mutations_and_0xb5_safety():
    items = reader._make_items(7, 4, 'answer', 'fixed', 'dev')
    bl = reader.evaluate_blind_baselines(items, 'answer')
    assert set(bl) == {'first_row', 'last_row', 'prior_only'}
    assert all(v['denominator'] == len(items) and v['family_rate'] <= 0.5 for v in bl.values())
    # Invalid UTF-8 byte 0xB5 never crashes _decode_batch, _span_parse, or decode_bytes_safe.
    bad_tok = 4 + 0xB5
    class BadByteModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.spec = type('Spec', (), {'context_length': 512})()
        def forward_tokens(self, x):
            logits = torch.full((*x.shape, UTF8.vocabulary_size), -100.0)
            logits[:, :, bad_tok] = 100.0
            return logits, torch.zeros(*x.shape, 4), []
    dec = reader._decode_batch(BadByteModel(), items[:1])
    assert dec[0][0] is None
    assert isinstance(reader.decode_bytes_safe([bad_tok, 0xB5]), str)
    parsed, _ = reader._span_parse(items[0], [bad_tok], [1])
    assert parsed is None


def test_u0000_reader_and_answer_forward_backward_autocast():
    from experiments.avendi_y4.models import AvendiY4ByteModel, get_y4_byte_spec
    spec = get_y4_byte_spec('S3', cpu_canary=True)
    for task, obj in (('reader', 'pointer'), ('answer', 'ce')):
        model = AvendiY4ByteModel(spec, seed=7, use_aux=True)
        batch = reader._make_items(7, 1, task, 'fixed', 'train')[:2]
        with torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=False):
            lv = reader.loss(model, batch, obj)
        assert torch.isfinite(lv)
        lv.backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        ev = reader.evaluate(model, reader._make_items(7, 1, task, 'fixed', 'dev')[:2], task)
        assert ev['denominator'] == 2 and 'blind_baselines' in ev

