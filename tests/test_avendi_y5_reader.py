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
