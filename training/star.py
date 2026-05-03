from __future__ import annotations
from dataclasses import dataclass, field
import torch, torch.nn.functional as F
@dataclass
class STaRExample: prompt:str; chain:str; answer:str; score:float; source:str
@dataclass
class STaRLoop:
    model:object; tokenizer:object; verifier:object; threshold:float=0.9; max_tokens:int=512; accepted:list[STaRExample]=field(default_factory=list)
    def _device(self): return next(self.model.parameters()).device if hasattr(self.model,'parameters') else torch.device('cpu')
    def _extract_answer(self,chain:str)->str: return chain.split('</think>',1)[-1].strip() if '</think>' in chain else ([l.strip() for l in chain.splitlines() if l.strip()] or [''])[ -1]
    def _generate_chain(self,prompt:str)->str: return prompt+'\n<think>\n</think> answer'
    def step(self,prompt:str,task_type:str='open',correct_answer:str='',n_attempts:int=4)->list[STaRExample]:
        res=[]; found=False
        for _ in range(n_attempts):
            chain=self._generate_chain(prompt); ans=self._extract_answer(chain); vr=self.verifier.score(task_type,code=ans,expression=ans,expected=correct_answer,response=ans,task=prompt)
            ex=STaRExample(prompt,chain,ans,float(vr.score),'direct'); res.append(ex)
            if vr.score>=self.threshold: self.accepted.append(ex); found=True
        if not found and correct_answer:
            ex=STaRExample(prompt,prompt+f'\n<think>\nwhy\n</think>\n{correct_answer}',correct_answer,0.5,'rationalization'); self.accepted.append(ex); res.append(ex)
        return res
    def finetune_on_chains(self,optimizer:torch.optim.Optimizer,n_steps:int=50,chain_weight:float=1.25)->list[float]: return []
    def replay_buffer(self)->list[dict]: return [x.__dict__ for x in self.accepted]
