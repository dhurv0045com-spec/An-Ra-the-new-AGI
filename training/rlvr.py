from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn.functional as F
@dataclass
class RLVRTask: prompt:str; task_type:str; test_code:str=''; expected:str=''; task_id:str=''
@dataclass
class RLVRStep: task:RLVRTask; completions:list[str]; rewards:list[float]; advantages:list[float]; loss:float; mean_reward:float
class RLVRTrainer:
    def __init__(self, model, tokenizer, optimizer: torch.optim.Optimizer, verifier, G:int=4, kl_coeff:float=0.04, max_new_tokens:int=256, grad_clip:float=1.0):
        self.model=model; self.tokenizer=tokenizer; self.optimizer=optimizer; self.verifier=verifier; self.G=G; self.kl_coeff=kl_coeff; self.max_new_tokens=max_new_tokens; self.grad_clip=grad_clip; self._ref_state=None
    def _snapshot_reference(self): self._ref_state={k:v.detach().clone() for k,v in self.model.state_dict().items()}
    def _device(self): return next(self.model.parameters()).device
    @torch.no_grad()
    def _generate_completions(self,prompt:str,n:int)->list[str]:
        return [self.tokenizer.decode(self.tokenizer.encode(prompt)[-16:]) for _ in range(n)]
    def _log_prob(self,prompt:str,completion:str)->torch.Tensor:
        ids=(self.tokenizer.encode(prompt)+self.tokenizer.encode(completion))[-self.model.block_size:]
        if len(ids)<2: return torch.tensor(0.0,device=self._device(),requires_grad=True)
        x=torch.tensor([ids[:-1]],dtype=torch.long,device=self._device()); y=torch.tensor([ids[1:]],dtype=torch.long,device=self._device())
        logits,_=self.model(x); lp=F.log_softmax(logits,dim=-1)
        n_prompt=min(len(self.tokenizer.encode(prompt)), len(ids)-1)
        comp=lp[0,n_prompt:,:]; yt=y[0,n_prompt:]
        return comp.gather(1,yt.unsqueeze(1)).squeeze(1).sum() if yt.numel()>0 else torch.tensor(0.0,device=self._device(),requires_grad=True)
    def _ref_log_prob(self,prompt:str,completion:str)->torch.Tensor:
        if self._ref_state is None: self._snapshot_reference(); return self._log_prob(prompt,completion).detach()
        cur={k:v.clone() for k,v in self.model.state_dict().items()}; self.model.load_state_dict(self._ref_state)
        with torch.no_grad(): lp=self._log_prob(prompt,completion).detach()
        self.model.load_state_dict(cur); return lp
    def train_step(self, task: RLVRTask) -> RLVRStep:
        if self._ref_state is None: self._snapshot_reference()
        completions=self._generate_completions(task.prompt,self.G)
        rewards=[]
        for c in completions:
            vr=self.verifier.score(task.task_type, code=c, test_code=task.test_code, expression=c, expected=task.expected, response=c, task=task.prompt); rewards.append(float(vr.score))
        r=torch.tensor(rewards,dtype=torch.float32); mean_r=r.mean(); std_r=r.std()+1e-8; adv=((r-mean_r)/std_r).tolist()
        self.model.train(); self.optimizer.zero_grad(); pol=torch.tensor(0.0,device=self._device()); kl=torch.tensor(0.0,device=self._device())
        for c,a in zip(completions,adv): lp=self._log_prob(task.prompt,c); rlp=self._ref_log_prob(task.prompt,c); pol=pol+(-a*lp); kl=kl+(lp-rlp)
        loss=pol/self.G + self.kl_coeff*(kl/self.G); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.grad_clip); self.optimizer.step()
        return RLVRStep(task,completions,rewards,adv,float(loss.item()),float(mean_r.item()))
