from __future__ import annotations
import math, torch
import torch.nn as nn
class ESVModule(nn.Module):
    def __init__(self,d_model:int=512,d_esv:int=64,momentum_pos:float=0.85,momentum_neg:float=0.95)->None:
        super().__init__(); self.d_model=d_model; self.d_esv=d_esv; self.momentum_pos=momentum_pos; self.momentum_neg=momentum_neg
        self.predictor=nn.Sequential(nn.Linear(d_esv,32),nn.SiLU(),nn.Linear(32,3),nn.Tanh())
        for m in self.predictor.modules():
            if isinstance(m, nn.Linear): nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.register_buffer('state', torch.zeros(3))
    def forward(self,h:torch.Tensor)->torch.Tensor:
        esv=h.mean(dim=(0,1))[-self.d_esv:]; new=self.predictor(esv); mom=self.momentum_neg if float(new[0].item())<0 else self.momentum_pos; self.state=mom*self.state+(1-mom)*new.detach(); return self.state
    @property
    def valence(self)->float: return float(self.state[0].item())
    @property
    def arousal(self)->float: return float(self.state[1].item())
    @property
    def dominance(self)->float: return float(self.state[2].item())
    def attention_temperature(self,tau0:float=1.0)->float: return tau0*math.exp(-0.5*self.arousal)
    def memory_write_threshold(self)->float: return 0.8-0.2*abs(self.valence)
    def ssm_modulation(self)->torch.Tensor: return self.state.clone()
    def dgsa_gate(self)->tuple[float,float]:
        gssm=1/(1+math.exp(-2*self.dominance)); gatt=1/(1+math.exp(2*self.dominance)); return gssm,gatt
    def reset(self)->None: self.state.zero_()
    def as_dict(self)->dict[str,float]: return {'valence':self.valence,'arousal':self.arousal,'dominance':self.dominance}
