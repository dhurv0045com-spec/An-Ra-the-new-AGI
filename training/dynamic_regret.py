from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class DynamicRegretScheduler:
    def __init__(self, optimizer: "torch.optim.Optimizer", eta_base: float = 3e-4, min_lr: float = 1e-5, max_lr: float = 3e-3) -> None:
        self.optimizer=optimizer; self.eta_base=float(eta_base); self.min_lr=float(min_lr); self.max_lr=float(max_lr)
        self.V_total=0.0; self.T_total=0; self._session_start_loss=None
    def session_start(self, val_loss: float)->None: self._session_start_loss=float(val_loss)
    def session_end(self, val_loss: float, steps_this_session: int)->float:
        if self._session_start_loss is not None: self.V_total += abs(float(val_loss)-self._session_start_loss)
        self.T_total += max(0,int(steps_this_session)); lr=self._compute_lr(); self._set_lr(lr); self._session_start_loss=None; return lr
    def _compute_lr(self)->float:
        if self.T_total==0: return self.eta_base
        lr=self.eta_base*((self.V_total/self.T_total)**(1/3)); return float(max(self.min_lr,min(self.max_lr,lr)))
    def current_lr(self)->float: return self._compute_lr()
    def _set_lr(self,lr:float)->None:
        for pg in self.optimizer.param_groups: pg['lr']=lr
    def state_dict(self)->dict: return {"V_total":self.V_total,"T_total":self.T_total,"current_lr":self._compute_lr(),"eta_base":self.eta_base,"min_lr":self.min_lr,"max_lr":self.max_lr}
    def load_state_dict(self,d:dict)->None:
        self.V_total=float(d.get('V_total',0.0)); self.T_total=int(d.get('T_total',0)); self.eta_base=float(d.get('eta_base',self.eta_base)); self.min_lr=float(d.get('min_lr',self.min_lr)); self.max_lr=float(d.get('max_lr',self.max_lr)); self._set_lr(float(d.get('current_lr',self.eta_base)))
    def save(self,path:Path)->None:
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(self.state_dict(),indent=2),encoding='utf-8')
    def load(self,path:Path)->None:
        path=Path(path)
        if path.exists():
            try: self.load_state_dict(json.loads(path.read_text(encoding='utf-8')))
            except Exception as exc: print(f"[DRS] Could not load state from {path}: {exc}. Starting fresh.")
