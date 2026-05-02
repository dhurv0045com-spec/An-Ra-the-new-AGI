from __future__ import annotations
from pathlib import Path
import torch
import torch.nn.functional as F
class CIVGuard:
    def __init__(self, model, tokenizer, identity_path: Path, layer_idx: int = 4, threshold: float = 0.92) -> None:
        self.model=model; self.tokenizer=tokenizer; self.identity_path=Path(identity_path); self.layer_idx=layer_idx; self.threshold=threshold; self.baseline=None
    def _extract_activation(self,prompts:list[str])->torch.Tensor:
        device=next(self.model.parameters()).device; captured=[]
        def hook_fn(module, inp, out): captured.append(out.detach().mean(dim=1))
        block=self.model.blocks[self.layer_idx]; h=block.register_forward_hook(hook_fn)
        self.model.eval(); all_caps=[]
        with torch.no_grad():
            for prompt in prompts:
                ids=self.tokenizer.encode(prompt)[:self.model.block_size]
                if not ids: continue
                x=torch.tensor([ids],dtype=torch.long,device=device)
                captured.clear(); self.model(x)
                if captured: all_caps.append(captured[0])
        h.remove()
        if not all_caps: return torch.zeros(self.model.n_embd, device=device)
        return torch.cat(all_caps,dim=0).mean(dim=0)
    def _load_prompts(self,max_pairs:int=512)->list[str]:
        if not self.identity_path.exists(): return ["Who are you?","What is your purpose?","What are you?"]
        out=[]
        for line in self.identity_path.read_text(encoding='utf-8',errors='replace').splitlines():
            line=line.strip()
            if line.startswith('H:'): out.append(line[2:].strip())
            if len(out)>=max_pairs: break
        return out or ["Who are you?"]
    def compute_baseline(self)->torch.Tensor:
        self.baseline=self._extract_activation(self._load_prompts()); return self.baseline
    def save_baseline(self,path:Path)->None:
        if self.baseline is None: raise RuntimeError('Call compute_baseline() first.')
        Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(self.baseline.cpu(), path)
    def load_baseline(self,path:Path)->None: self.baseline=torch.load(path,map_location=next(self.model.parameters()).device)
    def verify(self)->tuple[float,bool]:
        if self.baseline is None: raise RuntimeError('CIV baseline not set. Call compute_baseline() or load_baseline().')
        current=self._extract_activation(self._load_prompts()); base=self.baseline.to(current.device)
        sim=float(F.cosine_similarity(base.unsqueeze(0),current.unsqueeze(0)).item()); return sim, sim>=self.threshold
    def similarity(self)->float: return self.verify()[0]
