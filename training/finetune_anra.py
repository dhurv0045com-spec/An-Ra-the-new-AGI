from __future__ import annotations

import json
import pickle
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from anra_paths import ROOT, inject_all_paths, ensure_dirs
from training.optimizations import AdaptiveScheduler, GradientCheckpointedOuroboros, MultiScaleHardSampleDetector
from training.curriculum import get_phase

inject_all_paths()
ensure_dirs()

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from anra_brain import CausalTransformer, CharTokenizer
from generate import GenerationConfig, generate

try:
    from symbolic_bridge import (  # type: ignore
        SymbolicBridge, MathRouter, LogicRouter, CodeRouter
    )
except Exception:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(ROOT / "phase3" / "symbolic_bridge (45Q)"))
    import symbolic_bridge as _symbolic_bridge  # type: ignore

    class SymbolicBridge:
        def solve_math(self, text: str):
            return _symbolic_bridge.query_math(text).answer_text

        def solve_logic(self, text: str):
            return _symbolic_bridge.query_logic(text).answer_text

        def validate_code(self, text: str):
            res = _symbolic_bridge.query_code(text)
            return bool(getattr(res, "confidence", 0.0) >= 0.5)

        def verify_answer(self, answer: str, expected: str):
            return str(answer).strip().lower() == str(expected).strip().lower()

    class MathRouter:
        def __init__(self, engine: str = "sympy"):
            self.engine = engine

        def is_math_problem(self, text: str) -> bool:
            import re
            return bool(re.search(r"\d+\s*[-+*/=^%]", text))

    class LogicRouter:
        def __init__(self, engine: str = "dpll"):
            self.engine = engine

        def is_logic_problem(self, text: str) -> bool:
            import re
            return bool(re.search(r"\b(if|then|therefore|implies|proof|logic)\b", text.lower()))

    class CodeRouter:
        def __init__(self, engine: str = "ast"):
            self.engine = engine

        def is_code_block(self, text: str) -> bool:
            return any(k in text for k in ["def ", "class ", "for ", "while ", "import "])

CONFIG = {
    "base_checkpoint": "anra_brain.pt",
    "identity_checkpoint": "anra_brain_identity.pt",
    "tokenizer_path": "tokenizer/tokenizer.pkl",
    "data_path": "training_data/anra_dataset_v6_1.txt",
    "fallback_data_path": "training_data/anra_dataset_v6_1.txt",
    "drive_dir": "/content/drive/MyDrive/AnRa/",
    "epochs": 12,
    "batch_size": 32,
    "seq_len": 256,
    "val_split": 0.1,
    "patience": 3,
    "mixed_precision": True,
    "base_lr": 3e-5,
    "lm_head_lr": 6e-5,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "grad_accum_steps": 4,
    "block_size": 128,
    "n_embd": 256,
    "n_head": 4,
    "n_layer": 4,
    "shuffle_seed": 1337,
}

PROBE_PROMPTS = ["I am", "My purpose is", "An-Ra is", "Who created you", "What are you"]
IDENTITY_KEYWORDS = ["I am", "An-Ra", "my purpose", "I was", "I exist", "Who created", "What are you"]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    milestone: str


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_tokenizer(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def parse_identity_data(raw_text: str) -> Tuple[List[Tuple[str, str]], Dict[str, int]]:
    pairs: List[Tuple[str, str]] = []
    pair_count, mono_count = 0, 0

    block_pattern = re.compile(r"H:\s*(.*?)\nANRA:\s*(.*?)(?=\nH:|\Z)", re.S)
    for m in block_pattern.finditer(raw_text):
        h = m.group(1).strip()
        a = m.group(2).strip()
        if h and a:
            pairs.append((h, a))
            pair_count += 1

    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("H:") or ln.startswith("ANRA:"):
            continue
        pairs.append((ln, ln))
        mono_count += 1

    fmt = "mixed" if pair_count > 0 and mono_count > 0 else ("pairs" if pair_count > 0 else "monologue")
    dist = {"pairs": pair_count, "monologue": mono_count, "mixed": 1 if fmt == "mixed" else 0}
    return pairs, dist


def curriculum_subset(pairs: Sequence[Tuple[str, str]], phase: int) -> List[Tuple[str, str]]:
    if phase == 1:
        subset = [
            p for p in pairs
            if any(k.lower() in (p[0] + p[1]).lower() for k in IDENTITY_KEYWORDS)
        ]
        if len(subset) < 50:
            raise ValueError(
                f"Phase 1 subset too small ({len(subset)} pairs). "
                f"Identity keywords not found in training data. "
                f"Check combined_identity_data.txt has identity exchanges."
            )
        if len(subset) < 80:
            general_pairs = [p for p in pairs if p not in subset]
            padding = general_pairs[:80 - len(subset)]
            subset = subset + padding
            print(f"Phase 1 padded with {len(padding)} general pairs to reach {len(subset)} total")
        return subset
    if phase == 2:
        return list(pairs)
    return list(pairs)


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[str, str]], tok, seq_len: int):
        self.samples: List[List[int]] = []
        self.seq_len = seq_len
        for h, a in pairs:
            text = f"H: {h}\nANRA: {a}" if h else f"ANRA: {a}"
            ids = tok.encode(text)
            if len(ids) < 8:
                continue
            stride = max(1, seq_len // 2)
            for i in range(0, max(1, len(ids) - 1), stride):
                segment = ids[i : i + seq_len + 1]
                if len(segment) >= 8:
                    self.samples.append(segment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        ids = self.samples[index]
        if len(ids) < self.seq_len + 1:
            ids = ids + [0] * (self.seq_len + 1 - len(ids))
        x = torch.tensor(ids[: self.seq_len], dtype=torch.long)
        y = torch.tensor(ids[1 : self.seq_len + 1], dtype=torch.long)
        return x, y


def _load_model(vocab_size: int, device: torch.device) -> CausalTransformer:
    model = CausalTransformer(vocab_size, int(CONFIG["n_embd"]), int(CONFIG["n_head"]), int(CONFIG["n_layer"]), int(CONFIG["block_size"]))
    identity = Path(str(CONFIG["identity_checkpoint"]))
    base = Path(str(CONFIG["base_checkpoint"]))
    ckpt = identity if identity.exists() else base
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        elif isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {ckpt}")
    return model.to(device)


def _freeze_policy(model: CausalTransformer) -> Tuple[int, int, List[torch.nn.Parameter]]:
    for p in model.token_embedding_table.parameters():
        p.requires_grad = False
    total_blocks = len(model.blocks)
    freeze_n = max(1, total_blocks // 4)
    for i, block in enumerate(model.blocks):
        trainable = i >= freeze_n
        for p in block.parameters():
            p.requires_grad = trainable
    for p in model.ln_f.parameters():
        p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return frozen_count, trainable_count, trainable_params


def _optimizer(model: CausalTransformer) -> AdamW:
    base_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("lm_head"):
            head_params.append(p)
        else:
            base_params.append(p)
    return AdamW(
        [{"params": base_params, "lr": float(CONFIG["base_lr"])}, {"params": head_params, "lr": float(CONFIG["lm_head_lr"])}],
        weight_decay=float(CONFIG["weight_decay"]),
    )


def _milestone(val_loss: float) -> str:
    if val_loss > 2.0:
        return "Adapting..."
    if 1.5 <= val_loss <= 2.0:
        return "Identity forming"
    if 1.0 <= val_loss < 1.5:
        return "✓ Consistent voice present"
    return "★ Deep conditioning achieved"


def _evaluate(model: CausalTransformer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            b, t, c = logits.shape
            losses.append(F.cross_entropy(logits.view(b * t, c), y.view(b * t)).item())
    return float(sum(losses) / max(len(losses), 1))


def _identity_probe(tag: str) -> Dict[str, str]:
    out = {}
    print(f"\n[{tag}] identity probe")
    for p in PROBE_PROMPTS:
        text = generate(f"H: {p}\nANRA:", strategy="nucleus", max_tokens=100)
        out[p] = text
        print(f"- {p:16} | {text[:80]}")
    return out


from typing import Any
def _save_training_report(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")




def verify_training_pair(h_text: str, anra_text: str, symbolic: SymbolicBridge, math_router: MathRouter, logic_router: LogicRouter, code_router: CodeRouter) -> dict:
    result = {
      'valid': True,
      'type': 'natural_language',
      'verified': False,
      'correction': None
    }

    if math_router.is_math_problem(h_text):
        sympy_result = symbolic.solve_math(h_text)
        if sympy_result and not symbolic.verify_answer(anra_text, sympy_result):
            result['valid'] = False
            result['correction'] = sympy_result
            result['type'] = 'math'
        else:
            result['verified'] = True
            result['type'] = 'math'

    elif logic_router.is_logic_problem(h_text):
        dpll_result = symbolic.solve_logic(h_text)
        if dpll_result:
            result['verified'] = True
            result['type'] = 'logic'

    elif code_router.is_code_block(anra_text):
        ast_valid = symbolic.validate_code(anra_text)
        result['type'] = 'code'
        if not ast_valid:
            result['valid'] = False
            result['type'] = 'code_invalid'
        else:
            result['verified'] = True

    return result

def main() -> None:
    repo = ROOT
    if not (repo / str(CONFIG["data_path"])).exists():
        raise FileNotFoundError(f"Dataset not found: {repo / str(CONFIG['data_path'])}")
    data_path = repo / str(CONFIG["data_path"])
    if not data_path.exists():
        data_path = repo / str(CONFIG["fallback_data_path"])
    if not data_path.exists():
        print(f"IDENTITY DATA NOT FOUND: {data_path}")
        print("Run: python anra_merge_identity.py first")
        print(f"TXT files in repo: {sorted(Path('.').glob('*.txt'))}")
        import sys
        sys.exit(1)
    else:
        print(f"Identity dataset: {data_path} ({data_path.stat().st_size / 1e3:.1f}KB)")

    tokenizer = _load_tokenizer(repo / str(CONFIG["tokenizer_path"]))
    raw = data_path.read_text(encoding="utf-8", errors="replace")
    pairs, fmt_dist = parse_identity_data(raw)
    print(f"Format distribution: {fmt_dist}")

    symbolic = SymbolicBridge()
    math_router = MathRouter(engine='sympy')
    logic_router = LogicRouter(engine='dpll')
    code_router = CodeRouter(engine='ast')

    verified_pairs = []
    rejected_pairs = []
    corrected_pairs = []
    symbolic_stats = {
        "math_pairs_verified": 0,
        "logic_pairs_verified": 0,
        "code_pairs_validated": 0,
        "pairs_corrected": 0,
        "pairs_rejected": 0,
    }

    for h, anra in pairs:
        check = verify_training_pair(h, anra, symbolic, math_router, logic_router, code_router)
        if check['type'] == 'math' and check.get('verified'):
            symbolic_stats["math_pairs_verified"] += 1
        if check['type'] == 'logic' and check.get('verified'):
            symbolic_stats["logic_pairs_verified"] += 1
        if check['type'] in {'code', 'code_invalid'}:
            symbolic_stats["code_pairs_validated"] += 1

        if check['valid']:
            verified_pairs.append((h, anra))
        elif check['correction']:
            corrected_pairs.append((h, str(check['correction'])))
            symbolic_stats["pairs_corrected"] += 1
        else:
            rejected_pairs.append((h, anra))
            symbolic_stats["pairs_rejected"] += 1

    print("Symbolic verification complete:")
    print(f"  Verified: {len(verified_pairs)} pairs")
    print(f"  Corrected: {len(corrected_pairs)} pairs (wrong answers fixed)")
    print(f"  Rejected: {len(rejected_pairs)} pairs (unfixable)")
    print(f"  Training on {len(verified_pairs) + len(corrected_pairs)} pairs")

    verified_and_corrected = len(verified_pairs) + len(corrected_pairs)
    if verified_and_corrected < 100:
        raise ValueError(
            f"Only {verified_and_corrected} usable pairs after symbolic "
            f"verification (rejected: {len(rejected_pairs)}). "
            f"Add more training data or lower rejection thresholds."
        )

    rejection_rate = len(rejected_pairs) / max(len(pairs), 1)
    if rejection_rate > 0.5:
        print(f"WARNING: Symbolic bridge rejected {rejection_rate:.1%} of training pairs. Data quality may be low.")
        print("Consider reviewing combined_identity_data.txt for incorrect math/logic answers or malformed code blocks.")

    print(
        f"Symbolic verification: {len(verified_pairs)} verified, {len(corrected_pairs)} corrected, "
        f"{len(rejected_pairs)} rejected ({rejection_rate:.1%} rejection rate)"
    )

    pairs = verified_pairs + corrected_pairs

    rng = random.Random(CONFIG["shuffle_seed"])
    rng.shuffle(pairs)

    split = max(1, int(len(pairs) * CONFIG["val_split"]))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]

    device = _device()
    model = _load_model(tokenizer.vocab_size, device)
    ouroboros_model = GradientCheckpointedOuroboros(model, passes=1).to(device)
    hard_detector = MultiScaleHardSampleDetector()
    frozen, trainable, trainable_params = _freeze_policy(model)
    print(f"Frozen params: {frozen:,} | Trainable params: {trainable:,}")

    opt = _optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(CONFIG["mixed_precision"]) and device.type == "cuda")
    total_training_steps = int(CONFIG["epochs"]) * max((len(train_pairs) // int(CONFIG["batch_size"])) + 1, 1)
    warmup_steps = int(0.05 * total_training_steps)
    adaptive_scheduler = AdaptiveScheduler(float(CONFIG["base_lr"]), warmup_steps, total_training_steps)

    before_probe = _identity_probe("before")

    best = float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()
    history: List[EpochMetrics] = []
    loss_curve: List[float] = []

    for epoch in range(1, int(CONFIG["epochs"]) + 1):
        phase = get_phase(epoch - 1)
        if phase.name == "warmup":
            epoch_pairs = curriculum_subset(train_pairs, 1)
        elif phase.name == "ramp":
            epoch_pairs = curriculum_subset(train_pairs, 2)
        else:
            epoch_pairs = curriculum_subset(train_pairs, 3)
        print(f"CURRICULUM PHASE: {phase.name} | passes={phase.ouroboros_passes}")
        ouroboros_model.passes = phase.ouroboros_passes

        train_ds = PairDataset(epoch_pairs, tokenizer, int(CONFIG["seq_len"]))
        val_ds = PairDataset(val_pairs, tokenizer, int(CONFIG["seq_len"]))
        train_loader = DataLoader(train_ds, batch_size=int(CONFIG["batch_size"]), shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=int(CONFIG["batch_size"]), shuffle=False)

        model.train()
        losses = []
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            current_step = (epoch - 1) * len(train_loader) + step
            adaptive_lr = adaptive_scheduler.get_lr(current_step, None)
            for param_group in opt.param_groups:
                param_group['lr'] = adaptive_lr

            sample_text = tokenizer.decode(x[0].tolist()) if x.shape[0] > 0 else ""
            is_hard, difficulty = hard_detector.detect(sample_text)
            if is_hard and difficulty >= 2:
                ouroboros_model.passes = min(3, difficulty)
            else:
                ouroboros_model.passes = phase.ouroboros_passes

            with torch.amp.autocast("cuda", enabled=bool(CONFIG["mixed_precision"]) and device.type == "cuda"):
                _, loss = ouroboros_model(x, y)
                from typing import cast
                loss = cast(torch.Tensor, loss) / float(CONFIG["grad_accum_steps"])
            scaler.scale(loss).backward()

            if step % int(CONFIG["grad_accum_steps"]) == 0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(CONFIG["grad_clip"]))
                if float(grad_norm) > 2.0:
                    print(f"WARNING: Gradient norm {float(grad_norm):.2f} > 2.0 at step {step} — instability detected")
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                adaptive_scheduler.get_lr(current_step, loss.item() * float(CONFIG["grad_accum_steps"]))
            losses.append(float(loss.item() * float(CONFIG["grad_accum_steps"])))

        train_loss = float(sum(losses) / max(len(losses), 1))
        val_loss = _evaluate(model, val_loader, device)
        milestone = _milestone(val_loss)
        loss_curve.append(val_loss)
        history.append(EpochMetrics(epoch, train_loss, val_loss, float(CONFIG["base_lr"]), milestone))
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {milestone}")

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            stale = 0
            out = repo / str(CONFIG["identity_checkpoint"])
            ckpt_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch": epoch,
                "best_val_loss": best,
            }
            torch.save(ckpt_payload, out)
            drive_dir = Path(str(CONFIG["drive_dir"]))
            drive_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt_payload, drive_dir / str(CONFIG["identity_checkpoint"]))
            print(f"Saved best checkpoint to {out} and {drive_dir / str(CONFIG['identity_checkpoint'])}")
        else:
            stale += 1
            if stale >= int(CONFIG["patience"]):
                print("Early stopping triggered")
                break

    after_probe = _identity_probe("after")
    elapsed = time.time() - start
    report = {
        "epochs_completed": len(history),
        "final_train_loss": history[-1].train_loss if history else None,
        "best_val_loss": best,
        "best_epoch": best_epoch,
        "curriculum_phases": {"phase1_epochs": "1-4", "phase2_epochs": "5-8", "phase3_epochs": "9-12"},
        "params_trained": trainable,
        "params_frozen": frozen,
        "training_time_seconds": elapsed,
        "identity_probe_before": before_probe,
        "identity_probe_after": after_probe,
        "loss_curve": loss_curve,
        "train_loss_curve": [m.train_loss for m in history],
        "val_loss_curve": [m.val_loss for m in history],
        "format_distribution": fmt_dist,
        "symbolic_verification": symbolic_stats,
    }

    local_report = repo / "output" / "finetune_report.json"
    _save_training_report(local_report, report)
    drive_report = Path(str(CONFIG["drive_dir"])) / "finetune_report.json"
    drive_report.parent.mkdir(parents=True, exist_ok=True)
    _save_training_report(drive_report, report)
    print(f"Training report saved to {local_report} and {drive_report}")


if __name__ == "__main__":
    main()
