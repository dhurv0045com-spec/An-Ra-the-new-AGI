"""ARK-001: micro lift-off mapping. Self-contained; preregistered in PLAN.md.

Arms: T1-COMPACT, T1-BYTE, T2-COMPACT, T0-COMPACT, T1-COMPACT-LARGE.
Metrics: train/test exact (greedy), per-position digit accuracy, loss,
dose at first lift-off (train exact >= 0.9), wall time, parameter count.
Baselines computed before any model result: majority answer, per-position
marginal-mode exact rate (the best score achievable by fitting answer
marginals only — the red-team quantity for "distributional fit").
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- data

COMPACT_VOCAB = ["<pad>", "<bos>", "<eos>", "0", "1", "2", "3", "4", "5",
                 "6", "7", "8", "9", "+", "-", "*", "/", "=", " "]


def build_compact_vocab():
    return {token: index for index, token in enumerate(COMPACT_VOCAB)}


class ByteVocab:
    """Frozen cymek byte-level tokenizer (24,576) with explicit specials."""

    PAD, BOS, EOS = 0, 2, 3

    def __init__(self, artifact: Path) -> None:
        from tokenizers import Tokenizer

        self.backend = Tokenizer.from_str(
            gzip.decompress(artifact.read_bytes()).decode("utf-8")
        )
        self.size = 24_576

    def encode(self, text: str) -> list[int]:
        # BOS prefix keeps answer-sequence semantics identical to CompactVocab
        return [self.BOS] + list(self.backend.encode(text).ids)

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode([i for i in ids if i not in (0, 2, 3)])


class CompactVocab:
    PAD, BOS, EOS = 0, 1, 2

    def __init__(self) -> None:
        self.table = build_compact_vocab()
        self.size = len(COMPACT_VOCAB)

    def encode(self, text: str) -> list[int]:
        return [self.BOS] + [self.table[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        inverse = {i: t for t, i in self.table.items()}
        return "".join(
            inverse.get(i, "") for i in ids if i not in (self.PAD, self.BOS, self.EOS)
        )


def t1_pool() -> list[tuple[str, str]]:
    """All 100 single-digit additions, canon template."""
    return [(f"{a} + {b} = ", f"{a + b}") for a in range(10) for b in range(10)]


def t0_pool() -> list[tuple[str, str]]:
    """Trivial tier: x+0 and x*1 over a 0..5999 band (citadel T0 semantics)."""
    rows = []
    for x in range(0, 6000, 3):
        rows.append((f"{x} + 0 = ", f"{x}"))
        rows.append((f"{x} * 1 = ", f"{x}"))
    return rows


def t2_rows(split: str, n: int) -> list[tuple[str, str]]:
    """Two-digit no-carry add with citadel-style structural bands.

    train: tens of a in 1..5 ; test: tens of a in 6..7 (structurally
    disjoint band) ; ones chosen so that ones(a) + ones(b) <= 9 (no carry),
    b tens in 1..(9 - a_tens) so the tens column never carries.
    """
    import random

    rng = random.Random(13 if split == "train" else 29)
    tens_a = range(1, 6) if split == "train" else range(6, 8)
    rows = []
    seen = set()
    while len(rows) < n:
        ta = rng.choice(list(tens_a))
        ua = rng.randrange(0, 10)
        tb = rng.randrange(1, 10 - ta)  # keep tens sum <= 9 (no carry)
        ub = rng.randrange(0, 10 - ua)
        a, b = ta * 10 + ua, tb * 10 + ub
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((f"{a} + {b} = ", f"{a + b}"))
    return rows



def t3_rows(split: str, n: int) -> list[tuple[str, str]]:
    """Two-digit add WITH carry (compositional tier), structural tens bands.

    train: tens of a in 1..5 ; test: tens of a in 6..7. Ones digits are
    unconstrained (carry happens); the tens answer requires ta+tb+carry.
    """
    import random

    rng = random.Random(13 if split == "train" else 29)
    tens_a = range(1, 6) if split == "train" else range(6, 8)
    rows = []
    seen = set()
    while len(rows) < n:
        ta = rng.choice(list(tens_a))
        ua = rng.randrange(0, 10)
        tb = rng.randrange(1, 10 - ta)
        ub = rng.randrange(0, 10)
        a, b = ta * 10 + ua, tb * 10 + ub
        if (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((f"{a} + {b} = ", f"{a + b}"))
    return rows

# ---------------------------------------------------------------- model

class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class Block(nn.Module):
    def __init__(self, width: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, ffn, bias=False)
        self.up = nn.Linear(width, ffn, bias=False)
        self.down = nn.Linear(ffn, width, bias=False)

    def forward(self, x):
        b, t, w = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        hd = w // self.heads
        q = q.view(b, t, self.heads, hd).transpose(1, 2)
        k = k.view(b, t, self.heads, hd).transpose(1, 2)
        v = v.view(b, t, self.heads, hd).transpose(1, 2)
        pos = torch.arange(t, device=x.device)
        inv = 10000.0 ** (-torch.arange(0, hd, 2, device=x.device).float() / hd)
        phase = pos.float()[:, None] * inv[None, :]
        cos, sin = phase.cos()[None, None], phase.sin()[None, None]
        q2 = torch.stack((q[..., 0::2], q[..., 1::2]), -1).flatten(-2)
        k2 = torch.stack((k[..., 0::2], k[..., 1::2]), -1).flatten(-2)

        def apply_rope(z):
            # z: [b, heads, t, hd]; pairwise even/odd
            ze, zo = z[..., 0::2], z[..., 1::2]
            cos_e = cos.squeeze(0).squeeze(0)  # [t, hd/2]
            sin_e = sin.squeeze(0).squeeze(0)
            cos_f = torch.repeat_interleave(cos_e, 2, dim=-1)[None, None]
            sin_f = torch.repeat_interleave(sin_e, 2, dim=-1)[None, None]
            return torch.cat((ze * cos_f[..., 0::2] - zo * sin_f[..., 0::2],
                              ze * sin_f[..., 0::2] + zo * cos_f[..., 0::2]), dim=-1)

        q, k = apply_rope(q), apply_rope(k)
        mask = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, t, w)
        x = x + self.proj(out)
        h = self.norm2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Micro(nn.Module):
    def __init__(self, vocab_size: int, width: int, layers: int = 4, ffn: int = 512) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList(Block(width, 4, ffn) for _ in range(layers))
        self.norm = RMSNorm(width)
        self.width = width

    def head_weight(self):
        return self.embed.weight

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        # tied output head: project hidden states onto the vocabulary
        return self.norm(x) @ self.embed.weight.T


# ---------------------------------------------------------------- eval

def encode_batch(vocab, rows: list[tuple[str, str]], device):
    prompts = [vocab.encode(p) for p, _ in rows]
    answers = [vocab.encode(a) + [vocab.EOS] for _, a in rows]
    length = max(len(p) + len(a) for p, a in zip(prompts, answers))
    pad = vocab.PAD
    tokens = torch.full((len(rows), length), pad, dtype=torch.long)
    prompt_len = torch.zeros(len(rows), dtype=torch.long)
    for i, (p, a) in enumerate(zip(prompts, answers)):
        tokens[i, : len(p)] = torch.tensor(p)
        tokens[i, len(p): len(p) + len(a)] = torch.tensor(a)
        prompt_len[i] = len(p)
    return tokens.to(device), prompt_len.to(device)


def loss_and_positions(model, vocab, rows, device):
    tokens, prompt_len = encode_batch(vocab, rows, device)
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    positions = torch.arange(tokens.shape[1] - 1, device=device)[None, :]
    mask = positions >= (prompt_len - 1)[:, None]  # supervise answer + EOS
    if mask.sum() == 0:
        raise ValueError("no supervised targets")
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
        reduction="none",
    ).view(targets.shape)
    return (losses * mask).sum() / mask.sum(), mask.sum()


@torch.no_grad()
def greedy_exact(model, vocab, rows, device, max_answer=6):
    """Batched greedy decode, grouped by exact prompt length (no padding).

    All rows in a group share prompt length, so the right edge stays aligned
    and positional ids stay exact. Returns exact rate + per-position accuracy.
    """

    model.eval()
    groups: dict[int, list[int]] = {}
    for index, (prompt, _) in enumerate(rows):
        groups.setdefault(len(vocab.encode(prompt)), []).append(index)
    text_of: dict[int, str] = {}
    for same_length in groups.values():
        batch_rows = [rows[i] for i in same_length]
        tokens = torch.tensor([vocab.encode(p) for p, _ in batch_rows], device=device)
        batch = len(batch_rows)
        finished = torch.zeros(batch, dtype=torch.bool)
        generated: list[list[int]] = [[] for _ in batch_rows]
        for _ in range(max_answer):
            logits = model(tokens)[:, -1]
            next_ids = torch.argmax(logits, dim=-1)
            tokens = torch.cat(
                [tokens, torch.full((batch, 1), vocab.PAD, dtype=torch.long, device=device)],
                dim=1,
            )
            all_finished = True
            for i in range(batch):
                if finished[i]:
                    continue
                token = int(next_ids[i].item())
                if token in (vocab.EOS, vocab.PAD):
                    finished[i] = True
                else:
                    generated[i].append(token)
                    tokens[i, -1] = token
                    all_finished = False
            if all_finished:
                break
        for i, (prompt, _answer) in enumerate(batch_rows):
            text_of[same_length[i]] = vocab.decode(generated[i]).strip()
    correct = 0
    position_stats: dict[int, list[float]] = {}
    for index, (_, answer) in enumerate(rows):
        text = text_of[index]
        ok = text == answer
        correct += ok
        answer_digits = list(answer)
        gen_digits = list(text)
        for pos in range(max(len(answer_digits), len(gen_digits))):
            a = answer_digits[pos] if pos < len(answer_digits) else "<missing>"
            g = gen_digits[pos] if pos < len(gen_digits) else "<missing>"
            position_stats.setdefault(pos, []).append(1.0 if a == g else 0.0)
    model.train()
    per_position = {pos: sum(v) / len(v) for pos, v in sorted(position_stats.items())}
    return correct / len(rows), per_position


def marginal_mode_baseline(rows_train, rows_test):
    """Best exact rate achievable by per-position marginal mode answers."""
    from collections import Counter

    lengths = Counter(len(a) for _, a in rows_train)
    best_len = max(lengths, key=lambda l: lengths[l])
    pool = [a for _, a in rows_train if len(a) == best_len]
    mode = "".join(Counter(row[pos] for row in pool).most_common(1)[0][0]
                   for pos in range(best_len))
    exact = sum(1 for _, a in rows_test if a == mode) / len(rows_test)
    return mode, exact


def majority_baseline(rows_train, rows_test):
    from collections import Counter

    mode = Counter(a for _, a in rows_train).most_common(1)[0][0]
    return mode, sum(1 for _, a in rows_test if a == mode) / len(rows_test)


# ---------------------------------------------------------------- arms

def run_arm(name: str, *, rows_train, rows_eval, vocab, width: int,
            steps: int, batch: int, lr: float, wall_box_s: float,
            device, seed: int, out_dir: Path = None, log=print) -> dict:
    torch.manual_seed(seed)
    model = Micro(vocab.size, width).to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    majority = majority_baseline(rows_train, rows_eval)
    marginal = marginal_mode_baseline(rows_train, rows_eval)
    history = []
    lift_off_step = None
    started = time.perf_counter()
    log(f"[{name}] training loop start", flush=True)
    rng = torch.Generator().manual_seed(seed)
    model.train()
    aborted = False
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(rows_train), (batch,), generator=rng)
        batch_rows = [rows_train[i] for i in idx]
        loss, count = loss_and_positions(model, vocab, batch_rows, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 200 == 0 or step == 1:
            train_exact, per_pos = greedy_exact(model, vocab, rows_train[:256], device)
            eval_exact, eval_per_pos = greedy_exact(model, vocab, rows_eval[:256], device)
            history.append({
                "step": step, "loss": float(loss.item()),
                "train_exact": train_exact, "test_exact": eval_exact,
                "train_per_position": per_pos, "test_per_position": eval_per_pos,
                "exposures_per_example": step * batch / max(1, len(rows_train)),
            })
            if lift_off_step is None and train_exact >= 0.9:
                lift_off_step = step
            if step % 500 == 0:
                print(f"[{name}] step {step} loss {loss.item():.3f} "
                      f"train {train_exact:.2f} test {eval_exact:.2f}", flush=True)
        if time.perf_counter() - started > wall_box_s:
            aborted = True
            print(f"[{name}] WALL BOX hit at step {step}", flush=True)
            break
    elapsed = time.perf_counter() - started
    final = history[-1] if history else {}
    tokens_consumed = step * batch * max(len(p) + len(a) for p, a in rows_train)
    return {
        "arm": name,
        "status": "ABORTED_WALL_BOX" if aborted else "COMPLETED",
        "steps_run": step,
        "parameters": params,
        "vocab_size": vocab.size,
        "train_pool_size": len(rows_train),
        "eval_pool_size": len(rows_eval),
        "wall_seconds": elapsed,
        "tokens_consumed_estimate": tokens_consumed,
        "baselines": {
            "majority_answer": {"answer": majority[0], "exact_on_eval": majority[1]},
            "marginal_mode": {"answer": marginal[0], "exact_on_eval": marginal[1]},
        },
        "lift_off_step": lift_off_step,
        "lift_off_exposures_per_example": (
            lift_off_step * batch / max(1, len(rows_train)) if lift_off_step else None),
        "final": final,
        "history_tail": history[-6:],
        "history_full": history,
        "seed": seed,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> int:
    torch.set_num_threads(1)
    torch.set_num_threads(1)  # tiny models: intra-op threading only adds contention
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wall-box", type=float, default=12 * 60)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--arms", default="T1-COMPACT,T1-BYTE,T2-COMPACT,T0-COMPACT,T1-COMPACT-LARGE")
    parser.add_argument("--tokenizer-artifact", default=None)
    parser.add_argument("--out", default="experiments/ARK-001/RESULT.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    repo = Path(__file__).resolve().parents[2]
    artifact = Path(args.tokenizer_artifact) if args.tokenizer_artifact else (
        repo / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
    )
    compact = CompactVocab()
    byte_vocab = None

    arm_specs = {
        "T1-COMPACT": dict(rows_train=t1_pool(), rows_eval=t1_pool(), vocab=compact, width=128),
        "T2-COMPACT": dict(rows_train=t2_rows("train", 500), rows_eval=t2_rows("test", 200),
                           vocab=compact, width=128),
        "T0-COMPACT": dict(rows_train=t0_pool(), rows_eval=t0_pool()[:500], vocab=compact, width=128),
        "T1-COMPACT-LARGE": dict(rows_train=t1_pool(), rows_eval=t1_pool(), vocab=compact, width=256),
        "T3-COMPACT": dict(rows_train=t3_rows("train", 500), rows_eval=t3_rows("test", 200),
                           vocab=compact, width=128),
    }
    if "T1-BYTE" in args.arms:
        byte_vocab = ByteVocab(artifact)
        arm_specs["T1-BYTE"] = dict(rows_train=t1_pool(), rows_eval=t1_pool(),
                                    vocab=byte_vocab, width=128)

    results = []
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        spec = arm_specs[name]
        print(f"=== arm {name} ===", flush=True)
        result = run_arm(
            name, rows_train=spec["rows_train"], rows_eval=spec["rows_eval"], log=print,
            vocab=spec["vocab"], width=spec["width"], steps=args.steps,
            batch=args.batch, lr=args.lr, wall_box_s=args.wall_box,
            device=device, seed=args.seed, out_dir=repo / "experiments/ARK-001",
        )
        results.append(result)
        out_path = repo / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "schema": "arkenstone-ark001-result/v1",
            "preregistration": "experiments/ARK-001/PLAN.md",
            "seed": args.seed,
            "device": str(device),
            "torch": torch.__version__,
            "arms": results,
        }, indent=2) + "\n", encoding="utf-8")
    print("done ->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
