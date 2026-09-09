from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPECTED_SCIENCE_SHA = "b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5"
EXPECTED_BIRTH_SHA = "8f17d092897f41df45100d227ecf7a2391ed5a5cfd3d3bcf373ad86c94b0c4f4"
EXPECTED_BIRTH_BYTES = 17_906_590
SCIENCE_DRIVE_PATH = Path("/content/drive/MyDrive/genisis-arkenstone/data_15.parquet")
DRIVE_ROOT = Path("/content/drive/MyDrive/genisis-arkenstone/ARK018_SCIENCE_BIRTH_V1")
LOCAL_ROOT = Path("/content/ark018_work")
CONTEXT = 256
EFFECTIVE_SEQS = 32
TARGET_TOKENS_PER_UPDATE = CONTEXT * EFFECTIVE_SEQS
VOCAB_SIZE = 8192
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode())


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def stream_hash_and_copy(src: Path, dst: Path) -> tuple[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    n = 0
    tmp = dst.with_suffix(dst.suffix + ".partial")
    with src.open("rb") as fi, tmp.open("wb") as fo:
        while True:
            b = fi.read(8 << 20)
            if not b:
                break
            h.update(b)
            fo.write(b)
            n += len(b)
    tmp.replace(dst)
    return h.hexdigest(), n


def normalize_text(x: str) -> str:
    return x.replace("\r\n", "\n")


def split_from_hash(hex_digest: str) -> str:
    bucket = int(hex_digest[:16], 16) % 10000
    if bucket < 9000:
        return "train"
    if bucket < 9500:
        return "control"
    return "sealed"


def iter_parquet_text(path: Path, batch_size: int = 512) -> Iterator[tuple[int, str]]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    idx = 0
    for rb in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        arr = rb.column(0).to_pylist()
        for x in arr:
            if isinstance(x, str) and x:
                yield idx, normalize_text(x)
            else:
                yield idx, ""
            idx += 1


def parquet_schema_receipt(path: Path) -> dict:
    import pyarrow
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    return {
        "pyarrow_version": pyarrow.__version__,
        "rows": int(pf.metadata.num_rows),
        "row_groups": int(pf.metadata.num_row_groups),
        "columns": [{"name": f.name, "type": str(f.type)} for f in schema],
        "has_text": "text" in schema.names,
    }


def build_tokenizer(sample_texts: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from tokenizers.pre_tokenizers import ByteLevel
    tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN, byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=[PAD_TOKEN, EOS_TOKEN, UNK_TOKEN],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(sample_texts, trainer=trainer, length=len(sample_texts))
    return tok


def tokenizer_json_bytes(tok) -> bytes:
    # Stable JSON emitted by tokenizers; compact it once more for a canonical hash.
    obj = json.loads(tok.to_str())
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def encode_batch(tok, texts: list[str]) -> list[list[int]]:
    return [e.ids for e in tok.encode_batch(texts)]


def append_u16(path: Path, ids: Iterable[int]) -> int:
    arr = np.fromiter((int(x) for x in ids), dtype=np.uint16)
    with path.open("ab") as f:
        arr.tofile(f)
    return int(arr.size)


def memmap_u16(path: Path) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def fixed_start(seed: int, step: int, n_tokens: int, need: int) -> int:
    if n_tokens < need:
        raise RuntimeError(f"token cache too small: {n_tokens} < {need}")
    h = hashlib.sha256(f"ark018:{seed}:{step}".encode()).digest()
    return int.from_bytes(h[:8], "big") % (n_tokens - need + 1)


def batch_from_buffer(buf: np.ndarray, start: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    need = TARGET_TOKENS_PER_UPDATE + 1
    n = len(buf)
    if n < need:
        raise RuntimeError("cache too short for one effective update")
    if start + need <= n:
        raw = np.asarray(buf[start:start + need], dtype=np.int64)
    else:
        # deterministic wrap for sequential Birth/replay cursors
        first = np.asarray(buf[start:], dtype=np.int64)
        second = np.asarray(buf[:need - len(first)], dtype=np.int64)
        raw = np.concatenate([first, second])
    x = torch.from_numpy(raw[:-1].copy()).to(device).view(EFFECTIVE_SEQS, CONTEXT)
    y = torch.from_numpy(raw[1:].copy()).to(device).view(EFFECTIVE_SEQS, CONTEXT)
    return x, y


def source_for_step(arm: str, step: int) -> str:
    r = ((int(step) - 1) % 100) + 1
    if arm == "SCIENCE_ONLY":
        return "science"
    if arm == "BIRTH_NATURAL_2PCT":
        return "birth" if r in {25, 75} else "science"
    special = {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}
    if arm == "BIRTH_REHEARSAL_10PCT":
        return "birth" if r in special else "science"
    if arm == "SCIENCE_REPLAY_10PCT_CONTROL":
        return "science_replay" if r in special else "science"
    raise ValueError(f"unknown arm {arm}")


def count_sources(arm: str, horizon: int) -> Counter:
    return Counter(source_for_step(arm, s) for s in range(1, horizon + 1))


def source_occurrence_index(arm: str, step: int, source: str) -> int:
    # Number of same-source updates strictly before this step.
    return sum(source_for_step(arm, s) == source for s in range(1, step))


def ceil100(x: int) -> int:
    return ((int(x) + 99) // 100) * 100


def choose_horizon(birth_tokens: int) -> int:
    # Ten Birth updates per 100 normal steps; each Birth update consumes 8192 targets.
    full_pass_updates = math.ceil(birth_tokens / TARGET_TOKENS_PER_UPDATE)
    total_needed = ceil100(full_pass_updates * 10)
    return min(8000, max(6200, total_needed))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int = 384, heads: int = 6):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(z):
            return z.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int = 384, heads: int = 6, ffn: int = 1536):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Ark018GPT(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, context: int = CONTEXT):
        super().__init__()
        d_model = 384
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList([Block() for _ in range(10)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, return_hidden: bool = False):
        b, t = idx.shape
        if t > self.context:
            raise ValueError(f"sequence length {t} > context {self.context}")
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        h = self.ln_f(x)
        logits = self.lm_head(h)
        return (logits, h) if return_hidden else logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        a = t.detach().cpu().contiguous().numpy()
        h.update(a.tobytes())
    return h.hexdigest()


def optimizer_for(model: nn.Module, lr: float = 3e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)


def lr_at(step: int, horizon: int) -> float:
    peak, floor = 3e-4, 3e-5
    warm = max(1, int(round(0.02 * horizon)))
    if step <= warm:
        return peak * step / warm
    q = (step - warm) / max(1, horizon - warm)
    q = min(max(q, 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * q))


def autocast_ctx(device: torch.device):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", dtype=torch.float16)


def make_scaler(device: torch.device):
    enabled = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def lm_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


@torch.no_grad()
def eval_buffer(model: nn.Module, buf: np.ndarray, device: torch.device, seed: int, sequences: int = 64) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    for i in range(sequences):
        need = CONTEXT + 1
        st = fixed_start(seed, i + 1, len(buf), need)
        raw = np.asarray(buf[st:st + need], dtype=np.int64)
        x = torch.from_numpy(raw[:-1].copy()).to(device)[None, :]
        y = torch.from_numpy(raw[1:].copy()).to(device)[None, :]
        with autocast_ctx(device):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(-1) == y).sum().item())
        total_tokens += y.numel()
    nll = total_loss / max(1, total_tokens)
    return {
        "tokens": total_tokens,
        "nll": nll,
        "perplexity": float(math.exp(min(20.0, nll))),
        "next_token_accuracy": total_correct / max(1, total_tokens),
    }


def continuation_score(model: nn.Module, tok, prompt: str, answer: str, device: torch.device) -> float:
    p = tok.encode(prompt).ids
    a = tok.encode(answer).ids
    ids = p + a
    if len(ids) < 2 or not a:
        return -1e9
    if len(ids) > CONTEXT:
        drop = len(ids) - CONTEXT
        ids = ids[drop:]
        p_len = max(1, len(p) - drop)
    else:
        p_len = len(p)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device)[None, :]
    targets = torch.tensor(ids[1:], dtype=torch.long, device=device)
    with torch.no_grad(), autocast_ctx(device):
        logits = model(x)[0]
        lp = F.log_softmax(logits.float(), dim=-1)
    # Target positions corresponding to answer tokens.
    first = max(0, p_len - 1)
    vals = []
    for j in range(first, len(targets)):
        vals.append(float(lp[j, targets[j]].item()))
    return sum(vals) / max(1, len(vals))


def score_mcq(model: nn.Module, tok, items: list[dict], device: torch.device) -> dict:
    model.eval()
    by_family: dict[str, list[int]] = {}
    details = []
    correct = 0
    for it in items:
        scores = [continuation_score(model, tok, it["prompt"] + "\nAnswer: ", str(c), device) for c in it["choices"]]
        pred = int(np.argmax(scores))
        hit = int(pred == int(it["answer"]))
        correct += hit
        by_family.setdefault(it.get("family", "unknown"), []).append(hit)
        details.append({"id": it["id"], "pred": pred, "answer": int(it["answer"]), "correct": bool(hit), "scores": scores})
    return {
        "n": len(items),
        "accuracy": correct / max(1, len(items)),
        "by_family": {k: sum(v) / len(v) for k, v in by_family.items()},
        "details": details,
    }


def algorithmic_items(seed: int = 18018) -> list[dict]:
    rng = random.Random(seed)
    out = []
    # Deliberately beyond major Birth Book table boundaries; narrow transfer diagnostic only.
    for i in range(20):
        n = rng.randint(16385, 24000)
        ans = format(n, "x")
        vals = [ans, format(n + 1, "x"), format(max(0, n - 1), "x"), format(n ^ 0x10, "x")]
        rng.shuffle(vals)
        out.append({"id": f"hex_{i}", "family": "hex_ood", "prompt": f"Convert decimal {n} to hexadecimal:", "choices": vals, "answer": vals.index(ans)})
    for i in range(20):
        a = rng.randint(250, 999)
        b = rng.randint(250, 999)
        ans = str(a + b)
        vals = [ans, str(a + b + 1), str(a + b - 1), str(abs(a - b))]
        rng.shuffle(vals)
        out.append({"id": f"add_{i}", "family": "addition_ood", "prompt": f"Compute {a} + {b} =", "choices": vals, "answer": vals.index(ans)})
    for i in range(20):
        n = rng.randint(16385, 24000)
        ans = bin(n)[2:]
        vals = [ans, bin(n + 1)[2:], bin(n - 1)[2:], ans[::-1]]
        rng.shuffle(vals)
        out.append({"id": f"bin_{i}", "family": "binary_ood", "prompt": f"Convert decimal {n} to binary:", "choices": vals, "answer": vals.index(ans)})
    return out


def projected_parameter_names(model: nn.Module) -> list[str]:
    preferred = [
        "tok.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.mlp.2.weight",
        "blocks.9.attn.qkv.weight",
        "blocks.9.mlp.2.weight",
        "ln_f.weight",
    ]
    names = {n for n, _ in model.named_parameters()}
    return [n for n in preferred if n in names]


def capture_projected_params(model: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    params = dict(model.named_parameters())
    return {n: params[n].detach().float().cpu().clone() for n in names}


def projected_delta_norm(before: dict[str, torch.Tensor], model: nn.Module) -> float:
    params = dict(model.named_parameters())
    sq = 0.0
    for n, old in before.items():
        d = params[n].detach().float().cpu() - old
        sq += float((d * d).sum().item())
    return math.sqrt(sq)


def full_displacement(model: nn.Module, initial: dict[str, torch.Tensor]) -> float:
    state = model.state_dict()
    sq = 0.0
    for n, old in initial.items():
        d = state[n].detach().float().cpu() - old.float()
        sq += float((d * d).sum().item())
    return math.sqrt(sq)


def projected_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor, names: list[str], device: torch.device) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    with autocast_ctx(device):
        loss = lm_loss(model, x, y)
    loss.backward()
    params = dict(model.named_parameters())
    chunks = []
    for n in names:
        g = params[n].grad
        if g is not None:
            chunks.append(g.detach().float().flatten().cpu())
    model.zero_grad(set_to_none=True)
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    return float(torch.dot(a, b).item() / denom) if denom > 0 else 0.0


def setup_reproducibility() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(0)
    np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def device_now() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("ARK-018 V3 requires CUDA")
    return torch.device("cuda")


def package_versions() -> dict:
    out = {"torch": torch.__version__, "numpy": np.__version__}
    for name in ["pyarrow", "tokenizers", "datasets"]:
        try:
            m = __import__(name)
            out[name] = getattr(m, "__version__", "unknown")
        except Exception:
            out[name] = None
    return out
