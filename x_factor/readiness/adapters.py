"""CoreSubjectAdapter contract (Mission 8/9/11).

Triquetra consumes subjects through this SMALL abstraction. V4 is one
historical adapter; V5 becomes the primary adapter when its checkpoint (and
its own tokenizer/code) arrives. Do NOT duplicate Cymek's model
implementation here and do NOT maintain a subtly different V5 Transformer:
the V5 adapter imports canonical Cymek packages when available and otherwise
fails with actionable guidance.

Tokenizer rule (Mission 11): the subject MUST be evaluated with exactly the
tokenizer it trained with. No V4-32k fallback, no byte fallback, no
approximation. Missing/unverifiable tokenizer identity -> the pipeline
returns READINESS_UNRESOLVED (fail closed).
"""

from __future__ import annotations

from pathlib import Path

import sys as _sys

_XF = Path(__file__).resolve().parents[1]
if str(_XF) not in _sys.path:
    _sys.path.insert(0, str(_XF))

# Expected V5A center (read-only mirror of cymek frozen spec; identity of any
# REAL checkpoint still comes from its own files, never these constants).
V5A_EXPECT = {"vocabulary_size": 24576, "width": 896, "layers": 26,
              "query_heads": 14, "kv_heads": 7, "head_dimension": 64,
              "ffn_width": 2368, "context_length": 4096}
V5_SPECIAL_IDS = {"pad": 0, "unk": 1, "bos": 2, "eos": 3}


def assert_v5_tokenizer_identity(ident: dict) -> dict:
    """Mirror of cymek TokenizerIdentity.assert_valid (read-only contract)."""
    if ident.get("vocabulary_size") != 24576:
        raise ValueError("V5 tokenizer identity must declare 24576 entries")
    if dict(ident.get("special_token_ids", {})) != V5_SPECIAL_IDS:
        raise ValueError("V5 reserves exactly PAD 0, UNK 1, BOS 2, EOS 3")
    for name in ("artifact_sha256", "trainer_config_sha256", "corpus_manifest_sha256"):
        value = ident.get(name, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    return {"tokenizer_verified": True, "vocabulary_size": 24576}


class CoreSubjectAdapter:
    """Minimal subject surface consumed by Triquetra."""

    architecture_family: str = "unknown"

    def model_identity(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def tokenizer_identity(self) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def load_checkpoint(self, path: str, device: str):  # pragma: no cover
        raise NotImplementedError

    def score_candidate_suffixes(self, model, tok, prompt: str,
                                 candidates: list[str], device: str) -> list[float]:
        raise NotImplementedError

    def generate_free(self, model, tok, prompt: str, device: str,
                      max_new: int = 12) -> str:
        raise NotImplementedError

    def generate_constrained(self, model, tok, prompt: str, candidates: list[str],
                             device: str) -> str:
        """Constrained to emit one of the visible candidates (A1)."""
        raise NotImplementedError


class V4Adapter(CoreSubjectAdapter):
    architecture_family = "anra_v4_rope_interleaved_v1"

    def __init__(self, payload: dict):
        self._payload = payload

    def model_identity(self) -> dict:
        mc = self._payload.get("model_config", {})
        return {"family": self.architecture_family, "config": mc,
                "global_step": self._payload.get("global_step")}

    def tokenizer_identity(self) -> dict:
        return {"family": "v4-canonical-32k", "vocabulary_size": 32768,
                "note": "legacy canonical path; V5 subjects require sidecar identity"}

    def load_checkpoint(self, path: str, device: str):
        from checkpoint_identity import load_core

        return load_core(path, device)

    def score_candidate_suffixes(self, model, tok, prompt, candidates, device):
        import torch

        out = []
        with torch.no_grad():
            p_ids = tok.encode(prompt)
            for cand in candidates:
                c_ids = tok.encode(f" {cand}.")
                ids = torch.tensor([[tok.bos_token_id, *p_ids, *c_ids]],
                                   dtype=torch.long, device=device)
                lp = torch.log_softmax(model(ids)[0].float(), -1)
                out.append(sum(float(lp[pos - 1, ids[0, pos]])
                               for pos in range(1 + len(p_ids), ids.shape[1])))
        return out

    def generate_free(self, model, tok, prompt, device, max_new: int = 12) -> str:
        import torch

        ids = [tok.bos_token_id, *tok.encode(prompt)]
        cur, out = list(ids), []
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(torch.tensor([cur], dtype=torch.long, device=device))[:, -1, :]
                nxt = int(logits.argmax(dim=-1))
                if nxt == tok.eos_token_id:
                    break
                out.append(nxt)
                cur.append(nxt)
        return tok.decode(out)

    def generate_constrained(self, model, tok, prompt, candidates, device) -> str:
        scores = self.score_candidate_suffixes(model, tok, prompt, candidates, device)
        return candidates[int(__import__("numpy").argmax(scores))]


class V5Adapter(CoreSubjectAdapter):
    """Primary future adapter. Requires canonical Cymek packages + sidecar identity."""

    architecture_family = "anra_v5_dense_decoder_v1"

    def __init__(self, cymek_root: str | None = None,
                 tokenizer_sidecar: dict | None = None):
        self._root = cymek_root
        self._sidecar = tokenizer_sidecar

    def _cymek(self, module: str):
        import importlib

        if self._root and self._root not in _sys.path:
            _sys.path.insert(0, self._root)
        try:
            return importlib.import_module(module)
        except ImportError as e:
            raise UnsupportedSubject(
                f"V5 runtime package '{module}' not importable. Provide the canonical "
                f"cymek tree (cymek_root) — Triquetra will not vendor a copy. ({e})")

    def model_identity(self) -> dict:
        ms = self._cymek("v5_contracts.model_spec")
        return {"family": self.architecture_family,
                "spec": "V5A_250M mirror; real identity bound per-checkpoint at load"}

    def tokenizer_identity(self) -> dict:
        if self._sidecar is None:
            raise UnsupportedSubject(
                "V5 tokenizer sidecar identity missing: refusing V4-32k fallback. "
                "Supply artifact/trainer-config/corpus-manifest SHAs (READINESS_UNRESOLVED until then).")
        return assert_v5_tokenizer_identity(self._sidecar)

    def load_checkpoint(self, path: str, device: str):
        self.tokenizer_identity()  # fail first on tokenizer, before weights
        core = self._cymek("v5_model.core")
        ms = self._cymek("v5_contracts.model_spec")
        raise UnsupportedSubject(
            "V5 weight-format binding not yet implemented: register identity now, "
            "implement loader against the arrived checkpoint's exact format "
            f"(spec module: {ms.__name__}, core module: {core.__name__}).")


class UnsupportedSubject(ValueError):
    """Subject cannot be evaluated yet (missing runtime/sidecar/format binding).

    Distinct from BAD_CHECKPOINT (corrupt file) and UNSUPPORTED_ARCHITECTURE
    (config-level rejection): the instrument is waiting on subject-side inputs.
    """
