"""CPU FP32 reference path composed from Cymek model and training contracts."""

from collections.abc import Sequence
from dataclasses import asdict, replace
from hashlib import sha256
import io
from pathlib import Path
import platform
import random

import torch

from v5_contracts.model_spec import V5A_250M
from v5_model.core import initialize, packed_layout
from v5_objectives.causal_lm import causal_lm_loss
from v5_training.checkpoint import CheckpointStore
from v5_training.optimizer import build_adamw_optimizer, optimizer_group_receipt
from v5_training.state import CURSOR_SCHEMA, IDENTITY_SCHEMA, CursorState, IdentityBindings, TrainingState

from .benchmark import Example
from .codec import BOS, CODEC_SHA, EOS, PAD, VOCAB_SIZE, Prediction, answer_tokens, canonical, digest, encode
from .publication import quarantine_interrupted_publication, writer_lock


MODEL = replace(V5A_250M, vocabulary_size=VOCAB_SIZE, width=48, layers=2,
                query_heads=3, kv_heads=1, head_dimension=16, ffn_width=96, context_length=32)
BATCH_SIZE = 16
LR = 0.003
CLIP_NORM = 1.0
MAX_STEPS = 400
SCHEDULE = {"kind": "constant", "lr": LR, "clip_norm": CLIP_NORM}
SAMPLER = {"kind": "sample-without-replacement-per-update", "batch_size": BATCH_SIZE,
           "seed": "SHA256 of run seed and zero-based update"}


def runtime_identity() -> dict:
    return {"torch": str(torch.__version__), "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "intraop_threads": torch.get_num_threads(), "dtype": str(torch.get_default_dtype()),
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def configure_cpu() -> None:
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)


def data_sha(rows: Sequence[Example]) -> str:
    return digest([(asdict(row.visible), asdict(row.truth)) for row in rows])


def serialize(row: Example) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    prompt = (BOS, *encode(row.visible.prompt))
    answer = answer_tokens(row.truth.gold)
    return prompt + answer, (False,) * len(prompt) + (True,) * len(answer)


def batch_tensors(samples: Sequence[tuple[tuple[int, ...], tuple[bool, ...]]]) -> tuple:
    length = max(len(tokens) for tokens, _ in samples)
    padding = [length - len(tokens) for tokens, _ in samples]
    tokens = torch.tensor([ids + (PAD,) * n for (ids, _), n in zip(samples, padding)], device="cpu")
    segments = torch.tensor([(0,) * len(ids) + (-1,) * n
                             for (ids, _), n in zip(samples, padding)], device="cpu")
    eligible = torch.tensor([mask + (False,) * n for (_, mask), n in zip(samples, padding)], device="cpu")
    return tokens, segments, eligible


def logits_for(model, tokens: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    positions, mask = packed_layout(segments, torch_module=torch)
    return model(tokens, positions, mask)


def _dump(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load(payload: bytes):
    return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)


class Trainer:
    """One CPU writer and an index-addressed sampler. Restore creates a fresh instance."""

    def __init__(self, rows: Sequence[Example], *, seed: int, steps: int,
                 root: Path, lineage: str, source_commit: str, resume: bool = False):
        if type(steps) is not int or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be between 1 and {MAX_STEPS}")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if (torch.get_num_threads() != 1 or not torch.are_deterministic_algorithms_enabled()
                or torch.get_default_dtype() != torch.float32):
            raise ValueError("configure_cpu() is required before training")
        if not BATCH_SIZE <= len(rows) <= 1024:
            raise ValueError("training fixture size outside CPU reference bounds")
        if any(row.visible.split != "train" or row.visible.task_id != row.truth.task_id for row in rows):
            raise ValueError("trainer accepts only joined training rows")
        if len({row.visible.task_id for row in rows}) != len(rows):
            raise ValueError("duplicate training task ID")
        self.samples = tuple(serialize(row) for row in rows)
        lengths = {len(ids) - 1 for ids, _ in self.samples}  # All non-BOS, non-padding tokens.
        if len(lengths) != 1 or max(lengths) >= MODEL.context_length:
            raise ValueError("reference updates require equal unpadded sequence lengths within context")
        self.seed = seed
        self.model = initialize(MODEL, seed).cpu().float()
        self.optimizer = build_adamw_optimizer(self.model, lr=LR)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        identity = data_sha(rows)
        tokens_per_update = BATCH_SIZE * lengths.pop()
        run_sha = digest({"seed": seed, "steps": steps, "runtime": runtime_identity()})
        bindings = IdentityBindings(
            schema=IDENTITY_SCHEMA, source_commit=source_commit, model_spec_sha256=MODEL.sha256(),
            tokenizer_sha256=CODEC_SHA, data_manifest_sha256=identity, pack_manifest_sha256=identity,
            run_spec_sha256=run_sha, optimizer_spec_sha256=optimizer_group_receipt(
                self.model, self.optimizer)["sha256"],
            schedule_spec_sha256=digest(SCHEDULE), curriculum_spec_sha256=digest("single-stage"),
            sampler_spec_sha256=digest(SAMPLER),
        )
        self.store = CheckpointStore(root, lineage)
        self.parent = self.store.latest_sha256()
        self.faulted = False
        self.published_update = 0
        self.state = TrainingState.initial(
            lineage_id=lineage, token_budget=steps * tokens_per_update, tokens_per_update=tokens_per_update,
            cursor=CursorState(CURSOR_SCHEMA, identity, 0, 0, 0),
            rng_state_sha256=sha256(self._rng_bytes()).hexdigest(), curriculum_phase="development",
            identities=bindings,
        )
        if resume:
            self._restore()
        elif self.parent is not None:
            raise ValueError("checkpoint lineage already exists; explicit resume required")

    def _rng_bytes(self) -> bytes:
        return bytes(self.rng.get_state().tolist())

    def _restore(self) -> None:
        if self.parent is None:
            raise ValueError("no committed checkpoint exists")
        state, payloads = self.store.restore(self.parent)
        if state.identities != self.state.identities:
            raise ValueError("checkpoint identity mismatch")
        expected_cursor = replace(self.state.cursor, sequence_ordinal=BATCH_SIZE * state.global_update)
        if (state.cursor != expected_cursor or state.tokens_per_update != self.state.tokens_per_update
                or state.token_budget != self.state.token_budget
                or state.cumulative_tokens != state.global_update * state.tokens_per_update
                or payloads["scheduler.json"] != canonical(SCHEDULE)
                or sha256(payloads["rng.bin"]).hexdigest() != state.rng_state_sha256):
            raise ValueError("checkpoint continuation metadata mismatch")
        self.model.load_state_dict(_load(payloads["model.bin"]), strict=True)
        self.optimizer.load_state_dict(_load(payloads["optimizer.bin"]))
        self._verify_optimizer_step(state.global_update)
        self.rng.set_state(torch.tensor(list(payloads["rng.bin"]), dtype=torch.uint8, device="cpu"))
        self.state = state
        self.published_update = state.global_update

    def _verify_optimizer_step(self, update: int) -> None:
        if (len(self.optimizer.state) != len(list(self.model.parameters()))
                or any(value["step"].item() != update for value in self.optimizer.state.values())):
            raise ValueError("optimizer counter does not match the completed update")

    def sample_indices(self) -> tuple[int, ...]:
        rng = random.Random(int(digest([self.seed, self.state.global_update]), 16))
        return tuple(rng.sample(range(len(self.samples)), BATCH_SIZE))

    def step(self) -> dict:
        if self.state.complete or self.faulted:
            raise ValueError("cannot step a complete or failed trainer")
        # An exception after optimizer mutation requires restore, never a same-instance retry.
        self.faulted = True
        self.model.train()
        indices = self.sample_indices()
        tokens, segments, eligible = batch_tensors([self.samples[index] for index in indices])
        self.optimizer.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.rng.get_state())
            logits = logits_for(self.model, tokens, segments)
            loss, supervised = causal_lm_loss(
                logits, tokens, segments, bos_id=BOS, pad_id=PAD, eligible=eligible)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("nonfinite logits or loss")
            loss.backward()
            self.rng.set_state(torch.get_rng_state())
        pre_clip = torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_NORM, error_if_nonfinite=True)
        post_clip = torch.linalg.vector_norm(torch.stack([
            parameter.grad.norm() for parameter in self.model.parameters()
        ]))
        if post_clip.item() > CLIP_NORM + 1e-6:
            raise FloatingPointError("post-clip gradient exceeds the frozen bound")
        self.optimizer.step()
        update = self.state.global_update + 1
        self._verify_optimizer_step(update)
        if any(not torch.isfinite(parameter).all() for parameter in self.model.parameters()):
            raise FloatingPointError("nonfinite parameter after update")
        self.state = self.state.advance(
            tokens_by_source={"development": int((segments >= 0).sum()) - BATCH_SIZE},
            cursor=replace(self.state.cursor, sequence_ordinal=BATCH_SIZE * update),
            rng_state_sha256=sha256(self._rng_bytes()).hexdigest(), parent_checkpoint_sha256=self.parent,
        )
        self.faulted = False
        return {"update": update, "loss": loss.item(), "lr": LR, "supervised_tokens": supervised,
                "pre_clip": pre_clip.item(), "post_clip": post_clip.item(), "indices": indices}

    def save(self) -> str:
        if self.faulted or self.state.global_update == 0:
            raise ValueError("cannot publish without a successful update")
        if self.published_update >= self.state.global_update:
            raise ValueError("cannot publish the same update twice")
        state = replace(self.state, parent_checkpoint_sha256=self.parent)
        payloads = {
            "model.bin": _dump(self.model.state_dict()),
            "optimizer.bin": _dump(self.optimizer.state_dict()),
            "rng.bin": self._rng_bytes(),
            "scheduler.json": canonical(SCHEDULE),
            "cursor.json": canonical(asdict(state.cursor)),
            "ledger.json": canonical(dict(state.tokens_by_source)),
            "training_state.json": canonical(state.canonical()),
        }
        with writer_lock(self.store.lineage_root):
            if self.store.latest_sha256() != self.parent:
                raise ValueError("writer fence rejected stale parent")
            quarantine_interrupted_publication(self.store.lineage_root, state.generation)
            self.parent = self.store.publish(
                state=state, payloads=payloads, expected_parent_sha256=self.parent)
        self.published_update = state.global_update
        self.state = state
        return self.parent


@torch.inference_mode()
def generate(model, prompts: Sequence[tuple[str, str]], *, max_new_tokens: int) -> tuple[tuple[Prediction, ...], dict]:
    """Greedy autoregression from prompt strings only, with a per-row context bound."""
    if not 1 <= max_new_tokens <= MODEL.context_length or not prompts:
        raise ValueError("generation requires prompts and a positive bounded token limit")
    model.eval()
    predictions = []
    forwards = evaluated_positions = 0
    for task_id, prompt in prompts:
        ids = [BOS, *encode(prompt)]
        if len(ids) + max_new_tokens - 1 > MODEL.context_length:
            raise ValueError("generation exceeds native context")
        generated = []
        for _ in range(max_new_tokens):
            tokens = torch.tensor([ids], device="cpu")
            logits = logits_for(model, tokens, torch.zeros_like(tokens))
            if not torch.isfinite(logits[0, -1]).all():
                raise FloatingPointError("nonfinite inference logits")
            token = int(logits[0, -1].argmax())
            forwards += 1
            evaluated_positions += len(ids)
            generated.append(token)
            if token == EOS:
                break
            ids.append(token)
        predictions.append(Prediction(task_id, tuple(generated)))
    return tuple(predictions), {"forward_calls": forwards, "evaluated_positions": evaluated_positions}
