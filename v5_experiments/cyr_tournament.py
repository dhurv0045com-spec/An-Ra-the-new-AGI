"""CYR-GPU-001 tournament runner (version-controlled science, thin notebook).

Conventions (frozen for this campaign):
- Records render answer-LAST: text = context + query + "Answer: " + answer.
- Every record encodes to EXACTLY row_width-2 content tokens (renderer pads
  with extra in-distribution distractor facts to an exact size, fail-closed
  otherwise), so every packed row is exactly full: token accounting is exact
  everywhere, no partial tails, no padding ambiguity.
- Single-segment rows; the treatment IS batch assembly, so production
  multi-segment packing is deliberately not used here (proven elsewhere).
  Objective math, backend certification, training state, and checkpoints are
  the REAL Cymek paths at proxy scale (backend ctx API + train() + store).
- Research LR schedules are explicit isolated variants, never canonical.

Modes: --mode smoke (default; tiny CPU-safe plumbing check) vs --mode full
(refuses unless COLAB_GPU=1 or --allow-non-colab, never used locally).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


RUNNER_SCHEMA = "anra-cyr-tournament/v1"
WORLD_VERSION = "cyr-worlds/v1"
RENDER_SUFFIX = "Answer: "

RESEARCH_LRS = {"HIGH": 3e-4, "MID": 3e-5, "LOW": 3e-6}

BUNDLE_FILES = ("SESSION_MANIFEST.json", "ENVIRONMENT.json",
                "RESOLVED_PREREGISTRATION.json", "DATA_MANIFEST.json",
                "SPLIT_MANIFEST.json", "SMOKE.json")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# -- proxy ladder (frozen research specs; resolver picks at runtime) --------

def proxy_ladder() -> dict[str, dict[str, Any]]:
    return {
        "TINY": {"layers": 2, "width": 64, "query_heads": 4, "kv_heads": 2,
                 "head_dimension": 16, "ffn_width": 128, "context_length": 512,
                 "purpose": "smoke only"},
        "MICRO": {"layers": 4, "width": 256, "query_heads": 4, "kv_heads": 2,
                  "head_dimension": 64, "ffn_width": 512, "context_length": 512,
                  "purpose": "fast arms"},
        "MIDI": {"layers": 8, "width": 384, "query_heads": 6, "kv_heads": 3,
                 "head_dimension": 64, "ffn_width": 1024, "context_length": 512,
                 "purpose": "main tournament"},
        "P35": {"layers": 16, "width": 384, "query_heads": 6, "kv_heads": 3,
                "head_dimension": 64, "ffn_width": 1024, "context_length": 512,
                "purpose": "scale transfer only"},
    }


def resolve_proxy(*, micro_tokens_per_sec: float, free_vram_gb: float,
                  smoke: bool = False) -> dict[str, Any]:
    """Preregistered hardware-only budget resolver (throughput/VRAM in).

    Never sees accuracy, loss, or arm performance. Picks the largest proxy
    that leaves headroom plus per-arm token budgets from a frozen table.
    """

    if smoke:
        return {"proxy": "TINY", "tokens_per_arm": 4096, "reason": "smoke mode"}
    if micro_tokens_per_sec <= 0 or free_vram_gb < 0:
        raise ValueError("calibration produced non-positive hardware figures")
    if micro_tokens_per_sec >= 20000 and free_vram_gb >= 10:
        return {"proxy": "MIDI", "tokens_per_arm": 400_000,
                "reason": "fast hardware: MIDI main budget"}
    if micro_tokens_per_sec >= 5000 and free_vram_gb >= 4:
        return {"proxy": "MICRO", "tokens_per_arm": 150_000,
                "reason": "standard hardware: MICRO main budget"}
    return {"proxy": "MICRO", "tokens_per_arm": 60_000,
            "reason": "slow hardware: reduced MICRO budget, replication kept"}


# -- research LR schedules (explicit variants, never canonical) --------------

def constant_lr(value: float) -> Callable[[int], float]:
    if value <= 0:
        raise ValueError("research LR must be positive")

    def schedule(cumulative_tokens: int) -> float:
        if cumulative_tokens < 0:
            raise ValueError("cumulative tokens cannot be negative")
        return float(value)

    schedule.__name__ = f"research-constant-{value}"
    return schedule


# -- task worlds (self-contained renderer; no cognition-stack import) --------

_DISTRACTOR_POOL = ("Extra registry entry {k} carries payload {v}.",
                    "Archived revision notes {k} as {v}.",
                    "Transfer log shows {k} reaching {v}.")

_UNIT_CANDIDATES = ("x", "a", "z", "q", "0", "7", ".", "-", "qx")


def _append_unit(text: str, encode: Callable[[str], list[int]]) -> str:
    """Append exactly one token to live text (candidate probe, fail-closed).

    Merges are context-dependent, so candidates are probed against the
    actual current tail, not a fixed probe string.
    """

    current = len(encode(text))
    for candidate in _UNIT_CANDIDATES:
        if len(encode(text + candidate)) == current + 1:
            return text + candidate
    raise ValueError("no single-token extension for this tokenizer tail")


def render_worlds(*, family: str, split_seeds: Mapping[str, int],
                  worlds_per_split: int | dict[str, int],
                  encode: Callable[[str], list[int]] | None = None,
                  row_content_tokens: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """Render counterfactual-paired worlds per split with zero world overlap.

    Registry (binding/retrieval) and transfer (compositional hops) modes.
    Each world contributes a base record and its counterfactual twin sharing
    the context. Splits use disjoint seeds; world ids (hence hashes) never
    cross splits. With encode+row_content_tokens, every record's FULL text
    (content + answer, excl. BOS/EOS) pads with in-distribution distractor
    facts to EXACTLY row_content_tokens (fail-closed otherwise), so
    downstream rows are exactly full.
    """

    if family not in ("registry", "transfer"):
        raise ValueError("tournament families are registry and transfer")
    counts = (worlds_per_split if isinstance(worlds_per_split, dict)
              else {split: worlds_per_split for split in split_seeds})
    if set(counts) != set(split_seeds):
        raise ValueError("per-split world counts must cover every split")
    if any(count <= 0 for count in counts.values()):
        raise ValueError("worlds per split must be positive")
    if len(set(split_seeds.values())) != len(split_seeds):
        raise ValueError("split seeds must be distinct")
    if (encode is None) != (row_content_tokens is None):
        raise ValueError("encode and row_content_tokens go together")
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, seed in split_seeds.items():
        rng = random.Random(seed)
        worlds = []
        for index in range(counts[split]):
            keys = [f"k{seed}x{index}-{j}" for j in range(4)]
            values = [f"V{seed}{index}{j}" for j in range(4)]
            target = rng.randrange(4)
            if family == "registry":
                facts = [f"Registry entry {key} carries payload {value}."
                         for key, value in zip(keys, values)]
                query = f"Return the payload assigned to registry entry {keys[target]}."
                answer = values[target]
                twin_query = (f"Return the payload assigned to registry entry "
                              f"{keys[(target + 1) % 4]}.")
                twin_answer = values[(target + 1) % 4]
            else:
                facts = [f"{keys[0]} transfers to {keys[1]}.",
                         f"{keys[1]} transfers to {keys[2]}.",
                         f"{keys[2]} stores payload {values[2]}.",
                         f"Unrelated {keys[3]} stores payload {values[3]}."]
                query = (f"Which payload is reached by following transfers "
                         f"from {keys[0]}?")
                answer = values[2]
                twin_query = f"Which payload is stored by unrelated {keys[3]}?"
                twin_answer = values[3]

            def render(extra: list[str], ask: str, ans: str, tag: str) -> dict[str, str]:
                # Units tune the pre-answer content only: the answer itself
                # is never corrupted. A final exact check guards BPE merges
                # across the unit/answer boundary.
                body = list(facts + extra)
                context = "\n".join(body) + "\n" + ask
                framing = lambda parts: "\n".join(parts) + "\n" + ask + "\n" + RENDER_SUFFIX
                if encode is not None and row_content_tokens is not None:
                    answer_len = len(encode(ans))
                    target_prefix = row_content_tokens - answer_len
                    if len(encode(framing(body))) > target_prefix:
                        raise ValueError(f"record core exceeds exact size: {tag}")
                    pad_index = 0
                    while len(encode(framing(body))) < target_prefix:
                        body.append(
                            f"Supplemental log {tag}-{pad_index} cites "
                            f"{keys[pad_index % 4]}.")
                        pad_index += 1
                        if pad_index > 2000:
                            raise ValueError(f"cannot bulk-size {tag}")
                    while len(encode(framing(body))) > target_prefix:
                        if not body[-1].startswith("Supplemental"):
                            raise ValueError(f"core exceeds exact size: {tag}")
                        body.pop()
                    text = framing(body)
                    guard = 0
                    while len(encode(text)) < target_prefix:
                        text = _append_unit(text, encode)
                        guard += 1
                        if guard > 64:
                            raise ValueError(f"cannot unit-size {tag}")
                    if len(encode(text)) != target_prefix:
                        raise ValueError(f"inexact sizing for {tag}")
                    text = text + ans
                    if len(encode(text)) != row_content_tokens:
                        raise ValueError(f"answer-boundary merge in {tag}")
                else:
                    text = f"{chr(10).join(body)}\n{ask}\n{RENDER_SUFFIX}{ans}"
                    context = "\n".join(body) + "\n" + ask
                return {"text": text, "answer": ans, "context": context,
                        "query": ask}

            worlds.append({
                "world_id": f"{family}/{split}/{seed}/{index}",
                "facts": list(facts),
                "query": query,
                "twin_query": twin_query,
                "base": render([], query, answer, f"{split}-{index}-b"),
                "twin": render([], twin_query, twin_answer, f"{split}-{index}-t"),
            })
        splits[split] = worlds
    return splits


def split_manifest(splits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    body = {"schema": "anra-cyr-split-manifest/v1",
            "world_version": WORLD_VERSION,
            "splits": {name: [world["world_id"] for world in worlds]
                       for name, worlds in splits.items()}}
    body["sha256"] = _sha256_hex(_canonical_json(body))
    return body


def assert_split_firewall(splits: Mapping[str, list[dict[str, Any]]]) -> None:
    seen: dict[str, str] = {}
    for name, worlds in splits.items():
        for world in worlds:
            wid = world["world_id"]
            if wid in seen:
                raise ValueError(f"latent world {wid} in splits {seen[wid]} and {name}")
            seen[wid] = name


def pair_batches(worlds: list[dict[str, Any]], *, group_pairs: bool,
                 seed: int) -> list[dict[str, str]]:
    """Order records for microbatching; treatment keeps twins adjacent.

    Control shuffles the same multiset (same records, same counts).
    Deterministic in seed. Microbatches slice consecutive records at even
    boundaries with an even row count, so grouped twins share a microbatch
    guaranteed (asserted downstream; pair_split_rate is receipted).
    """

    records = []
    for world in worlds:
        records.append({**world["base"], "world_id": world["world_id"]})
        records.append({**world["twin"], "world_id": world["world_id"]})
    rng = random.Random(seed)
    if group_pairs:
        order = list(range(len(worlds)))
        rng.shuffle(order)
        batched = []
        for i in order:
            batched.append(records[2 * i])
            batched.append(records[2 * i + 1])
        for position in range(0, len(batched), 2):
            assert batched[position]["world_id"] == batched[position + 1]["world_id"]
        return batched
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled


# -- metrics -----------

def sustained(flags: list[bool], *, required: int = 3) -> bool:
    """True iff the trailing `required` evaluations all pass (no cherry-picks)."""

    return len(flags) >= required and all(flags[-required:])


def heuristic_baselines() -> dict[str, Callable[[str, str], str]]:
    """Trivial deterministic policies a model must beat to claim query control."""

    def copy_first(context: str, query: str) -> str:
        del query
        line = next((line for line in context.split("\n") if line.strip()), "")
        return line.split()[-1] if line.split() else ""

    def copy_last(context: str, query: str) -> str:
        del query
        words = context.split()
        return words[-1] if words else ""

    def most_frequent(context: str, query: str) -> str:
        del query
        counts: dict[str, int] = {}
        for word in context.split():
            counts[word] = counts.get(word, 0) + 1
        return max(counts, key=lambda word: (counts[word], word)) if counts else ""

    def fixed_position(context: str, query: str) -> str:
        del query
        words = context.split()
        return words[1] if len(words) > 1 else (words[0] if words else "")

    return {"copy_first": copy_first, "copy_last": copy_last,
            "most_frequent": most_frequent, "fixed_position": fixed_position}


# -- hysteretic controller (preregistered thresholds, receipted decisions) ----

@dataclass
class HysteresisController:
    enter_retention: float
    reenter_plasticity: float
    confirmations: int
    mode: str = "plasticity"
    streak: int = 0
    decisions: tuple = ()

    def assert_valid(self) -> None:
        if not self.reenter_plasticity < self.enter_retention:
            raise ValueError("hysteresis requires reenter < enter thresholds")
        if self.confirmations <= 0:
            raise ValueError("confirmations must be positive")
        if self.mode not in ("plasticity", "retention"):
            raise ValueError("unknown controller mode")

    def observe(self, *, metric: float, threshold_note: str,
                token_position: int, lr_before: float,
                lr_plasticity: float, lr_retention: float) -> dict[str, Any]:
        """Feed one dev-controller measurement; returns the decision receipt."""

        self.assert_valid()
        if not 0.0 <= metric <= 1.0:
            raise ValueError("controller metric must be a rate in [0, 1]")
        hit = (metric >= self.enter_retention if self.mode == "plasticity"
               else metric < self.reenter_plasticity)
        self.streak = self.streak + 1 if hit else 0
        before = self.mode
        if self.streak >= self.confirmations:
            self.mode = "retention" if before == "plasticity" else "plasticity"
            self.streak = 0
        lr_after = lr_retention if self.mode == "retention" else lr_plasticity
        receipt = {"observed_metric": metric, "threshold_note": threshold_note,
                   "state_before": before, "state_after": self.mode,
                   "lr_before": lr_before, "lr_after": lr_after,
                   "token_position": token_position,
                   "controller_split": "dev-controller only"}
        self.decisions = self.decisions + (receipt,)
        return receipt


# -- red team (deterministic; failures veto headlines) ------------------------

def redteam_overlap(train_ids: list[str], eval_ids: list[str]) -> dict[str, Any]:
    overlap = sorted(set(train_ids) & set(eval_ids))
    return {"check": "latent_world_overlap", "pass": not overlap,
            "overlap_count": len(overlap)}


def redteam_near_dup(texts: list[str], *, threshold: float = 0.80) -> dict[str, Any]:
    from v5_data.near_dedup import cluster_near_duplicates
    clusters = cluster_near_duplicates({str(i): text for i, text in enumerate(texts)},
                                       threshold=threshold)
    multi = [c for c in clusters if len(c.members) > 1]
    return {"check": "near_duplicates", "pass": not multi,
            "cluster_count": len(multi)}


def redteam_exposure(arms: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    equal = len({counts["tokens"] for counts in arms.values()}) == 1 and \
        len({counts["updates"] for counts in arms.values()}) == 1
    return {"check": "equal_exposure", "pass": equal,
            "detail": {arm: dict(counts) for arm, counts in arms.items()}}


def redteam_freezing(displacement_high: float, displacement_low: float,
                     *, ratio_floor: float = 5.0) -> dict[str, Any]:
    if displacement_low <= 0:
        return {"check": "freezing", "pass": False,
                "reason": "low arm did not move"}
    ratio = displacement_high / displacement_low
    return {"check": "freezing", "pass": ratio >= ratio_floor,
            "displacement_ratio": ratio}


def redteam_non_mutation(before_sha: str, after_sha: str) -> dict[str, Any]:
    return {"check": "parameter_mutation", "pass": before_sha != after_sha}


# -- environment ---------------------------------------------------------------

def discover_environment() -> dict[str, Any]:
    env: dict[str, Any] = {"schema": "anra-cyr-environment/v1",
                           "colab_gpu": os.environ.get("COLAB_GPU", "0") == "1"}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_model"] = props.name
            env["vram_gb"] = round(props.total_memory / (1 << 30), 2)
            env["free_vram_gb"] = round(
                (props.total_memory - torch.cuda.memory_reserved(0)) / (1 << 30), 2)
            env["bf16"] = bool(props.major >= 8)
        try:
            import psutil
            env["host_ram_gb"] = round(psutil.virtual_memory().total / (1 << 30), 2)
        except ImportError:
            env["host_ram_gb"] = "unknown"
    except ImportError as exc:
        env["torch"] = f"missing: {exc}"
    return env


def package_bundle(out: Path, *, manifest_extra: Mapping[str, Any] | None = None,
                   zip_name: str = "CYMEK_GPU_RESEARCH_RESULTS.zip") -> Path:
    """Zip the result bundle for the operator to return."""

    manifest = {"schema": "anra-cyr-session-manifest/v1",
                "files": sorted(path.name for path in out.iterdir()
                                if path.is_file() and path.suffix == ".json")}
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    (out / "SESSION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    archive = out / zip_name
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.iterdir()):
            if path.is_file() and path.name != zip_name:
                handle.write(path, path.name)
    return archive


def guard_full_mode(args: Any) -> None:
    """Refuse full campaigns outside Colab (override exists, never use locally)."""

    if args.mode == "full" and os.environ.get("COLAB_GPU", "0") != "1" \
            and not args.allow_non_colab:
        raise ValueError(
            "full campaign refuses non-Colab execution: set COLAB_GPU=1 on "
            "the Colab GPU runtime or pass --allow-non-colab explicitly "
            "(never use the override for local science)")


# -- frozen CYR-GPU-001 campaign constants (code and PLAN.md read these) -----

CYR_ID = "CYR-GPU-001"
CYR_WALL_MIN_MINUTES = 60
CYR_WALL_TARGET_MINUTES = (90, 150)
CYR_WALL_HARD_MINUTES = 180
CYR_SEEDS = (101, 202)
CYR_SPLIT_SEEDS = {"train": 1001, "dev_controller": 2002,
                   "dev_measurement": 3003, "sealed_reserved": 4004}
CYR_DEV_WORLDS = 64
CYR_EVAL_DENSE_WORLDS = 64
CYR_EVAL_BATCH = 32
CYR_FREE_GEN_WORLDS = 8
CYR_FREE_GEN_CAP = 16
CYR_ROW_CONTENT_TOKENS = 300
CYR_MICROBATCH_ROWS = 8
CYR_EVAL_EVERY_UPDATES = 10
CYR_SUSTAINED_REQUIRED = 3
CYR_ENTER_RETENTION = 0.90
CYR_REENTER_GAP = 0.15
CYR_FIXED_SWITCH_FRACTION = 0.5


__all__ = ["BUNDLE_FILES", "CYR_ENTER_RETENTION",
           "CYR_DEV_WORLDS", "CYR_EVAL_BATCH", "CYR_EVAL_DENSE_WORLDS",
           "CYR_EVAL_EVERY_UPDATES", "CYR_FIXED_SWITCH_FRACTION", "CYR_FREE_GEN_CAP",
           "CYR_FREE_GEN_WORLDS", "CYR_ID",
           "CYR_MICROBATCH_ROWS", "CYR_REENTER_GAP", "CYR_ROW_CONTENT_TOKENS",
           "CYR_SEEDS", "CYR_SPLIT_SEEDS", "CYR_SUSTAINED_REQUIRED",
           "CYR_WALL_HARD_MINUTES", "CYR_WALL_MIN_MINUTES",
           "CYR_WALL_TARGET_MINUTES",
           "HysteresisController", "RUNNER_SCHEMA",
           "WORLD_VERSION", "assert_split_firewall",
           "constant_lr", "discover_environment",
           "guard_full_mode", "heuristic_baselines",
           "package_bundle", "pair_batches", "proxy_ladder",
           "redteam_exposure", "redteam_freezing", "redteam_near_dup",
           "redteam_non_mutation", "redteam_overlap", "render_worlds",
           "resolve_proxy", "split_manifest", "sustained"]
