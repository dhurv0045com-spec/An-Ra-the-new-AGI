"""CYR-GPU-002 tournament primitives (pure logic; no torch, no training).

Version-controlled science core: shared-context world rendering,
orthogonal factorial variants, task-aware baselines, generation-report
scoring, learnability transitions, dose/wall resolver, hysteretic
controller with persisted history, red-team gates, packaging. Everything
here is deterministic and unit-tested without models or GPUs.
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


RUNNER_SCHEMA = "anra-cyr-tournament/v2"
WORLD_VERSION = "cyr-worlds/v2"
RENDER_SUFFIX = "Answer: "

RESEARCH_LRS = {"HIGH": 3e-4, "MID": 3e-5, "LOW": 3e-6}

BUNDLE_FILES = ("SESSION_MANIFEST.json", "ENVIRONMENT.json",
                "PREREGISTRATION.json", "RESOLVED_PREREGISTRATION.json",
                "CALIBRATION.json", "DATA_MANIFEST.json", "SPLIT_MANIFEST.json",
                "SMOKE.json", "NEGATIVE_CONTROL_TESTS.json")


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
        "RESEARCH_SMALL": {"layers": 4, "width": 128, "query_heads": 4,
                           "kv_heads": 2, "head_dimension": 32,
                           "ffn_width": 512, "context_length": 512,
                           "purpose": "fast arms"},
        "MICRO": {"layers": 4, "width": 256, "query_heads": 4, "kv_heads": 2,
                  "head_dimension": 64, "ffn_width": 512, "context_length": 512,
                  "purpose": "standard arms"},
        "MIDI": {"layers": 8, "width": 384, "query_heads": 6, "kv_heads": 3,
                 "head_dimension": 64, "ffn_width": 1024, "context_length": 512,
                 "purpose": "main tournament"},
        "P35": {"layers": 16, "width": 384, "query_heads": 6, "kv_heads": 3,
                "head_dimension": 64, "ffn_width": 1024, "context_length": 512,
                "purpose": "scale transfer only"},
    }


def proxy_spec_kwargs(proxy: Mapping[str, Any], *, vocab_size: int) -> dict[str, Any]:
    """Validated ModelSpec kwargs for a ladder entry (no model built)."""

    from v5_contracts.model_spec import ModelSpec
    spec = ModelSpec(
        schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
        vocabulary_size=int(vocab_size), width=int(proxy["width"]),
        layers=int(proxy["layers"]), query_heads=int(proxy["query_heads"]),
        kv_heads=int(proxy["kv_heads"]),
        head_dimension=int(proxy["head_dimension"]),
        ffn_width=int(proxy["ffn_width"]),
        context_length=int(proxy["context_length"]),
        rope_base=10_000.0, norm_epsilon=1e-5, tied_embeddings=True,
        qk_norm=True, qk_norm_affine=True, linear_bias=False, dropout=0.0)
    spec.assert_valid()
    return {"spec": spec, "parameters": spec.parameter_receipt().total}


# -- dose + wall-time resolver (hardware figures in, never outcomes) ---------

MIN_ACQUISITION_DOSE_TOKENS = 2_000_000
TARGET_ACQUISITION_DOSE_TOKENS = 4_000_000
TRAINING_WALL_MINUTES = 115.0
STAGE_FRACTIONS = {"acquisition": 0.30, "pair_query": 0.25,
                   "lr_retention": 0.30, "transfer": 0.15}


def resolve_campaign(*, tokens_per_sec: float, free_vram_gb: float,
                     microbatch_tokens: int,
                     smoke: bool = False) -> dict[str, Any]:
    """Preregistered hardware-only campaign resolver.

    Inputs: measured throughput, free VRAM. Outputs: proxy, per-arm
    updates, stage token splits. Never sees accuracy, loss, or curves.
    Policy: largest proxy that affords the dose floor inside the wall
    budget; proxy-downshift before replication cuts; replication cuts in
    preregistered priority order (transfer second seed, then MID arm).
    """

    if smoke:
        return {"proxy": "TINY", "updates_per_arm": 2, "tokens_per_arm": 4096,
                "reason": "smoke mode", "dose_floor_met": False,
                "dropped": []}
    if tokens_per_sec <= 0 or free_vram_gb < 0:
        raise ValueError("calibration produced non-positive hardware figures")
    training_minutes = TRAINING_WALL_MINUTES
    dropped: list[str] = []
    for proxy_name, vram_need, tps_factor in (
            ("MIDI", 10.0, 1.0), ("MICRO", 4.0, 2.5),
            ("RESEARCH_SMALL", 2.0, 5.0)):
        if free_vram_gb < vram_need:
            continue
        rate = tokens_per_sec * tps_factor
        wall_tokens = rate * training_minutes * 60.0
        # 40 arm-runs share the budget (S1 4 + S2 8 + S3 24 + S4 4).
        per_arm = wall_tokens / 40.0
        if per_arm >= MIN_ACQUISITION_DOSE_TOKENS:
            dose = min(per_arm, TARGET_ACQUISITION_DOSE_TOKENS * 1.5)
            updates = max(4, int(dose // microbatch_tokens))
            return {"proxy": proxy_name, "updates_per_arm": updates,
                    "tokens_per_arm": updates * microbatch_tokens,
                    "reason": f"{proxy_name} affords dose floor in wall budget",
                    "dose_floor_met": True, "dropped": dropped}
        dropped.append(f"{proxy_name}-dose-below-floor")
    # Fallback: smallest proxy at dose floor even if wall overruns is
    # dishonest; instead cut replication first (documented priority).
    updates = max(4, MIN_ACQUISITION_DOSE_TOKENS // microbatch_tokens)
    return {"proxy": "RESEARCH_SMALL", "updates_per_arm": updates,
            "tokens_per_arm": updates * microbatch_tokens,
            "reason": "floor dose on smallest proxy; replication cuts required",
            "dose_floor_met": True,
            "dropped": dropped + ["transfer-second-seed", "mid-arm"]}


def resolve_proxy(*, micro_tokens_per_sec: float, free_vram_gb: float,
                  smoke: bool = False) -> dict[str, Any]:
    """Legacy thin wrapper (kept for notebook compatibility)."""

    resolved = resolve_campaign(tokens_per_sec=micro_tokens_per_sec,
                                free_vram_gb=free_vram_gb,
                                microbatch_tokens=2416, smoke=smoke)
    return {"proxy": resolved["proxy"],
            "tokens_per_arm": resolved["tokens_per_arm"],
            "reason": resolved["reason"]}


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


# -- task worlds: ONE shared context per world, variants differ by name ------

def render_worlds(*, family: str, split_seeds: Mapping[str, int],
                  worlds_per_split: int | dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Render counterfactual-paired worlds per split with zero world overlap.

    Registry (binding/retrieval) and transfer (compositional hops) modes.
    Each world owns ONE shared latent context (facts + values); base and
    twin records differ ONLY in query and required answer. Splits use
    disjoint seeds; world ids never cross splits. Records carry full
    structure (facts, relevant indices, values) so eval variants and
    task-aware baselines operate on the grammar, not on strings.
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
    splits: dict[str, list[dict[str, Any]]] = {}
    for split, seed in split_seeds.items():
        rng = random.Random(seed)
        worlds = []
        for index in range(counts[split]):
            keys = [f"k{seed}x{index}-{j}" for j in range(4)]
            values = [f"V{seed:05d}{index:03d}{j}" for j in range(4)]
            target = rng.randrange(4)
            if family == "registry":
                facts = [f"Registry entry {key} carries payload {value}."
                         for key, value in zip(keys, values)]
                query = f"Return the payload assigned to registry entry {keys[target]}."
                answer = values[target]
                twin_query = (f"Return the payload assigned to registry entry "
                              f"{keys[(target + 1) % 4]}.")
                twin_answer = values[(target + 1) % 4]
                relevant = [target]
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
                relevant = [0, 1, 2]

            def render(ask: str, ans: str) -> dict[str, str]:
                context = "\n".join(facts) + "\n" + ask
                return {"text": f"{context}\n{RENDER_SUFFIX}{ans}",
                        "answer": ans, "context": context, "query": ask}

            worlds.append({
                "world_id": f"{family}/{split}/{seed}/{index}",
                "facts": list(facts),
                "values": list(values),
                "relevant": list(relevant),
                "query": query,
                "twin_query": twin_query,
                "base": render(query, answer),
                "twin": render(twin_query, twin_answer),
            })
        splits[split] = worlds
    return splits


def split_manifest(splits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    body = {"schema": "anra-cyr-split-manifest/v2",
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
    """Order records; treatment keeps counterfactual twins adjacent.

    Control shuffles the same multiset. Deterministic in seed.
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


# -- orthogonal factorial (each variant changes exactly what it names) -------

def eval_variant_texts(world: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """BASE / QUERY_ONLY / ORDER_ONLY / QUERY_AND_ORDER /
    RELEVANT_VALUE_ONLY / IRRELEVANT_VALUE_ONLY / RENDERING_ONLY."""

    facts = list(world["facts"])
    values = list(world["values"])
    relevant = set(world["relevant"])
    base_answer = world["base"]["answer"]
    twin_answer = world["twin"]["answer"]

    def rebuild(ordered_facts: list[str], ask: str, ans: str) -> dict[str, str]:
        context = "\n".join(ordered_facts) + "\n" + ask
        return {"text": context + "\n" + RENDER_SUFFIX + ans, "answer": ans,
                "context": context, "query": ask}

    variants: dict[str, dict[str, str]] = {
        "base": dict(world["base"]),
        "query_only": dict(world["twin"]),
        "order_only": rebuild(list(reversed(facts)), world["query"], base_answer),
        "query_and_order": rebuild(list(reversed(facts)), world["twin_query"],
                                   twin_answer),
    }
    # Relevant-value swap: change ONLY the queried value -> answer changes.
    rel_changed = False
    for index in sorted(relevant):
        old = values[index]
        new = old + "X"
        swapped = [fact.replace(old, new) if position == index else fact
                   for position, fact in enumerate(facts)]
        if sum(new in fact for fact in swapped) == 1:
            variants["relevant_value_only"] = rebuild(
                swapped, world["query"], base_answer + "X")
            rel_changed = True
            break
    if not rel_changed:
        raise ValueError("no clean relevant-value swap available")
    # Irrelevant-value swap: change a distractor value -> answer invariant.
    irrelevants = [i for i in range(len(facts)) if i not in relevant]
    inv_changed = False
    for index in irrelevants:
        old = values[index]
        new = old + "X"
        swapped = [fact.replace(old, new) if position == index else fact
                   for position, fact in enumerate(facts)]
        if sum(new in fact for fact in swapped) == 1 and \
                all(old not in fact or position == index
                    for position, fact in enumerate(swapped)):
            variants["irrelevant_value_only"] = rebuild(
                swapped, world["query"], base_answer)
            inv_changed = True
            break
    if not inv_changed:
        raise ValueError("no clean irrelevant-value swap available")
    # Rendering-only: same content, space-joined surface (weak surface probe,
    # documented as such — no paraphraser exists).
    variants["rendering_only"] = rebuild(
        [" ".join(facts)], world["query"], base_answer)
    return variants


# -- task-aware shortcut baselines (grammar, with precomputed expectations) ---

def task_baselines() -> dict[str, Callable[[Mapping[str, Any]], str]]:
    """Deterministic policies over world STRUCTURE (never the query answer)."""

    def first(mapping: Mapping[str, Any], world: Mapping[str, Any]) -> str:
        del mapping
        return str(world["values"][0])

    def last(mapping: Mapping[str, Any], world: Mapping[str, Any]) -> str:
        del mapping
        return str(world["values"][-1])

    def fixed_1(mapping: Mapping[str, Any], world: Mapping[str, Any]) -> str:
        del mapping
        return str(world["values"][1])

    def most_frequent(mapping: Mapping[str, Any], world: Mapping[str, Any]) -> str:
        del mapping
        counts: dict[str, int] = {}
        for value in world["values"]:
            counts[str(value)] = counts.get(str(value), 0) + 1
        return max(counts, key=lambda value: (counts[value], value))

    def query_blind_canonical(mapping: Mapping[str, Any], world: Mapping[str, Any]) -> str:
        del mapping
        return str(world["values"][0])

    def second_to_last(mapping: Mapping[str, Any], world: Mapping[str, Any]) -> str:
        del mapping
        return str(world["values"][-2])

    return {"COPY_FIRST_FACT_VALUE": first, "COPY_LAST_FACT_VALUE": last,
            "FIXED_FACT_POSITION": fixed_1, "MOST_FREQUENT_VALUE": most_frequent,
            "QUERY_BLIND_CANONICAL_POSITION": query_blind_canonical,
            "SECOND_TO_LAST_VALUE": second_to_last}


def heuristic_baselines() -> dict[str, Callable[[str, str], str]]:
    """Legacy word-position policies (kept for comparison; NOT the red team)."""

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


# -- generation-report scoring (pure; free generation is primary) --------------

def score_generated(*, generated: list[str], expected: list[str],
                    stops: list[str]) -> dict[str, Any]:
    """Score free-generation outputs: complete exact, EOS/stop rates, extras."""

    if not (len(generated) == len(expected) == len(stops)):
        raise ValueError("generation report needs aligned lists")
    complete = sum(1 for got, want in zip(generated, expected) if got == want)
    prefix_extra = sum(1 for got, want in zip(generated, expected)
                       if got != want and got.startswith(want) and want)
    eos = sum(1 for stop in stops if stop == "eos")
    capped = sum(1 for stop in stops if stop == "cap")
    invalid = sum(1 for got in generated if not got.strip())
    total = len(generated)
    return {"complete_exact": complete / total if total else 0.0,
            "prefix_correct_but_extra": prefix_extra / total if total else 0.0,
            "eos_stop_rate": eos / total if total else 0.0,
            "max_tokens_rate": capped / total if total else 0.0,
            "invalid_output_rate": invalid / total if total else 0.0,
            "total": total}


def both_correct_rate(*, base_ok: list[bool], twin_ok: list[bool]) -> float:
    if len(base_ok) != len(twin_ok) or not base_ok:
        raise ValueError("both-correct needs aligned nonempty lists")
    return sum(1 for a, b in zip(base_ok, twin_ok) if a and b) / len(base_ok)


# -- learnability transitions (onset + confirmation, all denominations) --------

def find_transitions(*, flags: list[bool], updates: list[int],
                     tokens_per_update: int, started_wall_s: float,
                     eval_wall_s: list[float], threshold_name: str,
                     required: int = 3) -> dict[str, Any]:
    """Onset (first hit) + confirmation (sustained) in updates/tokens/wall."""

    if not (len(flags) == len(updates) == len(eval_wall_s)):
        raise ValueError("transition inputs must align")
    onset = next((index for index, flag in enumerate(flags) if flag), None)
    confirmation = None
    if onset is not None:
        for index in range(onset, len(flags) - required + 1):
            if all(flags[index:index + required]):
                confirmation = index
                break
    def locate(index: int | None) -> dict[str, Any] | None:
        if index is None:
            return None
        return {"eval_index": index, "update": updates[index],
                "tokens": updates[index] * tokens_per_update,
                "wall_s": round(eval_wall_s[index] - started_wall_s, 1)}
    return {"threshold": threshold_name, "required": required,
            "onset": locate(onset), "confirmation": locate(confirmation)}


def sustained(flags: list[bool], *, required: int = 3) -> bool:
    """True iff the trailing `required` evaluations all pass (no cherry-picks)."""

    return len(flags) >= required and all(flags[-required:])


# -- hysteretic controller (observes BOTH states; history persists) ------------

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
        """Feed one dev-controller measurement IN EITHER MODE.

        Retention→plasticity re-entry works because observation never stops:
        in retention mode a metric below the lower threshold builds the
        re-entry streak. History (mode, streak, decisions) is plain data and
        can be checkpointed across chunks and resume.
        """

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

    def snapshot(self) -> dict[str, Any]:
        """Persistable controller history (chunks, checkpoints, resume)."""

        self.assert_valid()
        return {"mode": self.mode, "streak": self.streak,
                "decisions": [dict(decision) for decision in self.decisions],
                "enter_retention": self.enter_retention,
                "reenter_plasticity": self.reenter_plasticity,
                "confirmations": self.confirmations}

    @classmethod
    def restore(cls, snapshot: Mapping[str, Any]) -> "HysteresisController":
        controller = cls(
            enter_retention=float(snapshot["enter_retention"]),
            reenter_plasticity=float(snapshot["reenter_plasticity"]),
            confirmations=int(snapshot["confirmations"]),
            mode=str(snapshot["mode"]), streak=int(snapshot["streak"]))
        controller.decisions = tuple(dict(decision) for decision in snapshot["decisions"])
        controller.assert_valid()
        return controller


# -- frozen CYR-GPU-002 campaign constants (code and PLAN.md read these) -----

CYR_ID = "CYR-GPU-002"
CYR_WALL_MIN_MINUTES = 60
CYR_WALL_TARGET_MINUTES = (120, 160)
CYR_WALL_HARD_MINUTES = 175
CYR_SEEDS = (101, 202)
CYR_SPLIT_SEEDS = {"train": 1001, "dev_controller": 2002,
                   "dev_measurement": 3003, "sealed_reserved": 4004}
CYR_DEV_WORLDS = 64
CYR_EVAL_DENSE_WORLDS = 64
CYR_EVAL_BATCH = 32
CYR_FREE_GEN_WORLDS = 8
CYR_FREE_GEN_CAP = 16
CYR_EVAL_EVERY_UPDATES = 10
CYR_SUSTAINED_REQUIRED = 3
CYR_ENTER_RETENTION = 0.90
CYR_REENTER_GAP = 0.15
CYR_FIXED_SWITCH_FRACTION = 0.5
CYR_MICROBATCH_ROWS = 8
CYR_ROW_CONTENT_TOKENS = 300


def guard_full_mode(args: Any) -> None:
    """Refuse full campaigns outside Colab (override exists, never use locally)."""

    if args.mode == "full" and os.environ.get("COLAB_GPU", "0") != "1" \
            and not args.allow_non_colab:
        raise ValueError(
            "full campaign refuses non-Colab execution: set COLAB_GPU=1 on "
            "the Colab GPU runtime or pass --allow-non-colab explicitly "
            "(never use the override for local science)")


def package_bundle(out: Path, *, manifest_extra: Mapping[str, Any] | None = None,
                   zip_name: str = "CYMEK_GPU_RESEARCH_RESULTS.zip") -> Path:
    """Zip the result bundle for the operator to return (flat layout)."""

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
            if path.is_file() and path.name not in (zip_name, "SESSION_MANIFEST.json"):
                handle.write(path, path.name)
        handle.write(out / "SESSION_MANIFEST.json", "SESSION_MANIFEST.json")
    return archive


def package_bundle_v2(out: Path, *,
                      zip_name: str = "CYMEK_GPU_RESEARCH_V2_RESULTS.zip") -> Path:
    """Zip the V2 result bundle (stage subdirectories + top-level manifests).

    Stage files route by filename prefix; anything unmatched lands in
    MISC/ rather than silently dropping.
    """

    routing = (("BASELINE", "ACQUISITION"), ("PAIR", "PAIR_QUERY"),
               ("FACTORIAL", "PAIR_QUERY"), ("LR", "LR_RETENTION"),
               ("TRANSFER", "TRANSFER"), ("SCALE", "SCALE"),
               ("REDTEAM", "REDTEAM"), ("NEGATIVE_CONTROL", "REDTEAM"),
               ("SEALED", "REDTEAM"))
    staged: dict[str, list[str]] = {}
    top_level: list[str] = []
    for path in sorted(out.iterdir()):
        if not path.is_file() or path.suffix != ".json" or path.name == zip_name:
            continue
        placed = False
        for prefix, stage in routing:
            if path.name.startswith(prefix):
                staged.setdefault(stage, []).append(path.name)
                placed = True
                break
        if not placed:
            top_level.append(path.name)
    manifest = {"schema": "anra-cyr-session-manifest/v2",
                "stages": {stage: sorted(names) for stage, names in sorted(staged.items())},
                "top_level": sorted(top_level)}
    (out / "SESSION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    archive = out / zip_name
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path.name in (zip_name, "SESSION_MANIFEST.json"):
                continue
            if path.suffix == ".bin":
                continue  # tensor payloads stay on the runner; receipts ship
            relative = path.relative_to(out)
            parts = (("CHECKPOINTS",) + relative.parts[1:]
                     if relative.parts and relative.parts[0] == "stores"
                     else relative.parts)
            if parts[0] in ("CHECKPOINTS", "ACQUISITION", "PAIR_QUERY",
                            "LR_RETENTION", "TRANSFER", "SCALE", "REDTEAM"):
                arcname = str(Path(*parts))
            elif path.parent == out and path.suffix == ".json":
                arcname = path.name
                for prefix, stage in routing:
                    if path.name.startswith(prefix):
                        arcname = f"{stage}/{path.name}"
                        break
            else:
                arcname = f"MISC/{path.name}"
            handle.write(path, arcname)
        handle.write(out / "SESSION_MANIFEST.json", "SESSION_MANIFEST.json")
    return archive


# -- frozen arm sets (prereg and runner read the same constant) ---------------

CYR_ARMS = {"s2": ("shuffled", "paired"),
            "s3": ("high", "mid", "low", "fixed", "state", "hysteretic")}
CYR_LR_SCHEDULES = ("HIGH", "MID", "LOW")


# -- frozen milestone thresholds (train/dev, sustained-3) --------------------

MILESTONE_THRESHOLDS = {"M99": ("train_exact", 0.99),
                        "G50": ("both_correct", 0.50),
                        "G90": ("both_correct", 0.90),
                        "G95": ("both_correct", 0.95)}


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


def exposure_matched(*, high_tokens: int, low_tokens: int,
                     tolerance: float = 2.0) -> dict[str, Any]:
    """Prospective low-LR exposure comparison with confound flagging."""

    if high_tokens <= 0 or low_tokens <= 0:
        raise ValueError("exposure comparison needs positive exposures")
    ratio = high_tokens / low_tokens
    return {"check": "low_lr_exposure_matched", "high_tokens": high_tokens,
            "low_tokens": low_tokens, "ratio": ratio,
            "pass": ratio <= tolerance,
            "note": "comparison confounded" if ratio > tolerance else "exposure matched"}


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


__all__ = ["BUNDLE_FILES", "CYR_ARMS", "CYR_DEV_WORLDS", "CYR_ENTER_RETENTION",
           "CYR_EVAL_BATCH", "CYR_EVAL_DENSE_WORLDS",
           "CYR_EVAL_EVERY_UPDATES", "CYR_FIXED_SWITCH_FRACTION", "CYR_ID",
           "CYR_LR_SCHEDULES",
           "CYR_MICROBATCH_ROWS", "CYR_REENTER_GAP",
           "CYR_SEEDS", "CYR_SPLIT_SEEDS", "CYR_SUSTAINED_REQUIRED",
           "CYR_WALL_HARD_MINUTES", "CYR_WALL_MIN_MINUTES",
           "CYR_WALL_TARGET_MINUTES", "HysteresisController",
           "MILESTONE_THRESHOLDS",
           "MIN_ACQUISITION_DOSE_TOKENS", "RUNNER_SCHEMA",
           "STAGE_FRACTIONS", "TARGET_ACQUISITION_DOSE_TOKENS",
           "TRAINING_WALL_MINUTES", "WORLD_VERSION", "assert_split_firewall",
           "both_correct_rate", "constant_lr", "discover_environment",
           "eval_variant_texts", "exposure_matched",
           "find_transitions", "guard_full_mode", "heuristic_baselines",
           "package_bundle", "package_bundle_v2", "pair_batches", "proxy_ladder",
           "proxy_spec_kwargs", "redteam_exposure", "redteam_freezing",
           "redteam_near_dup", "redteam_non_mutation", "redteam_overlap",
           "render_worlds", "resolve_campaign", "resolve_proxy",
           "score_generated", "split_manifest", "sustained",
           "task_baselines"]
