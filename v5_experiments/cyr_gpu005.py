"""CYR-GPU-005 pure campaign core: shared-parent fork contract.

Design contract (preregistered, immutable after the preregistration freeze):

For each acquisition seed EXACTLY ONE acquisition run happens. At the
G90_CONFIRMED checkpoint the parent's bytes are captured once; every
continuation arm restores THE SAME bytes and consumes THE SAME future
examples in THE SAME order. Only the registered optimization policy may
differ. No arm ever acquires for itself; LOW is applied only AFTER
capability exists.

This module is pure (no torch): registry, task worlds, manifest, leak
audit, future stream, policies, verdict rules, readiness gate. The torch
orchestrator lives in ``anra_v5.cyr_gpu005_run`` and must import its truth
from here.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Callable, Mapping

CORE_SCHEMA = "anra-cyr-gpu005-core/v1"
MANIFEST_SCHEMA = "anra-cyr-gpu005-data-manifest/v1"
SPLIT_SCHEMA = "anra-cyr-gpu005-split-manifest/v1"
PARENT_SCHEMA = "anra-cyr-gpu005-parent-equivalence/v1"
DECISION_SCHEMA = "anra-cyr-gpu005-decision/v1"
READINESS_SCHEMA = "anra-cyr-gpu005-run-readiness/v1"

CYR5_ID = "CYR-GPU-005"
CYR5_WALL_MIN_MINUTES = 60.0
CYR5_WALL_TARGET_MINUTES = (120.0, 150.0)
CYR5_WALL_HARD_MINUTES = 175.0

# Preregistered arms (section 21). HIGH/LOW values follow the replicated
# Arkenstone ARK-007R / ARK-010 evidence (fresh acquisitions 909/1010/1111):
# HIGH 1e-3 for plasticity, LOW 1e-5 for retention.
CYR5_LRS = {"HIGH": 1e-3, "LOW": 1e-5}
CYR5_ARMS = ("HIGH_CONTINUE", "LOW_CONTINUE",
             "FIXED_TIME_HIGH_TO_LOW", "HYSTERETIC_HIGH_LOW")
CYR5_FIXED_TIME_FRACTION = 0.5  # of ACTUAL continuation tokens, not steps

# Acquisition parents: 3 independent seeds, forks run only from G90 parents.
CYR5_PARENT_SEEDS = (707, 808, 909)
CYR5_ACQ_DOSE_MIN_TOKENS = 2_000_000
CYR5_ACQ_DOSE_TARGET_TOKENS = 4_000_000
CYR5_FORK_DOSE_MIN_TOKENS = 500_000
CYR5_FORK_DOSE_TARGET_TOKENS = 2_000_000

# Candidate-free gates (section 14): generated answers with valid stops only.
CYR5_GATES = {"M99": ("train_probe_complete", 0.99),
              "G50": ("dev_controller_complete", 0.50),
              "G90": ("dev_controller_complete", 0.90),
              "G95": ("dev_controller_complete", 0.95)}
CYR5_GATE_CONFIRMATIONS = 3

# Hysteresis (section 23): observes BOTH states; lower threshold below upper.
CYR5_HYSTERESIS = {"enter_retention": 0.90, "reenter_plasticity": 0.50,
                   "confirmations": 3}

# T2 arithmetic task: train tens 1-5, held-out eval tens 6-7 (Arkenstone T2).
CYR5_TRAIN_TENS = (1, 2, 3, 4, 5)
CYR5_EVAL_TENS = (6, 7)
CYR5_SPLIT_SEEDS = {"train": 5001, "dev_controller": 6002,
                    "dev_measurement": 6003, "sealed_reserved": 6004}
CYR5_WORLDS_PER_SPLIT = {"train": 500, "dev_controller": 96,
                         "dev_measurement": 112, "sealed_reserved": 48}

# Transfer family (section 25): binding/registry (ARK-009 lesson: the old
# diagnostic confounded query swap with fact-order reversal; the registry
# world renders query-only variants, so a zero-event transfer design cannot
# hide inside an order artifact).
CYR5_TRANSFER_FAMILY = "registry"


# -- one canonical research proxy registry (section 11) ----------------------

PROXY_ROLES = {
    "TINY": "plumbing only",
    "RESEARCH_SMALL": "cheap scientific screening",
    "MICRO": "standard replicated research",
    "MIDI": "main GPU candidate",
    "P35": "larger finalist only",
}


def proxy_registry(vocab_size: int = 24576) -> dict[str, dict[str, Any]]:
    """The ONE canonical registry. Geometry matches the frozen CYR ladder;
    parameter counts below are mechanically verified by
    ``assert_proxy_registry`` against the real ``ModelSpec`` receipt and by
    building the model in the registry test. A proxy whose verified count
    leaves its band may not be used under that name."""

    from v5_contracts.model_spec import ModelSpec

    geometry = {
        "TINY": {"layers": 2, "width": 64, "query_heads": 4, "kv_heads": 2,
                 "head_dimension": 16, "ffn_width": 128, "context_length": 512},
        "RESEARCH_SMALL": {"layers": 4, "width": 128, "query_heads": 4,
                           "kv_heads": 2, "head_dimension": 32,
                           "ffn_width": 512, "context_length": 512},
        "MICRO": {"layers": 4, "width": 256, "query_heads": 4, "kv_heads": 2,
                  "head_dimension": 64, "ffn_width": 512, "context_length": 512},
        "MIDI": {"layers": 8, "width": 384, "query_heads": 6, "kv_heads": 3,
                 "head_dimension": 64, "ffn_width": 1024, "context_length": 512},
        "P35": {"layers": 16, "width": 384, "query_heads": 6, "kv_heads": 3,
                "head_dimension": 64, "ffn_width": 1024, "context_length": 512},
    }
    registry: dict[str, dict[str, Any]] = {}
    for name, kwargs in geometry.items():
        spec = ModelSpec(
            schema="anra-v5-model-spec/v1", family="dense-decoder-transformer",
            vocabulary_size=int(vocab_size), width=kwargs["width"],
            layers=kwargs["layers"], query_heads=kwargs["query_heads"],
            kv_heads=kwargs["kv_heads"], head_dimension=kwargs["head_dimension"],
            ffn_width=kwargs["ffn_width"], context_length=kwargs["context_length"],
            rope_base=10_000.0, norm_epsilon=1e-5, tied_embeddings=True,
            qk_norm=True, qk_norm_affine=True, linear_bias=False, dropout=0.0)
        spec.assert_valid()
        receipt = spec.parameter_receipt()
        registry[name] = {"schema": "anra-cyr-gpu005-proxy/v1", "name": name,
                          "role": PROXY_ROLES[name], "spec": spec,
                          "parameter_receipt": receipt.as_dict(),
                          "parameters": receipt.total}
    return registry


def assert_proxy_in_registry(name: str, actual_parameters: int,
                             registry: Mapping[str, Mapping[str, Any]],
                             *, tolerance: float = 0.01) -> None:
    """A model may run under a registry name only at the registry size."""

    entry = registry.get(name)
    if entry is None:
        raise ValueError(f"proxy {name!r} is not in the canonical registry")
    claimed = int(entry["parameters"])
    if abs(actual_parameters - claimed) > tolerance * claimed:
        raise ValueError(
            f"proxy {name} claims {claimed} parameters but the built model has "
            f"{actual_parameters}: refusing mislabeled scale")


# -- T2 arithmetic worlds (candidate-free, generation-scored) ----------------

def t2_universe(tens: tuple[int, ...]) -> list[tuple[int, int]]:
    """Every valid (a, b) pair for first-operand tens bands `tens`.

    Mirrors the frozen T2 grammar: a = ta*10 + ua with ua in 0..9;
    b = tb*10 + ub with tb in 1..(9-ta) and ub in 0..(9-ua). Finite by
    construction — the sampler below never rejection-loops.
    """

    universe: list[tuple[int, int]] = []
    for ta in tens:
        for ua in range(10):
            for tb in range(1, 10 - ta):
                for ub in range(0, 10 - ua):
                    universe.append((ta * 10 + ua, tb * 10 + ub))
    return universe


def render_t2_worlds(*, split_seeds: Mapping[str, int] | None = None,
                     worlds_per_split: Mapping[str, int] | None = None,
                     train_tens: tuple[int, ...] = CYR5_TRAIN_TENS,
                     eval_tens: tuple[int, ...] = CYR5_EVAL_TENS,
                     ) -> dict[str, list[dict[str, Any]]]:
    """Deterministic two-digit addition worlds per split role.

    HOLDOUT AXIS (preregistered): the FIRST operand's tens band. Train
    draws bands 1-5; every eval split draws ONLY 6-7, so the ordered
    question "6x + y"/"7x + y" never appears in training. Because
    addition is commutative and the grammar is closed under canonical
    reordering (the reversed rendering of an eval row is always a legal
    train row), canonical-pair disjointness is mathematically unavailable
    for this task family; the audit verifies the ordered-row holdout and
    DECLARES the closure instead of pretending it away. Ordered (a, b)
    rows are globally unique across all splits; eval rows are selected
    answer-balanced with a per-answer usage cap; nothing rejection-loops.
    """

    seeds = dict(split_seeds or CYR5_SPLIT_SEEDS)
    counts = dict(worlds_per_split or CYR5_WORLDS_PER_SPLIT)
    if set(seeds) != {"train", "dev_controller", "dev_measurement",
                      "sealed_reserved"}:
        raise ValueError("T2 splits must be exactly the four preregistered roles")
    if len(set(seeds.values())) != len(seeds):
        raise ValueError("split seeds must be distinct")
    splits: dict[str, list[dict[str, Any]]] = {}
    used_rows: set[tuple[int, int]] = set()
    canonical_per_split: dict[str, set[tuple[int, int]]] = {}
    for split in ("train", "dev_controller", "dev_measurement",
                  "sealed_reserved"):
        seed = seeds[split]
        tens = train_tens if split == "train" else eval_tens
        owned_canonical = canonical_per_split.setdefault(split, set())
        # In bands 1-5 BOTH orderings of a pair are legal rows, so a split
        # may not take a row whose canonical form it already holds. Eval
        # splits are canonical-unique by grammar (the reversal of an eval
        # row has a 6-7 tens first operand and cannot be an eval row).
        universe = [pair for pair in t2_universe(tens)
                    if pair not in used_rows
                    and (min(pair), max(pair)) not in owned_canonical]
        if len(universe) < counts[split]:
            raise ValueError(
                f"split {split} needs {counts[split]} rows but only "
                f"{len(universe)} unused universe rows remain")
        rng = random.Random(seed)
        rows: list[dict[str, Any]] = []
        if split == "train":
            rng.shuffle(universe)
            chosen = []
            for pair in universe:
                canon = (min(pair), max(pair))
                if canon in owned_canonical:
                    continue
                owned_canonical.add(canon)
                chosen.append(pair)
                if len(chosen) == counts[split]:
                    break
            if len(chosen) < counts[split]:
                raise ValueError(
                    f"split {split} could only fill {len(chosen)}/"
                    f"{counts[split]} rows under canonical uniqueness")
        else:
            # Proportional stratified allocation: each answer contributes
            # rows in proportion to its share of the REMAINING universe
            # (largest-remainder rounding, +1 capped at availability). The
            # grammar's answer distribution is inherently triangular
            # (answers <= 99; pair counts 1..20), so the honest invariant
            # is distribution matching, not uniformity.
            by_answer: dict[int, list[tuple[int, int]]] = {}
            for pair in universe:
                if (min(pair), max(pair)) in owned_canonical:
                    continue
                by_answer.setdefault(sum(pair), []).append(pair)
            for answer in sorted(by_answer):
                rng.shuffle(by_answer[answer])
            total_remaining = len(universe)
            deficit = counts[split]
            targets: dict[int, int] = {}
            remainders: list[tuple[float, int]] = []
            for answer in sorted(by_answer):
                exact = deficit * len(by_answer[answer]) / total_remaining
                base = int(exact)
                if base > len(by_answer[answer]):
                    base = len(by_answer[answer])
                targets[answer] = base
                remainders.append((exact - base, answer))
            for _fraction, answer in sorted(remainders, reverse=True):
                if deficit - sum(targets.values()) <= 0:
                    break
                if targets[answer] < len(by_answer[answer]):
                    targets[answer] += 1
            chosen: list[tuple[int, int]] = []
            for answer in sorted(by_answer):
                take = by_answer[answer][:targets[answer]]
                chosen.extend(take)
            if len(chosen) != counts[split]:
                raise ValueError(
                    f"stratified allocation produced {len(chosen)} rows for "
                    f"{split}, expected {counts[split]}")
        for a, b in chosen:
            used_rows.add((a, b))
            pair = (min(a, b), max(a, b))
            owned_canonical.add(pair)
            rows.append({
                "world_id": f"t2/{split}/{seed}/{len(rows)}",
                "prompt": f"{a} + {b} = ", "answer": str(a + b),
                "a": a, "b": b,
                "canonical_pair": [pair[0], pair[1]],
                "tens_band": a // 10,
            })
        splits[split] = rows
    return splits


def canonical_row_bytes(row: Mapping[str, Any]) -> bytes:
    """Exact row serialization bound into the manifest (no drift)."""

    body = {"world_id": row["world_id"], "prompt": row["prompt"],
            "answer": row["answer"], "a": row["a"], "b": row["b"],
            "canonical_pair": row["canonical_pair"],
            "tens_band": row["tens_band"]}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def build_data_manifest(splits: Mapping[str, list[dict[str, Any]]], *,
                        generator_version: str = "cyr-gpu005-t2/v1",
                        seed: int = 5001) -> dict[str, Any]:
    """Manifest over the ACTUAL rendered bytes (section 16 option B).

    The SHA is computed from the exact serialized rows this campaign will
    train and evaluate on; nothing is claimed by constant.
    """

    body: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "generator": "t2-two-digit-addition",
        "generator_version": generator_version, "seed": seed,
        "splits": {name: [json.loads(canonical_row_bytes(row).decode("utf-8"))
                          for row in rows]
                   for name, rows in splits.items()}}
    body["sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    return body


def assert_manifest_sha(manifest: Mapping[str, Any]) -> None:
    claimed = manifest["sha256"]
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    if claimed != actual:
        raise ValueError("data manifest sha256 does not match its own rows")


def split_manifest(splits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    body = {"schema": SPLIT_SCHEMA,
            "splits": {name: [row["world_id"] for row in rows]
                       for name, rows in splits.items()}}
    body["sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")).hexdigest()
    return body


# -- commutation / structural leak audit (section 17) ------------------------

def commutation_audit(splits: Mapping[str, list[dict[str, Any]]],
                      *, train_tens: tuple[int, ...] = CYR5_TRAIN_TENS,
                      eval_tens: tuple[int, ...] = CYR5_EVAL_TENS,
                      tv_bound: float = 0.20,
                      ) -> dict[str, Any]:
    """Fail closed on any structural leak against the DECLARED holdout.

    The preregistered T2 holdout axis is the ordered question: train never
    contains a first operand with tens band 6-7. Checks: ordered-row
    crossing across splits, in-split duplicates, reversed duplicates within
    a split, answer-distribution shift on eval splits (total-variation
    distance against the grammar's own universe distribution; the frozen
    full-campaign bound is 0.20 — tiny smoke fixtures pass an explicit,
    receipted, larger bound because a 16-row sample cannot resolve a
    30-answer distribution), first-operand tens-band overlap, and
    canonical-reordering closure (declared property, reported — never
    silently ignored).
    """

    findings: dict[str, list[str] | str] = {}
    reported: dict[str, Any] = {}
    seen_row: dict[tuple[int, int], str] = {}
    canonical_owners: dict[tuple[int, int], set[str]] = {}
    for split, rows in splits.items():
        for row in rows:
            ordered = (int(row["a"]), int(row["b"]))
            first = seen_row.get(ordered)
            if first is not None:
                findings.setdefault("row_crossing_or_duplicate", []).append(
                    f"{list(ordered)} in {first} and {split}")
            seen_row[ordered] = split
            canonical_owners.setdefault(
                (min(ordered), max(ordered)), set()).add(split)
            if int(row["a"]) != int(row["b"]) \
                    and (int(row["b"]), int(row["a"])) in seen_row \
                    and seen_row[(int(row["b"]), int(row["a"]))] == split:
                findings.setdefault("reversed_duplicate_pairs", []).append(
                    f"{list(ordered)} and its reversal both in {split}")
    crossed = {pair: sorted(owners) for pair, owners in canonical_owners.items()
               if len(owners) > 1}
    # Declared, audited property — NOT a finding: for commutative addition
    # the grammar is closed under canonical reordering, so eval canonical
    # pairs also exist as reversed train rows. The holdout claim is the
    # ordered question (first-operand tens band), verified below.
    reported["canonical_reordering_crossing"] = {
        "crossed_canonical_pairs": len(crossed),
        "declaration": "closed under commutative reordering; the holdout "
                       "axis is the ordered first-operand tens band"}

    # Answer distribution: the grammar's own answer distribution is
    # inherently triangular (answers capped at 99; pair counts 1..20), so
    # the preregistered invariant is DISTRIBUTION MATCHING — each eval
    # split must track the eval universe (total-variation distance), not
    # uniformity over answers.
    answers: dict[str, dict[str, int]] = {}
    universe_counts: dict[str, int] = {}
    universe_total = 0
    for pair in t2_universe(eval_tens):
        universe_counts[str(sum(pair))] = universe_counts.get(str(sum(pair)), 0) + 1
        universe_total += 1
    for split in ("dev_controller", "dev_measurement", "sealed_reserved"):
        counter: dict[str, int] = {}
        for row in splits[split]:
            counter[row["answer"]] = counter.get(row["answer"], 0) + 1
        answers[split] = counter
        if counter and universe_total:
            distance = 0.5 * sum(
                abs(counter.get(answer, 0) / len(splits[split])
                    - universe_counts.get(answer, 0) / universe_total)
                for answer in set(counter) | set(universe_counts))
            if distance > tv_bound:
                findings.setdefault("answer_distribution_shift", []).append(
                    f"{split}: TV distance from the eval-universe answer "
                    f"distribution is {distance:.3f} (> {tv_bound:.2f})")

    train_bands = {int(row["tens_band"]) for row in splits["train"]}
    eval_bands = {int(row["tens_band"]) for split in (
                      "dev_controller", "dev_measurement", "sealed_reserved")
                  for row in splits[split]}
    overlap = sorted(train_bands & eval_bands)
    if overlap:
        findings["tens_band_overlap"] = (
            f"first-operand tens bands {overlap} appear in both train and "
            f"eval; the OOD claim would be unsupported (declared train "
            f"{sorted(train_tens)}, eval {sorted(eval_tens)})")
    passed = not findings
    return {"schema": "anra-cyr-gpu005-leak-audit/v1",
            "commutation_free": passed, "findings": findings,
            "reported": reported, "tv_bound": tv_bound,
            "holdout_axis": "first_operand_tens_band",
            "canonical_closure_declared": True,
            "train_tens": sorted(train_tens), "eval_tens": sorted(eval_tens)}


# -- exact future stream (section 8) -----------------------------------------

def build_future_stream(*, seed: int, world_count: int,
                        prefix_rows: int, tail_rows: int,
                        ) -> dict[str, Any]:
    """Pre-generate the FULL example-index stream before any training.

    The acquisition prefix and the continuation tail are one deterministic
    sequence: every fork consumes tail[0:] exactly. Hashes bind the bytes;
    no per-arm reseeding can ever diverge.
    """

    if prefix_rows <= 0 or tail_rows <= 0:
        raise ValueError("stream needs positive prefix and tail rows")
    rng = random.Random(seed)
    stream = [rng.randrange(world_count) for _ in range(prefix_rows + tail_rows)]
    def sha(rows: list[int]) -> str:
        return hashlib.sha256(json.dumps(rows).encode("utf-8")).hexdigest()
    return {"schema": "anra-cyr-gpu005-future-stream/v1", "seed": seed,
            "world_count": world_count, "prefix_rows": prefix_rows,
            "tail_rows": tail_rows, "fork_boundary": prefix_rows,
            "stream": stream, "prefix_sha256": sha(stream[:prefix_rows]),
            "tail_sha256": sha(stream[prefix_rows:]),
            "full_sha256": sha(stream)}


def batch_shas(*, stream: Mapping[str, Any], batch_size: int, count: int,
               ) -> list[str]:
    """Hashes of the first `count` continuation batches from the tail.

    Every arm must observe the identical sequence (tested mechanically).
    """

    tail = list(stream["stream"])[stream["fork_boundary"]:]
    if batch_size <= 0 or count <= 0 or count * batch_size > len(tail):
        raise ValueError("batch hash window exceeds the continuation tail")
    return [hashlib.sha256(json.dumps(
        tail[index * batch_size:(index + 1) * batch_size]).encode("utf-8")
    ).hexdigest() for index in range(count)]


def assert_future_tail_equality(arm_shas: Mapping[str, list[str]]) -> dict[str, Any]:
    """Every arm's first-N batch hashes must be byte-identical."""

    if not arm_shas:
        raise ValueError("no arm hashes to compare")
    reference_name, reference = next(iter(arm_shas.items()))
    for name, shas in arm_shas.items():
        if shas != reference:
            raise ValueError(
                f"future-tail mismatch: arm {name} diverges from "
                f"{reference_name} — the fork contract is broken")
    return {"schema": "anra-cyr-gpu005-tail-equality/v1",
            "reference_arm": reference_name, "arms": sorted(arm_shas),
            "batches_compared": len(reference),
            "identical": True}


# -- policies (sections 21, 22, 23) ------------------------------------------

def fixed_time_switch_point(*, continuation_target_tokens: int,
                            fraction: float = CYR5_FIXED_TIME_FRACTION,
                            ) -> int:
    """FIXED_TIME switches on ACTUAL continuation tokens, never steps."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("fixed-time fraction must be in (0, 1)")
    if continuation_target_tokens <= 0:
        raise ValueError("continuation target must be positive")
    return int(continuation_target_tokens * fraction)


def arm_policies() -> dict[str, dict[str, Any]]:
    return {
        "HIGH_CONTINUE": {"lr": CYR5_LRS["HIGH"], "policy": "constant_high"},
        "LOW_CONTINUE": {"lr": CYR5_LRS["LOW"], "policy": "constant_low"},
        "FIXED_TIME_HIGH_TO_LOW": {
            "lr_high": CYR5_LRS["HIGH"], "lr_low": CYR5_LRS["LOW"],
            "policy": "fixed_time",
            "switch_fraction_of_actual_tokens": CYR5_FIXED_TIME_FRACTION},
        "HYSTERETIC_HIGH_LOW": {
            "lr_plasticity": CYR5_LRS["HIGH"], "lr_retention": CYR5_LRS["LOW"],
            "policy": "hysteretic", **dict(CYR5_HYSTERESIS)},
    }


def lr_for_token(arm: str, continuation_tokens: int, *, switch_point: int,
                 controller: Any | None = None) -> float:
    """Registered optimization policy as a pure function of actual tokens."""

    if arm == "HIGH_CONTINUE":
        return CYR5_LRS["HIGH"]
    if arm == "LOW_CONTINUE":
        return CYR5_LRS["LOW"]
    if arm == "FIXED_TIME_HIGH_TO_LOW":
        return CYR5_LRS["HIGH"] if continuation_tokens < switch_point else CYR5_LRS["LOW"]
    if arm == "HYSTERETIC_HIGH_LOW":
        if controller is None:
            raise ValueError("hysteretic arm requires its controller")
        snapshot = controller.snapshot() if hasattr(controller, "snapshot") else controller
        return (CYR5_LRS["LOW"] if snapshot["mode"] == "retention"
                else CYR5_LRS["HIGH"])
    raise ValueError(f"unregistered arm {arm!r}")


# -- verdict rules (sections 35, 36) -----------------------------------------

def decide_verdict(*, arm_receipts: Mapping[str, Mapping[str, Any]],
                   parent_equivalence: Mapping[str, Any],
                   future_tail: Mapping[str, Any],
                   leak_audit: Mapping[str, Any],
                   parent_status: str,
                   transfer: Mapping[str, Any] | None = None,
                   ) -> dict[str, Any]:
    """Matched comparisons are valid only under the full contract.

    A TIMEBOXed, exposure-starved, or fork-broken arm can never be
    promoted; missing prerequisites return INCONCLUSIVE with reasons.
    """

    reasons: list[str] = []
    if parent_status != "G90_CONFIRMED":
        reasons.append(f"parent_status={parent_status}: retention forks are "
                       "undefined without an acquired capability")
    if not parent_equivalence.get("identical", False):
        reasons.append("parent equivalence failed: forks did not start from "
                       "the same bytes")
    if not future_tail.get("identical", False):
        reasons.append("future-tail equality failed: arms did not consume the "
                       "same examples in the same order")
    if not leak_audit.get("commutation_free", False):
        reasons.append("commutation audit found structural leaks")
    if reasons:
        return {"schema": DECISION_SCHEMA, "verdict": "INCONCLUSIVE",
                "winner": None, "reasons": reasons}
    complete: dict[str, Mapping[str, Any]] = {}
    for arm, receipt in arm_receipts.items():
        if receipt.get("status") != "COMPLETE":
            reasons.append(f"{arm}: status={receipt.get('status')} (timeboxed "
                           "or starved arms are not comparable)")
            continue
        if not receipt.get("shares_valid_parent", False):
            reasons.append(f"{arm}: parent not valid")
            continue
        if not receipt.get("redteam_pass", False):
            reasons.append(f"{arm}: red team failed")
            continue
        complete[arm] = receipt
    if len(complete) < 2:
        return {"schema": DECISION_SCHEMA, "verdict": "INCONCLUSIVE",
                "winner": None, "reasons": reasons or ["fewer than two "
                "contract-valid arms"]}
    scored = {arm: (receipt["retention_ret90"], receipt.get("final_g", 0.0))
              for arm, receipt in complete.items()}
    winner = max(scored, key=lambda arm: scored[arm])
    best_other = max(value[0] for arm, value in scored.items() if arm != winner)
    margin = scored[winner][0] - best_other
    return {"schema": DECISION_SCHEMA, "verdict": "DECIDED" if margin > 0 else "TIED",
            "winner": winner if margin > 0 else None,
            "ranking": sorted(scored, key=lambda arm: scored[arm], reverse=True),
            "ret90_margin": round(margin, 4), "reasons": reasons,
            "transfer": transfer}


# -- RUN_READINESS gate (section 59) -----------------------------------------

READINESS_CONDITIONS = (
    "exact_cymek_v5_used", "canonical_proxy_identity", "production_tokenizer",
    "data_manifest_verified", "commutation_leak_audit_pass",
    "acquisition_once_per_seed", "forks_restore_exact_same_parent",
    "optimizer_state_matches_at_fork", "rng_cursor_state_matches",
    "future_minibatches_hash_identical", "actual_token_targeting",
    "candidate_free_g90", "controller_measurement_split", "fixed_time_control",
    "hysteresis_tested_locally", "timebox_safe", "checkpoint_recovery_works",
    "evidence_packaging_works", "full_compressed_pipeline_passes",
    "static_analysis_passes", "freeze_checkout_simulation_passes",
    "xla_accumulation_oracle_passes",
)


def run_readiness_gate(results: Mapping[str, bool]) -> dict[str, Any]:
    """ready=true only when EVERY preregistered condition holds."""

    missing = [name for name in READINESS_CONDITIONS if not results.get(name)]
    unknown = sorted(set(results) - set(READINESS_CONDITIONS))
    if unknown:
        raise ValueError(f"unknown readiness conditions: {unknown}")
    return {"schema": READINESS_SCHEMA, "experiment": CYR5_ID,
            "ready": not missing, "conditions": dict(results),
            "blockers": missing}


# -- notebook freeze contract (sections 43, 44) -------------------------------

def assert_freeze_contract(*, preregistration: Mapping[str, Any],
                           executable_sha: str,
                           head_sha: str,
                           file_hashes: Mapping[str, str]) -> dict[str, Any]:
    """CELL 0's gate: prereg must name the checked-out executable exactly.

    The notebook copies the preregistration OUTSIDE the repo, checks out
    EXECUTABLE_SHA, asserts HEAD, verifies every bound file hash, and only
    then runs with the external copy. Any mismatch fails closed.
    """

    if preregistration.get("executable_sha256") != executable_sha:
        raise ValueError(
            "preregistration binds a different executable: refusing to run "
            "(CYR-GPU-005 would be SUPERSEDED_BEFORE_EXECUTION)")
    if head_sha != executable_sha:
        raise ValueError(f"HEAD {head_sha} is not the frozen executable "
                         f"{executable_sha}")
    for path, expected in preregistration.get("executable_files", {}).items():
        if file_hashes.get(path) != expected:
            raise ValueError(f"frozen file hash mismatch: {path}")
    return {"schema": "anra-cyr-gpu005-freeze-contract/v1",
            "executable_sha256": executable_sha, "head_sha256": head_sha,
            "files_verified": sorted(file_hashes), "verified": True}


def notebook_cell0_sequence(preregistration: Mapping[str, Any], *,
                            executable_sha: str) -> dict[str, Any]:
    """The exact CELL 0 order, asserted once in code and once in tests."""

    return {"order": ["read_preregistration_from_commit_B_head",
                      "copy_outside_repo_/content/CYR-GPU-005-PREREGISTRATION.json",
                      "checkout_executable_sha_A",
                      "assert_HEAD_eq_A",
                      "verify_executable_file_hashes",
                      "run_with_external_preregistration"],
            "executable_sha256": executable_sha,
            "preregistration_sha256": hashlib.sha256(json.dumps(
                preregistration, sort_keys=True, separators=(",", ":"),
                ensure_ascii=False).encode("utf-8")).hexdigest()}
