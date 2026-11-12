"""V7.16 falsifiable mechanism implementations (H1, H2+H2B, H3+H3B, H4, H5, H6).

Every mechanism is a standalone, CPU-testable component. None of them reads
evaluator truth: masking, canonicalization, permutation, counterfactual
construction, rollout selection, pointer heads and replay swaps derive only
from public state, deterministic seeds and recorded losses. Importing this
module launches no work.

Isolation contract (master directive section 3): each mechanism changes one
factor at a time against the blocked cross-entropy baseline while holding
epochs, optimizer updates, supervised tokens and unpadded input positions
exactly fixed.
"""
from dataclasses import dataclass, replace
import math
import random

import torch

from .codec import digest
from .cognition import (CognitiveState, ExperienceEvent, Goal, MemoryVersion,
                        PublicEnvironment, Truth, belief)
from .foundation_training import TokenSample
from .representation import UTF8, SpecialToken
from .v6_data import (CHANNEL, MAX_SEQUENCE, SYMBOLS, VERSION, Surface, V6Case,
                      V6Example, _supervision, action_text, case_to_record,
                      public_action, render_public)

# UTF8 byte-level vocabulary: PAD=0, BOS=1, EOS=2, ANSWER delimiter=3, then
# one entry per byte b at id b+4 (total 260).
BYTE_OFFSET = 4
EOS_ID = UTF8.eos_id
ANSWER_DELIMITER_ID = UTF8.special(SpecialToken.ANSWER)
ACTION_LETTER_IDS = tuple(ord(symbol) + BYTE_OFFSET for symbol in SYMBOLS)
ABSTAIN_ID = ord("!") + BYTE_OFFSET
QUERY_ID = ord("Q") + BYTE_OFFSET
RETRIEVE_ID = ord("R") + BYTE_OFFSET
DELIBERATE_ID = ord("~") + BYTE_OFFSET
FIRST_ACTION_ALLOWED_DEFAULT = frozenset(ACTION_LETTER_IDS) | {ABSTAIN_ID, DELIBERATE_ID}


@dataclass(frozen=True)
class PublicStateSpec:
    """Public decision state at one supervised decision; never contains truth."""

    channels: int
    memory_available: bool

    def __post_init__(self):
        if type(self.channels) is not int or self.channels < 0:
            raise ValueError("channel count must be a nonnegative integer")
        if type(self.memory_available) is not bool:
            raise ValueError("memory availability must be boolean")


def public_state_specs(cases):
    """Map task_id -> PublicStateSpec by replaying the public teacher loop.

    Mirrors v6_data._supervision exactly, recording the advertised channels and
    memory availability at each decision. Contains no evaluator information.
    """
    result = {}
    for case in cases:
        state = CognitiveState(case.goal).observe(case.environment.initial)
        channels = tuple(case.environment.channels)
        available = case.memory is not None
        for decision in range(4):
            result[f"{case.case_id}/{decision}"] = PublicStateSpec(
                channels=len(channels), memory_available=available)
            operation, argument = public_action(state, channels, case.surface, available)
            if operation == "QUERY":
                state = state.observe(case.environment.channels[argument])
                channels = ()
            elif operation == "RETRIEVE":
                state = state.observe(case.memory.recall(case.goal))
                available = False
            elif operation == "DELIBERATE":
                state = replace(state, hypothesis=argument)
            else:
                break
    return result


# ---------------------------------------------------------------------------
# H1: training-time controller logit masking (co-adaptation hypothesis)
# ---------------------------------------------------------------------------

class CoAdaptedControllerMaskLoss:
    """Mask illegal first actions and grammar continuations during training.

    The mask depends only on the public state spec and the already-visible
    input prefix, never on any future (target) token. Masked logits are filled
    with a finite large-negative value so both fp32 and autocast fp16 forward
    passes stay finite, while exp() underflow removes them from the softmax.
    """

    schema = "elendril-v716-coadapt-mask-v1"
    MASKED_LOGIT = -1e4
    MAX_ABS_LOGIT = 1000.0

    def __init__(self, *, allow_query=True, allow_retrieve=True, allow_deliberate=True):
        for flag in (allow_query, allow_retrieve, allow_deliberate):
            if type(flag) is not bool:
                raise ValueError("mask policy flags must be booleans")
        self.allow_query = allow_query
        self.allow_retrieve = allow_retrieve
        self.allow_deliberate = allow_deliberate

    def first_action_ids(self, spec: PublicStateSpec):
        allowed = set(ACTION_LETTER_IDS)
        allowed.add(ABSTAIN_ID)
        if self.allow_query and spec.channels:
            allowed.add(QUERY_ID)
        if self.allow_retrieve and spec.memory_available:
            allowed.add(RETRIEVE_ID)
        if self.allow_deliberate:
            allowed.add(DELIBERATE_ID)
        return frozenset(allowed)

    def legal_predictions_at(self, position, tokens, spec):
        """Legal token ids for predicting tokens[position] from tokens[:position]."""
        if position <= 0:
            raise ValueError("prediction positions start at 1 (after BOS)")
        return self._legal(position, tuple(tokens), spec)

    def _legal(self, position, tokens, spec):
        first = self._first_action_position(tokens)
        if first is None or position < first:
            return None  # prompt region: no mask
        if position == first:
            return self.first_action_ids(spec)
        previous = tokens[position - 1]
        if previous == DELIBERATE_ID:
            return frozenset(ACTION_LETTER_IDS)
        return frozenset({EOS_ID})

    def _first_action_position(self, tokens):
        """First position after the ANSWER delimiter; purely input-derived."""
        try:
            return tokens.index(ANSWER_DELIMITER_ID) + 1
        except ValueError:
            return None

    def span_violations(self, tokens, eligible, spec):
        """Targets that the public-state grammar already forbids (teacher bug)."""
        tokens = tuple(tokens)
        eligible = tuple(bool(x) for x in eligible)
        violations = []
        for position, is_target in enumerate(eligible):
            if not is_target:
                continue
            legal = self._legal(position, tokens, spec)
            if legal is not None and tokens[position] not in legal:
                violations.append(position)
        return violations

    def action_masks(self, tokens, eligible, spec):
        """Boolean mask aligned to tokens: True where prediction is grammar-legal.

        Causal: entry t depends only on tokens[:t], spec and the delimiter
        position found in the input. Identical prompts with different legal
        targets yield identical masks.
        """
        tokens = tuple(int(x) for x in tokens)
        if tokens[0] != UTF8.bos_id:
            raise ValueError("mask requires BOS-framed token rows")
        mask = [False] * len(tokens)
        first = self._first_action_position(tokens)
        if first is None:
            return tuple(mask)
        for position in range(first, len(tokens)):
            legal = self._legal(position, tokens, spec)
            mask[position] = tokens[position] in legal if legal else False
            if tokens[position] == EOS_ID and position > first:
                break
        return tuple(mask)

    def __call__(self, logits, tokens, eligible, specs):
        """Return masked logits; eligible marks supervised target positions."""
        if logits.ndim != 3 or tokens.shape != logits.shape[:2] or eligible.shape != tokens.shape:
            raise ValueError("mask requires aligned [batch, length] tensors and [batch, length, vocab] logits")
        if specs is None or len(specs) != logits.shape[0]:
            raise ValueError("one public state spec per batch row is required")
        if not torch.isfinite(logits).all().item() or logits.abs().max().item() > self.MAX_ABS_LOGIT:
            raise ValueError("logits exceed the finite masking margin; refusing to mask")
        mask = torch.zeros_like(logits)
        for row in range(logits.shape[0]):
            row_tokens = tuple(int(x) for x in tokens[row].tolist())
            row_eligible = tuple(bool(x) for x in eligible[row].tolist())
            for position, is_target in enumerate(row_eligible):
                if not is_target:
                    continue
                legal = self._legal(position, row_tokens, specs[row])
                if legal is None:
                    raise ValueError("eligible target outside the action span")
                if row_tokens[position] not in legal:
                    raise ValueError("teacher target violates the public action grammar")
                mask[row, position - 1, :] = self.MASKED_LOGIT
                mask[row, position - 1, list(legal)] = 0.
        return logits + mask


# ---------------------------------------------------------------------------
# H2 / H2B: surface invariance curriculum and canonical normalization
# ---------------------------------------------------------------------------

def _world_semantic(family, subject, initial_rows, hidden_rows, remembered_rows):
    """Mirror v6_data world-identity construction over raw (s, src, t, v) rows."""
    hops = 2 if family == "route" else 0
    return [family, hops, subject,
            sorted((a, b, c, d or "<NULL>") for a, b, c, d in initial_rows),
            sorted((a, b, c, d or "<NULL>") for a, b, c, d in hidden_rows),
            sorted((a, b, c, d or "<NULL>") for a, b, c, d in remembered_rows)]


def _rows(events):
    return [(event.subject, event.source, event.tick, event.value) for event in events]


def _rebuild(case, *, initial_rows, hidden_rows, remembered_rows, expected,
             subject=None, slots=None, surface=None, panel=None, pair_id=None):
    subject = subject or case.goal.subject
    family = case.family
    semantic = _world_semantic(family, subject, initial_rows, hidden_rows, remembered_rows)
    world = digest([VERSION, "semantic-world", semantic])
    scope, reusable = "world-" + world, "memory-" + world
    event = lambda row, custody: ExperienceEvent(custody, *row)
    initial = tuple(event(row, scope) for row in initial_rows)
    channels = {CHANNEL: tuple(event(row, scope) for row in hidden_rows)} if hidden_rows else {}
    memory = MemoryVersion(tuple(event(row, reusable) for row in remembered_rows)) if remembered_rows else None
    slots = tuple(slots) if slots is not None else tuple(dict.fromkeys(
        [subject, *(row[0] for row in initial_rows), *(row[0] for row in remembered_rows)]))
    goal = Goal(world, family, subject, scope, reusable, slots)
    all_events = [*initial, *(e for values in channels.values() for e in values),
                  *(memory.events if memory else ())]
    truth = Truth(world, expected, tuple(e.identity for e in all_events))
    return V6Case(world, world, family, case.kind, goal,
                  PublicEnvironment(initial, channels), truth,
                  surface or case.surface, memory,
                  pair_id if pair_id is not None else case.pair_id,
                  panel or case.panel)


def _split_safe(case):
    """True when every transformed observation state still partitions to train."""
    _, keys = _supervision(case)
    from .v6_data import _partition
    return all(_partition(key) == "train" for key in keys)


class Ark014SurfaceAugmenter:
    """Deterministic cursor-bound per-epoch public-surface transformations.

    Epoch 1 keeps the canonical blocked-CE rendering, so every arm shares the
    exact baseline epoch. Epochs 2-4 apply reversed rows, a uniform timestamp
    offset that preserves both rank order and rendered digit width
    (split-safe and length-safe, therefore parity-safe) and a balanced symbol
    permutation. Any transformed world that would cross the procedural split
    hash, exceed a digit-width ceiling or collide with a reserved world falls
    back to the canonical rendering of the same case, keeping per-epoch
    example, supervised-token and input-position counts exactly equal.
    """

    schema = "elendril-v716-ark014-surface-augmenter-v1"
    EPOCH_PLAN = ("canonical", "reversed-rows", "shifted-timestamps", "symbol-permutation")
    MAX_ATTEMPTS = 8

    def __init__(self, cases, *, seed, epochs=4, reserved_world_ids=frozenset()):
        cases = tuple(cases)
        if not cases or type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("augmenter requires cases and a bounded seed")
        if type(epochs) is not int or not 1 <= epochs <= len(self.EPOCH_PLAN):
            raise ValueError("augmenter supports one to four epochs")
        if not all(_split_safe(case) for case in cases):
            raise ValueError("augmenter requires base cases that partition to train")
        self.cases = cases
        self.seed = seed
        self.epochs = epochs
        self.reserved_world_ids = frozenset(reserved_world_ids)
        self.fallbacks = {plan: 0 for plan in self.EPOCH_PLAN}
        self.permutations_admitted = 0

    def epoch_cases(self, epoch):
        if type(epoch) is not int or not 1 <= epoch <= self.epochs:
            raise ValueError("epoch outside the registered plan")
        cases, fallbacks, admitted = self._evaluate_epoch(epoch)
        plan = self.EPOCH_PLAN[epoch - 1]
        self.fallbacks[plan] += fallbacks
        self.permutations_admitted += admitted
        return cases

    def _evaluate_epoch(self, epoch):
        """One deterministic plan evaluation that touches no ledger state.

        Returns the epoch's cases plus the fallback and admitted deltas a
        recorded consumption of this evaluation would accumulate. Used both
        by ``epoch_cases`` (which applies the deltas) and by ``audit`` (which
        reports a single canonical pass instead of re-counting state
        accumulated by its callers).
        """
        plan = self.EPOCH_PLAN[epoch - 1]
        if plan == "canonical":
            return self.cases, 0, 0
        if plan == "reversed-rows":
            return tuple(replace(case, surface=replace(case.surface, order="reverse"))
                         for case in self.cases), 0, 0
        result, used, fallbacks, admitted = [], set(), 0, 0
        for case in self.cases:
            transformed = (self._shifted(case, epoch) if plan == "shifted-timestamps"
                           else self._permuted(case, epoch))
            if transformed is None or transformed.world_id in self.reserved_world_ids \
                    or transformed.world_id in used or not _split_safe(transformed):
                fallbacks += 1
                result.append(case)
                used.add(case.world_id)
                continue
            if plan == "symbol-permutation":
                admitted += 1
            result.append(transformed)
            used.add(transformed.world_id)
        return tuple(result), fallbacks, admitted

    def _shifted(self, case, epoch):
        """Uniform rank-preserving tick offset that preserves rendered width.

        The offset must keep every tick inside its original digit count so the
        rendered prompt length (and therefore the unpadded input positions) is
        unchanged. Worlds whose ticks already touch a digit-width ceiling
        return None and fall back to the canonical rendering.
        """
        all_rows = [*_rows(case.environment.initial),
                    *(row for values in case.environment.channels.values() for row in _rows(values)),
                    *(_rows(case.memory.events) if case.memory is not None else [])]
        room = min(10 ** len(str(tick)) - 1 - tick for _, _, tick, _ in all_rows)
        if room < 1:
            return None
        delta = 1 + int(digest([self.seed, epoch, "v716-tick-shift", case.world_id]), 16) % room
        initial_rows = [(s, src, t + delta, v) for s, src, t, v in _rows(case.environment.initial)]
        hidden_rows = [(s, src, t + delta, v) for rows in case.environment.channels.values()
                       for s, src, t, v in _rows(rows)]
        remembered_rows = [(s, src, t + delta, v) for s, src, t, v in _rows(case.memory.events)] \
            if case.memory is not None else []
        return _rebuild(case, initial_rows=initial_rows, hidden_rows=hidden_rows,
                        remembered_rows=remembered_rows, expected=case.truth.expected)

    def _permuted(self, case, epoch):
        for attempt in range(self.MAX_ATTEMPTS):
            rng = random.Random(int(digest([self.seed, epoch, "v716-permute",
                                            case.world_id, attempt]), 16))
            shuffled = list(SYMBOLS)
            rng.shuffle(shuffled)
            mapping = dict(zip(SYMBOLS, shuffled))
            mapped = lambda symbol: mapping[symbol]
            initial_rows = [(mapped(s), src, t, None if v is None else mapped(v))
                            for s, src, t, v in _rows(case.environment.initial)]
            hidden_rows = [(mapped(s), src, t, None if v is None else mapped(v))
                           for rows in case.environment.channels.values()
                           for s, src, t, v in _rows(rows)]
            remembered_rows = [(mapped(s), src, t, None if v is None else mapped(v))
                               for s, src, t, v in _rows(case.memory.events)] \
                if case.memory is not None else []
            expected = None if case.truth.expected is None else mapped(case.truth.expected)
            candidate = _rebuild(case, initial_rows=initial_rows, hidden_rows=hidden_rows,
                                 remembered_rows=remembered_rows, expected=expected,
                                 subject=mapped(case.goal.subject),
                                 slots=tuple(mapped(s) for s in case.goal.slots),
                                 surface=case.surface)
            if candidate.world_id not in self.reserved_world_ids:
                return candidate
        return None

    def epoch_examples(self, epoch):
        from .v6_data import _supervision as supervision
        return tuple(example for case in self.epoch_cases(epoch)
                     for example in supervision(case)[0])

    def audit(self):
        """Single-pass plan audit; the ledger is read, never written.

        ``epoch_cases`` records every genuine consumption in the fallback
        ledger. The committed provenance must describe the registered plan
        itself, so the audit evaluates each epoch exactly once through the
        pure evaluator and reports that pass's counters. Re-counting state
        accumulated by callers would inflate the committed fallback counts
        by the number of redundant evaluations (previously three per epoch
        here, on top of the training row build).
        """
        from .v6_data import _supervision as supervision
        epochs, records, fallbacks, admitted = {}, [], {}, 0
        for epoch in range(1, self.epochs + 1):
            cases, epoch_fallbacks, epoch_admitted = self._evaluate_epoch(epoch)
            plan = self.EPOCH_PLAN[epoch - 1]
            fallbacks[plan] = fallbacks.get(plan, 0) + epoch_fallbacks
            admitted += epoch_admitted
            records.extend(case_to_record(case) for case in cases)
            examples = tuple(example for case in cases
                             for example in supervision(case)[0])
            samples = tuple(example.sample for example in examples)
            epochs[plan] = {
                "epoch": epoch, "cases": len(cases),
                "examples": len(examples),
                "supervised_tokens": sum(sum(s.eligible) for s in samples),
                "input_positions": sum(len(s.tokens) for s in samples),
            }
        reference = epochs[self.EPOCH_PLAN[0]]
        parity = all(value["examples"] == reference["examples"]
                     and value["supervised_tokens"] == reference["supervised_tokens"]
                     and value["input_positions"] == reference["input_positions"]
                     for value in epochs.values())
        return {"schema": self.schema, "epochs": epochs, "exposure_parity": parity,
                "fallbacks": {plan: fallbacks.get(plan, 0)
                              for plan in self.EPOCH_PLAN},
                "permutations_admitted": admitted,
                "determinism_sha256": digest(records)}


def canonical_events_key(event):
    """Canonical public-row ordering: entity, then tick ascending, then source."""
    return (event.subject, event.tick, event.source, event.value or "")


class CanonicalRowNormalizer:
    """H2B controller canonicalization: sort observation rows before rendering.

    Sorting is applied at render time to the already-observed public state, so
    latent semantics, observation-state keys and targets are untouched. This
    adds no learned parameters and cannot leak the answer: it only removes the
    presentation-order shortcut the reversed-rows failure exposes.
    """

    schema = "elendril-v716-canonical-row-normalizer-v1"

    @staticmethod
    def sorted_events(events):
        return tuple(sorted(events, key=canonical_events_key))

    @classmethod
    def render(cls, state, channels, surface, memory_available=False):
        return render_public(replace(state, events=cls.sorted_events(state.events)),
                             channels, surface, memory_available)

    @classmethod
    def supervision(cls, case):
        """Rebuild the case's supervised examples with canonically ordered rows."""
        state = CognitiveState(case.goal).observe(case.environment.initial)
        channels = tuple(case.environment.channels)
        available = case.memory is not None
        examples = []
        for decision in range(4):
            examples.append(V6Example(
                f"{case.case_id}/{decision}", case.case_id, case.world_id, case.kind,
                "initial" if not decision else f"observed-{decision}",
                cls.render(state, channels, case.surface, available),
                action_text(*public_action(state, channels, case.surface, available))))
            operation, argument = public_action(state, channels, case.surface, available)
            if operation == "QUERY":
                state = state.observe(case.environment.channels[argument])
                channels = ()
            elif operation == "RETRIEVE":
                state = state.observe(case.memory.recall(case.goal))
                available = False
            elif operation == "DELIBERATE":
                state = replace(state, hypothesis=argument)
            else:
                break
        return tuple(examples)


# ---------------------------------------------------------------------------
# H3 / H3B: equal-length counterfactual history worlds and order contrast
# ---------------------------------------------------------------------------

class CounterfactualMemoryGenerator:
    """Shape-matched paired counterfactual history worlds (C-P01 / D3).

    Only one-decision revision presentations are eligible, because the stale-
    overwrite (A), consistent-control (B) and unresolved-contradiction (C)
    variants each produce exactly one sample with a one-byte action target,
    keeping supervised-token parity exact when 25% of eligible presentations
    are replaced. Variants that would cross the split hash, collide with a
    reserved world or duplicate an in-epoch world fall back to the base case.
    """

    schema = "elendril-v716-counterfactual-memory-v1"
    VARIANT_PLAN = ("stale-overwrite", "consistent-control", "unresolved-contradiction")

    def __init__(self, *, seed):
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("counterfactual generator requires a bounded seed")
        self.seed = seed

    @staticmethod
    def eligible(case):
        if case.family != "revision":
            return False
        try:
            examples, _ = _supervision(case)
        except ValueError:
            return False
        return len(examples) == 1 and len(examples[0].target) == 1

    @staticmethod
    def _subject_rows(case):
        return [row for row in _rows(case.environment.initial)
                if row[0] == case.goal.subject]

    def stale_overwrite(self, case):
        """Attach a memory carrying the oldest subject value.

        The public rows already contain a newer overwrite for the same subject,
        so the teacher answer follows public evidence while a stale retrieval
        remains available. Cases that already carry a memory return None: the
        base presentation is then already the stale-overwrite arm.
        """
        rows = self._subject_rows(case)
        if len({row[2] for row in rows}) < 2 or case.memory is not None:
            return None
        oldest = min(rows, key=lambda row: row[2])
        return _rebuild(case, initial_rows=_rows(case.environment.initial),
                        hidden_rows=[r for rows in case.environment.channels.values()
                                     for r in _rows(rows)],
                        remembered_rows=[oldest],
                        expected=case.truth.expected, panel="v716-stale-overwrite")

    def consistent_control(self, case):
        """Rewrite every stale subject row onto a distractor entity.

        All non-latest subject observations are re-attributed to one fresh
        symbol, leaving a single-latest consistent subject history to answer
        from. The oldest row the paired stale-overwrite arm attaches to
        memory is always among the moved rows, so the control neutralizes
        exactly the value the treatment plants. Rows of other entities are
        never touched.
        """
        subject_rows = self._subject_rows(case)
        if len(subject_rows) < 2:
            return None
        keeper = max(subject_rows, key=lambda row: row[2])
        stale = {row for row in subject_rows
                 if row is not keeper and row != keeper}
        slots = set(case.goal.slots) | {case.truth.expected}
        distractor = next((s for s in SYMBOLS if s not in slots), None)
        if distractor is None:
            return None
        moved = lambda row: row[0] == case.goal.subject and row in stale
        initial_rows = [(distractor, row[1], row[2], row[3]) if moved(row) else row
                        for row in _rows(case.environment.initial)]
        hidden_rows = [r for values in case.environment.channels.values() for r in _rows(values)]
        return _rebuild(case, initial_rows=initial_rows, hidden_rows=hidden_rows,
                        remembered_rows=_rows(case.memory.events) if case.memory else [],
                        expected=case.truth.expected, panel="v716-consistent-control")

    def unresolved_contradiction(self, case):
        rows = self._subject_rows(case)
        if not rows:
            return None
        value = next((row[3] for row in rows if row[3] is not None), None)
        if value is None:
            return None
        slots = set(case.goal.slots) | {value}
        alternative = next((s for s in SYMBOLS if s not in slots), None)
        if alternative is None:
            return None
        tick = max(row[2] for row in rows)
        source = rows[0][1]
        other_source = "s1" if source != "s1" else "s0"
        contradiction = [(case.goal.subject, source, tick, value),
                         (case.goal.subject, other_source, tick, alternative)]
        initial_rows = [row for row in _rows(case.environment.initial)
                        if row[0] != case.goal.subject] + contradiction
        variant = _rebuild(case, initial_rows=initial_rows, hidden_rows=[],
                           remembered_rows=[], expected=None,
                           panel="v716-unresolved-contradiction")
        # Contradictory public evidence leaves no publicly obtainable answer;
        # the honest record label is the safe-action kind, not the base kind.
        return replace(variant, kind="unavailable")

    def variant(self, case, plan):
        builder = {"stale-overwrite": self.stale_overwrite,
                   "consistent-control": self.consistent_control,
                   "unresolved-contradiction": self.unresolved_contradiction}[plan]
        variant = builder(case)
        if variant is None:
            return None
        return replace(variant, case_id=digest([case.case_id, "v716-h3", plan]),
                       pair_id=digest([case.world_id, "v716-counterfactual", plan]),
                       panel=f"v716-{plan}")

    def epoch_replacements(self, cases, *, epoch, fraction=0.25,
                           reserved_world_ids=frozenset()):
        if type(epoch) is not int or epoch < 1 or type(fraction) not in (int, float) \
                or not 0 < fraction <= 1:
            raise ValueError("invalid counterfactual replacement request")
        cases = tuple(cases)
        eligible = sorted((case for case in cases if self.eligible(case)),
                          key=lambda case: case.case_id)
        count = min(len(eligible), int(math.floor(fraction * len(eligible))))
        if count:
            stride = len(eligible) / count
            selected = {eligible[min(len(eligible) - 1, int(index * stride))].case_id
                        for index in range(count)}
        else:
            selected = set()
        plan = self.VARIANT_PLAN[(epoch - 1) % len(self.VARIANT_PLAN)]
        replacements, used, audit = {}, set(), {"selected": 0, "replaced": 0,
                                                "unsupported": 0, "fallbacks": 0}
        for case in cases:
            if case.case_id not in selected:
                continue
            audit["selected"] += 1
            variant = self.variant(case, plan)
            if variant is None:
                audit["unsupported"] += 1
                continue
            if variant.world_id in reserved_world_ids or variant.world_id in used \
                    or not _split_safe(variant):
                audit["fallbacks"] += 1
                continue
            replacements[case.case_id] = variant
            used.add(variant.world_id)
            audit["replaced"] += 1
        return replacements, audit


class OrderContrastiveLoss:
    """Cosine-margin penalty between decision-token hidden states.

    Applied to paired renderings of the same semantic world in reversed and
    chronological row order so both map to the same latent point before the
    ANSWER head. Purely representational: no target token is involved.
    """

    schema = "elendril-v716-order-contrastive-v1"

    def __init__(self, *, margin=0.5):
        if type(margin) not in (int, float) or not math.isfinite(margin) or margin < 0:
            raise ValueError("contrastive margin must be finite and nonnegative")
        self.margin = float(margin)

    def __call__(self, hidden_a, hidden_b):
        if hidden_a.shape != hidden_b.shape or hidden_a.dim() != 2 or not hidden_a.shape[0]:
            raise ValueError("contrastive loss requires aligned [pairs, width] states")
        if not torch.isfinite(hidden_a).all() or not torch.isfinite(hidden_b).all():
            raise FloatingPointError("nonfinite hidden states in contrastive loss")
        cosine = torch.nn.functional.cosine_similarity(hidden_a, hidden_b, dim=-1)
        return torch.relu(self.margin - cosine).mean()


# ---------------------------------------------------------------------------
# H4: outcome-coupled bounded deliberation (expert iteration / RLOO selection)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RolloutRecord:
    """One completed bounded episode trace with public-only cost accounting."""

    task_id: str
    trace_id: str
    actions: str
    correct: bool
    knowable: bool
    verified_queries: int
    deliberations: int
    illegal: int
    supervised_tokens: int

    def utility(self, *, query_cost=0.1, deliberate_cost=0.0, illegal_cost=0.05):
        for name, value in (("query", query_cost), ("deliberate", deliberate_cost),
                            ("illegal", illegal_cost)):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} cost must be finite and nonnegative")
        return (1.0 if self.correct else 0.0) - query_cost * self.verified_queries \
            - deliberate_cost * self.deliberations - illegal_cost * self.illegal


class OutcomeCoupledDeliberationTrainer:
    """Select self-generated traces above the RLOO baseline for CE updates.

    The trainer performs selection and bookkeeping only. Token updates run
    through the qualified cross-entropy engine; the campaign supplies executed
    rollouts from the real environment and enforces the shape-matched exposure
    ledger, so the registered update count can never change.
    """

    schema = "elendril-v716-outcome-coupled-deliberation-v1"
    ROLLOUT_CHOICES = (1, 2, 4)

    def __init__(self, *, rollouts=2, query_cost=0.1, deliberate_cost=0.0,
                 illegal_cost=0.05):
        if rollouts not in self.ROLLOUT_CHOICES:
            raise ValueError("rollout budget must be 1, 2 or 4")
        self.rollouts = rollouts
        self.query_cost = query_cost
        self.deliberate_cost = deliberate_cost
        self.illegal_cost = illegal_cost

    def utility(self, record: RolloutRecord):
        return record.utility(query_cost=self.query_cost,
                              deliberate_cost=self.deliberate_cost,
                              illegal_cost=self.illegal_cost)

    def rloo_baseline(self, utilities, index):
        utilities = tuple(utilities)
        if len(utilities) != self.rollouts or not 0 <= index < len(utilities):
            raise ValueError("RLOO baseline requires the full rollout group")
        others = utilities[:index] + utilities[index + 1:]
        return sum(others) / len(others) if others else 0.0

    def select(self, groups):
        if not isinstance(groups, dict) or not groups:
            raise ValueError("selection requires rollout groups")
        kept = {}
        for task_id, records in sorted(groups.items()):
            records = tuple(records)
            if len(records) != self.rollouts:
                raise ValueError(f"rollout group for {task_id} is incomplete")
            utilities = [self.utility(record) for record in records]
            chosen = tuple(record for index, record in enumerate(records)
                           if utilities[index] > self.rloo_baseline(utilities, index)
                           and utilities[index] > 0.)
            if chosen:
                kept[task_id] = chosen
        return kept

    def audit(self, groups):
        kept = self.select(groups)
        total = sum(len(records) for records in groups.values())
        kept_count = sum(len(records) for records in kept.values())
        utilities = [self.utility(record) for records in groups.values() for record in records]
        return {"schema": self.schema, "prompts": len(groups), "rollouts": total,
                "kept": kept_count, "kept_fraction": kept_count / total if total else 0.,
                "mean_utility": sum(utilities) / len(utilities) if utilities else 0.,
                "query_cost": self.query_cost, "rollouts_per_prompt": self.rollouts}


def rollout_sample(record: RolloutRecord, prompt, *, max_tokens=MAX_SEQUENCE):
    """Frame one kept trace as a TokenSample for the qualified CE engine."""
    if not isinstance(record, RolloutRecord) or not record.actions:
        raise ValueError("rollout samples require an executed action trace")
    frame = UTF8.frame(prompt, record.actions, max_length=max_tokens)
    return TokenSample(record.task_id, frame.ids, frame.loss_mask, record.task_id.split("/")[0],
                       "rollout")


# ---------------------------------------------------------------------------
# H5: minimal pointer-copy and two-read relational composition head
# ---------------------------------------------------------------------------

class CausalPointerTwoReadDecoder(torch.nn.Module):
    """Parameter-matched pointer-copy gate plus two-read relational pooling.

    Read 1 pools prompt hidden states by a learned score; read 2 pools them by
    dot-product similarity against the read-1 vector, giving the copy query a
    composed (two-hop) key without any additional projection parameters. The
    copy distribution is the attention mass scattered onto observed token ids,
    mixed with the tied-embedding vocabulary distribution by a learned gate.

    The forward output is a log-probability tensor (the backend contract
    admits normalized log probabilities); callers must use NLL, not
    ``cross_entropy``, which would apply a second softmax.
    """

    VERSION = "elendril-v716-pointer-two-read-v1"
    PARAMETER_CEILING = 3_090_000
    MASK_FILL = -1e9
    LOG_EPSILON = 1e-12

    def __init__(self, core, *, pad_id=0):
        super().__init__()
        if not hasattr(core, "config") or not hasattr(core, "embedding"):
            raise ValueError("pointer decoder wraps an initialized v5 core")
        self.core = core
        self.pad_id = pad_id
        width = core.config.width
        self.copy_gate = torch.nn.Linear(width, 1, bias=False)
        self.read_pool = torch.nn.Linear(width, 1, bias=False)
        self.register_buffer("gate_offset", torch.tensor(-4.0))
        with torch.no_grad():
            self.copy_gate.weight.zero_()
            self.read_pool.weight.zero_()

    def parameter_delta(self):
        width = self.core.config.width
        return {"copy_gate": width, "read_pool": width, "gate_offset_buffer": 1}

    def parameter_receipt(self):
        base = self.core.spec.parameter_receipt().total
        delta = sum(self.parameter_delta().values())
        return {"schema": "elendril-v716-pointer-receipt-v1", "base": base,
                "delta": self.parameter_delta(), "total": base + delta,
                "ceiling": self.PARAMETER_CEILING,
                "within_ceiling": base + delta <= self.PARAMETER_CEILING}

    def forward(self, token_ids, positions, mask, use_activation_checkpointing=False):
        import torch.nn.functional as functional

        hidden = self.core.forward_hidden(token_ids, positions, mask,
                                          use_activation_checkpointing=use_activation_checkpointing)
        vocab_logits = functional.linear(hidden, self.core.embedding.weight)
        return self._mix(hidden, vocab_logits, token_ids, mask)

    def copy_distribution(self, token_ids, positions, mask):
        """Frozen-model diagnostic: the copy component alone, without vocab mixing.

        Its support is exactly the observed prompt token ids of each row, so
        the E-family copy analysis can verify that copied mass can never point
        outside the prompt.
        """
        import torch.nn.functional as functional

        hidden = self.core.forward_hidden(token_ids, positions, mask)
        width = hidden.shape[-1]
        content = token_ids != self.pad_id
        scale = math.sqrt(width)
        scores1 = self.read_pool(hidden).squeeze(-1).masked_fill(~content, self.MASK_FILL)
        weights1 = functional.softmax(scores1, dim=-1)
        read1 = torch.einsum("bl,blw->bw", weights1, hidden)
        scores2 = (torch.einsum("blw,bw->bl", hidden, read1) / scale).masked_fill(
            ~content, self.MASK_FILL)
        weights2 = functional.softmax(scores2, dim=-1)
        read2 = torch.einsum("bl,blw->bw", weights2, hidden)
        keys = hidden + read2.unsqueeze(1)
        copy_scores = torch.einsum("bqw,bkw->bqk", hidden, keys) / scale
        allowed = mask[:, 0] & content[:, None, :]
        copy_scores = copy_scores.masked_fill(~allowed, self.MASK_FILL)
        copy_attention = functional.softmax(copy_scores, dim=-1)
        one_hot = functional.one_hot(token_ids, self.core.config.vocabulary_size) \
            .to(hidden.dtype)
        return torch.bmm(copy_attention, one_hot)

    def _mix(self, hidden, vocab_logits, token_ids, mask):
        functional = torch.nn.functional
        width = hidden.shape[-1]
        vocabulary = vocab_logits.shape[-1]
        content = token_ids != self.pad_id
        scale = math.sqrt(width)
        scores1 = self.read_pool(hidden).squeeze(-1).masked_fill(~content, self.MASK_FILL)
        weights1 = functional.softmax(scores1, dim=-1)
        read1 = torch.einsum("bl,blw->bw", weights1, hidden)
        scores2 = (torch.einsum("blw,bw->bl", hidden, read1) / scale).masked_fill(
            ~content, self.MASK_FILL)
        weights2 = functional.softmax(scores2, dim=-1)
        read2 = torch.einsum("bl,blw->bw", weights2, hidden)
        keys = hidden + read2.unsqueeze(1)
        copy_scores = torch.einsum("bqw,bkw->bqk", hidden, keys) / scale
        allowed = mask[:, 0] & content[:, None, :]
        copy_scores = copy_scores.masked_fill(~allowed, self.MASK_FILL)
        copy_attention = functional.softmax(copy_scores, dim=-1)
        one_hot = functional.one_hot(token_ids, vocabulary).to(hidden.dtype)
        copy_distribution = torch.bmm(copy_attention, one_hot)
        gate = torch.sigmoid(self.copy_gate(hidden) + self.gate_offset)
        mixed = (1. - gate) * functional.softmax(vocab_logits, dim=-1) \
            + gate * copy_distribution
        return torch.log(mixed.clamp_min(self.LOG_EPSILON))


# ---------------------------------------------------------------------------
# H6: ARK-017 high-surprise sparse replay with strictly fixed update count
# ---------------------------------------------------------------------------

class Ark017SurpriseReplayBuffer:
    """Swap a fixed fraction of each update for highest-surprise A episodes.

    The buffer never adds optimizer updates: it only substitutes samples
    inside an already-registered batch. Every swapped-in Skill-A episode and
    its supervised tokens are written to a public ledger so replay exposure is
    never silently inflated or denied.
    """

    schema = "elendril-v716-ark017-sparse-replay-v1"

    def __init__(self, replay_samples, *, fraction=0.1, ema=0.3):
        replay_samples = tuple(replay_samples)
        if not replay_samples or type(fraction) not in (int, float) \
                or not 0 < fraction < 1 or type(ema) not in (int, float) \
                or not 0 < ema <= 1:
            raise ValueError("replay buffer requires samples, 0<fraction<1 and 0<ema<=1")
        if any(not isinstance(sample, TokenSample) for sample in replay_samples):
            raise ValueError("replay pool must contain TokenSample rows")
        self.samples = replay_samples
        self.fraction = float(fraction)
        self.ema = float(ema)
        self.surprise = [0.0] * len(replay_samples)
        self._swaps = 0

    def observe(self, position, loss):
        if not 0 <= position < len(self.surprise):
            raise ValueError("replay observation outside the pool")
        if type(loss) not in (int, float) or not math.isfinite(loss):
            raise ValueError("replay surprise must be finite")
        self.surprise[position] = (1. - self.ema) * self.surprise[position] \
            + self.ema * float(loss)

    def _rank(self, position):
        sample = self.samples[position]
        return (-round(self.surprise[position], 9), digest(sample.task_id))

    def plan_update(self, base_indices):
        base_indices = tuple(base_indices)
        if not base_indices:
            raise ValueError("replay planning requires a nonempty registered batch")
        count = int(math.floor(self.fraction * len(base_indices)))
        if count == 0:
            return (), (), self.ledger()
        pool = sorted(range(len(self.samples)), key=self._rank)[:count]
        swapped_out = base_indices[-count:]
        self._swaps += count
        return tuple(pool), tuple(swapped_out), self.ledger()

    def ledger(self):
        return {"schema": self.schema, "swaps": self._swaps,
                "fraction_requested": self.fraction,
                "pool_size": len(self.samples),
                "updates_added": 0,
                "rule": "sample-level substitution only; total optimizer updates stay fixed; "
                        "swapped-in Skill-A supervised tokens are charged to the replay ledger"}
