"""Complete raw-action V6 episodes and independent diagnostic scoring.

The policy receives only public state, advertised channels and renderer settings.
Gold joins after a complete trace. Engineered policies and privileged oracles are
explicit controls; none substitutes for model-origin success.
"""
from dataclasses import asdict, replace, dataclass
import math
import random
import hashlib
from pathlib import Path

from .codec import canonical, digest
from .cognition import (Budget, CognitiveState, ExperienceEvent, MemoryVersion,
                        OperationCandidate, belief, run_episode, verify)
from .representation import UTF8, LEGACY, SpecialToken
from .v6_registry import PANELS, REGISTRY, REGISTRY_SHA, formation_gate


DEFAULT_BUDGET = Budget(decisions=4, model_positions=12000, requests=2)
QUERY_COST = .1
THINK_COST = 0.
SYMBOLS = "ABCDEFGHIJKLMNOP"


@dataclass(frozen=True, slots=True)
class PublicCase:
    """Host adapter configuration; excludes truth, channels and stored values."""
    case_id: str
    surface: object
    memory: bool | None


def action_tokens(operation, argument=None):
    from .v6_data import action_text
    return (*UTF8.encode(action_text(operation, argument)), UTF8.eos_id)


def parse_action(tokens, channels, representation="utf8-raw"):
    """No repair, legality masking, implicit EOS, fuzzy parse or oracle fallback."""
    if representation not in ("utf8-raw", "utf8-ascii", "legacy-ascii"):
        raise ValueError("unknown decoder representation")
    tokenizer = LEGACY if representation == "legacy-ascii" else UTF8
    tokens = tuple(tokens)
    first_content = 3 if tokenizer is LEGACY else 4
    if (len(tokens) < 2 or tokens[-1] != tokenizer.eos_id
            or any(type(t) is not int or not first_content <= t < tokenizer.vocabulary_size for t in tokens[:-1])):
        return "INVALID", None
    try:
        text = tokenizer.decode(tokens[:-1])
    except ValueError:
        return "INVALID", None
    if text in SYMBOLS and len(text) == 1:
        return "ANSWER", text
    if text == "!":
        return "ABSTAIN", None
    if text == "Q":
        # An unavailable query remains invalid in the controller. Do not turn
        # it into an abstention or choose a different action on the model's behalf.
        return "QUERY", channels[0] if len(channels) == 1 else None
    if text == "R":
        return "RETRIEVE", None
    if len(text) == 2 and text[0] == ("~" if representation == "utf8-raw" else "-") and text[1] in SYMBOLS:
        return "DELIBERATE", text[1]
    return "INVALID", None


class NeuralPolicy:
    origin = "model"

    def __init__(self, backend, surface=None, *, memory_available=False, max_new_tokens=4):
        if type(max_new_tokens) is not int or not 2 <= max_new_tokens <= 8:
            raise ValueError("bounded action token limit must be2..8")
        if backend.binding.tokenizer_sha256 != UTF8.identity or backend.binding.vocabulary_size != UTF8.vocabulary_size:
            raise ValueError("V6 policy requires the qualified UTF8 tokenizer binding")
        self.backend, self.surface = backend, surface
        self.memory_available, self.max_new_tokens = memory_available, max_new_tokens
        self.model_identity = backend.binding.checkpoint_sha256 or backend.binding.identity
        self.prompts = []

    def for_case(self, case, *, memory_available=None):
        return NeuralPolicy(self.backend, case.surface,
            memory_available=case.memory is not None if memory_available is None else memory_available,
            max_new_tokens=self.max_new_tokens)

    def propose(self, state, legal, channels, positions):
        from .v6_data import render_public
        if self.surface is None:
            raise ValueError("bind a public case surface before decoding")
        prompt = render_public(state, channels, self.surface,
                               self.memory_available and "RETRIEVE" in legal)
        ids = [UTF8.bos_id, *UTF8.encode(prompt), UTF8.special(SpecialToken.ANSWER)]
        if len(ids) > self.backend.binding.context_length or len(ids) > positions:
            return None
        raw, probabilities, used = [], [], 0
        for _ in range(self.max_new_tokens):
            if len(ids) > self.backend.binding.context_length or used + len(ids) > positions:
                break
            distribution = self.backend.next_distribution(ids)
            token = int(distribution.argmax())
            probabilities.append(float(distribution[token]))
            used += len(ids)
            raw.append(token)
            if token in (UTF8.eos_id, UTF8.pad_id, UTF8.bos_id, UTF8.special(SpecialToken.ANSWER)):
                break
            ids.append(token)
        self.prompts.append(prompt)
        operation, argument = parse_action(raw, channels)
        return OperationCandidate(operation, argument, math.prod(probabilities), used, len(raw), tuple(raw))


class ControlPolicy:
    """Explicit public algorithms; only the named oracle receives Truth."""
    model_identity = None

    def __init__(self, mode="fixed", *, seed=7, surface=None, memory_available=False, truth=None):
        if mode not in {"fixed", "oracle", "always-query", "no-query", "random", "query-blind", "recency"}:
            raise ValueError("unknown control")
        if (truth is not None) != (mode == "oracle" and surface is not None):
            raise ValueError("truth is reserved for a bound privileged oracle")
        self.mode, self.seed, self.surface = mode, seed, surface
        self.memory_available, self.truth = memory_available, truth
        self.origin = "privileged_oracle" if mode == "oracle" else "fixed_baseline"
        self.prompts = []

    def for_case(self, case, *, memory_available=None):
        return ControlPolicy(self.mode, seed=self.seed, surface=case.surface,
            memory_available=case.memory is not None if memory_available is None else memory_available,
            truth=case.truth if self.mode == "oracle" else None)

    def propose(self, state, legal, channels, positions):
        from .v6_data import public_action
        if self.mode == "oracle":
            if state.goal.episode != self.truth.episode:
                raise ValueError("oracle episode binding mismatch")
            return OperationCandidate("ABSTAIN") if self.truth.expected is None else OperationCandidate("ANSWER", self.truth.expected)
        if self.mode == "always-query" and channels and "QUERY" in legal:
            return OperationCandidate("QUERY", channels[0])
        if self.mode == "query-blind":
            return OperationCandidate("ANSWER", "A")
        if self.mode == "recency":
            subject = state.goal.subject
            for _ in range(self.surface.hops if state.goal.family == "route" else 1):
                values = [event.value for event in state.events if event.subject == subject]
                subject = values[-1] if values else None
                if subject is None:
                    break
            return OperationCandidate("ABSTAIN") if subject is None else OperationCandidate("ANSWER", subject)
        if self.mode == "random":
            rng = random.Random(int(digest([self.seed, state.render(), channels, state.hypothesis]), 16))
            options = [("ANSWER", symbol) for symbol in SYMBOLS] + [("ABSTAIN", None)]
            if "QUERY" in legal and channels:
                options.append(("QUERY", channels[0]))
            if "RETRIEVE" in legal and self.memory_available:
                options.append(("RETRIEVE", None))
            if "DELIBERATE" in legal:
                options.extend(("DELIBERATE", s) for s in SYMBOLS)
            return OperationCandidate(*rng.choice(options))
        operation, argument = public_action(state, () if self.mode == "no-query" else channels,
            self.surface, self.memory_available and "RETRIEVE" in legal)
        return OperationCandidate(operation, argument)


def source_identity():
    """Bind evaluation to the actual source files used to build its receipts."""
    root = Path(__file__).parent
    # Git enforces LF; normalize an existing Windows worktree's equivalent
    # CRLF checkout too, so offline replay does not reject identical source.
    return digest({name: hashlib.sha256((root / name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                   for name in ("v6_evaluation.py", "v6_registry.py", "v6_data.py", "v6_training.py", "v6_supplement.py", "cognition.py", "representation.py")})


def _cases(data):
    return tuple(data.cases if hasattr(data, "cases") else data)


def _case_record(case):
    return {"id": case.case_id, "world": case.world_id, "family": case.family, "kind": case.kind,
            "goal": asdict(case.goal), "surface": asdict(case.surface),
            "initial": [asdict(e) for e in case.environment.initial],
            "channels": {k: [asdict(e) for e in v] for k, v in case.environment.channels.items()},
            "memory": None if case.memory is None else asdict(case.memory), "truth": asdict(case.truth),
            "pair_id": case.pair_id, "panel": case.panel}


def cases_identity(cases):
    return digest([_case_record(case) for case in cases])


def _metrics(rows, representation="utf8-raw"):
    eos_id = LEGACY.eos_id if representation == "legacy-ascii" else UTF8.eos_id
    def totals(subset):
        decisions = [decision for row in subset for decision in row["trace"]["decisions"]]
        return {"episodes": len(subset), "correct": sum(row["outcome"]["correct"] for row in subset),
            "requests": sum(row["trace"]["requests"] for row in subset),
            "verified_queries": sum(round((int(row["outcome"]["correct"]) - row["utility"]) / QUERY_COST) for row in subset),
            "memory_requests": sum(d["operation"] == "RETRIEVE" for d in decisions),
            "utility": sum(row["utility"] for row in subset),
            "invalid": sum(row["trace"]["status"] == "invalid_action" for row in subset),
            "model_calls": sum(d["model_calls"] for d in decisions),
            "model_positions": sum(d["model_positions"] for d in decisions),
            "thinking_decisions": sum(d["operation"] == "DELIBERATE" for d in decisions),
            "queries": sum(d["operation"] == "QUERY" for d in decisions),
            "retrievals": sum(d["operation"] == "RETRIEVE" for d in decisions),
            "first_query": sum(bool(r["trace"]["decisions"]) and r["trace"]["decisions"][0]["operation"] == "QUERY" for r in subset),
            "query_needed": sum(r["query_needed"] for r in subset),
            "useful_first_query": sum(r["query_needed"] and bool(r["trace"]["decisions"]) and r["trace"]["decisions"][0]["operation"] == "QUERY" for r in subset),
            "unsafe_initial_answer": sum(r["query_needed"] and bool(r["trace"]["decisions"]) and r["trace"]["decisions"][0]["operation"] == "ANSWER" for r in subset),
            "raw_eos": sum(bool(d["raw_tokens"]) and d["raw_tokens"][-1] == eos_id for d in decisions),
            "raw_actions": sum(bool(d["raw_tokens"]) for d in decisions),
            "abstained": sum(r["trace"]["status"] == "abstained" for r in subset)}
    pairs = {}
    for row in rows:
        if row["pair_id"]:
            pairs.setdefault(row["pair_id"], []).append(bool(row["outcome"]["correct"]))
    return {"all": totals(rows),
            "by_kind": {k: totals([r for r in rows if r["kind"] == k]) for k in sorted({r["kind"] for r in rows})},
            "by_family": {k: totals([r for r in rows if r["family"] == k]) for k in sorted({r["family"] for r in rows})},
            "pairs": {"both_correct": sum(len(v) == 2 and all(v) for v in pairs.values()),
                      "complete_pairs": sum(len(v) == 2 for v in pairs.values()),
                      "incomplete_pairs": sum(len(v) != 2 for v in pairs.values())}}


def evaluate(data, policy_factory, *, split=None, budget=DEFAULT_BUDGET, memory_mode="on"):
    from .v6_data import public_action
    if memory_mode not in ("on", "off", "prefetch"):
        raise ValueError("unknown memory evaluation mode")
    representation = getattr(policy_factory, "representation", "utf8-raw")
    if representation not in ("utf8-raw", "utf8-ascii", "legacy-ascii"):
        raise ValueError("unknown decoder representation")
    cases = _cases(data)
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("evaluation requires unique complete cases")
    rows = []
    for case in cases:
        memory = None if memory_mode == "off" else case.memory
        # Only the explicitly privileged oracle receives evaluator Truth.
        public_case = (case if getattr(policy_factory, 'mode', None) == 'oracle' else
                       PublicCase(case.case_id, case.surface, True if memory is not None else None))
        actor = (policy_factory.for_case(public_case, memory_available=memory is not None)
                 if hasattr(policy_factory, "for_case") else policy_factory(public_case))
        if public_case is case and actor.origin != 'privileged_oracle':
            raise ValueError('truth-bearing oracle configuration cannot be attributed to a model')
        state = CognitiveState(case.goal).observe(case.environment.initial)
        wanted, _ = public_action(state, tuple(case.environment.channels), case.surface, memory is not None)
        trace = run_episode(case.goal, case.environment, actor, budget=budget, memory=memory,
                            prefetch=memory_mode == "prefetch")
        outcome = verify(trace, case.truth)
        thinking = sum(d.operation == "DELIBERATE" for d in trace.decisions)
        confidence = trace.decisions[-1].predicted_success if trace.decisions else None
        rows.append({"case_id": case.case_id, "world_id": case.world_id, "family": case.family,
            "kind": case.kind, "pair_id": case.pair_id, "panel": case.panel,
            "trace": asdict(trace), "trace_sha256": trace.identity, "outcome": asdict(outcome),
            "truth_sha256": digest(asdict(case.truth)), "expected_public_sources": list(case.truth.admitted_public),
            "observed_public_sources": list(outcome.admitted_public), "query_needed": wanted == "QUERY",
            "utility": float(outcome.correct) - QUERY_COST * sum(d.operation == "QUERY" and d.argument in case.environment.channels for d in trace.decisions if d.operation == "QUERY" and not (trace.status == "invalid_action" and d is trace.decisions[-1])),
            "confidence": confidence, "prompts": list(getattr(actor, "prompts", []))})
    identities = sorted({row["trace"]["model_identity"] for row in rows if row["trace"]["model_identity"] is not None})
    if len(identities) > 1:
        raise ValueError("evaluation mixes model checkpoint identities")
    return {"schema": "elendril-v6-evaluation-v1", "registry_sha256": REGISTRY_SHA,
            "source_sha256": source_identity(), "decoder_schema": representation,
            "dataset_sha256": data.identity if hasattr(data, "identity") else cases_identity(cases),
            "cases_sha256": cases_identity(cases), "split": split or getattr(data, "split", "development"),
            "budget": asdict(budget), "memory_mode": memory_mode,
            "costs": {"verified_query": QUERY_COST, "thinking_decision": THINK_COST},
            "model_identity": identities[0] if identities else None,
            "policy": getattr(policy_factory, "mode", getattr(policy_factory, "origin", "declared_factory")),
            "metrics": _metrics(rows, representation), "episodes": rows,
            "qualification": "independent development outcome join; no promotion or broad cognition claim"}


def independent_rescore(data, record):
    """Replay every committed raw action on fresh public worlds and rejoin truth."""
    cases = _cases(data)
    if (record.get("schema") != "elendril-v6-evaluation-v1" or record.get("registry_sha256") != REGISTRY_SHA
            or record.get("source_sha256") != source_identity()
            or record.get("decoder_schema") not in ("utf8-raw", "utf8-ascii", "legacy-ascii")
            or record.get("cases_sha256") != cases_identity(cases)
            or record.get("dataset_sha256") != (data.identity if hasattr(data, "identity") else cases_identity(cases))
            or record.get("costs") != {"verified_query": QUERY_COST, "thinking_decision": THINK_COST}
            or len(record.get("episodes", [])) != len(cases)):
        raise ValueError("evaluation source/data/cost/denominator mismatch")
    indexed = {r["case_id"]: r for r in record["episodes"]}
    if len(indexed) != len(cases) or set(indexed) != {case.case_id for case in cases}:
        raise ValueError("duplicate or missing evaluation case")

    class ReplayFactory:
        mode = record["policy"]
        representation = record["decoder_schema"]

        def for_case(self, case, *, memory_available=None):
            reference = indexed[case.case_id]
            steps = reference["trace"]["decisions"]
            if record["memory_mode"] == "prefetch" and case.memory is not None:
                steps = steps[1:]

            class Replay:
                model_identity = record["model_identity"]
                origin = "privileged_oracle" if record['policy']=='oracle' else "model"

                def __init__(self):
                    self.index, self.prompts = 0, []

                def propose(self, state, legal, channels, positions):
                    from .v6_data import render_public
                    if self.index == len(steps):
                        return None
                    step = steps[self.index]
                    self.index += 1
                    self.origin = step["origin"]
                    if self.origin == "model":
                        operation, argument = parse_action(tuple(step["raw_tokens"]), channels, record["decoder_schema"])
                        if record["decoder_schema"] == "utf8-raw":
                            self.prompts.append(render_public(state, channels, case.surface,
                                bool(memory_available) and "RETRIEVE" in legal))
                        if operation != step["operation"] or argument != step["argument"]:
                            raise ValueError("raw model actions disagree with declared operation")
                    elif self.origin in ("fixed_baseline", "privileged_oracle") and record["model_identity"] is None:
                        operation, argument = step["operation"], step["argument"]
                    else:
                        raise ValueError("scripted choice cannot be attributed to model")
                    return OperationCandidate(operation, argument, step["predicted_success"], step["model_positions"],
                                              step["model_calls"], tuple(step["raw_tokens"]))
            return Replay()

    checked = evaluate(data, ReplayFactory(), split=record["split"], budget=Budget(**record["budget"]),
                       memory_mode=record["memory_mode"])
    if canonical(checked) != canonical(record):
        raise ValueError("raw-action replay disagrees with trace, confidence, prompts, cost, outcome or totals")
    return checked["metrics"]


def _probability_rows(record):
    rows = []
    for row in record["episodes"]:
        p = row["confidence"]
        if p is not None:
            if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid pre-outcome confidence")
            rows.append((float(p), int(row["outcome"]["correct"])))
    return rows


def calibration_metrics(probabilities, *, bins=10):
    pairs = tuple(probabilities)
    if not pairs:
        return {"status": "NO_CONFIDENCE", "n": 0, "brier": None, "ece": None, "auroc": None}
    if type(bins) is not int or not 2 <= bins <= 100:
        raise ValueError("calibration bins must be2..100")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 or y not in (0, 1) for p, y in pairs):
        raise ValueError("invalid confidence/endpoint pair")
    ece = 0.
    for index in range(bins):
        group = [(p, y) for p, y in pairs if min(int(p * bins), bins - 1) == index]
        if group:
            ece += len(group) / len(pairs) * abs(sum(p for p, _ in group) / len(group) - sum(y for _, y in group) / len(group))
    positive, negative = sum(y for _, y in pairs), sum(1 - y for _, y in pairs)
    # Pairwise rank interpretation handles confidence ties exactly. Diagnostic
    # panels are bounded; this never participates in model optimization.
    auc = (sum((p > q) + .5 * (p == q) for p, y in pairs if y for q, z in pairs if not z)
           / (positive * negative)) if positive and negative else None
    ordered = sorted(pairs, key=lambda item: item[0], reverse=True)
    coverage = []
    for fraction in (.25, .5, .75, 1.):
        keep = max(1, math.ceil(fraction * len(ordered)))
        coverage.append({"coverage": keep / len(ordered), "accuracy": sum(y for _, y in ordered[:keep]) / keep})
    return {"status": "DIAGNOSTIC", "n": len(pairs), "brier": sum((p - y)**2 for p, y in pairs) / len(pairs),
            "ece": ece, "auroc": auc, "auroc_status": "DEFINED" if auc is not None else "UNDEFINED_SINGLE_CLASS",
            "accuracy_coverage": coverage}


def fit_calibration(validation, *, bins=10):
    if validation["split"] != "validation" or validation["model_identity"] is None:
        raise ValueError("calibration fitting requires separate neural validation evidence")
    pairs = _probability_rows(validation)
    calibration_metrics(pairs, bins=bins)
    if not pairs:
        raise ValueError("cannot fit absent confidence")
    rates = []
    for index in range(bins):
        labels = [y for p, y in pairs if min(int(p * bins), bins - 1) == index]
        rates.append((sum(labels) + 1) / (len(labels) + 2))
    record = {"schema": "elendril-v6-validation-bin-calibration-v1", "bins": bins, "rates": rates,
        "validation_sha256": validation["dataset_sha256"], "model_identity": validation["model_identity"],
        "base_rate": (sum(y for _, y in pairs) + 1) / (len(pairs) + 2),
        "rule": "fixed equal-width bins with Beta(1,1) smoothing, fit only validation outcomes"}
    return {**record, "sha256": digest(record)}


def validation_risk_bound(validation, *, max_error=.1, delta=.05):
    """Select a confidence threshold on validation only with a finite-sample bound.

    Hoeffding plus a union bound covers the data-dependent choice among all
    distinct validation thresholds. This does not certify a shifted test set.
    """
    if validation["split"] != "validation" or validation["model_identity"] is None:
        raise ValueError("risk threshold requires neural validation")
    if not 0 < max_error < 1 or not 0 < delta < 1:
        raise ValueError("invalid risk bound parameters")
    pairs = _probability_rows(validation)
    thresholds = sorted({p for p, _ in pairs})
    selected = None
    for threshold in thresholds:
        group = [y for p, y in pairs if p >= threshold]
        error = 1 - sum(group) / len(group)
        upper = min(1., error + math.sqrt(math.log(len(thresholds) / delta) / (2 * len(group))))
        if upper <= max_error:
            selected = {"threshold": threshold, "samples": len(group), "error": error, "upper_bound": upper}
            break
    body = {"validation_sha256": validation["dataset_sha256"], "model_identity": validation["model_identity"],
            "delta": delta, "max_error": max_error, "selected": selected,
            "method": "Hoeffding union bound across validation thresholds"}
    return {**body, "sha256": digest(body)}


def apply_calibration(record, calibration):
    body = {k: v for k, v in calibration.items() if k != "sha256"}
    if (digest(body) != calibration.get("sha256") or record["split"] in ("train", "validation")
            or record["dataset_sha256"] == calibration["validation_sha256"]
            or record["model_identity"] != calibration["model_identity"]):
        raise ValueError("calibration must be fixed on separate validation for the same model")
    pairs = _probability_rows(record)
    calibrated = [(calibration["rates"][min(int(p * calibration["bins"]), calibration["bins"] - 1)], y) for p, y in pairs]
    return {"raw": calibration_metrics(pairs), "calibrated": calibration_metrics(calibrated),
            "constant_validation_base_rate": calibration_metrics([(calibration["base_rate"], y) for _, y in pairs]),
            "calibration_sha256": calibration["sha256"], "qualification": "error calibration, not awareness"}


def _summary(record, kinds=None):
    if record is None:
        return None
    if kinds is None:
        return record["metrics"]["all"]
    rows = [r for r in record["episodes"] if r["kind"] in kinds]
    return _metrics(rows, record.get("decoder_schema", "utf8-raw"))["all"]


def retention_panel(stages):
    """Refuse retention/transfer inference without the actual acquisition chain."""
    if stages is None:
        return {"status": "NOT_QUALIFIED", "reason": "no independently acquired A/B stage and matched C trajectories"}
    required = {"parent_checkpoint", "a_before", "a_after", "b_after", "c_scratch", "c_staged", "total_work"}
    if not isinstance(stages, dict) or not required <= set(stages):
        raise ValueError("incomplete retention stage evidence")
    if any(not isinstance(stages[key], dict) or "metrics" not in stages[key] for key in ("a_before", "a_after", "b_after")):
        raise ValueError("retention requires complete A/B evaluation receipts")
    if not formation_gate(stages["a_before"]) or not formation_gate(stages["b_after"], require_route=True):
        return {"status": "NOT_QUALIFIED", "reason": "A or B failed independent acquisition", "evidence": stages}
    if not stages["c_scratch"] or not stages["c_staged"] or stages["total_work"].get("matched") is not True:
        return {"status": "NOT_QUALIFIED", "reason": "missing matched C learning curves/cumulative spend", "evidence": stages}
    return {"status": "DEVELOPMENT_EVIDENCE", "evidence": stages,
            "qualification": "independent rescore and work matching still required before promotion"}


def _intervention_pairs(record, action):
    groups = {}
    for row in record["episodes"]:
        if row["pair_id"]:
            groups.setdefault(row["pair_id"], []).append(row)
    qualified = [v for v in groups.values() if len(v) == 2]
    def succeeds(pair):
        terminals = [(r["trace"]["status"], r["trace"]["answer"]) for r in pair]
        return (terminals[0] != terminals[1] and all(r["outcome"]["correct"] and any(
            d["operation"] == action and (d["evidence"] or action == "QUERY")
            for d in r["trace"]["decisions"]) for r in pair))
    return {"complete_pairs": len(qualified), "incomplete_pairs": len(groups) - len(qualified),
            "both_follow_intervention": sum(succeeds(pair) for pair in qualified)}


def panel_report(base, variants=None, *, positive_control=None, stage_records=None, calibration=None, controls=None, weight_identity=None):
    variants = variants or {}
    kinds = base["metrics"]["by_kind"]
    route = base["metrics"]["by_family"].get("route", {})
    route_formed = bool(route.get("episodes")) and route["correct"] / route["episodes"] >= .9 and route["invalid"] == 0
    reports = {p: {"name": name, "status": "SKIPPED", "reason": "required diagnostic evidence absent"}
               for p, name, _, _, _ in PANELS}
    if positive_control is not None:
        if not isinstance(positive_control, dict) or positive_control.get("source") is None:
            raise ValueError("positive control must carry actual source-bound evidence")
        reports["P01"].update(status="PASS" if positive_control.get("passed") is True else "FAIL", evidence=positive_control)
    for panel, selected in (("P02", ("known",)), ("P03", ("missing",)), ("P04", ("conflict",)),
                            ("P05", ("null", "unavailable")), ("P06", ("missing", "conflict"))):
        evidence = _summary(base, selected)
        reports[panel].update(status=("PASS" if evidence["correct"] == evidence["episodes"] and evidence["invalid"] == 0 else "FAIL") if evidence["episodes"] else "NOT_QUALIFIED", metrics=evidence)
    # P02-P04 additionally require the appropriate initial decision, not just terminal correctness.
    for panel in ("P02", "P03", "P04"):
        evidence = reports[panel].get("metrics", {})
        if evidence.get("episodes"):
            initial_ok = (evidence["first_query"] == 0 if panel == "P02" else
                          evidence["useful_first_query"] == evidence["query_needed"] and
                          evidence["query_needed"] == evidence["episodes"])
            if not initial_ok:
                reports[panel]["status"] = "FAIL"
    # Revision requires a paired observation intervention, not ordinary queried episodes.
    reports["P06"]["status"] = "NOT_QUALIFIED"
    reports["P06"]["reason"] = "paired post-query payload intervention receipt required"
    if "payload-swap" in variants:
        paired = _intervention_pairs(variants["payload-swap"], "QUERY")
        reports["P06"].update(status=("PASS" if paired["both_follow_intervention"] == paired["complete_pairs"] and not paired["incomplete_pairs"] else "FAIL") if paired["complete_pairs"] else "NOT_QUALIFIED", metrics=paired, reason="paired query-response intervention")
    reports["P07"].update(status="NOT_QUALIFIED", metrics=base["metrics"]["all"],
                          note="compare separately attributed fixed/always-query/no-query/random/oracle records; report GPU work separately")
    if controls and all(k in controls for k in ("fixed", "always-query", "no-query")):
        relevant = {k: controls[k] for k in ("fixed", "always-query", "no-query")}
        case_ids = {r["case_id"] for r in base["episodes"]}
        if all(c.get("policy") == k and c.get("cases_sha256") == base["cases_sha256"] and
               c.get("source_sha256") == base["source_sha256"] and c.get("budget") == base["budget"] and
               {r["case_id"] for r in c["episodes"]} == case_ids for k,c in relevant.items()):
            utilities = {k: c["metrics"]["all"]["utility"] / len(c["episodes"]) for k,c in relevant.items()}
            own = base["metrics"]["all"]["utility"] / len(base["episodes"])
            reports["P07"].update(status="PASS" if all(own > u for u in utilities.values()) else "FAIL",
                comparison={"candidate": own, "controls": utilities}, reason="matched-case primary utility comparison")
    reports["P08"].update(status="NOT_QUALIFIED", metrics=calibration_metrics(_probability_rows(base)),
                          note="raw joint output likelihood is not calibrated uncertainty")
    if calibration is not None:
        fitted = apply_calibration(base, calibration)
        reports["P08"]["validation_fitted"] = fitted
        measured = fitted["calibrated"]
        baseline = fitted["constant_validation_base_rate"]
        if measured["n"]:
            reports["P08"].update(status="PASS" if measured["brier"] <= baseline["brier"] and measured["ece"] <= .10 else "FAIL", reason="descriptive held-out Brier and ECE rule; no awareness claim")
    reports["P09"].update(status="PASS" if base["metrics"]["all"]["invalid"] == 0 and base["metrics"]["all"]["raw_actions"] > 0 and base["metrics"]["all"]["raw_eos"] == base["metrics"]["all"]["raw_actions"] else "FAIL", metrics=base["metrics"]["all"])
    for panel, key in (("P10", "query-swap"), ("P11", "time-order")):
        if key in variants:
            pairs = variants[key]["metrics"]["pairs"]
            reports[panel].update(status=("PASS" if pairs["both_correct"] == pairs["complete_pairs"] and not pairs["incomplete_pairs"] else "FAIL") if pairs["complete_pairs"] else "NOT_QUALIFIED", metrics=variants[key]["metrics"])
    if "time-order" in variants and "row-order" in variants:
        temporal = variants["time-order"]["metrics"]["pairs"]
        ordered = variants["row-order"]["metrics"]["pairs"]
        valid = all(p["complete_pairs"] and not p["incomplete_pairs"] for p in (temporal, ordered))
        reports["P11"]["status"] = ("PASS" if all(p["both_correct"] == p["complete_pairs"] for p in (temporal, ordered)) else "FAIL") if valid else "NOT_QUALIFIED"
        reports["P11"]["order_intervention"] = variants["row-order"]["metrics"]
    else:
        reports["P11"]["status"] = "NOT_QUALIFIED"
        reports["P11"]["reason"] = "both timestamp and serialized-order pairs required"
    ood = {key: r["metrics"] for key, r in variants.items() if key in ("alias", "template", "distractor", "combined")}
    if ood and base.get("decoder_schema", "utf8-raw") == "utf8-raw":
        reports["P12"].update(status=("PASS" if all(v["all"]["correct"] == v["all"]["episodes"] and v["all"]["invalid"] == 0 for v in ood.values()) else "FAIL") if len(ood) == 4 else "NOT_QUALIFIED", variants=ood)
    if "three-hop" in variants:
        reports["P13"].update(status=("PASS" if variants["three-hop"]["metrics"]["all"]["correct"] == variants["three-hop"]["metrics"]["all"]["episodes"] else "FAIL") if route_formed else "NOT_QUALIFIED",
            reason="two-hop formation qualified" if route_formed else "two-hop acquisition floor failed",
            diagnostic=variants["three-hop"]["metrics"])
    memory = {key: value["metrics"] for key, value in variants.items() if key.startswith("memory-")}
    if memory:
        reports["P14"].update(status="NOT_QUALIFIED",
            diagnostic=memory, note="prefetch is engineered; model-selected retrieval is reported separately; weights must be hashed before/after")
    required_memory = {"memory-on", "memory-off", "memory-irrelevant", "memory-swap"}
    if required_memory <= set(variants) and isinstance(weight_identity, dict):
        before, after = weight_identity.get("before"), weight_identity.get("after")
        records = [variants[k] for k in required_memory]
        base_ids = {r["case_id"] for r in variants["memory-on"]["episodes"]}
        matched = all({r["case_id"] for r in variants[k]["episodes"]} == base_ids for k in ("memory-off", "memory-irrelevant"))
        checkpoint = weight_identity.get("checkpoint", before)
        if before and before == after and checkpoint == base["model_identity"] and matched and all(r["model_identity"] == checkpoint and r["source_sha256"] == base["source_sha256"] for r in records):
            on, off, irrelevant = (variants[k]["metrics"]["all"] for k in ("memory-on", "memory-off", "memory-irrelevant"))
            swap = _intervention_pairs(variants["memory-swap"], "RETRIEVE")
            qualifies = swap["complete_pairs"] and not swap["incomplete_pairs"]
            succeeds = on["correct"] > off["correct"] and on["correct"] > irrelevant["correct"] and swap["both_follow_intervention"] == swap["complete_pairs"]
            reports["P14"].update(status=("PASS" if succeeds else "FAIL") if qualifies else "NOT_QUALIFIED", causal_swap=swap, weight_identity=weight_identity, reason="matched frozen-weight memory intervention")
    # These conclusions are conditional on actual positive-control learning.
    # Keep the raw outcomes visible even when the gate has not been met.
    if reports["P01"]["status"] != "PASS":
        for panel_id in ("P02", "P03", "P04", "P05", "P06", "P07", "P08"):
            reports[panel_id]["status"] = "NOT_QUALIFIED"
            reports[panel_id]["reason"] = "positive-control acquisition has not been established"
    reports["P15"].update(retention_panel(stage_records))
    if reports["P15"]["status"] not in ("PASS", "FAIL", "NOT_QUALIFIED", "SKIPPED"):
        reports["P15"]["status"] = "NOT_QUALIFIED"
    for panel_id, panel in reports.items():
        variant_key = {"P06": "payload-swap", "P10": "query-swap", "P11": "time-order", "P13": "three-hop", "P14": "memory-on"}.get(panel_id)
        rows = variants[variant_key]["episodes"] if variant_key in variants else base["episodes"]
        selected_kinds = {"P02": ("known",), "P03": ("missing",), "P04": ("conflict",), "P05": ("null", "unavailable")}.get(panel_id)
        if selected_kinds:
            rows = [r for r in rows if r["kind"] in selected_kinds]
        if panel_id == "P06" and "payload-swap" not in variants:
            rows = []
        if panel_id == "P01":
            rows = []  # positive-control receipt contains its own training/validation evidence
        if panel_id == "P11":
            rows = [r for k in ("time-order", "row-order") if k in variants for r in variants[k]["episodes"]]
        if panel_id == "P12":
            rows = [r for k in ("alias", "template", "distractor", "combined") if k in variants for r in variants[k]["episodes"]] if base.get("decoder_schema") == "utf8-raw" else []
        if panel_id == "P14":
            rows = [{**r, "_condition": k} for k in ("memory-on", "memory-off", "memory-irrelevant", "memory-swap") if k in variants for r in variants[k]["episodes"]]
        panel["samples"] = len(rows)
        panel["raw_predictions"] = [{"case_id": r["case_id"], "world_id": r["world_id"], "pair_id": r["pair_id"], "panel": r["panel"], "condition": r.get("_condition"), "trace": r["trace"], "correct": r["outcome"]["correct"], "expected_public_sources": r["expected_public_sources"], "observed_public_sources": r["observed_public_sources"]} for r in rows]
        n = len(rows)
        successes = sum(r["outcome"]["correct"] for r in rows)
        # Wilson interval is descriptive; correlated rows are not independent worlds.
        if n:
            z = 1.959963984540054
            phat = successes / n
            center = (phat + z*z/(2*n))/(1+z*z/n)
            half = z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/(1+z*z/n)
            clusters = {}
            for row in rows:
                clusters.setdefault(row["world_id"], []).append(int(row["outcome"]["correct"]))
            values = list(clusters.values())
            rng = random.Random(8675309)
            bootstrap = []
            for _ in range(500):
                sample = [values[rng.randrange(len(values))] for _ in values]
                bootstrap.append(sum(map(sum, sample)) / sum(map(len, sample)))
            bootstrap.sort()
            panel["uncertainty"] = {"method": "deterministic world-cluster bootstrap percentile 95%", "worlds": len(values),
                                    "interval": [bootstrap[12], bootstrap[487]],
                                    "episode_wilson_descriptive": [max(0., center-half), min(1., center+half)]}
        else:
            panel["uncertainty"] = None
        panel["source_sha256"] = base["source_sha256"]
        panel["dataset_sha256"] = base["dataset_sha256"]
    return reports


def diagnostic_cases(data, name):
    """Reconstruct exact variant cases for an independent raw-action replay.

    memory-on/off/prefetch share case identity; the receipt's memory_mode
    controls the actual episode. Irrelevant memory preserves case IDs but has
    a separately bound cases hash.
    """
    from .v6_data import panel_cases
    if name in ("memory-on", "memory-off", "memory-prefetch"):
        return panel_cases(data, "memory")
    if name != "memory-irrelevant":
        return panel_cases(data, name)
    cases = panel_cases(data, "memory")
    irrelevant = []
    for case in cases:
        state = CognitiveState(case.goal).observe(case.environment.initial)
        trail, subject = set(), case.goal.subject
        for _ in range(case.surface.hops):
            trail.add(subject)
            subject = belief(state.events, subject).value
            if subject is None:
                break
        choices = [slot for slot in case.goal.slots if slot not in trail and belief(state.events, slot).status == "derived"]
        if choices:
            original = case.memory.events[0]
            subject = choices[0]
            event = ExperienceEvent(case.goal.reusable_scope, subject, original.source, original.tick, belief(state.events, subject).value)
            irrelevant.append(replace(case, memory=MemoryVersion((event,))))
    return tuple(irrelevant) if len(irrelevant) == len(cases) else ()


def diagnose(data, policy_factory, *, variant_names=("payload-swap", "query-swap", "time-order", "row-order", "alias", "template", "distractor", "combined", "three-hop", "memory", "memory-swap"),
             positive_control=None, stage_records=None, calibration=None, controls=None, weight_identity=None, budget=DEFAULT_BUDGET, base_record=None):
    base = evaluate(data, policy_factory, budget=budget) if base_record is None else base_record
    if base_record is not None:
        independent_rescore(data, base_record)
    if controls:
        for control in controls.values():
            independent_rescore(data, control)
    variants = {}
    for name in variant_names:
        if name == "memory":
            for mode in ("on", "off", "prefetch", "irrelevant"):
                cases = diagnostic_cases(data, "memory-" + mode)
                if cases:
                    variants["memory-" + mode] = evaluate(cases, policy_factory, split=getattr(data, "split", "development"), budget=budget, memory_mode=mode if mode != "irrelevant" else "on")
        else:
            cases = diagnostic_cases(data, name)
            if cases:
                variants[name] = evaluate(cases, policy_factory, split=getattr(data, "split", "development"), budget=budget)
    return {"schema": "elendril-v6-checkpoint-diagnostics-v1", "registry_sha256": REGISTRY_SHA,
            "base": base, "variants": variants,
            "panels": panel_report(base, variants, positive_control=positive_control, stage_records=stage_records, calibration=calibration, controls=controls, weight_identity=weight_identity)}
