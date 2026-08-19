"""Connector failure-ablation loop: mechanical credit assignment over Core.

This is not part of the stable Core namespace. Core remains IDs → logits.
The experimenter treats Core as f(context, decode) → tokens, scores a fixed
verifier, and maps the flip pattern to one update target.

Ties fail closed to ``model_limitation``. Knowledge writes are refused unless
the class uniquely implicates K.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Literal

FAILURE_CLASSES = (
    "missing_knowledge",
    "wrong_knowledge",
    "bad_retrieval",
    "bad_planning",
    "weak_reasoning",
    "tool_execution_failure",
    "context_limit",
    "model_limitation",
    "representation_failure",
)
FailureClass = Literal[
    "missing_knowledge",
    "wrong_knowledge",
    "bad_retrieval",
    "bad_planning",
    "weak_reasoning",
    "tool_execution_failure",
    "context_limit",
    "model_limitation",
    "representation_failure",
]

INTERVENTION_ARMS = (
    "k_add",
    "k_swap",
    "k_retrieve",
    "plan_change",
    "decode_change",
    "tool_change",
    "truncated_k",
    "empty_k",
)

ARM_TO_CLASS: dict[str, FailureClass] = {
    "k_add": "missing_knowledge",
    "k_swap": "wrong_knowledge",
    "k_retrieve": "bad_retrieval",
    "plan_change": "bad_planning",
    "decode_change": "weak_reasoning",
    "tool_change": "tool_execution_failure",
    "truncated_k": "context_limit",
}

UPDATE_BY_CLASS: dict[FailureClass, str] = {
    "missing_knowledge": "write_memory",
    "wrong_knowledge": "correct_memory",
    "bad_retrieval": "adjust_retrieval",
    "bad_planning": "store_plan_template",
    "weak_reasoning": "raise_candidates_verify",
    "tool_execution_failure": "fix_tool_adapter",
    "context_limit": "compress_pack",
    "model_limitation": "queue_training_change_nothing",
    "representation_failure": "stop_do_not_learn",
}
KNOWLEDGE_WRITES = frozenset({"write_memory", "correct_memory"})


@dataclass(frozen=True, slots=True)
class DecodePolicy:
    temperature: float = 0.0
    top_p: float = 0.92
    candidates: int = 1
    seed: int = 0
    max_new_tokens: int = 16


@dataclass(frozen=True, slots=True)
class AttemptPack:
    knowledge: str
    plan: str
    question: str
    expected: str
    tool_ok: bool | None = None
    truncated: bool = False
    filler: bool = False

    def render(self) -> str:
        tool = ""
        if self.tool_ok is False:
            tool = "\n<tool>ERROR</tool>"
        elif self.tool_ok is True:
            tool = "\n<tool>OK</tool>"
        return (
            f"<k>{self.knowledge}</k>\n"
            f"<plan>{self.plan}</plan>"
            f"{tool}\n"
            f"<q>{self.question}</q>\n"
            "<answer>"
        )


@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    success: bool
    text: str = ""
    representation_error: bool = False


@dataclass(frozen=True, slots=True)
class Diagnosis:
    failure_class: FailureClass
    update: str
    flips: tuple[str, ...]
    baseline_success: bool
    write_knowledge: bool


@dataclass(frozen=True, slots=True)
class PlantedItem:
    item_id: str
    planted_class: FailureClass
    question: str
    expected: str
    gold_fact: str
    poison_fact: str
    distractor: str
    good_plan: str
    bad_plan: str
    filler: str = ""


@dataclass
class SuiteReport:
    n: int
    accuracy: float
    always_model_limitation: float
    false_knowledge_rate: float
    confusion: dict[str, dict[str, int]]
    by_class: dict[str, float]
    predictions: list[tuple[str, str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "always_model_limitation": self.always_model_limitation,
            "false_knowledge_rate": self.false_knowledge_rate,
            "by_class": self.by_class,
            "confusion": self.confusion,
        }


def classify_from_arms(results: dict[str, ArmResult]) -> Diagnosis:
    """Map a completed battery to one class. Ties → model_limitation."""
    baseline = results.get("baseline")
    if baseline is None:
        raise ValueError("battery requires a baseline arm")
    if baseline.representation_error or any(
        item.representation_error for item in results.values()
    ):
        return Diagnosis(
            "representation_failure",
            UPDATE_BY_CLASS["representation_failure"],
            (),
            False,
            False,
        )
    flips = tuple(
        arm
        for arm in INTERVENTION_ARMS
        if arm in results
        and results[arm].success
        and not baseline.success
        and arm != "empty_k"
    )
    if len(flips) == 1:
        failure_class = ARM_TO_CLASS[flips[0]]
    else:
        failure_class = "model_limitation"
    update = UPDATE_BY_CLASS[failure_class]
    return Diagnosis(
        failure_class=failure_class,
        update=update,
        flips=flips,
        baseline_success=baseline.success,
        write_knowledge=update in KNOWLEDGE_WRITES,
    )


def verify_answer(text: str, expected: str) -> bool:
    needle = expected.strip().lower()
    if not needle:
        return False
    return needle in text.strip().lower()


def planted_suite() -> tuple[PlantedItem, ...]:
    """80 items, 10 per actionable class, closed-domain exact verifiers."""
    capitals = (
        ("France", "Paris", "Lyon", "France is in Europe"),
        ("Germany", "Berlin", "Munich", "Germany is in Europe"),
        ("Japan", "Tokyo", "Osaka", "Japan is in Asia"),
        ("Italy", "Rome", "Milan", "Italy is in Europe"),
        ("Spain", "Madrid", "Barcelona", "Spain is in Europe"),
        ("Canada", "Ottawa", "Toronto", "Canada is in North America"),
        ("Australia", "Canberra", "Sydney", "Australia is an island"),
        ("Egypt", "Cairo", "Alexandria", "Egypt is in Africa"),
        ("Peru", "Lima", "Cusco", "Peru is in South America"),
        ("Sweden", "Stockholm", "Gothenburg", "Sweden is in Europe"),
    )
    items: list[PlantedItem] = []
    for index, (country, capital, decoy, region) in enumerate(capitals, start=1):
        gold = f"The capital of {country} is {capital}."
        poison = f"The capital of {country} is {decoy}."
        question = f"What is the capital of {country}?"
        plan_read = f"Read the capital fact for {country}."
        plan_bad = f"Name any large city in {country}."
        items.append(
            PlantedItem(
                f"missing-{index:02d}",
                "missing_knowledge",
                question,
                capital,
                gold,
                poison,
                region,
                plan_read,
                plan_bad,
            )
        )
        items.append(
            PlantedItem(
                f"wrong-{index:02d}",
                "wrong_knowledge",
                question,
                capital,
                gold,
                poison,
                region,
                plan_read,
                plan_bad,
            )
        )
        items.append(
            PlantedItem(
                f"retr-{index:02d}",
                "bad_retrieval",
                question,
                capital,
                gold,
                poison,
                region + ".",
                plan_read,
                plan_bad,
            )
        )
        items.append(
            PlantedItem(
                f"ctx-{index:02d}",
                "context_limit",
                question,
                capital,
                gold,
                poison,
                region,
                plan_read,
                plan_bad,
                filler=" ".join(["note"] * 40),
            )
        )

    arith = (
        ("3+4 then *2", "14", "add 3 and 4 then multiply by 2", "multiply 3 and 4 then add 2"),
        ("5+1 then *3", "18", "add 5 and 1 then multiply by 3", "multiply 5 and 1 then add 3"),
        ("2+6 then *2", "16", "add 2 and 6 then multiply by 2", "multiply 2 and 6 then add 2"),
        ("7+3 then *2", "20", "add 7 and 3 then multiply by 2", "multiply 7 and 3 then add 2"),
        ("4+4 then *3", "24", "add 4 and 4 then multiply by 3", "multiply 4 and 4 then add 3"),
        ("9+1 then *2", "20", "add 9 and 1 then multiply by 2", "multiply 9 and 1 then add 2"),
        ("8+2 then *2", "20", "add 8 and 2 then multiply by 2", "multiply 8 and 2 then add 2"),
        ("6+6 then *2", "24", "add 6 and 6 then multiply by 2", "multiply 6 and 6 then add 2"),
        ("1+4 then *5", "25", "add 1 and 4 then multiply by 5", "multiply 1 and 4 then add 5"),
        ("2+3 then *4", "20", "add 2 and 3 then multiply by 4", "multiply 2 and 3 then add 4"),
    )
    for index, (question, expected, good, bad) in enumerate(arith, start=1):
        items.append(
            PlantedItem(
                f"plan-{index:02d}",
                "bad_planning",
                f"Compute: {question}",
                expected,
                f"numbers {question}",
                "ignore order",
                "random arithmetic",
                good,
                bad,
            )
        )

    riddles = (
        ("Echo the word ping.", "ping"),
        ("Echo the word oak.", "oak"),
        ("Echo the word zinc.", "zinc"),
        ("Echo the word ember.", "ember"),
        ("Echo the word quartz.", "quartz"),
        ("Echo the word linen.", "linen"),
        ("Echo the word marble.", "marble"),
        ("Echo the word barley.", "barley"),
        ("Echo the word copper.", "copper"),
        ("Echo the word cedar.", "cedar"),
    )
    for index, (question, expected) in enumerate(riddles, start=1):
        items.append(
            PlantedItem(
                f"dec-{index:02d}",
                "weak_reasoning",
                question,
                expected,
                f"The required echo is {expected}.",
                "",
                "",
                "repeat the required echo word",
                "invent a different word",
            )
        )

    tools = (
        ("add 20 and 22", "42"),
        ("add 10 and 7", "17"),
        ("add 15 and 15", "30"),
        ("add 8 and 9", "17"),
        ("add 11 and 14", "25"),
        ("add 3 and 33", "36"),
        ("add 19 and 2", "21"),
        ("add 40 and 5", "45"),
        ("add 6 and 12", "18"),
        ("add 1 and 99", "100"),
    )
    for index, (question, expected) in enumerate(tools, start=1):
        items.append(
            PlantedItem(
                f"tool-{index:02d}",
                "tool_execution_failure",
                f"Calculator: {question}",
                expected,
                f"tool result {expected}",
                "",
                "",
                "read the calculator result",
                "guess",
            )
        )

    hard = (
        ("17 + 25", "42"),
        ("128 * 47", "6016"),
        ("def add(a,b):", "return a+b"),
        ("SHA256 of an-ra", "not memorized"),
        ("integral of x^2", "x^3/3"),
        ("13^3", "2197"),
        ("gcd(252, 105)", "21"),
        ("binary of 1024", "10000000000"),
        ("9th Mersenne prime", "2305843009213693951"),
        ("sort unique token ids", "monotonic"),
    )
    for index, (question, expected) in enumerate(hard, start=1):
        items.append(
            PlantedItem(
                f"lim-{index:02d}",
                "model_limitation",
                question,
                expected,
                "",
                "",
                "",
                "solve exactly",
                "solve exactly",
            )
        )

    assert len(items) == 80, len(items)
    counts = Counter(item.planted_class for item in items)
    assert all(counts[name] == 10 for name in FAILURE_CLASSES if name != "representation_failure")
    return tuple(items)


def arms_for(item: PlantedItem) -> dict[str, tuple[AttemptPack, DecodePolicy]]:
    """Build baseline + 8 interventions. Exactly one arm is intended to flip."""
    greedy = DecodePolicy()
    sampled = DecodePolicy(temperature=0.8, candidates=4, seed=1)
    read = item.good_plan
    empty = AttemptPack("", read, item.question, item.expected)
    gold = AttemptPack(item.gold_fact, read, item.question, item.expected)
    poison = AttemptPack(item.poison_fact, read, item.question, item.expected)
    distract = AttemptPack(item.distractor, read, item.question, item.expected)
    bad_plan = AttemptPack(item.gold_fact, item.bad_plan, item.question, item.expected)
    filled = AttemptPack(
        f"{item.filler} {item.gold_fact}".strip(),
        read,
        item.question,
        item.expected,
        filler=True,
    )
    truncated = AttemptPack(
        item.gold_fact, read, item.question, item.expected, truncated=True
    )
    tool_fail = AttemptPack(
        item.gold_fact, read, item.question, item.expected, tool_ok=False
    )
    tool_ok = AttemptPack(
        item.gold_fact, read, item.question, item.expected, tool_ok=True
    )

    planted = item.planted_class
    if planted == "missing_knowledge":
        baseline = empty
    elif planted == "wrong_knowledge":
        baseline = poison
    elif planted == "bad_retrieval":
        baseline = distract
    elif planted == "bad_planning":
        baseline = bad_plan
    elif planted == "weak_reasoning":
        baseline = gold
    elif planted == "tool_execution_failure":
        baseline = tool_fail
    elif planted == "context_limit":
        baseline = filled
    else:
        baseline = gold if item.gold_fact else empty

    if planted == "bad_planning":
        plan_pack = AttemptPack(
            baseline.knowledge,
            item.good_plan,
            item.question,
            item.expected,
            baseline.tool_ok,
            baseline.truncated,
            baseline.filler,
        )
    else:
        plan_pack = baseline
    return {
        "baseline": (baseline, greedy),
        "k_add": (gold if planted == "missing_knowledge" else baseline, greedy),
        "k_swap": (gold if planted == "wrong_knowledge" else baseline, greedy),
        "k_retrieve": (gold if planted == "bad_retrieval" else baseline, greedy),
        "plan_change": (plan_pack, greedy),
        "decode_change": (baseline, sampled if planted == "weak_reasoning" else greedy),
        "tool_change": (tool_ok if planted == "tool_execution_failure" else baseline, greedy),
        "truncated_k": (truncated if planted == "context_limit" else baseline, greedy),
        "empty_k": (empty, greedy),
    }


def oracle_success(item: PlantedItem, pack: AttemptPack, policy: DecodePolicy) -> bool:
    """Planted physics: success depends on pack contents, not on arm name."""
    planted = item.planted_class
    if planted == "model_limitation":
        return False
    if planted == "missing_knowledge":
        return item.gold_fact in pack.knowledge
    if planted == "wrong_knowledge":
        return item.gold_fact in pack.knowledge and item.poison_fact not in pack.knowledge
    if planted == "bad_retrieval":
        return item.gold_fact in pack.knowledge and item.distractor not in pack.knowledge
    if planted == "bad_planning":
        return pack.plan == item.good_plan
    if planted == "weak_reasoning":
        return policy.temperature > 0.0 or policy.candidates > 1
    if planted == "tool_execution_failure":
        return pack.tool_ok is True
    if planted == "context_limit":
        return pack.truncated and item.gold_fact in pack.knowledge and not pack.filler
    return False


Completer = Callable[[PlantedItem, AttemptPack, DecodePolicy], ArmResult]


def oracle_completer(item: PlantedItem, pack: AttemptPack, policy: DecodePolicy) -> ArmResult:
    success = oracle_success(item, pack, policy)
    return ArmResult("arm", success, item.expected if success else "")


def run_battery(item: PlantedItem, completer: Completer) -> dict[str, ArmResult]:
    results: dict[str, ArmResult] = {}
    for arm, (pack, policy) in arms_for(item).items():
        try:
            raw = completer(item, pack, policy)
        except Exception as exc:
            name = type(exc).__name__
            representation = "Representation" in name or "ContextOverflow" in name
            results[arm] = ArmResult(arm, False, "", representation_error=representation)
            continue
        results[arm] = ArmResult(arm, raw.success, raw.text, raw.representation_error)
    return results


def _decode_from_logits(
    executor: object,
    tokenizer: object,
    state: object,
    logits: object,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    import torch

    from .executor import CoreExecutor
    from .state import CoreState
    from .tokenizer import V4Tokenizer

    assert isinstance(executor, CoreExecutor)
    assert isinstance(tokenizer, V4Tokenizer)
    assert isinstance(state, CoreState)
    device = executor.device
    generated: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed)
    current = logits
    for _ in range(max_new_tokens):
        if temperature <= 0:
            next_id = int(current.argmax(dim=-1).item())
        else:
            probabilities = torch.softmax(current / temperature, dim=-1)
            sorted_probs, sorted_ids = probabilities.sort(descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)
            remove = cumulative - sorted_probs > top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0)
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            choice = torch.multinomial(sorted_probs, 1, generator=generator)
            next_id = int(sorted_ids.gather(-1, choice).item())
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)
        pred = executor.forward_step(next_id, state=state)
        current = pred.logits[:, -1, :]
    return tokenizer.decode(generated)


def run_core_battery(
    item: PlantedItem,
    executor: object,
    tokenizer: object,
) -> dict[str, ArmResult]:
    """Prefill baseline once; fork for the decode arm when the prompt is identical."""
    import torch

    from .errors import CoreError, RepresentationIncompatibleError
    from .executor import CoreExecutor
    from .generate import generate
    from .tokenizer import V4Tokenizer

    if not isinstance(executor, CoreExecutor) or not isinstance(tokenizer, V4Tokenizer):
        raise TypeError("run_core_battery requires CoreExecutor and V4Tokenizer")

    arms = arms_for(item)
    results: dict[str, ArmResult] = {}
    baseline_pack, baseline_policy = arms["baseline"]
    prompt = baseline_pack.render()
    prompt_ids = tokenizer.encode(prompt)
    required = 1 + len(prompt_ids) + baseline_policy.max_new_tokens
    parent = executor.create_state(capacity=min(required, executor.model.config.block_size))
    try:
        ids = torch.tensor(
            [[tokenizer.bos_token_id, *prompt_ids]],
            dtype=torch.long,
            device=executor.device,
        )
        pred = executor.prefill(ids, state=parent)
        greedy_state = executor.fork_state(parent)
        try:
            greedy_text = _decode_from_logits(
                executor,
                tokenizer,
                greedy_state,
                pred.logits[:, -1, :],
                max_new_tokens=baseline_policy.max_new_tokens,
                temperature=0.0,
                top_p=baseline_policy.top_p,
                seed=baseline_policy.seed,
            )
            results["baseline"] = ArmResult(
                "baseline",
                verify_answer(greedy_text, baseline_pack.expected),
                greedy_text,
            )
        finally:
            if not greedy_state.is_released:
                executor.release_state(greedy_state)

        decode_pack, decode_policy = arms["decode_change"]
        if decode_pack.render() == prompt:
            child = executor.fork_state(parent)
            try:
                texts = []
                for index in range(max(1, decode_policy.candidates)):
                    sample_state = executor.fork_state(child)
                    try:
                        texts.append(
                            _decode_from_logits(
                                executor,
                                tokenizer,
                                sample_state,
                                pred.logits[:, -1, :],
                                max_new_tokens=decode_policy.max_new_tokens,
                                temperature=decode_policy.temperature,
                                top_p=decode_policy.top_p,
                                seed=decode_policy.seed + index,
                            )
                        )
                    finally:
                        if not sample_state.is_released:
                            executor.release_state(sample_state)
                text = next(
                    (row for row in texts if verify_answer(row, decode_pack.expected)),
                    texts[0],
                )
                results["decode_change"] = ArmResult(
                    "decode_change",
                    verify_answer(text, decode_pack.expected),
                    text,
                )
            finally:
                if not child.is_released:
                    executor.release_state(child)
    except RepresentationIncompatibleError:
        results["baseline"] = ArmResult("baseline", False, "", True)
        results["decode_change"] = ArmResult("decode_change", False, "", True)
    except CoreError:
        results.setdefault("baseline", ArmResult("baseline", False, ""))
        results.setdefault("decode_change", ArmResult("decode_change", False, ""))
    finally:
        if not parent.is_released:
            executor.release_state(parent)

    for arm, (pack, policy) in arms.items():
        if arm in results:
            continue
        try:
            text = generate(
                executor,
                tokenizer,
                pack.render(),
                max_new_tokens=policy.max_new_tokens,
                temperature=policy.temperature,
                top_p=policy.top_p,
                seed=policy.seed,
            )
            results[arm] = ArmResult(arm, verify_answer(text, pack.expected), text)
        except RepresentationIncompatibleError:
            results[arm] = ArmResult(arm, False, "", True)
        except CoreError:
            results[arm] = ArmResult(arm, False, "")
    return results


def diagnose_item(
    item: PlantedItem,
    completer: Completer | None = None,
    *,
    battery: dict[str, ArmResult] | None = None,
) -> Diagnosis:
    if battery is None:
        if completer is None:
            raise ValueError("diagnose_item requires completer or battery")
        battery = run_battery(item, completer)
    return classify_from_arms(battery)


def evaluate_suite(
    completer: Completer | None = None,
    items: tuple[PlantedItem, ...] | None = None,
    *,
    core: tuple[object, object] | None = None,
) -> SuiteReport:
    items = items or planted_suite()
    if completer is None and core is None:
        raise ValueError("evaluate_suite requires completer or core")
    confusion: dict[str, dict[str, int]] = {
        truth: {pred: 0 for pred in FAILURE_CLASSES} for truth in FAILURE_CLASSES
    }
    predictions: list[tuple[str, str, str]] = []
    correct = 0
    false_writes = 0
    ineligible = 0
    always_lim = 0
    per_class_hits: Counter[str] = Counter()
    per_class_n: Counter[str] = Counter()
    for item in items:
        if core is not None:
            diagnosis = diagnose_item(
                item, battery=run_core_battery(item, core[0], core[1])
            )
        else:
            diagnosis = diagnose_item(item, completer)
        predictions.append((item.item_id, item.planted_class, diagnosis.failure_class))
        confusion[item.planted_class][diagnosis.failure_class] += 1
        per_class_n[item.planted_class] += 1
        if diagnosis.failure_class == item.planted_class:
            correct += 1
            per_class_hits[item.planted_class] += 1
        if diagnosis.failure_class == "model_limitation":
            always_lim += 1
        if item.planted_class not in {"missing_knowledge", "wrong_knowledge"}:
            ineligible += 1
            if diagnosis.write_knowledge:
                false_writes += 1
    n = len(items)
    by_class = {
        name: (per_class_hits[name] / per_class_n[name] if per_class_n[name] else 0.0)
        for name in FAILURE_CLASSES
        if per_class_n[name]
    }
    return SuiteReport(
        n=n,
        accuracy=correct / n if n else 0.0,
        always_model_limitation=always_lim / n if n else 0.0,
        false_knowledge_rate=false_writes / ineligible if ineligible else 0.0,
        confusion=confusion,
        by_class=by_class,
        predictions=predictions,
    )


def core_completer(executor: object, tokenizer: object) -> Completer:
    """Real Core generate. Decode-policy arm forks after a shared prefill."""
    from .errors import CoreError, RepresentationIncompatibleError
    from .executor import CoreExecutor
    from .generate import generate
    from .tokenizer import V4Tokenizer

    if not isinstance(executor, CoreExecutor) or not isinstance(tokenizer, V4Tokenizer):
        raise TypeError("core_completer requires CoreExecutor and V4Tokenizer")

    def complete(item: PlantedItem, pack: AttemptPack, policy: DecodePolicy) -> ArmResult:
        prompt = pack.render()
        try:
            if policy.candidates > 1 or policy.temperature > 0:
                texts = [
                    generate(
                        executor,
                        tokenizer,
                        prompt,
                        max_new_tokens=policy.max_new_tokens,
                        temperature=policy.temperature,
                        top_p=policy.top_p,
                        seed=policy.seed + index,
                    )
                    for index in range(max(1, policy.candidates))
                ]
                text = next((row for row in texts if verify_answer(row, pack.expected)), texts[0])
            else:
                text = generate(
                    executor,
                    tokenizer,
                    prompt,
                    max_new_tokens=policy.max_new_tokens,
                    temperature=0.0,
                    top_p=policy.top_p,
                    seed=policy.seed,
                )
        except RepresentationIncompatibleError:
            return ArmResult("arm", False, "", representation_error=True)
        except CoreError:
            return ArmResult("arm", False, "", representation_error=False)
        return ArmResult("arm", verify_answer(text, pack.expected), text)

    return complete


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the An-Ra failure-ablation suite")
    parser.add_argument("--oracle", action="store_true", help="evaluate planted physics, no Core")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.oracle:
        report = evaluate_suite(oracle_completer)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    if not args.checkpoint:
        parser.error("provide --oracle or --checkpoint")
    from .executor import CoreExecutor

    executor = CoreExecutor.from_checkpoint(args.checkpoint, device=args.device)
    tokenizer = executor.tokenizer
    if tokenizer is None:
        raise RuntimeError("executor did not bind a tokenizer")
    report = evaluate_suite(core=(executor, tokenizer))
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
