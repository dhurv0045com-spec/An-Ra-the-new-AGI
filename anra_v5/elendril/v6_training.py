"""V6 recipes reuse V5 updates, packed masks and transactional checkpoint state.

No import launches work. Production workers are CUDA-only; the explicit CPU
canary option is reserved for tiny mechanical tests, never the campaign CLI.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import math

from .codec import canonical, digest, encode as legacy_encode, decode as legacy_decode, BOS, EOS
from .cognition import CognitiveState, OperationCandidate
from .foundation_training import FoundationTrainer, Recipe, TokenSample, StageTransition
from .representation import UTF8, LEGACY
from .training import configure_device, runtime_identity
from .v6_registry import MODEL_PROFILES, LEGACY_MODEL_PROFILES


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    arm: str
    seed: int
    profile: str = "S3"
    epochs: int = 2
    representation: str = "utf8-raw"
    family: str = "all"

    def validate(self):
        import re
        if (re.fullmatch(r"[A-Za-z0-9_-]{1,90}", self.job_id) is None
                or self.stage not in ("qualification", "screen", "replicate", "scale", "sequential")
                or self.arm not in ("A1", "A2", "A3", "A4", "A5", "R0", "R1")
                or self.profile not in MODEL_PROFILES or type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.epochs) is not int or not 1 <= self.epochs <= 4
                or self.representation not in ("utf8-raw", "legacy-ascii", "utf8-ascii")
                or self.family not in ("all", "revision", "route")):
            raise ValueError("unregistered V6 job")
        if (self.arm == "R0") != (self.representation == "legacy-ascii"):
            raise ValueError("R0 is the separate legacy vocabulary comparison")
        if self.arm == "R1" and self.representation != "utf8-ascii":
            raise ValueError("R1 must match R0's public ASCII semantics")


class V6Trainer(FoundationTrainer):
    def indices(self, update):
        import random
        per_epoch = math.ceil(len(self.samples) / self.recipe.batch_size)
        epoch, batch = divmod(update, per_epoch)
        if epoch not in self._epoch_orders:
            order = list(range(len(self.samples)))
            random.Random(int(digest([self.recipe.seed, epoch, "foundation-order-v1"]), 16)).shuffle(order)
            if self.recipe.sampling == "uncertainty-first" and epoch == 0:
                order.sort(key=lambda i: self.samples[i].category not in ("missing", "conflict"))
            self._epoch_orders[epoch] = tuple(order)
        start = batch * self.recipe.batch_size
        return self._epoch_orders[epoch][start:start + self.recipe.batch_size]

    def sample_cursor(self, completed_updates):
        per_epoch = math.ceil(len(self.samples) / self.recipe.batch_size)
        epoch, batch = divmod(completed_updates, per_epoch)
        return epoch * len(self.samples) + min(batch * self.recipe.batch_size, len(self.samples))

    def runtime_binding(self):
        observed = runtime_identity(self.device)
        self.observed_runtime = observed
        if self.device.type == "cpu":
            return observed
        # GPU numerical state cannot depend on a disposable VM's CPU ISA/kernel
        # string. Keep CUDA/PyTorch/Python/numerical policy bound; record full
        # host identity externally. Cross-UUID continuation still needs measured
        # tensor/optimizer equivalence, never an asserted bitwise guarantee.
        return {"policy": "elendril-cuda-fp32-portable-v1", "torch": observed["torch"],
                "python": observed["python"], "dtype": observed["dtype"],
                "deterministic": observed["deterministic"], "threads": observed["intraop_threads"],
                "cuda": observed["cuda"]}


def ascii_prompt(state, channels, surface, memory_available=False):
    """Same raw evidence for both vocabulary arms, without derived beliefs.

    Public source labels receive fixed one-letter codes; list order is preserved.
    No alias/OOD renderer is silently claimed to fit the symbolic vocabulary.
    """
    sources = {"operator": "O", "sensor": "S", "directory": "D", "ledger": "L",
               "map": "M", "publication": "P"}
    rows = []
    events = state.events if surface.order == "original" else tuple(reversed(state.events))
    for event in events:
        # Unknown source names have a collision-free public index within this
        # prompt. Their actual spelling is not the question in this comparison.
        if event.source not in sources:
            used = set(sources.values())
            sources[event.source] = next(x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if x not in used)
        rows.append(f"{event.subject}:{event.value or '-'}:{event.tick}:{sources[event.source]}")
    scratch = "" if state.hypothesis is None else ";H:" + state.hypothesis
    return (f"{'G' if state.goal.family == 'route' else 'V'}:{surface.hops}:{state.goal.subject};"
            f"Q:{int(bool(channels))};R:{int(memory_available)};" + ";".join(rows) + scratch + ">")


def represented_samples(data, representation):
    from .v6_data import action_text
    if representation == "utf8-raw":
        return data.samples, data.manifest, UTF8
    tokenizer = LEGACY if representation == "legacy-ascii" else UTF8
    samples = []
    # Replay supervised public transitions rather than translating natural text
    # into a different oracle status code. Targets are the same actions.
    from .v6_data import public_action, CHANNEL
    for case in data.cases:
        state = CognitiveState(case.goal).observe(case.environment.initial)
        channels = tuple(case.environment.channels)
        memory = case.memory is not None
        for stage in range(4):
            operation, arg = public_action(state, channels, case.surface, memory)
            target = action_text(operation, arg).replace("~", "-")
            prompt = ascii_prompt(state, channels, case.surface, memory)
            framed = tokenizer.frame(prompt, target, max_length=MODEL_PROFILES["S3"].context_length)
            samples.append(TokenSample(case.case_id + "/ascii-" + str(stage), framed.ids, framed.loss_mask,
                                       case.world_id, case.kind))
            if operation == "QUERY":
                state = state.observe(case.environment.channels[arg]); channels = ()
            elif operation == "RETRIEVE":
                state = state.observe(case.memory.recall(case.goal)); memory = False
            elif operation == "DELIBERATE":
                state = replace(state, hypothesis=arg)
            else:
                break
    provenance = {**data.manifest, "representation": representation,
                  "semantic_parent": data.identity, "public_renderer": "raw-ascii-v1",
                  "comparison_limit": "same world/action targets; token counts and vocab parameters differ"}
    return tuple(samples), provenance, tokenizer


def create_trainer(data, job: Job, *, source, root, device="cuda:0", resume=False,
                   parent_store=None, transition=None, cpu_canary=False, spec_override=None,
                   updates_override=None):
    from .v6_data import transform_dataset, V6Dataset, _dataset
    job.validate()
    if device != "cuda:0" and not (cpu_canary is True and device == "cpu"):
        raise ValueError("V6 production workers require an isolated CUDA device; no CPU fallback")
    if job.family != "all":
        selected = tuple(c for c in data.cases if c.family == job.family)
        if not selected:
            raise ValueError("empty stage family")
        data = _dataset("train", selected, seed=data.manifest["seed"], parent=data.identity)
    if job.arm in ("A4", "A5"):
        data = transform_dataset(data, augmentation=job.arm == "A4", scratch=job.arm == "A5")
    samples, provenance, tokenizer = represented_samples(data, job.representation)
    batch = min(16, len(samples))
    updates = updates_override or math.ceil(len(samples) / batch) * job.epochs
    if not 1 <= updates <= 9216:
        raise ValueError("full epochs exceed the qualified update bound; reduce predeclared epochs before launch")
    spec = spec_override or (LEGACY_MODEL_PROFILES if tokenizer is LEGACY else MODEL_PROFILES)[job.profile]
    configure_device(device)
    recipe = Recipe(seed=job.seed, total_updates=updates, batch_size=batch,
                    micro_batch_size=min(2, batch), lr=3e-4,
                    objective="action-weighted" if job.arm == "A2" else "ce",
                    sampling="uncertainty-first" if job.arm == "A3" else "uniform",
                    schedule="cosine", precision="fp32", activation_checkpointing=False)
    provenance = {**provenance, "v6_job": asdict(job), "split": "train"}
    return V6Trainer(samples, spec=spec, recipe=recipe, source_commit=source, root=Path(root),
        lineage=job.job_id, data_provenance=provenance, tokenizer_manifest=tokenizer.manifest.record(),
        device=device, resume=resume, transition=transition, parent_store=parent_store)


class AsciiPolicy:
    """Independent equivalent-input representation control, no output repair."""
    origin = "model"
    def __init__(self, backend, tokenizer, surface=None, memory_available=False):
        self.backend, self.tokenizer, self.surface, self.memory = backend, tokenizer, surface, memory_available
        self.representation = "legacy-ascii" if tokenizer is LEGACY else "utf8-ascii"
        tokenizer.require_binding(backend.binding.tokenizer_sha256, backend.binding.vocabulary_size)
        self.model_identity = backend.binding.checkpoint_sha256 or backend.binding.identity

    def for_case(self, case, *, memory_available=None):
        return AsciiPolicy(self.backend, self.tokenizer, case.surface,
            case.memory is not None if memory_available is None else memory_available)

    def propose(self, state, legal, channels, positions):
        from .representation import SpecialToken
        prompt = ascii_prompt(state, channels, self.surface, self.memory and "RETRIEVE" in legal)
        ids = (self.tokenizer.bos_id, *self.tokenizer.encode(prompt))
        if self.tokenizer is UTF8:
            ids += (UTF8.special(SpecialToken.ANSWER),)
        if len(ids) + 3 > self.backend.binding.context_length or 4 * len(ids) + 6 > positions:
            return None
        generated = self.backend.generate((ids,), max_new_tokens=4)[0]
        op, arg = "INVALID", None
        if generated.ended_eos:
            try:
                text = self.tokenizer.decode(generated.tokens[:-1])
            except ValueError:
                text = ""
            if len(text) == 1 and text in "ABCDEFGHIJKLMNOP": op, arg = "ANSWER", text
            elif text == "Q": op, arg = "QUERY", channels[0] if len(channels) == 1 else None
            elif text == "!": op = "ABSTAIN"
            elif text == "R": op = "RETRIEVE"
            elif len(text) == 2 and text[0] == "-" and text[1] in "ABCDEFGHIJKLMNOP": op, arg = "DELIBERATE", text[1]
        return OperationCandidate(op, arg, math.prod(generated.token_probabilities),
                                  generated.evaluated_positions, generated.forward_calls, generated.tokens)


def policy_for(trainer, job):
    if job.representation == "utf8-raw":
        from .v6_evaluation import NeuralPolicy
        return NeuralPolicy(trainer.backend)
    return AsciiPolicy(trainer.backend, LEGACY if job.representation == "legacy-ascii" else UTF8)
