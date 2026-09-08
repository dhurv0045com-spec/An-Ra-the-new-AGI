"""CYR-GPU-005 torch orchestrator: shared-parent fork campaign.

ONE orchestrator executes every stage in both modes; only the hardware
resolver's outputs differ:

    startup -> tokenizer identity -> data render + manifest + leak audit ->
    calibration -> hardware resolver -> per seed: ONE acquisition to
    G90_CONFIRMED (or NOT_QUALIFIED, no forks) -> parent checkpoint ->
    four fork restorations (byte-equality receipt) -> HIGH / LOW /
    FIXED_TIME / HYST continuation arms on the shared future tail ->
    transfer stage -> red team -> decision -> evidence bundle.

Actual tokens are the only exposure currency. Wall budget is one absolute
campaign deadline. Full mode fails closed without CUDA. Smoke mode runs
TINY locally for plumbing evidence and is the only mode allowed a
PLUMBING_OVERRIDE of the G90 gate (labeled in every receipt, forbidden in
full mode — the override exercises the gate CODE PATH, it never fabricates
a science result).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from v5_experiments import cyr_gpu005 as core  # noqa: E402
from v5_experiments.cyr_tournament import (  # noqa: E402
    HysteresisController,
    sustained,
)

CAMPAIGN_SCHEMA = "anra-cyr-gpu005-campaign/v1"
PARENT_EQUIVALENCE_SCHEMA = core.PARENT_SCHEMA
NOTEBOOK_PATH = "notebooks/cymek_colab_gpu_research_v5.ipynb"
TOKENIZER_ARTIFACT = "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
BUNDLE_NAME = "CYMEK_GPU_RESEARCH_V5_RESULTS.zip"


# -- environment / identity ---------------------------------------------------

def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def environment_receipt(*, torch: Any, device: Any, mode: str) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False
    receipt: dict[str, Any] = {
        "schema": "anra-cyr-gpu005-environment/v1", "mode": mode,
        "torch": torch.__version__, "cuda_available": cuda_available,
        "device": str(device)}
    if cuda_available:
        receipt["gpu_name"] = torch.cuda.get_device_name(0)
        receipt["vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return receipt


def production_tokenizer(repo: Path = REPO) -> tuple[Any, dict[str, Any]]:
    """Load the FROZEN 24,576 tokenizer through the validating adapter and
    cross-check the special IDs the artifact itself declares (section 12).
    Hardcoded EOS is a defect; the identity receipt binds artifact bytes and
    the IDs read from those bytes."""

    from v5_data.corpus_loading import _load_tokenizer
    artifact = repo / TOKENIZER_ARTIFACT
    tokenizer, _evaluation = _load_tokenizer(repo.resolve())
    identity = tokenizer.identity
    import gzip
    body = json.loads(gzip.decompress(artifact.read_bytes()).decode("utf-8"))
    added = {entry["content"]: entry["id"]
             for entry in body.get("added_tokens", [])}
    for required in ("<pad>", "<unk>", "<bos>", "<eos>"):
        if required not in added:
            raise ValueError(f"tokenizer artifact does not declare {required}")
    if sha256_file(artifact) != identity.artifact_sha256:
        raise ValueError("tokenizer artifact bytes do not match its identity")
    receipt = {
        "schema": "anra-cyr-gpu005-tokenizer-identity/v1",
        "artifact": TOKENIZER_ARTIFACT,
        "artifact_sha256": identity.artifact_sha256,
        "trainer_config_sha256": identity.trainer_config_sha256,
        "corpus_manifest_sha256": identity.corpus_manifest_sha256,
        "vocabulary_size": int(tokenizer.vocabulary_size),
        "pad_id": added["<pad>"], "unk_id": added["<unk>"],
        "bos_id": added["<bos>"], "eos_id": added["<eos>"],
        "special_ids_identity": dict(identity.special_token_ids),
        "source": "read from artifact added_tokens, cross-checked against "
                  "the frozen TokenizerIdentity; never hardcoded"}
    if receipt["vocabulary_size"] != 24576:
        raise ValueError("frozen production tokenizer must carry 24,576 slots")
    if (receipt["pad_id"], receipt["bos_id"], receipt["eos_id"]) \
            != (identity.special_token_ids["pad"],
                identity.special_token_ids["bos"],
                identity.special_token_ids["eos"]):
        raise ValueError("artifact special IDs disagree with the frozen identity")
    return tokenizer, receipt


# -- batches and generation ----------------------------------------------------

def render_batch(tokenizer: Any, rows: list[dict[str, Any]], *, torch: Any,
                 device: Any, special: Mapping[str, int],
                 ) -> tuple[Any, Any, Any, dict[str, int]]:
    """One padded batch. Target = answer + EOS (EOS carries loss); the
    prompt and all padding carry none (section 13). ``eligible`` indexes
    TARGET positions: entry i supervises token i for i >= 1."""

    bos, eos, pad = special["bos_id"], special["eos_id"], special["pad_id"]
    encoded: list[tuple[list[int], int]] = []
    for row in rows:
        prompt_ids = tokenizer.encode(row["prompt"])
        answer_ids = tokenizer.encode(row["answer"])
        encoded.append(([bos, *prompt_ids, *answer_ids, eos],
                        1 + len(prompt_ids)))
    width = max(len(ids) for ids, _prompt_len in encoded)
    tokens = torch.tensor(
        [ids + [pad] * (width - len(ids)) for ids, _ in encoded],
        dtype=torch.long, device=device)
    segment_ids = torch.tensor(
        [[0] * len(ids) + [-1] * (width - len(ids)) for ids, _ in encoded],
        dtype=torch.long, device=device)
    eligible_rows = []
    for ids, prompt_len in encoded:
        eligible_rows.append([False] * prompt_len
                             + [True] * (len(ids) - prompt_len)
                             + [False] * (width - len(ids)))
    eligible = torch.tensor(eligible_rows, dtype=torch.bool, device=device)
    return tokens, segment_ids, eligible, {
        "real_tokens": int((segment_ids >= 0).sum().item()),
        "supervised_tokens": int(eligible.sum().item())}


def generate_rates(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *,
                   torch: Any, device: Any, special: Mapping[str, int],
                   max_new_tokens: int = 8) -> dict[str, Any]:
    """Candidate-free evaluation: prompt only, greedy generate answer + EOS
    (section 14). Reports the five preregistered rates (section 13)."""

    from v5_model.core import packed_layout
    bos, eos, pad = special["bos_id"], special["eos_id"], special["pad_id"]
    was_training = model.training
    model.eval()
    content_exact = complete_exact = eos_stops = capped = prefix_extra = 0
    for row in rows:
        ids = [bos, *tokenizer.encode(row["prompt"])]
        generated: list[int] = []
        stopped = False
        hit_cap = False
        with torch.no_grad():
            for _ in range(max_new_tokens):
                current = torch.tensor([ids + generated], device=device)
                positions, mask = packed_layout(
                    torch.tensor([[0] * current.shape[1]], device=device),
                    torch_module=torch)
                logits = model(current, positions, mask)[0, -1]
                next_id = int(torch.argmax(logits).item())
                if next_id == eos:
                    stopped = True
                    break
                if next_id == pad:
                    break
                generated.append(next_id)
            else:
                hit_cap = True
        text = tokenizer.decode(generated) if generated else ""
        expected = row["answer"]
        if text == expected:
            content_exact += 1
            if stopped:
                complete_exact += 1
        if stopped:
            eos_stops += 1
        if hit_cap:
            capped += 1
        if text != expected and expected.startswith(text) and text:
            prefix_extra += 1
    if was_training:
        model.train()
    total = len(rows)
    return {"content_exact": round(content_exact / total, 4) if total else 0.0,
            "complete_exact_with_valid_stop": round(complete_exact / total, 4) if total else 0.0,
            "eos_rate": round(eos_stops / total, 4) if total else 0.0,
            "max_tokens_rate": round(capped / total, 4) if total else 0.0,
            "prefix_correct_extra": round(prefix_extra / total, 4) if total else 0.0,
            "total": total}


# -- checkpoints ----------------------------------------------------------------

def save_parent_checkpoint(path: str | Path, *, model: Any, optimizer: Any,
                           torch: Any, counters: Mapping[str, Any]) -> dict[str, Any]:
    from anra_v5.cyr_execute import save_research_checkpoint
    return save_research_checkpoint(path, model=model, optimizer=optimizer,
                                    torch=torch, counters=dict(counters))


def load_parent_checkpoint(path: str | Path, *, model: Any, optimizer: Any,
                           torch: Any) -> dict[str, Any]:
    from anra_v5.cyr_execute import load_research_checkpoint
    return load_research_checkpoint(path, model=model, optimizer=optimizer,
                                    torch=torch)


def state_fingerprint(model: Any, optimizer: Any, *, torch: Any) -> dict[str, str]:
    """Byte-exact hashes of model and optimizer state (fork equality)."""

    model_buffer, optim_buffer = io.BytesIO(), io.BytesIO()
    torch.save(model.state_dict(), model_buffer)
    torch.save(optimizer.state_dict(), optim_buffer)
    return {"model_sha256": hashlib.sha256(model_buffer.getvalue()).hexdigest(),
            "optimizer_sha256": hashlib.sha256(optim_buffer.getvalue()).hexdigest()}


def parent_equivalence_receipt(forks: Mapping[str, tuple[Any, Any]], *,
                               torch: Any) -> dict[str, Any]:
    """All four forks must hold THE SAME model and optimizer bytes before
    their first continuation update (sections 5, 7). Counters (token
    counters, stream cursor, RNG-state and parameter-snapshot hashes
    captured at the checkpoint) must match too — same checkpoint bytes."""

    fingerprints = {name: state_fingerprint(model, optimizer, torch=torch)
                    for name, (model, optimizer) in forks.items()}
    reference = next(iter(fingerprints.values()))
    identical = all(entry == reference for entry in fingerprints.values())
    receipt = {"schema": PARENT_EQUIVALENCE_SCHEMA,
               "forks": fingerprints, "identical": identical,
               "checked": "model state_dict bytes + optimizer state bytes "
                          "+ checkpoint counters (RNG-state hash, cursor, "
                          "parameter snapshot)"}
    if not identical:
        raise ValueError("parent equivalence FAILED: forks did not restore "
                         "the same bytes — refusing to run the campaign")
    return receipt


# -- acquisition ------------------------------------------------------------------

ARM_SEED_OFFSETS = {arm: 101 * (index + 1)
                    for index, arm in enumerate(core.CYR5_ARMS)}


def acquire_parent(*, seed: int, spec: Any, proxy_name: str,
                   tokenizer: Any, torch: Any,
                   device: Any, special: Mapping[str, int],
                   train_rows: list[dict[str, Any]],
                   controller_rows: list[dict[str, Any]],
                   probe_rows: list[dict[str, Any]],
                   target_actual_tokens: int,
                   stream: Mapping[str, Any],
                   store_root: Path, run_id: str,
                   deadline_s: float | None,
                   lr: float = core.CYR5_LRS["HIGH"],
                   eval_every_updates: int = 4,
                   microbatch_rows: int = 16,
                   plumbing_override: bool = False,
                   progress: Callable[[str], None] | None = None,
                   ) -> dict[str, Any]:
    """EXACTLY ONE acquisition run per parent seed (sections 5, 6).

    Consumes the acquisition prefix of the shared stream. G90 requires the
    candidate-free generated metric sustained for 3 consecutive evaluations
    on DEV_CONTROLLER. On confirmation the parent checkpoint captures model,
    optimizer, RNG, cursor, counters, and a flat parameter snapshot. An
    unqualified parent is reported NOT_QUALIFIED and forks are refused.
    """

    from anra_v5.cyr_execute import optimizer_moment_norms
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.state import CURSOR_SCHEMA, CursorState

    torch.manual_seed(seed)
    model = initialize(spec, seed, torch_module=torch).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    core.assert_proxy_in_registry(proxy_name, parameters,
                                  core.proxy_registry(vocab_size=spec.vocabulary_size))
    optimizer = build_adamw_optimizer(model, torch_module=torch, lr=lr)
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=special["bos_id"],
        pad_id=special["pad_id"], device=device,
        schedule=lambda cumulative_tokens: lr, bfloat16_autocast=False,
        torch_module=torch, activation_checkpointing=False)

    prefix = list(stream["stream"])[:stream["fork_boundary"]]
    data_sha = hashlib.sha256(_canonical_json(
        [row["prompt"] + row["answer"] for row in train_rows])).hexdigest()
    eval_trace: list[dict[str, Any]] = []
    controller_flags: list[bool] = []
    probe_flags: list[bool] = []
    consumed = 0
    updates = 0
    status = "RUNNING"
    onset = confirm = None
    cursor_offset = 0
    parent_path = store_root / run_id / "parent"
    while consumed < target_actual_tokens:
        if deadline_s is not None and time.monotonic() >= deadline_s:
            status = "TIMEBOX"
            break
        batch_indices = prefix[cursor_offset:cursor_offset + microbatch_rows]
        if not batch_indices:
            cursor_offset = 0
            batch_indices = prefix[:microbatch_rows]
        cursor_offset += len(batch_indices)
        rows = [train_rows[index] for index in batch_indices]
        tokens, segment_ids, eligible, counted = render_batch(
            tokenizer, rows, torch=torch, device=device, special=special)
        ctx = backend.begin_update(type("S", (), {"cumulative_tokens": consumed})())
        ctx = backend.accumulate_microstep(
            ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
            tokens_by_source={"t2_train": counted["supervised_tokens"]},
            planned_total=counted["supervised_tokens"])
        backend.finish_update(
            type("S", (), {"cumulative_tokens": consumed})(), ctx,
            planned_total=counted["supervised_tokens"],
            cursor=CursorState(CURSOR_SCHEMA, data_sha, updates + 1,
                               cursor_offset, 0))
        consumed += counted["real_tokens"]
        updates += 1
        if progress is not None and updates % 10 == 0:
            progress(f"{run_id}: update {updates} real_tokens {consumed}")
        if updates % eval_every_updates == 0:
            controller = generate_rates(model, tokenizer, controller_rows,
                                        torch=torch, device=device,
                                        special=special)
            probe = generate_rates(model, tokenizer, probe_rows,
                                   torch=torch, device=device, special=special)
            eval_trace.append({
                "update": updates, "real_tokens": consumed,
                "dev_controller": controller, "train_probe": probe,
                "moments": optimizer_moment_norms(optimizer=optimizer,
                                                  torch=torch)})
            controller_flags.append(
                controller["complete_exact_with_valid_stop"] >= 0.90)
            probe_flags.append(probe["complete_exact_with_valid_stop"] >= 0.99)
            if onset is None and any(controller_flags):
                onset = updates
            if sustained(controller_flags, required=3):
                confirm = updates
                status = "G90_CONFIRMED"
                break
    if status == "RUNNING":
        status = "NO_G90"
        if plumbing_override:
            # SMOKE MODE ONLY: label the parent as plumbing-qualified so the
            # fork/restore/continuation CODE PATHS execute with tiny budgets.
            # The metric gate itself still ran on real generated behavior;
            # full mode refuses this override structurally.
            status = "G90_CONFIRMED"
            confirm = updates
    if status != "G90_CONFIRMED":
        return {"schema": "anra-cyr-gpu005-acquisition/v1", "run_id": run_id,
                "seed": seed, "status": "NOT_QUALIFIED", "parent_status":
                "NOT_QUALIFIED", "updates": updates,
                "actual_real_tokens": consumed, "parameters": parameters,
                "eval_trace": eval_trace, "gate_override":
                "PLUMBING_SMOKE_ONLY" if plumbing_override else None}
    rng_hashes = {"cpu": hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()).hexdigest()}
    if device is not None and getattr(device, "type", None) == "cuda":
        rng_hashes["cuda"] = hashlib.sha256(
            torch.cuda.get_rng_state().numpy().tobytes()).hexdigest()
    save_parent_checkpoint(parent_path, model=model, optimizer=optimizer,
                           torch=torch,
                           counters={"run_id": run_id, "seed": seed,
                                     "updates": updates,
                                     "actual_real_tokens": consumed,
                                     "cursor_offset": cursor_offset,
                                     "stream_tail_sha256": stream["tail_sha256"],
                                     "lr_at_confirmation": lr,
                                     "rng_state_sha256": rng_hashes,
                                     "parameter_flat_sha256": hashlib.sha256(
                                         torch.cat([parameter.detach().reshape(-1)
                                                    for parameter in model.parameters()])
                                         .numpy().tobytes()
                                     ).hexdigest(),
                                     "status": status})
    from v5_training.production_backend import capture_evidence
    evidence = capture_evidence(model, optimizer, torch=torch)
    return {"schema": "anra-cyr-gpu005-acquisition/v1", "run_id": run_id,
            "seed": seed, "status": status, "parent_status": "G90_CONFIRMED",
            "updates": updates, "actual_real_tokens": consumed,
            "parameters": parameters, "eval_trace": eval_trace,
            "g90_onset_update": onset, "g90_confirm_update": confirm,
            "parent_checkpoint": str(parent_path),
            "parent_model_sha256": state_fingerprint(model, optimizer,
                                                     torch=torch)["model_sha256"],
            "parameter_sha256": evidence.parameter_sha256,
            "cursor_offset": cursor_offset,
            "gate_override": "PLUMBING_SMOKE_ONLY" if plumbing_override else None}


# -- continuation arms --------------------------------------------------------

def continuation_arm(*, arm: str, parent_receipt: Mapping[str, Any],
                     spec: Any, tokenizer: Any, torch: Any, device: Any,
                     special: Mapping[str, int],
                     train_rows: list[dict[str, Any]],
                     controller_rows: list[dict[str, Any]],
                     measurement_rows: list[dict[str, Any]],
                     stream: Mapping[str, Any],
                     target_actual_tokens: int,
                     store_root: Path, run_id: str,
                     deadline_s: float | None,
                     switch_point: int | None = None,
                     eval_every_updates: int = 2,
                     microbatch_rows: int = 16,
                     parent_flat: Any | None = None,
                     progress: Callable[[str], None] | None = None,
                     ) -> dict[str, Any]:
    """One fork: restore THE SAME parent bytes, consume tail[0:] exactly,
    apply ONLY the registered optimization policy, stop on ACTUAL tokens."""

    from anra_v5.cyr_execute import optimizer_moment_norms
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.state import CURSOR_SCHEMA, CursorState

    if parent_receipt.get("parent_status") != "G90_CONFIRMED":
        raise ValueError(f"{run_id}/{arm}: forks require a G90 parent")
    seed = int(parent_receipt["seed"]) + ARM_SEED_OFFSETS[arm]
    torch.manual_seed(seed)
    model = initialize(spec, seed, torch_module=torch).to(device)
    optimizer = build_adamw_optimizer(model, torch_module=torch)
    load_parent_checkpoint(parent_receipt["parent_checkpoint"],
                           model=model, optimizer=optimizer, torch=torch)
    controller = None
    if arm == "HYSTERETIC_HIGH_LOW":
        controller = HysteresisController(
            enter_retention=core.CYR5_HYSTERESIS["enter_retention"],
            reenter_plasticity=core.CYR5_HYSTERESIS["reenter_plasticity"],
            confirmations=core.CYR5_HYSTERESIS["confirmations"])
        controller.assert_valid()

    tail = list(stream["stream"])[stream["fork_boundary"]:]
    data_sha = hashlib.sha256(_canonical_json(
        [row["prompt"] + row["answer"] for row in train_rows])).hexdigest()
    consumed_batch_shas: list[str] = []
    schedule = lambda cumulative_tokens: 0.0  # placeholder; LR set per update below
    backend = ProductionTrainingBackend(
        model=model, optimizer=optimizer, bos_id=special["bos_id"],
        pad_id=special["pad_id"], device=device, schedule=schedule,
        bfloat16_autocast=False, torch_module=torch,
        activation_checkpointing=False)

    consumed = 0
    updates = 0
    tail_offset = 0
    status = "RUNNING"
    lr_trace: list[float] = []
    eval_trace: list[dict[str, Any]] = []
    high_tokens = low_tokens = 0
    checkpoint_paths: list[str] = []
    while consumed < target_actual_tokens:
        if deadline_s is not None and time.monotonic() >= deadline_s:
            status = "TIMEBOX"
            break
        lr_now = core.lr_for_token(arm, consumed,
                                   switch_point=switch_point or 0,
                                   controller=controller)
        backend.schedule = lambda cumulative_tokens, _lr=lr_now: _lr
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        batch_indices = tail[tail_offset:tail_offset + microbatch_rows]
        if not batch_indices:
            tail_offset = 0
            batch_indices = tail[:microbatch_rows]
        tail_offset += len(batch_indices)
        consumed_batch_shas.append(hashlib.sha256(_canonical_json(
            batch_indices)).hexdigest())
        rows = [train_rows[index] for index in batch_indices]
        tokens, segment_ids, eligible, counted = render_batch(
            tokenizer, rows, torch=torch, device=device, special=special)
        ctx = backend.begin_update(type("S", (), {"cumulative_tokens": consumed})())
        ctx = backend.accumulate_microstep(
            ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
            tokens_by_source={"t2_continuation": counted["supervised_tokens"]},
            planned_total=counted["supervised_tokens"])
        backend.finish_update(
            type("S", (), {"cumulative_tokens": consumed})(), ctx,
            planned_total=counted["supervised_tokens"],
            cursor=CursorState(CURSOR_SCHEMA, data_sha, updates + 1,
                               tail_offset, 0))
        consumed += counted["real_tokens"]
        updates += 1
        lr_trace.append(lr_now)
        if lr_now >= core.CYR5_LRS["HIGH"]:
            high_tokens += counted["real_tokens"]
        else:
            low_tokens += counted["real_tokens"]
        if progress is not None and updates % 10 == 0:
            progress(f"{run_id}/{arm}: update {updates} real {consumed}/{target_actual_tokens}")
        if updates % eval_every_updates == 0 or consumed >= target_actual_tokens:
            controller_metric = generate_rates(model, tokenizer,
                                               controller_rows, torch=torch,
                                               device=device, special=special)
            measurement = generate_rates(model, tokenizer, measurement_rows,
                                         torch=torch, device=device,
                                         special=special)
            displacement = _displacement(model, parent_flat)
            eval_trace.append({
                "update": updates, "continuation_real_tokens": consumed,
                "lr": lr_now,
                "dev_controller": controller_metric,
                "dev_measurement": measurement,
                "parameter_displacement": displacement,
                "moments": optimizer_moment_norms(optimizer=optimizer,
                                                  torch=torch)})
            if controller is not None:
                controller.observe(
                    metric=controller_metric["complete_exact_with_valid_stop"],
                    threshold_note="dev_controller generated complete-exact",
                    token_position=consumed, lr_before=lr_now,
                    lr_plasticity=core.CYR5_LRS["HIGH"],
                    lr_retention=core.CYR5_LRS["LOW"])
        if updates % 25 == 0:
            path = store_root / run_id / f"arm-{arm}" / f"checkpoint-{updates}"
            save_parent_checkpoint(path, model=model, optimizer=optimizer,
                                   torch=torch,
                                   counters={"arm": arm, "updates": updates,
                                             "continuation_real_tokens": consumed})
            checkpoint_paths.append(str(path))
    if status == "RUNNING":
        status = "COMPLETE"
    final_path = store_root / run_id / f"arm-{arm}" / "final"
    save_parent_checkpoint(final_path, model=model, optimizer=optimizer,
                           torch=torch,
                           counters={"arm": arm, "updates": updates,
                                     "continuation_real_tokens": consumed,
                                     "status": status})
    trajectory = [entry["dev_measurement"]["complete_exact_with_valid_stop"]
                  for entry in eval_trace]
    ret90 = (sum(1 for value in trajectory if value >= 0.90) / len(trajectory)
             if trajectory else 0.0)
    final_g = trajectory[-1] if trajectory else 0.0
    redteam = red_team_metrics(eval_trace=eval_trace,
                               high_tokens=high_tokens, low_tokens=low_tokens)
    return {"schema": "anra-cyr-gpu005-arm/v1", "run_id": run_id, "arm": arm,
            "seed": parent_receipt["seed"], "status": status,
            "updates": updates, "actual_real_tokens": consumed,
            "target_actual_real_tokens": target_actual_tokens,
            "shares_valid_parent": True,
            "parent_model_sha256": parent_receipt["parent_model_sha256"],
            "future_tail_sha256": stream["tail_sha256"],
            "consumed_batch_shas": consumed_batch_shas,
            "lr_trace": lr_trace, "eval_trace": eval_trace,
            "retention_ret90": round(ret90, 4), "final_g": final_g,
            "high_exposure_tokens": high_tokens,
            "low_exposure_tokens": low_tokens,
            "redteam": redteam, "redteam_pass": redteam["pass"],
            "controller_snapshot": (controller.snapshot()
                                    if controller is not None else None),
            "checkpoints": checkpoint_paths,
            "final_checkpoint": str(final_path)}


def _displacement(model: Any, parent_flat: Any) -> dict[str, float]:
    import torch
    current = torch.cat([parameter.detach().reshape(-1)
                         for parameter in model.parameters()])
    if parent_flat is None:
        return {"l2": 0.0, "relative": 0.0}
    delta = (current - parent_flat).norm().item()
    return {"l2": round(delta, 5),
            "relative": round(delta / max(parent_flat.norm().item(), 1e-8), 6)}


def red_team_metrics(*, eval_trace: list[dict[str, Any]], high_tokens: int,
                     low_tokens: int) -> dict[str, Any]:
    """Near-freezing red team (section 24): LOW may win by barely moving.

    Pass requires measured displacement/moments on every evaluation; the
    interpretation (consolidation vs freeze) stays with DECISION.json.
    """

    if not eval_trace:
        return {"pass": False, "reason": "no evaluations to audit"}
    last = eval_trace[-1]
    displacement = last.get("parameter_displacement", {})
    moments = last.get("moments", {})
    measured = bool(displacement) and bool(moments)
    return {"schema": "anra-cyr-gpu005-redteam/v1",
            "parameter_displacement": displacement,
            "relative_displacement": displacement.get("relative"),
            "update_norms_measured": measured,
            "adam_moment_norms": moments,
            "high_exposure_tokens": high_tokens,
            "low_exposure_tokens": low_tokens,
            "note": "LOW protective effects must be read against relative "
                    "displacement; near-freezing is reported, not hidden",
            "pass": measured}


# -- calibration + resolver -----------------------------------------------------

def calibrate(*, spec: Any, tokenizer: Any, torch: Any, device: Any,
              special: Mapping[str, int], train_rows: list[dict[str, Any]],
              ) -> dict[str, Any]:
    """CELL 0 hardware benchmark on the REAL proxy + tokenizer + batch
    builder + forward/backward + AdamW (section 29)."""

    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    start = time.monotonic()
    model = initialize(spec, 123, torch_module=torch).to(device)
    build_adamw_optimizer(model, torch_module=torch)
    cold_s = time.monotonic() - start
    rows = train_rows[:8]
    tokens, segment_ids, eligible, counted = render_batch(
        tokenizer, rows, torch=torch, device=device, special=special)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.monotonic()
    updates = 3
    for _ in range(updates):
        model.zero_grad(set_to_none=True)
        from v5_training.production_backend import causal_lm_loss
        from v5_model.core import packed_layout
        positions, mask = packed_layout(segment_ids, torch_module=torch)
        logits = model(tokens, positions.to(device), mask.to(device))
        loss, _supervised = causal_lm_loss(
            logits, tokens, segment_ids, bos_id=special["bos_id"],
            pad_id=special["pad_id"], eligible=eligible, torch_module=torch)
        loss.backward()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = max(time.monotonic() - start, 1e-9)
    peak_vram = (torch.cuda.max_memory_allocated() / 1e9
                 if hasattr(torch, "cuda") and torch.cuda.is_available() else 0.0)
    return {"schema": "anra-cyr-gpu005-calibration/v1",
            "cold_startup_s": round(cold_s, 3),
            "updates_benchmarked": updates,
            "updates_per_sec": round(updates / wall, 3),
            "real_tokens_per_update": counted["real_tokens"],
            "real_tokens_per_sec": round(
                updates * counted["real_tokens"] / wall, 1),
            "peak_vram_gb": round(peak_vram, 3)}


def resolve_hardware(*, calibration: Mapping[str, Any], mode: str,
                     ) -> dict[str, Any]:
    """Hardware-ONLY decisions (section 30): never sees accuracy/loss."""

    if mode == "smoke":
        return {"schema": "anra-cyr-gpu005-resolver/v1", "mode": "smoke",
                "proxy": "TINY", "parents": 1, "updates_acquisition": 2,
                "updates_continuation": 2, "target_actual_tokens_acquisition": 4096,
                "target_actual_tokens_continuation": 4096,
                "wall_budget_minutes": 10.0,
                "eval_every_acquisition": 1, "eval_every_continuation": 1,
                "transfer_enabled": True, "reason": "smoke plumbing"}
    if mode != "full":
        raise ValueError("mode must be smoke or full")
    tokens_per_sec = float(calibration["real_tokens_per_sec"])
    vram = float(calibration.get("peak_vram_gb", 0.0)) or 1.0
    registry = core.proxy_registry()
    budget_minutes = core.CYR5_WALL_TARGET_MINUTES[0]
    # 3 parents x (acquisition + 4 forks) + transfer shares the wall.
    arm_slots = 3 * 5 + 2
    per_arm_tokens = tokens_per_sec * budget_minutes * 60.0 / arm_slots
    for name in ("MIDI", "MICRO", "RESEARCH_SMALL"):
        entry = registry[name]
        per_update_tokens = 4096
        updates = int(per_arm_tokens // per_update_tokens)
        if per_arm_tokens >= core.CYR5_ACQ_DOSE_MIN_TOKENS:
            acq = min(core.CYR5_ACQ_DOSE_TARGET_TOKENS, int(per_arm_tokens * 0.7))
            fork = min(core.CYR5_FORK_DOSE_TARGET_TOKENS, int(per_arm_tokens * 0.3))
            return {"schema": "anra-cyr-gpu005-resolver/v1", "mode": "full",
                    "proxy": name, "parents": 3,
                    "updates_acquisition": max(4, acq // per_update_tokens),
                    "updates_continuation": max(4, fork // per_update_tokens),
                    "target_actual_tokens_acquisition": acq,
                    "target_actual_tokens_continuation": fork,
                    "wall_budget_minutes": budget_minutes,
                    "eval_every_acquisition": 50, "eval_every_continuation": 25,
                    "transfer_enabled": True,
                    "parameters": entry["parameters"],
                    "reason": f"{name} affords the dose floor inside the wall"}
    raise ValueError("hardware cannot afford the preregistered dose floor: "
                     "refusing to run a starved campaign")


# -- transfer stage (section 25) ----------------------------------------------

def transfer_stage(*, baseline_receipt: Mapping[str, Any],
                   finalist_receipt: Mapping[str, Any],
                   spec: Any, tokenizer: Any, torch: Any, device: Any,
                   special: Mapping[str, int], store_root: Path,
                   run_id: str, deadline_s: float | None,
                   target_actual_tokens: int = 65536,
                   ) -> dict[str, Any]:
    """Binding/registry family: baseline parent + strongest finalist,
    query-only variants (no fact-order confound, per ARK-009)."""

    from v5_experiments.cyr_tournament import render_worlds
    from v5_model.core import initialize
    from v5_training.optimizer import build_adamw_optimizer
    from v5_training.production_backend import ProductionTrainingBackend
    from v5_training.state import CURSOR_SCHEMA, CursorState

    splits = render_worlds(family=core.CYR5_TRANSFER_FAMILY,
                           split_seeds={"train": 9001, "heldout": 9002},
                           worlds_per_split={"train": 64, "heldout": 32})

    def as_row(record: Mapping[str, Any]) -> dict[str, str]:
        # Registry records render as "<context>\nAnswer: <answer>"; the
        # batch builder needs the prompt split off the answer.
        text = record["text"]
        return {"prompt": text[: len(text) - len(record["answer"])],
                "answer": record["answer"]}

    train_records = [as_row(world["base"]) for world in splits["train"]]
    heldout = [as_row(world["base"]) for world in splits["heldout"]]
    results: dict[str, Any] = {}
    for label, receipt in (("baseline", baseline_receipt),
                           ("finalist", finalist_receipt)):
        seed = int(receipt["seed"])
        torch.manual_seed(seed + 17)
        model = initialize(spec, seed + 17, torch_module=torch).to(device)
        from anra_v5.cyr_execute import load_research_checkpoint
        import torch as _torch
        source = finalist_receipt if label == "finalist" else baseline_receipt
        optimizer = build_adamw_optimizer(model, torch_module=torch)
        load_research_checkpoint(source["final_checkpoint"], model=model,
                                 optimizer=optimizer, torch=_torch)
        backend = ProductionTrainingBackend(
            model=model, optimizer=optimizer, bos_id=special["bos_id"],
            pad_id=special["pad_id"], device=device,
            schedule=lambda cumulative_tokens: core.CYR5_LRS["HIGH"],
            bfloat16_autocast=False, torch_module=torch,
            activation_checkpointing=False)
        consumed = 0
        updates = 0
        while consumed < target_actual_tokens:
            if deadline_s is not None and time.monotonic() >= deadline_s:
                results[label] = {"status": "TIMEBOX", "updates": updates,
                                  "actual_real_tokens": consumed}
                break
            batch = [train_records[(updates * 8 + i) % len(train_records)]
                     for i in range(8)]
            tokens, segment_ids, eligible, counted = render_batch(
                tokenizer, batch, torch=torch, device=device, special=special)
            ctx = backend.begin_update(type("S", (), {"cumulative_tokens": consumed})())
            ctx = backend.accumulate_microstep(
                ctx, tokens=tokens, segment_ids=segment_ids, eligible=eligible,
                tokens_by_source={"registry_train": counted["supervised_tokens"]},
                planned_total=counted["supervised_tokens"])
            backend.finish_update(
                type("S", (), {"cumulative_tokens": consumed})(), ctx,
                planned_total=counted["supervised_tokens"],
                cursor=CursorState(CURSOR_SCHEMA, "registry", updates + 1, 0, 0))
            consumed += counted["real_tokens"]
            updates += 1
        else:
            held = generate_rates(model, tokenizer, heldout, torch=torch,
                                  device=device, special=special)
            results[label] = {"status": "COMPLETE", "updates": updates,
                              "actual_real_tokens": consumed,
                              "heldout_generated": held}
    qualified = all(entry.get("status") == "COMPLETE" for entry in results.values())
    event_rate = min((entry.get("heldout_generated", {}).get("content_exact", 0.0)
                      for entry in results.values()), default=0.0)
    return {"schema": "anra-cyr-gpu005-transfer/v1",
            "family": core.CYR5_TRANSFER_FAMILY,
            "protocol": "query-only variants; nonzero event-rate design",
            "results": results, "complete": qualified,
            "event_rate_floor": round(event_rate, 4),
            "status": "COMPLETE" if qualified and event_rate > 0 else
                      ("TIMEBOX" if not qualified else "NOT_INFORMATIVE")}


# -- campaign -------------------------------------------------------------------

def run_campaign(*, out: Path, mode: str = "smoke", torch: Any = None,
                 device: Any = None, progress: Callable[[str], None] | None = None,
                 ) -> dict[str, Any]:
    """The ONE orchestrator (section 46): identical stage order in smoke and
    full; smoke runs TINY locally, full requires CUDA (section 42)."""

    if torch is None:
        import torch as torch_module
        torch = torch_module
    if mode == "full" and not (hasattr(torch, "cuda") and torch.cuda.is_available()):
        raise RuntimeError("CYR-GPU-005 full scientific mode requires "
                           "Google Colab GPU: refusing CPU fallback")
    device = device or torch.device("cpu")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    campaign: dict[str, Any] = {"schema": CAMPAIGN_SCHEMA,
                                "experiment": core.CYR5_ID, "mode": mode,
                                "started_utc": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    def note(message: str) -> None:
        if progress is not None:
            progress(message)

    tokenizer, tokenizer_identity = production_tokenizer(REPO)
    campaign["tokenizer_identity"] = tokenizer_identity
    special = {"pad_id": tokenizer_identity["pad_id"],
               "bos_id": tokenizer_identity["bos_id"],
               "eos_id": tokenizer_identity["eos_id"]}

    smoke_counts = ({"train": 48, "dev_controller": 24, "dev_measurement": 24,
                     "sealed_reserved": 16} if mode == "smoke" else None)
    splits = core.render_t2_worlds(worlds_per_split=smoke_counts)
    manifest = core.build_data_manifest(splits)
    core.assert_manifest_sha(manifest)
    audit = core.commutation_audit(
        splits, tv_bound=(0.60 if mode == "smoke" else 0.20))
    if not audit["commutation_free"]:
        raise ValueError(f"leak audit failed: {audit['findings']}")
    campaign["data_manifest_sha256"] = manifest["sha256"]
    campaign["leak_audit"] = audit
    note("data rendered, manifest bound, leak audit passed")

    registry = core.proxy_registry(vocab_size=tokenizer.vocabulary_size)
    campaign["proxy_registry"] = {name: {"role": entry["role"],
                                         "parameters": entry["parameters"]}
                                  for name, entry in registry.items()}
    smoke_spec = registry["TINY"]["spec"]
    calibration = calibrate(spec=smoke_spec, tokenizer=tokenizer, torch=torch,
                            device=device, special=special,
                            train_rows=splits["train"])
    resolved = resolve_hardware(calibration=calibration, mode=mode)
    campaign["calibration"] = calibration
    campaign["resolved"] = resolved
    spec = (smoke_spec if mode == "smoke"
            else registry[resolved["proxy"]]["spec"])
    proxy_name = "TINY" if mode == "smoke" else resolved["proxy"]
    wall_minutes = float(resolved["wall_budget_minutes"])
    deadline_s = started + wall_minutes * 60.0
    campaign["campaign_deadline_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline_s))

    smoke_stream_counts = ({"prefix_rows": 32, "tail_rows": 32}
                           if mode == "smoke" else
                           {"prefix_rows": max(1, resolved["target_actual_tokens_acquisition"] // 4096),
                            "tail_rows": max(1, resolved["target_actual_tokens_continuation"] // 4096)})
    parent_receipts: list[dict[str, Any]] = []
    arm_receipts: dict[str, dict[str, Any]] = {}
    parent_equivalences: list[dict[str, Any]] = []
    tail_equality_receipts: list[dict[str, Any]] = []
    qualified_parent: dict[str, Any] | None = None

    for parent_index in range(int(resolved["parents"])):
        seed = core.CYR5_PARENT_SEEDS[parent_index]
        stream = core.build_future_stream(
            seed=seed, world_count=len(splits["train"]),
            prefix_rows=smoke_stream_counts["prefix_rows"],
            tail_rows=smoke_stream_counts["tail_rows"])
        receipt = acquire_parent(
            seed=seed, spec=spec, proxy_name=proxy_name, tokenizer=tokenizer,
            torch=torch,
            device=device, special=special, train_rows=splits["train"],
            controller_rows=splits["dev_controller"],
            probe_rows=splits["train"][:16],
            target_actual_tokens=int(resolved["target_actual_tokens_acquisition"]),
            stream=stream, store_root=out, run_id=f"parent-{seed}",
            deadline_s=deadline_s,
            eval_every_updates=int(resolved.get("eval_every_acquisition", 4)),
            plumbing_override=(mode == "smoke"), progress=note)
        receipt["future_stream"] = {key: stream[key] for key in
                                    ("seed", "prefix_sha256", "tail_sha256",
                                     "fork_boundary")}
        parent_receipts.append(receipt)
        if receipt["parent_status"] == "G90_CONFIRMED":
            qualified_parent = receipt
            break
        note(f"parent {seed}: {receipt['parent_status']} — no forks (section 9)")

    if qualified_parent is not None:
        stream = core.build_future_stream(
            seed=qualified_parent["seed"], world_count=len(splits["train"]),
            prefix_rows=smoke_stream_counts["prefix_rows"],
            tail_rows=smoke_stream_counts["tail_rows"])
        import torch as _torch
        parent_state = _torch.load(
            io.BytesIO(Path(qualified_parent["parent_checkpoint"],
                            "model.bin").read_bytes()), map_location="cpu",
            weights_only=True)
        parent_flat = _torch.cat([value.detach().reshape(-1)
                                  for value in parent_state.values()])
        forks: dict[str, tuple[Any, Any]] = {}
        from v5_model.core import initialize
        from v5_training.optimizer import build_adamw_optimizer
        for arm in core.CYR5_ARMS:
            fork_seed = int(qualified_parent["seed"]) + ARM_SEED_OFFSETS[arm]
            fork_model = initialize(spec, fork_seed, torch_module=torch).to(device)
            fork_optimizer = build_adamw_optimizer(fork_model, torch_module=torch)
            load_parent_checkpoint(qualified_parent["parent_checkpoint"],
                                   model=fork_model, optimizer=fork_optimizer,
                                   torch=torch)
            forks[arm] = (fork_model, fork_optimizer)
        equivalence = parent_equivalence_receipt(forks, torch=torch)
        parent_equivalences.append(equivalence)
        switch_point = core.fixed_time_switch_point(
            continuation_target_tokens=int(
                resolved["target_actual_tokens_continuation"]))
        for arm in core.CYR5_ARMS:
            arm_receipts[arm] = continuation_arm(
                arm=arm, parent_receipt=qualified_parent, spec=spec,
                tokenizer=tokenizer, torch=torch, device=device,
                special=special, train_rows=splits["train"],
                controller_rows=splits["dev_controller"],
                measurement_rows=splits["dev_measurement"], stream=stream,
                target_actual_tokens=int(
                    resolved["target_actual_tokens_continuation"]),
                store_root=out, run_id=f"parent-{qualified_parent['seed']}",
                deadline_s=deadline_s, switch_point=switch_point,
                eval_every_updates=int(
                    resolved.get("eval_every_continuation", 2)),
                parent_flat=parent_flat, progress=note)
        # Future-tail equality from what the arms ACTUALLY consumed: the
        # recorded per-update batch index hashes must be identical across
        # arms for the compared prefix (section 8's hard test).
        compared_batches = min(len(receipt["consumed_batch_shas"])
                               for receipt in arm_receipts.values()) \
            if arm_receipts else 0
        if compared_batches:
            tail_equality_receipts.append(core.assert_future_tail_equality(
                {arm: receipt["consumed_batch_shas"][:compared_batches]
                 for arm, receipt in arm_receipts.items()}))
        note("forks restored byte-identically; arms complete")

    tail_receipt = (tail_equality_receipts[-1]
                    if tail_equality_receipts else {"identical": False})
    equivalence_receipt = (parent_equivalences[-1]
                           if parent_equivalences else {"identical": False})
    decision = core.decide_verdict(
        arm_receipts=arm_receipts, parent_equivalence=equivalence_receipt,
        future_tail=tail_receipt, leak_audit=audit,
        parent_status=(qualified_parent["parent_status"]
                       if qualified_parent else "NOT_QUALIFIED"))
    transfer = None
    if qualified_parent is not None and arm_receipts and resolved.get("transfer_enabled"):
        finalist = max(arm_receipts.items(),
                       key=lambda entry: entry[1]["retention_ret90"])[1]
        transfer = transfer_stage(
            baseline_receipt={"seed": qualified_parent["seed"],
                              "final_checkpoint": qualified_parent["parent_checkpoint"]},
            finalist_receipt=finalist, spec=spec, tokenizer=tokenizer,
            torch=torch, device=device, special=special, store_root=out,
            run_id=f"parent-{qualified_parent['seed']}",
            deadline_s=deadline_s)
        decision = core.decide_verdict(
            arm_receipts=arm_receipts, parent_equivalence=equivalence_receipt,
            future_tail=tail_receipt, leak_audit=audit,
            parent_status=qualified_parent["parent_status"],
            transfer=transfer)
    campaign["parents"] = parent_receipts
    campaign["arms"] = arm_receipts
    campaign["parent_equivalence"] = equivalence_receipt
    campaign["future_tail"] = tail_receipt
    campaign["decision"] = decision
    campaign["transfer"] = transfer
    campaign["wall_seconds"] = round(time.monotonic() - started, 1)
    campaign["status"] = "COMPLETE"
    bundle = package_bundle(out, campaign=campaign,
                            environment=environment_receipt(torch=torch,
                                                            device=device,
                                                            mode=mode),
                            preregistration=None, mode=mode)
    campaign["bundle"] = bundle
    (out / "campaign_receipt.json").write_text(
        json.dumps(campaign, indent=2, default=str) + "\n", encoding="utf-8")
    return campaign


# -- evidence bundle (sections 39, 40) -----------------------------------------

BUNDLE_ENTRIES = (
    "SESSION_MANIFEST.json", "ENVIRONMENT.json", "PREREGISTRATION.json",
    "RESOLVED_PREREGISTRATION.json", "CALIBRATION.json", "MODEL_REGISTRY.json",
    "DATA_MANIFEST.json", "SPLIT_MANIFEST.json", "REDTEAM.json",
    "NEGATIVE_CONTROL_TESTS.json", "DECISION.json",
)


def package_bundle(out: Path, *, campaign: Mapping[str, Any],
                   environment: Mapping[str, Any],
                   preregistration: Mapping[str, Any] | None,
                   mode: str) -> dict[str, Any]:
    """Section-39 bundle; failures still package (section 40)."""

    bundle_path = out / BUNDLE_NAME
    payload = {
        "SESSION_MANIFEST.json": {
            "schema": "anra-cyr-gpu005-session/v1", "experiment": core.CYR5_ID,
            "mode": mode, "campaign_status": campaign.get("status"),
            "wall_seconds": campaign.get("wall_seconds")},
        "ENVIRONMENT.json": dict(environment),
        "PREREGISTRATION.json": dict(preregistration or
                                     {"status": "smoke campaign — "
                                      "preregistration binding lives in the "
                                      "preregistration commit"}),
        "RESOLVED_PREREGISTRATION.json": dict(campaign.get("resolved", {})),
        "CALIBRATION.json": dict(campaign.get("calibration", {})),
        "MODEL_REGISTRY.json": dict(campaign.get("proxy_registry", {})),
        "DATA_MANIFEST.json": {"sha256": campaign.get("data_manifest_sha256")},
        "SPLIT_MANIFEST.json": {"schema": core.SPLIT_SCHEMA},
        "REDTEAM.json": {arm: receipt.get("redteam")
                         for arm, receipt in campaign.get("arms", {}).items()},
        "NEGATIVE_CONTROL_TESTS.json": negative_control_tests(),
        "DECISION.json": dict(campaign.get("decision", {})),
        "ACQUISITION/parents.json": [
            {key: value for key, value in receipt.items()
             if key != "eval_trace"}
            for receipt in campaign.get("parents", [])],
        "RETENTION/arms.json": {
            arm: {key: value for key, value in receipt.items()
                  if key not in ("eval_trace", "lr_trace")}
            for arm, receipt in campaign.get("arms", {}).items()},
        "CONTROLLER/hysteresis.json": {
            arm: receipt.get("controller_snapshot")
            for arm, receipt in campaign.get("arms", {}).items()},
        "CHECKPOINT_RECEIPTS/paths.json": {
            arm: {"checkpoints": receipt.get("checkpoints", []),
                  "final": receipt.get("final_checkpoint")}
            for arm, receipt in campaign.get("arms", {}).items()},
        "DIAGNOSTICS/exposure.json": {
            arm: {"high_tokens": receipt.get("high_exposure_tokens"),
                  "low_tokens": receipt.get("low_exposure_tokens")}
            for arm, receipt in campaign.get("arms", {}).items()},
    }
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, body in payload.items():
            bundle.writestr(name, json.dumps(body, indent=2, default=str))
    return {"schema": "anra-cyr-gpu005-bundle/v1", "path": str(bundle_path),
            "sha256": sha256_file(bundle_path),
            "entries": sorted(payload), "entry_count": len(payload)}


def negative_control_tests() -> dict[str, Any]:
    """Mechanical negative controls shipped with every bundle."""

    from v5_experiments.xla_accumulation_oracle import (
        run_xla_accumulation_oracle)
    oracle = run_xla_accumulation_oracle()
    splits = core.render_t2_worlds()
    clean = core.commutation_audit(splits)["commutation_free"]
    leaky = dict(splits)
    leaky["dev_controller"] = leaky["dev_controller"][:8] + [dict(leaky["train"][0])]
    catches_leak = not core.commutation_audit(leaky)["commutation_free"]
    stream = core.build_future_stream(seed=1, world_count=8, prefix_rows=8,
                                      tail_rows=8)
    arm_shas = core.batch_shas(stream=stream, batch_size=2, count=2)
    try:
        core.assert_future_tail_equality({"A": arm_shas,
                                          "B": list(reversed(arm_shas))})
        detects_tail_divergence = False
    except ValueError:
        detects_tail_divergence = True
    return {"schema": "anra-cyr-gpu005-negative-controls/v1",
            "xla_oracle_passes": oracle["passed"],
            "clean_data_passes_leak_audit": clean,
            "audit_catches_injected_leak": catches_leak,
            "tail_equality_detects_divergence": detects_tail_divergence}


# -- CLI -------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    campaign = run_campaign(out=Path(args.out), mode=args.mode,
                            progress=lambda message: print(message, flush=True))
    decision = campaign["decision"]
    print(json.dumps({"experiment": core.CYR5_ID, "mode": args.mode,
                      "status": campaign["status"],
                      "verdict": decision["verdict"],
                      "winner": decision.get("winner")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
