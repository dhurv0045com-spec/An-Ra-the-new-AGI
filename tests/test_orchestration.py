"""P0 orchestration tests: prove the REAL trainer path, not helpers.

Covers:
- prepare_training_state (the canonical pre-XLA lifecycle)
- real recovery/candidate checkpoint writers from train_xla
- pack_max_steps semantics (global vs pack steps can never collide)
- receipt schedule identity matches the actual pack-bound WSD schedule
- non-finite guards exist for loss/grad_norm/lr
- notebook Run-All symbol resolution
"""

import hashlib
import json

import pytest
import torch

from anra_core.config import CANONICAL_CONFIG
from anra_core.model import AnRaCore
from anra_core.tokenizer import V4Tokenizer


def _build_parent(
    tmp_path, updates: int = 2, schema_version: int = 1, *, legacy_contract: bool = False
):
    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 32))
        (-model(x).sum()).backward()
        optimizer.step()
    tok = V4Tokenizer.load_canonical()
    payload = {
        "checkpoint_artifact_class": "full_resume",
        "checkpoint_schema_version": schema_version,
        "global_step": 20_000,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "tokenizer_contract": (
            tok.identity(probe_count=500)
            if legacy_contract
            else {"available": True, **tok.identity(probe_count=500)}
        ),
    }
    path = tmp_path / "parent.pt"
    torch.save(payload, path)
    return path, model, optimizer


# --------------------------------------------------------------------------
# P0-7/P0-1/2/3: canonical preparation path.
# --------------------------------------------------------------------------


def test_prepare_training_state_restores_model_and_moments(tmp_path) -> None:
    from training.resume import prepare_training_state

    parent_path, parent_model, parent_opt = _build_parent(tmp_path, schema_version=9)

    prepared = prepare_training_state(
        parent_checkpoint=str(parent_path),
        model_config=CANONICAL_CONFIG,
        learning_rate=9e-4,  # must be overridden by restored param-group LR
        weight_decay=0.1,
        expected_resume_step=20_000,
        resume_mode="new_pack_parent",
    )

    # Model parameters exactly restored into the prepared model.
    for key in ("token_embedding_table.weight", "blocks.0.norm_1.weight"):
        assert torch.equal(
            prepared.model.state_dict()[key], parent_model.state_dict()[key]
        ), f"{key} not restored through the canonical preparation path"
    # Optimizer moments restored into the PREPARED optimizer (no second restore).
    live = prepared.optimizer.state_dict()["state"]
    expected = parent_opt.state_dict()["state"]
    first_param = next(iter(expected))
    assert torch.equal(live[first_param]["exp_avg"], expected[first_param]["exp_avg"])
    assert torch.equal(live[first_param]["exp_avg_sq"], expected[first_param]["exp_avg_sq"])
    # Metadata is explicit on the struct - no loose payload dict anywhere.
    assert prepared.global_step == 20_000
    assert prepared.optimizer_updates >= 1
    assert prepared.source_checkpoint == str(parent_path)
    assert prepared.resume_mode == "new_pack_parent"
    assert prepared.checkpoint_schema_version == 9


def test_legacy_contract_requires_and_honors_explicit_authorization(tmp_path) -> None:
    from anra_core.errors import RepresentationIncompatibleError
    from training.resume import prepare_training_state

    parent_path, _model, _opt = _build_parent(
        tmp_path, schema_version=9, legacy_contract=True
    )
    kwargs = {
        "parent_checkpoint": str(parent_path),
        "model_config": CANONICAL_CONFIG,
        "learning_rate": 1e-3,
        "weight_decay": 0.1,
        "expected_resume_step": 20_000,
        "resume_mode": "new_pack_parent",
    }
    with pytest.raises(RepresentationIncompatibleError):
        prepare_training_state(**kwargs)

    prepared = prepare_training_state(**kwargs, allow_legacy_checkpoint=True)
    assert prepared.global_step == 20_000
    assert prepared.checkpoint_schema_version == 9
    assert prepared.optimizer_restored


def test_restore_rebuilds_step20k_decay_parameter_groups(tmp_path) -> None:
    from training.resume import prepare_training_state

    model = AnRaCore(CANONICAL_CONFIG)
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1e-3,
        betas=(0.9, 0.95),
    )
    optimizer.zero_grad(set_to_none=True)
    (-model(torch.randint(0, CANONICAL_CONFIG.vocab_size, (1, 32))).sum()).backward()
    optimizer.step()
    tokenizer = V4Tokenizer.load_canonical()
    parent = tmp_path / "two_group_parent.pt"
    torch.save(
        {
            "checkpoint_artifact_class": "full_resume",
            "checkpoint_schema_version": 9,
            "global_step": 20_000,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_contract": tokenizer.identity(probe_count=500),
        },
        parent,
    )

    prepared = prepare_training_state(
        parent_checkpoint=str(parent),
        model_config=CANONICAL_CONFIG,
        learning_rate=2e-4,
        weight_decay=0.1,
        expected_resume_step=20_000,
        resume_mode="new_pack_parent",
        allow_legacy_checkpoint=True,
    )
    assert [len(group["params"]) for group in prepared.optimizer.param_groups] == [127, 37]
    assert [group["weight_decay"] for group in prepared.optimizer.param_groups] == [0.1, 0.0]
    assert sum(len(state) > 0 for state in prepared.optimizer.state.values()) == 164


def test_prepare_training_state_enforces_expected_resume_step(tmp_path) -> None:
    from training.resume import prepare_training_state

    parent_path, _m, _o = _build_parent(tmp_path)  # global_step 20000
    with pytest.raises(RuntimeError, match="below the expected"):
        prepare_training_state(
            parent_checkpoint=str(parent_path),
            model_config=CANONICAL_CONFIG,
            learning_rate=1e-3, weight_decay=0.1,
            expected_resume_step=25_000,  # wrong parent for this expectation
        )


def test_prepare_training_state_fresh_has_zero_metadata(tmp_path) -> None:
    from training.resume import prepare_training_state

    prepared = prepare_training_state(
        parent_checkpoint=None,
        model_config=CANONICAL_CONFIG,
        learning_rate=1e-3, weight_decay=0.1,
        expected_resume_step=0,
    )
    assert prepared.global_step == 0 and prepared.optimizer_updates == 0
    assert prepared.resume_mode == "fresh" and not prepared.optimizer_restored


# --------------------------------------------------------------------------
# P0-6: real checkpoint writers from train_xla (not manual payload building).
# --------------------------------------------------------------------------


def test_real_candidate_writer_roundtrip_and_immutability(tmp_path) -> None:
    from training.train_xla import save_candidate
    from training.wsd_scheduler import PackWsdSchedule

    tok = V4Tokenizer.load_canonical()
    output = tmp_path / "latest.pt"
    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training_config = {"pack_manifest_sha256": "c" * 64}
    first_parameter = next(model.parameters())
    optimizer.state[first_parameter] = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.zeros_like(first_parameter),
        "exp_avg_sq": torch.zeros_like(first_parameter),
    }
    training_state = {
        "schema": "anra-training-state/v2",
        "global_step": 5,
        "pack_step": 5,
        "pack_manifest_sha256": "c" * 64,
    }
    schedule = PackWsdSchedule(base_lr=2e-4, total_steps=100)

    p5 = save_candidate(output, model, optimizer, training_config, tok, 5,
                        {"loss": 2.0}, str(tmp_path / "parent.pt"), world_size=8,
                        training_state=training_state, schedule=schedule)
    assert p5.exists()
    payload5 = torch.load(p5, weights_only=False)
    assert payload5["checkpoint_artifact_class"] == "full_resume"
    assert payload5["optimizer_state_dict"]["state"], "candidate must be resumable"
    assert payload5["checkpoint_schema_version"] == 3
    assert payload5["global_step"] == 5
    assert payload5["parameter_sha256"]  # canonical hash recorded

    digest5 = hashlib.sha256(p5.read_bytes()).hexdigest()

    # Later candidate at step 10 must NOT overwrite step 5.
    p10 = save_candidate(output, model, optimizer, training_config, tok, 10,
                         {"loss": 1.5}, None, world_size=8,
                         training_state={**training_state, "global_step": 10, "pack_step": 10},
                         schedule=schedule)
    assert p5.exists() and p10.name.endswith("00010.pt")
    assert hashlib.sha256(p5.read_bytes()).hexdigest() == digest5

    # Model reloads from a candidate.
    reloaded = AnRaCore(CANONICAL_CONFIG)
    reloaded.load_state_dict(payload5["model_state_dict"], strict=True)


def test_recovery_payload_via_checkpoint_payload_function(tmp_path) -> None:
    """The recovery writer's payload builder: full_resume class, optimizer
    present, trainer_state present, parameter hash stable across reload."""
    from training.train_xla import _checkpoint_payload
    from training.resume import canonical_parameter_sha256

    tok = V4Tokenizer.load_canonical()
    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    training_config = {"pack_manifest_sha256": "d" * 64}
    trainer_state = {"schema": "anra-training-state/v2", "global_step": 100}

    payload = _checkpoint_payload(
        model, optimizer,
        model_config=CANONICAL_CONFIG,
        training_config=training_config,
        tokenizer=tok, step=100, metrics={"loss": 1.9},
        source_checkpoint="parent.pt", world_size=8,
        training_state=trainer_state,
    )
    assert payload["checkpoint_artifact_class"] == "full_resume"
    assert payload["checkpoint_schema_version"] == 2
    assert isinstance(payload["optimizer_state_dict"], dict) and payload["optimizer_state_dict"]
    assert payload["trainer_state"]["schema"] == "anra-training-state/v2"

    from training.wsd_scheduler import PackWsdSchedule

    wsd_payload = _checkpoint_payload(
        model, optimizer,
        model_config=CANONICAL_CONFIG,
        training_config=training_config,
        tokenizer=tok, step=100, metrics={"loss": 1.9},
        source_checkpoint="parent.pt", world_size=8,
        training_state={
            **trainer_state,
            "pack_step": 10,
            "pack_manifest_sha256": "d" * 64,
        },
        schedule=PackWsdSchedule(base_lr=2e-4, total_steps=100),
    )
    assert wsd_payload["checkpoint_schema_version"] == 3
    assert wsd_payload["lr_schedule"]["name"] == "wsd_pack_v1"

    # Round trip: saved tensors hash to the recorded canonical SHA.
    path = tmp_path / "rec.pt"
    torch.save(payload, path)
    reloaded = torch.load(path, weights_only=False)
    assert reloaded["parameter_sha256"] == canonical_parameter_sha256(
        reloaded["model_state_dict"]
    ), "parameter SHA must survive save/reload unchanged"


def test_real_save_latest_uses_runtime_training_config(tmp_path) -> None:
    from training.train_xla import _save_latest
    from training.wsd_scheduler import PackWsdSchedule

    class FakeXm:
        rendezvous_calls: list[str] = []

        @staticmethod
        def is_master_ordinal() -> bool:
            return True

        @staticmethod
        def save(payload, path, master_only=True) -> None:
            assert master_only is True
            torch.save(payload, path)

        @classmethod
        def rendezvous(cls, tag: str) -> None:
            cls.rendezvous_calls.append(tag)

    model = AnRaCore(CANONICAL_CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokenizer = V4Tokenizer.load_canonical()
    runtime_config = {"pack_manifest_sha256": "e" * 64}
    trainer_state = {
        "schema": "anra-training-state/v2",
        "global_step": 20_010,
        "pack_step": 10,
        "pack_manifest_sha256": "e" * 64,
    }
    output = tmp_path / "latest.pt"

    _save_latest(
        FakeXm, output, model, optimizer, runtime_config, tokenizer, 20_010,
        {"loss": 1.5}, "parent.pt", 8, trainer_state,
        PackWsdSchedule(base_lr=2e-4, total_steps=100),
    )

    payload = torch.load(output, weights_only=True)
    assert payload["pack_manifest_sha256"] == "e" * 64
    assert payload["trainer_state"] == trainer_state
    assert payload["checkpoint_schema_version"] == 3
    assert FakeXm.rendezvous_calls == ["checkpoint-written"]


# --------------------------------------------------------------------------
# P0-10: step semantics - global vs pack steps can never silently collide.
# --------------------------------------------------------------------------


def test_pack_max_steps_semantics_with_global_20k_parent() -> None:
    """The exact historic bug class: global 20,000 vs pack budget 2,500."""
    from training.resume import resolve_pack_horizon

    horizon = resolve_pack_horizon(
        global_step=20_000, restored_pack_step=0,
        token_budget=330_000_000, tokens_per_step=131_072,
    )
    pack_updates = horizon.updates_remaining
    assert pack_updates == 2_517
    final_global = 20_000 + pack_updates
    assert final_global == 22_517
    # The loop bound is pack-relative; it can never compare against global.
    executed = 0
    pack_step = horizon.start_pack_step
    while pack_step < horizon.pack_total_steps:
        pack_step += 1
        executed += 1
    assert executed == 2_517 > 0


# --------------------------------------------------------------------------
# P0-11: receipt schedule identity matches the actual scheduler.
# --------------------------------------------------------------------------


def test_receipt_schedule_identity_is_pack_bound_wsd(tmp_path) -> None:
    from training.train_xla import write_run_receipt

    identity_block = {"pack_manifest_sha256": "a" * 64, "parent_global_step": 20_000}
    config = {"batch_size": 1, "grad_accum_steps": 8, "learning_rate": 2e-4,
              "min_lr_ratio": 0.1, "lr_decay_steps": 1_000_000}
    receipt_path = write_run_receipt(
        tmp_path, identity_block=identity_block, config=config, world_size=8,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert isinstance(receipt["schedule"], dict), "schedule must be structured, not prose"
    assert receipt["schedule"]["name"] == "wsd_pack_v1"
    params = receipt["schedule"]["parameters"]
    assert params["base_lr"] == 2e-4 and params["min_lr_ratio"] == 0.1


# --------------------------------------------------------------------------
# P0-15: non-finite guard coverage (static proof on the real worker source).
# --------------------------------------------------------------------------


def test_worker_guards_loss_grad_norm_and_lr() -> None:
    import inspect

    from training import train_xla

    source = inspect.getsource(train_xla._worker)
    assert "NON-FINITE LOSS" in source
    assert "NON-FINITE GRADIENT NORM" in source
    assert "INVALID LEARNING RATE" in source


# --------------------------------------------------------------------------
# P0-12: notebook Run-All symbol audit.
# --------------------------------------------------------------------------


def test_notebook_run_all_symbols_resolve_in_order() -> None:
    """Every name used before definition fails this audit."""
    import re
    from pathlib import Path

    nb_path = Path(__file__).resolve().parents[1] / "core_vnext_tpu_training.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    defined: set[str] = set()
    builtins = set(dir(__builtins__)) | {
        "json", "os", "sys", "torch", "Path", "hashlib", "tarfile", "subprocess",
        "np", "print", "sorted", "len", "range", "open", "str", "int", "float",
        "enumerate", "zip", "list", "dict", "set", "tuple", "isinstance", "type",
        # imported names (from X import Y) are definitions too - the audit
        # tracks module-level constants; treat known repo imports as defined.
        "CANONICAL_CONFIG",
    }
    assign_re = re.compile(r"^\s*([A-Z_][A-Z0-9_]*)\s*=", re.M)
    use_re = re.compile(r"\b([A-Z_][A-Z0-9_]{3,})\b")

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        # Strip quoted strings AND string-keyed subscripts (env var names like
        # os.environ['PJRT_DEVICE'] are not symbol uses). Also strip comment
        # lines so prose like 'RUN_DIR already exists' is not audited.
        code_no_strings = re.sub(r"\[['\"][^'\"]*['\"]\]", "[]", src)
        code_no_strings = re.sub(r"#[^\n]*", "", code_no_strings)
        code_no_strings = re.sub(r"[\"'].*?[\"']", "", code_no_strings)
        uses = {m for m in use_re.findall(code_no_strings)}
        assigned_here = set(assign_re.findall(code_no_strings))
        undefined = {
            u for u in uses
            if u not in defined and u not in builtins and u not in assigned_here
        }
        assert not undefined, (
            f"notebook uses symbols before definition: {undefined}"
        )
        defined |= assigned_here


def test_xla_bounds_gradient_accumulation_graph_per_microbatch() -> None:
    from pathlib import Path

    source = (Path(__file__).parents[1] / "training" / "train_xla.py").read_text()
    backward = source.index("scaled.backward()")
    reduce_gradients = source.index("xm.reduce_gradients(optimizer)", backward)
    boundary = source.index("xm.mark_step()", backward, reduce_gradients)
    assert backward < boundary < reduce_gradients
