"""End-to-end recovery and fail-closed admission for tiny CPU mechanics only."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from anra_v5.elendril.foundation_training import Recipe
from anra_v5.elendril.v6_registry import MODEL_PROFILES
from anra_v5.elendril.v8.contracts import build_v8_model_spec
from anra_v5.elendril.v8.data import build_d1_training_dataset, D1_TRAIN_SEED
from anra_v5.elendril.v8.training import prepare_v8_training_data, V8TypedTrainer
from anra_v5.planthir import CartridgeArmSpec, CartridgeRunner, ExperimentCartridge, V8EnvironmentKernel, CausalContrastSpec
from anra_v5.planthir.cartridge import audit_experimental_design, _validate_cartridge
from anra_v5.planthir.causal import _probe_split_keys
from anra_v5.planthir.checkpoint_intelligence import CheckpointCatalog


@pytest.fixture
def cartridge():
    data = prepare_v8_training_data('D1', d1_worlds_per_family=16, sample_limit_for_cpu=12)
    train = build_d1_training_dataset(quota_per_family=16).worlds
    _, forbidden = _probe_split_keys(train)
    pool = build_d1_training_dataset(seed=D1_TRAIN_SEED + 7, quota_per_family=16).worlds
    evaluation = [w for w in pool if int(w['renderer_template']) == 0
                  and not (_probe_split_keys([w])[1] & forbidden)][:4]
    _, forbidden_eval = _probe_split_keys(evaluation)
    probes = [w for w in train if not (_probe_split_keys([w])[1] & forbidden_eval)][:4]
    assert len(evaluation) == 4 and len(probes) == 4
    base = replace(MODEL_PROFILES['S20'], width=32, layers=1, query_heads=2,
                   kv_heads=1, head_dimension=16, ffn_width=64, context_length=512)
    arm = CartridgeArmSpec('J', 'J', True, build_v8_model_spec('J', base_spec=base),
                           Recipe(total_updates=2, batch_size=4, micro_batch_size=2, lr=1e-4),
                           data.data_identity, milestone_updates=(1, 2))
    return ExperimentCartridge('RECOVERY', 'fixture', '21f57160343a54053278360e056b3ba4347f7c03',
        V8EnvironmentKernel(), arm, (), data, tuple(evaluation), (), tuple(probes))


def test_gate_rejects_undeclared_precision_and_capacity(cartridge):
    control = cartridge.control_arm
    contrast = CausalContrastSpec('C', 'architecture', 'J', 'F', (), 'head')
    treatment = replace(control, arm_id='F', is_control=False,
        v8_spec=build_v8_model_spec('F', base_spec=control.v8_spec.base_spec),
        recipe=replace(control.recipe, precision='fp16'), contrasts=(contrast,))
    report = audit_experimental_design(control, (treatment,))
    assert not report['passed'] and 'PRECISION_MISMATCH' in report['violations'][0]['codes']
    treatment = replace(treatment, recipe=control.recipe)
    # Omitting "matched" from held_constant must not disable the explicit capacity limit.
    report = audit_experimental_design(control, (treatment,), max_param_ratio=1.00001)
    assert not report['passed'] and 'PARAM_SCALE_CONFOUND' in report['violations'][0]['codes']


def test_declared_learning_rate_intervention_is_allowed(cartridge):
    control = cartridge.control_arm
    treatment = replace(control, arm_id='J_lr', is_control=False,
        recipe=replace(control.recipe, lr=2e-4),
        contrasts=(CausalContrastSpec('C', 'lr', 'J', 'J_lr', (), 'learning-rate'),))
    assert audit_experimental_design(control, (treatment,))['passed']
    assert not audit_experimental_design(control, (replace(treatment, order_seed=19),))['passed']


@pytest.mark.parametrize('mutation', ['empty', 'duplicate', 'identity', 'factory', 'leak', 'large_cpu'])
def test_invalid_cartridges_reject_before_training(cartridge, tmp_path, mutation):
    if mutation == 'empty':
        cartridge.eval_worlds = ()
    elif mutation == 'duplicate':
        cartridge.treatment_arms = (cartridge.control_arm,)
    elif mutation == 'identity':
        cartridge.control_arm = replace(cartridge.control_arm, dataset_identity='invented')
    elif mutation == 'factory':
        cartridge.control_arm = replace(cartridge.control_arm, trainer_factory=lambda **kw: None)
    elif mutation == 'leak':
        cartridge.probe_train_worlds = cartridge.eval_worlds
    else:
        cartridge.control_arm = replace(cartridge.control_arm, v8_spec=build_v8_model_spec('J'))
    with pytest.raises(ValueError):
        CartridgeRunner().run(cartridge, work_root=tmp_path/'w')
    assert not (tmp_path/'w'/'checkpoints').exists()


def test_interruption_resume_matches_uninterrupted_and_keeps_receipts(cartridge, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    reference = CartridgeRunner().run(cartridge, work_root=tmp_path/'reference', checkpoint_interval=1)
    original_step = V8TypedTrainer.step
    def fail_after_committed_update(self, *args, **kwargs):
        if self.state.global_update == 1:
            raise RuntimeError('injected interruption')
        return original_step(self, *args, **kwargs)
    monkeypatch.setattr(V8TypedTrainer, 'step', fail_after_committed_update)
    work = tmp_path/'resumed'
    with pytest.raises(RuntimeError, match='injected'):
        CartridgeRunner().run(cartridge, work_root=work, checkpoint_interval=1)
    status = json.loads((work/'RUN_STATUS.json').read_text())
    assert status['status'] == 'FAILED' and status['checkpoint_latest_paths']
    with pytest.raises(ValueError, match='resume required'):
        CartridgeRunner().run(cartridge, work_root=work, checkpoint_interval=1)
    monkeypatch.setattr(V8TypedTrainer, 'step', original_step)
    resumed = CartridgeRunner().run(cartridge, work_root=work, checkpoint_interval=1, resume=True)
    assert resumed['arms']['J']['checkpoint_sha256'] == reference['arms']['J']['checkpoint_sha256']
    assert json.loads((work/'RUN_STATUS.json').read_text())['status'] == 'COMPLETE'
    assert (work/'ATTEMPTS.jsonl').is_file()
    disk_brief = json.loads((work/'result_bundle'/'AI_HANDOFF_BRIEF.json').read_text())
    assert disk_brief == resumed['ai_handoff_brief']
    assert resumed['token_fuel_report']['handoff_brief_bytes'] == len((work/'result_bundle'/'AI_HANDOFF_BRIEF.json').read_bytes())
    # Changed recipes cannot be resumed under the same work identity.
    cartridge.control_arm = replace(cartridge.control_arm, recipe=replace(cartridge.control_arm.recipe, lr=2e-4))
    with pytest.raises(ValueError, match='identity mismatch'):
        CartridgeRunner().run(cartridge, work_root=work, checkpoint_interval=1, resume=True)


def test_export_failure_cannot_leave_complete_receipt(cartridge, tmp_path, monkeypatch):
    import anra_v5.planthir.cartridge as module
    def fail_export(*args, **kwargs):
        raise OSError('injected export failure')
    monkeypatch.setattr(module, 'export_result_bundle', fail_export)
    work = tmp_path/'w'
    with pytest.raises(OSError, match='export failure'):
        CartridgeRunner().run(cartridge, work_root=work, checkpoint_interval=1)
    assert json.loads((work/'RUN_STATUS.json').read_text())['status'] == 'FAILED'
    assert not (work/'FINAL_RECEIPT.json').exists()
    assert len(CheckpointCatalog.discover_checkpoints([work/'checkpoints']).records) >= 1


def test_b0_uses_registered_recipe_and_real_gradient_check(cartridge, tmp_path):
    control = cartridge.control_arm
    cartridge.control_arm = replace(control, arm_id='B0', architecture_family='B0',
        v8_spec=build_v8_model_spec('B0', base_spec=control.v8_spec.base_spec),
        recipe=replace(control.recipe, total_updates=1, lr=0.0002), milestone_updates=(1,))
    result = CartridgeRunner().run(cartridge, work_root=tmp_path/'b0')
    grad = json.loads((tmp_path/'b0'/'arm_state'/'B0.json').read_text())['grad_check']
    assert grad['passed'] and grad.get('note') != 'byte-lm-joint-ce'
    assert result['gates']['all_5_gates_passed']


def test_resume_repairs_committed_but_unmarked_milestone(cartridge, tmp_path, monkeypatch):
    from v5_training.checkpoint import CheckpointStore
    original = CheckpointStore.record_milestone
    def crash_before_binding(self, **kwargs):
        raise RuntimeError('milestone publication interrupted')
    monkeypatch.setattr(CheckpointStore, 'record_milestone', crash_before_binding)
    work = tmp_path/'w'
    with pytest.raises(RuntimeError, match='publication interrupted'):
        CartridgeRunner().run(cartridge, work_root=work, checkpoint_interval=1)
    monkeypatch.setattr(CheckpointStore, 'record_milestone', original)
    result = CartridgeRunner().run(cartridge, work_root=work, checkpoint_interval=1, resume=True)
    assert {r['global_update'] for r in result['arms']['J']['milestones']} == {1, 2}


def test_final_verification_failure_preserves_partial_bundle(cartridge, tmp_path, monkeypatch):
    import anra_v5.planthir.cartridge as module
    def reject_final(*args, **kwargs):
        raise ValueError('injected verification failure')
    monkeypatch.setattr(module, 'verify_result_bundle', reject_final)
    work = tmp_path/'w'
    with pytest.raises(ValueError, match='verification failure'):
        CartridgeRunner().run(cartridge, work_root=work)
    assert json.loads((work/'result_bundle'/'FINAL_STATUS.json').read_text())['status'] == 'PARTIAL'
    assert json.loads((work/'RUN_STATUS.json').read_text())['status'] == 'FAILED'
    assert not (work/'FINAL_RECEIPT.json').exists()


def test_full_restore_models_released_between_arms(cartridge, tmp_path, monkeypatch):
    import gc
    import weakref
    control = cartridge.control_arm
    cartridge.treatment_arms = (replace(control, arm_id='J_lr', is_control=False,
        recipe=replace(control.recipe, lr=2e-4),
        contrasts=(CausalContrastSpec('C', 'lr', 'J', 'J_lr', (), 'learning-rate'),)),)
    original = CheckpointCatalog.load_checkpoint_for_causal_probing
    restored_models = []
    def tracked_load(record, *args, **kwargs):
        # Gate 5 restores from its isolated verified extraction directory.
        is_final_restore = 'planthir-verify-' in str(record)
        if is_final_restore:
            gc.collect()
            assert all(ref() is None for ref in restored_models), 'previous full model remains live'
        loaded = original(record, *args, **kwargs)
        if is_final_restore:
            restored_models.append(weakref.ref(loaded['model']))
        return loaded
    monkeypatch.setattr(CheckpointCatalog, 'load_checkpoint_for_causal_probing', tracked_load)
    CartridgeRunner().run(cartridge, work_root=tmp_path/'two-arms')
    gc.collect()
    assert len(restored_models) == 2 and all(ref() is None for ref in restored_models)
