def test_rlvr_has_reference_model():
    """Reference model must be a frozen deepcopy."""
    import inspect

    from training.rlvr import RLVRTrainer

    src = inspect.getsource(RLVRTrainer.__init__)
    assert "deepcopy" in src, "RLVRTrainer must deepcopy model into _ref_model"
    assert "requires_grad_(False)" in src, "Reference model must be frozen"


def test_rlvr_has_optimizer():
    """Optimizer must be stored and used in train_step."""
    import inspect

    from training.rlvr import RLVRTrainer

    init_src = inspect.getsource(RLVRTrainer.__init__)
    step_src = inspect.getsource(RLVRTrainer.train_step)
    assert "optimizer" in init_src
    assert "backward" in step_src
    assert "optimizer.step" in step_src


def test_rlvr_grpo_advantages_normalized():
    """Advantages must be normalized by std dev, not just mean-subtracted."""
    import inspect

    from training.rlvr import RLVRTrainer

    src = inspect.getsource(RLVRTrainer.train_step)
    assert "std" in src, "GRPO advantages must divide by std"
    assert "1e-8" in src or "1e-6" in src, "Must have epsilon for numerical stability"


def test_rlvr_kl_uses_logprob_difference():
    """KL penalty must be log pi_theta - log pi_ref, not logp.pow(2)."""
    import inspect

    from training.rlvr import RLVRTrainer

    src = inspect.getsource(RLVRTrainer.train_step)
    assert "pow(2)" not in src, "KL must not be logp^2 - that is not KL divergence"
    assert "lp_ref" in src or "_ref_model" in src, "Must use reference model for KL"
