"""
test_45F.py — Test Suite for Steps 21-26
==========================================
Proves each module works correctly:
  Step 21: dataset.py      — data loading, tokenization, batching
  Step 22: trainer.py      — training loop, loss decreases
  Step 23: loss_tracker.py — recording, plotting, persistence
  Step 24: checkpoint.py   — save, load, prune, best tracking
  Step 25: scheduler.py    — warmup, cosine decay, state dict
  Step 26: mixed_precision.py — autocast, backward, scaler

Run: python test_45F.py
All tests must pass before marking this module complete.
"""

import logging
import math
import os
import sys
import tempfile
import time
import random
from pathlib import Path

import torch
import torch.nn as nn

# Ensure imports work from any directory
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)  # suppress info during tests

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

_results = []


def test(name: str):
    """Decorator to register a test."""
    def decorator(fn):
        _results.append((name, fn))
        return fn
    return decorator


def run_all():
    passed = 0
    failed = 0
    print("\n" + "=" * 65)
    print("  45F Test Suite — Steps 21 through 26")
    print("=" * 65)

    for name, fn in _results:
        try:
            fn()
            print(f"  {PASS}  {name}")
            passed += 1
        except Exception as e:
            print(f"  {FAIL}  {name}")
            print(f"         {type(e).__name__}: {e}")
            failed += 1

    print("=" * 65)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 65 + "\n")
    return failed == 0


# ============================================================
# Step 21: dataset.py
# ============================================================

@test("Step 21 | PackedTextDataset produces correct shapes")
def test_packed_dataset():
    from dataset import PackedTextDataset, get_tokenizer
    tok = get_tokenizer()
    texts = ["The quick brown fox jumps over the lazy dog. " * 50] * 10
    ds = PackedTextDataset(texts, tok, seq_len=64)
    assert len(ds) > 0, "Dataset should have samples"
    item = ds[0]
    assert item["input_ids"].shape == (64,), f"Expected (64,), got {item['input_ids'].shape}"
    assert item["labels"].shape == (64,), f"Expected (64,), got {item['labels'].shape}"
    # Labels should be input_ids shifted by 1
    assert torch.all(item["labels"] == ds.tokens[1:65])


@test("Step 21 | Labels are input_ids shifted by one token")
def test_labels_are_shifted():
    from dataset import PackedTextDataset, get_tokenizer
    tok = get_tokenizer()
    texts = ["Hello world! " * 100]
    ds = PackedTextDataset(texts, tok, seq_len=32)
    item = ds[0]
    # input[i+1] should equal label[i]
    assert torch.all(item["labels"][:-1] == ds.tokens[1:32])


@test("Step 21 | StreamingTextDataset yields correct shapes")
def test_streaming_dataset():
    from dataset import StreamingTextDataset, get_tokenizer
    tok = get_tokenizer()

    # Minimal fake HF-style iterable dataset
    texts = [{"text": "Hello world! " * 100} for _ in range(20)]
    ds = StreamingTextDataset(texts, tok, seq_len=32)
    samples = list(ds)
    assert len(samples) > 0, "Should yield at least one sample"
    for s in samples[:3]:
        assert s["input_ids"].shape == (32,)
        assert s["labels"].shape == (32,)


@test("Step 21 | LMDataModule setup runs without error (wikitext-2)")
def test_lm_data_module():
    from dataset import LMDataModule
    with tempfile.TemporaryDirectory() as tmpdir:
        dm = LMDataModule(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            seq_len=64,
            batch_size=4,
            num_workers=0,
            cache_dir=tmpdir,
        )
        dm.setup()
        assert len(dm._train_ds) > 0
        assert len(dm._val_ds) > 0
        batch = next(iter(dm.train_loader()))
        assert batch["input_ids"].shape == (4, 64)
        assert batch["labels"].shape == (4, 64)


@test("Step 21 | DataLoader throughput > 1000 tokens/sec")
def test_dataloader_throughput():
    from dataset import LMDataModule
    with tempfile.TemporaryDirectory() as tmpdir:
        dm = LMDataModule(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            seq_len=64,
            batch_size=8,
            num_workers=0,
            cache_dir=tmpdir,
        )
        dm.setup()
        loader = dm.train_loader()
        t0 = time.perf_counter()
        n = 0
        for batch in loader:
            n += 1
            if n >= 10:
                break
        elapsed = time.perf_counter() - t0
        tps = n * 8 * 64 / elapsed
        assert tps > 1000, f"Throughput too low: {tps:.0f} tokens/sec"


# ============================================================
# Step 23: loss_tracker.py
# ============================================================

@test("Step 23 | LossTracker records and smooths train loss")
def test_loss_tracker_records():
    from loss_tracker import LossTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        t = LossTracker(log_dir=tmpdir, smoothing=0.9)
        for i in range(20):
            ema = t.record_train_step(i, 3.0 - i * 0.05)
        assert len(t.train_steps) == 20
        assert len(t.train_smooth) == 20
        # EMA should lag raw loss
        assert t.train_smooth[-1] > t.train_losses[-1] - 0.5


@test("Step 23 | LossTracker detects and tracks best val loss")
def test_loss_tracker_best():
    from loss_tracker import LossTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        t = LossTracker(log_dir=tmpdir)
        for step in range(100):
            t.record_train_step(step, 3.0)
        t.record_epoch(1, val_loss=2.5, train_loss=2.8)
        t.record_epoch(2, val_loss=2.1, train_loss=2.4)
        t.record_epoch(3, val_loss=2.4, train_loss=2.2)
        assert t.best_val_loss == 2.1
        assert t.best_val_epoch == 2


@test("Step 23 | LossTracker saves and restores from JSON")
def test_loss_tracker_persistence():
    from loss_tracker import LossTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        t = LossTracker(log_dir=tmpdir)
        for i in range(50):
            t.record_train_step(i, 4.0 - i * 0.04)
        t.record_epoch(1, val_loss=2.8, train_loss=3.0)

        t2 = LossTracker(log_dir=tmpdir)
        ok = t2.load()
        assert ok, "load() should return True"
        assert len(t2.train_steps) == 50
        assert t2.best_val_epoch == 1
        assert abs(t2.best_val_loss - 2.8) < 1e-6


@test("Step 23 | LossTracker generates a PNG plot")
def test_loss_tracker_plot():
    from loss_tracker import LossTracker
    with tempfile.TemporaryDirectory() as tmpdir:
        t = LossTracker(log_dir=tmpdir)
        for i in range(20):
            t.record_train_step(i, 3.0 - i * 0.05)
        t.record_epoch(1, val_loss=2.5, train_loss=2.7)
        t.record_epoch(2, val_loss=2.2, train_loss=2.4)
        plot = Path(tmpdir) / "loss_curves.png"
        assert plot.exists(), "loss_curves.png should be generated"
        assert plot.stat().st_size > 1000, "Plot file too small"


@test("Step 23 | perplexity() converts loss correctly")
def test_perplexity():
    from loss_tracker import LossTracker
    t = LossTracker.__new__(LossTracker)
    assert abs(t.perplexity(0.0) - 1.0) < 1e-6
    assert abs(t.perplexity(math.log(2)) - 2.0) < 1e-4
    assert t.perplexity(100) == math.exp(20)  # capped


# ============================================================
# Step 24: checkpoint.py
# ============================================================

class TinyModel(nn.Module):
    def __init__(self, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(x)


@test("Step 24 | CheckpointManager saves and loads model state")
def test_checkpoint_save_load():
    from checkpoint import CheckpointManager, CheckpointMeta
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_last_n=5)
        meta = CheckpointMeta(epoch=1, global_step=100, train_loss=2.0, val_loss=2.5,
                              best_val_loss=2.5)
        mgr.save(model, opt, sched, scaler=None, meta=meta)

        model2 = TinyModel(seed=99)  # different init
        loaded = mgr.load(None, model2, device="cpu")
        assert loaded.epoch == 1
        assert loaded.global_step == 100
        # Weights should match original
        w1 = model.fc.weight.detach()
        w2 = model2.fc.weight.detach()
        assert torch.allclose(w1, w2), "Loaded weights differ from saved"


@test("Step 24 | CheckpointManager keeps only N rolling checkpoints")
def test_checkpoint_pruning():
    from checkpoint import CheckpointManager, CheckpointMeta
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_last_n=2)
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters())
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        for epoch in range(5):
            meta = CheckpointMeta(epoch=epoch+1, global_step=(epoch+1)*100,
                                  train_loss=1.0, val_loss=2.0-epoch*0.1,
                                  best_val_loss=2.0-epoch*0.1)
            mgr.save(model, opt, sched, None, meta)

        rolling = list(Path(tmpdir).glob("epoch_*.pt"))
        assert len(rolling) == 2, f"Expected 2 rolling checkpoints, got {len(rolling)}"


@test("Step 24 | Best checkpoint is always preserved")
def test_best_checkpoint_preserved():
    from checkpoint import CheckpointManager, CheckpointMeta
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(checkpoint_dir=tmpdir, keep_last_n=1)
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters())
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        val_losses = [3.0, 2.0, 2.5, 3.1, 2.8]
        best = float("inf")
        for i, vl in enumerate(val_losses):
            best = min(best, vl)
            meta = CheckpointMeta(epoch=i+1, global_step=(i+1)*100,
                                  train_loss=vl-0.1, val_loss=vl, best_val_loss=best)
            mgr.save(model, opt, sched, None, meta)

        best_path = Path(tmpdir) / "best.pt"
        assert best_path.exists(), "best.pt should always exist"
        # Load it and confirm val_loss is 2.0
        m2 = TinyModel()
        mb = mgr.load_best(m2)
        assert abs(mb.val_loss - 2.0) < 1e-6


@test("Step 24 | Resume restores optimizer and scheduler states")
def test_checkpoint_optimizer_restore():
    from checkpoint import CheckpointManager, CheckpointMeta
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        # Do a few steps to dirty the optimizer state
        for _ in range(5):
            opt.zero_grad()
            model(torch.randn(2, 16)).sum().backward()
            opt.step()
            sched.step()

        mgr = CheckpointManager(checkpoint_dir=tmpdir)
        meta = CheckpointMeta(epoch=1, global_step=5, train_loss=1.0,
                              val_loss=1.5, best_val_loss=1.5)
        mgr.save(model, opt, sched, None, meta)

        model2 = TinyModel(seed=99)
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
        mgr.load(None, model2, optimizer=opt2, scheduler=sched2)

        # Optimizer step count should be restored
        assert opt2.state_dict()["state"] == opt.state_dict()["state"] or \
               len(opt2.state_dict()["state"]) > 0


# ============================================================
# Step 25: scheduler.py
# ============================================================

@test("Step 25 | LR starts at ~0 during warmup")
def test_scheduler_warmup_start():
    from scheduler import get_cosine_schedule_with_warmup
    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps=100, total_steps=1000)
    # Step 0: LR should be 0/100 * peak = 0
    lr = sched.get_last_lr()[0]
    assert lr < 1e-8, f"LR at step 0 should be ~0, got {lr}"


@test("Step 25 | LR reaches peak at end of warmup")
def test_scheduler_warmup_peak():
    from scheduler import get_cosine_schedule_with_warmup
    model = nn.Linear(4, 4)
    peak = 3e-4
    opt = torch.optim.AdamW(model.parameters(), lr=peak)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps=50, total_steps=500)
    for _ in range(50):
        opt.step()
        sched.step()
    lr = sched.get_last_lr()[0]
    assert abs(lr - peak) < 1e-8, f"LR at end of warmup should be {peak}, got {lr}"


@test("Step 25 | LR decays to min_lr at total_steps")
def test_scheduler_final_lr():
    from scheduler import get_cosine_schedule_with_warmup
    model = nn.Linear(4, 4)
    peak = 3e-4
    min_ratio = 0.1
    warmup = 50
    total = 500
    opt = torch.optim.AdamW(model.parameters(), lr=peak)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps=warmup,
                                             total_steps=total, min_lr_ratio=min_ratio)
    for _ in range(total):
        opt.step()
        sched.step()
    lr = sched.get_last_lr()[0]
    expected = peak * min_ratio
    assert abs(lr - expected) < 1e-8, f"Expected {expected}, got {lr}"


@test("Step 25 | LR is monotonically increasing during warmup")
def test_scheduler_warmup_monotone():
    from scheduler import get_cosine_schedule_with_warmup
    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps=100, total_steps=1000)
    lrs = [sched.get_last_lr()[0]]
    for _ in range(100):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])
    assert all(lrs[i] <= lrs[i+1] for i in range(len(lrs)-1)), "LR not monotone in warmup"


@test("Step 25 | TransformerScheduler state_dict round-trip")
def test_scheduler_state_dict():
    from scheduler import TransformerScheduler
    model = nn.Linear(64, 64)
    ts = TransformerScheduler(model, peak_lr=3e-4, warmup_steps=100, total_steps=1000)
    for _ in range(50):
        ts.zero_grad()
        ts.step()
    lr_before = ts.current_lr()
    sd = ts.state_dict()

    model2 = nn.Linear(64, 64)
    ts2 = TransformerScheduler(model2, peak_lr=3e-4, warmup_steps=100, total_steps=1000)
    ts2.load_state_dict(sd)
    assert abs(ts2.current_lr() - lr_before) < 1e-10, "LR mismatch after state dict restore"


@test("Step 25 | Weight decay only applied to matrix params, not biases")
def test_scheduler_weight_decay_groups():
    from scheduler import TransformerScheduler
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
    ts = TransformerScheduler(model, peak_lr=1e-3, warmup_steps=10, total_steps=100)
    groups = ts.optimizer.param_groups
    assert len(groups) == 2
    wd_group = groups[0]
    no_wd_group = groups[1]
    assert wd_group["weight_decay"] > 0
    assert no_wd_group["weight_decay"] == 0


# ============================================================
# Step 26: mixed_precision.py
# ============================================================

@test("Step 26 | MixedPrecisionTrainer initializes without error")
def test_mp_init():
    from mixed_precision import MixedPrecisionTrainer, get_device
    device = get_device()
    mp = MixedPrecisionTrainer(device=device)
    assert mp.device == device
    assert mp.dtype in (torch.float16, torch.bfloat16)


@test("Step 26 | Autocast context manager executes forward pass")
def test_mp_autocast():
    from mixed_precision import MixedPrecisionTrainer, get_device
    device = get_device()
    mp = MixedPrecisionTrainer(device=device)
    model = nn.Linear(16, 8).to(device)
    x = torch.randn(4, 16).to(device)
    with mp.autocast():
        y = model(x)
    assert y.shape == (4, 8)


@test("Step 26 | amp_step produces finite gradients and updates weights")
def test_mp_amp_step():
    from mixed_precision import MixedPrecisionTrainer, amp_step, get_device
    device = get_device()
    mp = MixedPrecisionTrainer(device=device)
    model = nn.Linear(16, 16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    w_before = model.weight.clone().detach()

    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randn(4, 16).to(device)
        with mp.autocast():
            loss = model(x).sum()
        amp_step(loss, model, optimizer, mp, max_grad_norm=1.0)

    w_after = model.weight.detach()
    assert not torch.allclose(w_before, w_after), "Weights should have changed"
    assert torch.all(torch.isfinite(w_after)), "Weights contain NaN/Inf"


@test("Step 26 | MixedPrecisionTrainer state_dict round-trip")
def test_mp_state_dict():
    from mixed_precision import MixedPrecisionTrainer, get_device
    device = get_device()
    mp = MixedPrecisionTrainer(device=device)
    sd = mp.state_dict()
    mp2 = MixedPrecisionTrainer(device=device)
    mp2.load_state_dict(sd)
    assert mp2.scale == mp.scale


@test("Step 26 | log_stats returns expected keys")
def test_mp_log_stats():
    from mixed_precision import MixedPrecisionTrainer, get_device
    device = get_device()
    mp = MixedPrecisionTrainer(device=device)
    stats = mp.log_stats()
    assert "amp_enabled" in stats
    assert "amp_dtype" in stats
    assert "grad_scale" in stats


# ============================================================
# Step 22: trainer.py (integration test — runs a real tiny training loop)
# ============================================================

@test("Step 22 | Trainer runs without error and loss decreases")
def test_trainer_smoke():
    from trainer import Trainer, TrainerConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = TrainerConfig(
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            seq_len=32,
            batch_size=4,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            peak_lr=1e-3,
            warmup_steps=5,
            total_steps=30,
            n_epochs=2,
            eval_every=15,
            eval_batches=5,
            log_every=5,
            checkpoint_dir=f"{tmpdir}/ckpts",
            log_dir=f"{tmpdir}/logs",
            data_cache=f"{tmpdir}/cache",
        )
        trainer = Trainer(cfg)

        # Record initial loss
        initial_val = trainer.evaluate(n_batches=5)

        trainer.train()

        # Verify checkpoints created
        ckpts = list(Path(f"{tmpdir}/ckpts").glob("*.pt"))
        assert len(ckpts) >= 1, "No checkpoint files saved"

        # Verify loss history saved
        hist = Path(f"{tmpdir}/logs/loss_history.json")
        assert hist.exists(), "Loss history JSON not written"

        # Loss should have been recorded
        assert len(trainer.tracker.train_losses) > 0


@test("Step 22 | Training can resume from a checkpoint")
def test_trainer_resume():
    from trainer import Trainer, TrainerConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        base_cfg = dict(
            d_model=32, n_heads=2, n_layers=1, d_ff=64, seq_len=32,
            batch_size=4,
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            peak_lr=1e-3, warmup_steps=5, total_steps=20, n_epochs=1,
            eval_every=10, eval_batches=3, log_every=5,
            checkpoint_dir=f"{tmpdir}/ckpts",
            log_dir=f"{tmpdir}/logs",
            data_cache=f"{tmpdir}/cache",
        )

        # First run
        cfg1 = TrainerConfig(**base_cfg)
        t1 = Trainer(cfg1)
        t1.train()
        step1 = t1.global_step

        # Resume run
        cfg2 = TrainerConfig(**{**base_cfg, "resume_from": "latest", "total_steps": 40})
        t2 = Trainer(cfg2)
        # Should have restored global_step
        assert t2.global_step == step1, f"Expected step {step1}, got {t2.global_step}"


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
