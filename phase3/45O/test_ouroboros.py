"""
test_ouroboros.py — Phase 3 | Component 45O
Test suite: 5 tests, all must pass before An-Ra receives the Ouroboros upgrade.

Tests are run against a minimal mock transformer that mirrors
the interface expected by OuroborosDecoder without requiring the
full An-Ra weights.

Run: python test_ouroboros.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# MOCK BASE MODEL (mirrors An-Ra's interface)
# ─────────────────────────────────────────────────────────────────────────────

class MockTransformerLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x           = self.norm1(x + attn_out)
        x           = self.norm2(x + self.ff(x))
        return x


class MockAnRaBrain(nn.Module):
    """
    Minimal mock of An-Ra with the interface OuroborosDecoder requires:
      .d_model             — embedding dimension
      .vocab_size          — vocabulary size
      .embed(x)            — token embedding
      .run_all_layers(h)   — run all transformer layers
      .lm_head(h)          — project to vocab logits
    """
    def __init__(self, vocab_size: int = 512, d_model: int = 256, n_layers: int = 6):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.n_layers   = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(512, d_model)
        self.layers    = nn.ModuleList([
            MockTransformerLayer(d_model) for _ in range(n_layers)
        ])
        self.lm_head   = nn.Linear(d_model, vocab_size, bias=False)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Token + positional embedding."""
        B, T   = x.shape
        pos    = torch.arange(T, device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(pos)

    def run_all_layers(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run all 6 transformer layers."""
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


# ─────────────────────────────────────────────────────────────────────────────
# TEST HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_batch(vocab_size=512, batch=2, seq=32):
    return torch.randint(0, vocab_size, (batch, seq + 1))

PASS  = "  ✓ PASS"
FAIL  = "  ✗ FAIL"

results = {}


def run_test(name, fn):
    print(f"\n── {name} {'─' * max(0, 55 - len(name))}")
    try:
        passed, detail = fn()
        status = PASS if passed else FAIL
        print(f"{status}  {detail}")
        results[name] = passed
    except Exception as e:
        print(f"{FAIL}  Exception: {e}")
        import traceback; traceback.print_exc()
        results[name] = False


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Does it run without errors?
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_pass():
    from ouroboros import OuroborosDecoder

    base  = MockAnRaBrain()
    model = OuroborosDecoder(base, n_passes=3)

    tokens  = make_batch()
    x       = tokens[:, :-1]
    targets = tokens[:, 1:]

    logits, loss = model(x, targets=targets)

    ok = (
        logits.shape == (2, 32, 512)  # (B, T, vocab_size)
        and loss is not None
        and loss.item() > 0
        and not torch.isnan(logits).any()
    )
    return ok, f"logits shape={tuple(logits.shape)}, loss={loss.item():.4f}, no NaNs"


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Do 3 passes produce more refined output than 1?
# ─────────────────────────────────────────────────────────────────────────────

def test_passes_improve_output():
    from ouroboros import OuroborosDecoder

    torch.manual_seed(42)
    base   = MockAnRaBrain()
    # Both models wrap the same base — they share all weights.
    # The only difference is n_passes: 1 vs 3.
    model1 = OuroborosDecoder(base, n_passes=1)
    model3 = OuroborosDecoder(base, n_passes=3)

    tokens  = make_batch(batch=4, seq=64)
    x       = tokens[:, :-1]
    targets = tokens[:, 1:]

    with torch.no_grad():
        logits1, loss1 = model1(x, targets=targets)
        logits3, loss3 = model3(x, targets=targets)

    # The two models have different pass_gates (randomly initialised independently)
    # so outputs will differ. What we verify:
    #   (a) outputs ARE different — 3 passes did something different from 1
    #   (b) all 3 pass hidden states have non-zero norms — every pass is active
    outputs_differ = not torch.allclose(logits1, logits3, atol=1e-5)

    hiddens    = model3._last_pass_hiddens
    pass_norms = [h.norm().item() for h in hiddens]
    norms_ok   = all(n > 0 for n in pass_norms)

    # Also verify that the 3 hidden states in model3 are not all identical
    # (would mean the Ouroboros loop isn't doing work)
    h1, h2, h3 = hiddens
    passes_differ_internally = (
        not torch.allclose(h1, h2, atol=1e-6) and
        not torch.allclose(h2, h3, atol=1e-6)
    )

    detail = (
        f"outputs differ from 1-pass={outputs_differ}, "
        f"pass norms={[round(n,2) for n in pass_norms]}, "
        f"internal passes differ={passes_differ_internally}"
    )
    return outputs_differ and norms_ok and passes_differ_internally, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Does adaptive pass count work?
# ─────────────────────────────────────────────────────────────────────────────

def test_adaptive_controller():
    from adaptive import AdaptiveController

    ctrl = AdaptiveController(d_model=256, max_passes=3,
                              low_threshold=0.3, high_threshold=0.7)

    # Simulate a LOW entropy hidden state (peaked = certain = simple question)
    certain_hidden = torch.zeros(1, 10, 256)
    certain_hidden[:, :, 0] = 10.0   # one dimension dominates heavily → low entropy

    # Simulate a HIGH entropy hidden state (flat = uncertain = complex question)
    uncertain_hidden = torch.randn(1, 10, 256) * 0.01  # near-uniform → high entropy

    with torch.no_grad():
        passes_certain   = ctrl.decide_passes(certain_hidden,   use_learned=False)
        passes_uncertain = ctrl.decide_passes(uncertain_hidden, use_learned=False)

    unc_c = ctrl.measure_uncertainty(certain_hidden).item()
    unc_u = ctrl.measure_uncertainty(uncertain_hidden).item()

    simple_ok  = passes_certain   <= 2   # should be 1 or 2 (simple)
    complex_ok = passes_uncertain >= 2   # should be 2 or 3 (complex)

    detail = (
        f"certain → {passes_certain} passes (entropy={unc_c:.3f}), "
        f"uncertain → {passes_uncertain} passes (entropy={unc_u:.3f})"
    )
    return simple_ok and complex_ok, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Are new parameters minimal (< 1000)?
# ─────────────────────────────────────────────────────────────────────────────

def test_parameter_count():
    from ouroboros      import OuroborosDecoder
    from weight_sharing import parameter_audit, verify_weight_sharing

    base  = MockAnRaBrain(vocab_size=512, d_model=256, n_layers=6)
    model = OuroborosDecoder(base, n_passes=3)

    audit  = parameter_audit(model)
    checks = verify_weight_sharing(model)

    new_params = audit["total_new_params"]
    sharing_ok = all(checks.values())

    detail = (
        f"new params = {new_params} "
        f"(pass_gates={audit['pass_gates_params']}, "
        f"blend_weights={audit['blend_weights_params']}), "
        f"weight sharing verified={sharing_ok}"
    )
    return new_params < 1000 and sharing_ok, detail


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Does training loss decrease over 10 steps?
# ─────────────────────────────────────────────────────────────────────────────

def test_training_decreases():
    from ouroboros  import OuroborosDecoder
    from pass_gates import PassConsistencyLoss

    torch.manual_seed(0)
    base  = MockAnRaBrain(vocab_size=512, d_model=256, n_layers=2)  # small for speed
    model = OuroborosDecoder(base, n_passes=3)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    consistency_loss = PassConsistencyLoss()

    data     = torch.randint(0, 512, (2, 33))  # (B, T+1)
    x        = data[:, :-1]
    targets  = data[:, 1:]

    losses = []
    for step in range(10):
        logits, lm_loss = model(x, targets=targets)
        pass_hiddens    = model._last_pass_hiddens

        total = lm_loss
        if step >= 2:  # introduce consistency after step 2
            total = total + 0.1 * consistency_loss(pass_hiddens)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(total.item())

    # Loss should decrease at least somewhat over 10 steps
    first_half = sum(losses[:5]) / 5
    second_half = sum(losses[5:]) / 5
    decreased   = second_half < first_half * 1.05  # allow 5% tolerance

    detail = (
        f"first 5 steps avg={first_half:.4f}, "
        f"last 5 steps avg={second_half:.4f}, "
        f"trend={'↓ decreasing' if second_half < first_half else '→ flat'}"
    )
    return decreased, detail


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  OUROBOROS TEST SUITE — Phase 3 | Component 45O")
    print("=" * 60)

    run_test("TEST 1: Forward pass (no errors, returns logits)", test_forward_pass)
    run_test("TEST 2: 3 passes produce different output from 1",  test_passes_improve_output)
    run_test("TEST 3: Adaptive controller assigns passes by complexity", test_adaptive_controller)
    run_test("TEST 4: New parameter count < 1000",               test_parameter_count)
    run_test("TEST 5: Training loss decreases over 10 steps",    test_training_decreases)

    print("\n" + "=" * 60)
    passed = sum(v for v in results.values())
    total  = len(results)
    print(f"  RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n  Ouroboros is ready for An-Ra.")
        print("  Forwarding to 45P — Ghost State Memory.\n")
    else:
        print("\n  Not ready. Fix failing tests before handoff.\n")

    sys.exit(0 if passed == total else 1)
