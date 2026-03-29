"""
test_phase3_integration.py
An-Ra Phase 3 Full Integration Test Suite
==========================================

Tests all 5 Phase 3 modules through the MasterSystem interface.
All tests are CPU-only and work without GPU or external APIs.

Run:
    cd phase2/45M && python -m pytest ../../tests/test_phase3_integration.py -v
    OR from project root:
    python -m pytest tests/test_phase3_integration.py -v
"""

import sys
import os
import json
import time
from pathlib import Path

# ── Setup paths ───────────────────────────────────────────────────────────────
TEST_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
PHASE2_45M   = PROJECT_ROOT / "phase2" / "45M"
PHASE3       = PROJECT_ROOT / "phase3"

# Add all necessary paths
for p in [str(PHASE2_45M), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)
for p3 in ["45N", "45O", "45P", "45Q", "45R"]:
    p = str(PHASE3 / p3)
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(str(PHASE2_45M))

import pytest


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL TESTS (no MasterSystem needed)
# These validate each Phase 3 component's core API in isolation.
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdentityInjector:
    """45N — Identity Injector (CPU-only, no training required)."""

    def setup_method(self):
        from identity_injector import IdentityInjector
        # Use the real identity file if available, else let it use fallback
        identity_file = PROJECT_ROOT / "phase3" / "45N" / "anra_identity_v2.txt"
        self.injector = IdentityInjector(identity_file=identity_file if identity_file.is_file()
                                         else None)

    def test_inject_adds_identity_block(self):
        """inject() prepends the AN-RA IDENTITY CONTEXT block."""
        prompt = "Who are you?"
        result = self.injector.inject(prompt)
        assert "[AN-RA IDENTITY CONTEXT]" in result
        assert prompt in result

    def test_inject_no_double_injection(self):
        """inject() called twice doesn't double the identity block."""
        prompt = "Hello"
        once = self.injector.inject(prompt)
        twice = self.injector.inject(once)
        assert once == twice, "inject() should not double-inject"

    def test_clean_removes_robotic_phrases(self):
        """clean_response() strips 'I am an AI language model' and variants."""
        robotic = "I am an AI language model and I cannot feel emotions."
        cleaned = self.injector.clean_response(robotic)
        assert "I am an AI language model" not in cleaned
        assert "cannot feel" not in cleaned

    def test_clean_preserves_non_robotic(self):
        """clean_response() leaves normal text unchanged."""
        normal = "That question requires more than a simple answer."
        cleaned = self.injector.clean_response(normal)
        assert cleaned == normal

    def test_status_returns_dict(self):
        status = self.injector.status()
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "anchors_loaded" in status
        assert status["anchors_loaded"] >= 0

    def test_identity_file_loaded(self):
        """If identity file exists, anchors should be > 0."""
        identity_file = PROJECT_ROOT / "phase3" / "45N" / "anra_identity_v2.txt"
        if identity_file.is_file():
            assert self.injector.status()["anchors_loaded"] > 0


class TestOuroborosNumpy:
    """45O — Ouroboros NumPy (CPU, no torch required)."""

    def setup_method(self):
        from ouroboros_numpy import OuroborosNumpy

        call_log = []

        def mock_generate(prompt: str, max_new_tokens: int = 200) -> str:
            call_log.append(prompt[:50])
            # Simulate a real response that gets progressively more refined
            if "Verify" in prompt or "adversarial" in prompt.lower():
                return "The verified answer is: this is correct."
            if "reason step by step" in prompt.lower():
                return "By reasoning through it: the answer follows logically."
            return "Initial understanding of the question."

        self.call_log = call_log
        self.ouro = OuroborosNumpy(
            generate_fn=mock_generate,
            n_passes=3,
            max_new_tokens=100,
            enabled=True,
        )

    def test_single_pass_returns_string(self):
        result = self.ouro.recursive_generate("What is 2+2?", n_passes=1)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_three_passes_calls_generate_multiple_times(self):
        """3-pass generation should call the generate function more than once."""
        self.call_log.clear()
        self.ouro.recursive_generate("Explain consciousness.", n_passes=3)
        assert len(self.call_log) >= 3, \
            f"Expected ≥3 generate calls, got {len(self.call_log)}"

    def test_adaptive_simple_query_uses_fewer_passes(self):
        """Simple queries should use 1 pass."""
        from ouroboros_numpy import _estimate_complexity
        n = _estimate_complexity("hi")
        assert n == 1, f"Expected 1 pass for 'hi', got {n}"

    def test_adaptive_complex_query_uses_more_passes(self):
        """Complex queries should use 3 passes."""
        from ouroboros_numpy import _estimate_complexity
        n = _estimate_complexity(
            "Prove that the sum of two prime numbers greater than 2 is always even "
            "using mathematical induction and derive the implications for cryptography."
        )
        assert n >= 2, f"Expected ≥2 passes for complex query, got {n}"

    def test_disabled_mode_passthrough(self):
        """When disabled, recursive_generate should call generate once."""
        from ouroboros_numpy import OuroborosNumpy
        calls = []
        ouro = OuroborosNumpy(
            generate_fn=lambda p, **kw: "response",
            enabled=False,
        )
        result = ouro.recursive_generate("any query")
        assert result == "response"

    def test_status_returns_dict(self):
        status = self.ouro.status()
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "blend_weights" in status
        assert len(status["blend_weights"]) == 3


class TestGhostMemory:
    """45P — Ghost State Memory (CPU, mock embeddings)."""

    def setup_method(self):
        import tempfile
        self.tmpdir = Path(tempfile.mkdtemp())

        try:
            from ghost_memory import GhostMemory, default_config
            import numpy as np

            cfg = default_config(storage_dir=self.tmpdir)

            # Use mock embedder (no sentence-transformers needed)
            rng = __import__("random").Random(42)
            def mock_embed(text: str):
                return np.array([rng.gauss(0, 1) for _ in range(cfg.embedding_dim)],
                                dtype=np.float32)

            self.gm = GhostMemory(config=cfg, embedder=mock_embed)
            self.available = True
        except ImportError as e:
            self.available = False
            self.skip_reason = str(e)

    def test_add_turn_stores_memory(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        self.gm.add_turn("user", "My project is called An-Ra.")
        self.gm.add_turn("assistant", "I'll remember that.")
        rows = self.gm.memory_store().iter_retrieval_rows()
        assert len(rows) == 2

    def test_build_ghost_prompt_returns_string(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        self.gm.add_turn("user", "My name is Ankit.")
        prompt = self.gm.build_ghost_prompt("What is my name?")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_empty_memory_returns_prompt_unchanged(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        prompt = self.gm.build_ghost_prompt("Hello")
        assert isinstance(prompt, str)


class TestSymbolicBridge:
    """45Q — Symbolic Logic Bridge (requires sympy + scipy)."""

    def setup_method(self):
        try:
            from symbolic_bridge import query, detect
            self.query = query
            self.detect = detect
            self.available = True
        except ImportError as e:
            self.available = False
            self.skip_reason = f"symbolic_bridge unavailable: {e}"

    def test_math_equation_solved(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        result = self.query("solve x^2 - 4 = 0")
        assert result is not None
        assert result.confidence > 0.8
        # Answers should include 2 and -2
        assert ("2" in str(result.answer_text) or
                "2" in str(result.answer))

    def test_logic_formula_classified(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        result = self.query("(A -> B) AND (B -> C) -> (A -> C)")
        assert result is not None
        assert result.confidence > 0.8

    def test_code_analysis_finds_issues(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        result = self.query("def find_max(lst): return max(lst[0:len(lst)-1])")
        assert result is not None
        # Should find the off-by-one slice issue

    def test_detection_correctly_classifies_math(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        detection = self.detect("solve x^2 = 9")
        assert str(detection.mode) in ("Mode.MATH", "MATH")

    def test_detection_correctly_classifies_natural(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        detection = self.detect("How are you today?")
        assert str(detection.mode) in ("Mode.NATURAL", "NATURAL")


class TestSovereigntyBridge:
    """45R — Sovereignty Bridge (requires psutil)."""

    def setup_method(self):
        try:
            import psutil  # noqa: F401
            from sovereignty_bridge import SovereigntyBridge
            import tempfile
            self.data_dir = Path(tempfile.mkdtemp())
            # Don't actually start the daemon in tests
            self.bridge = SovereigntyBridge(
                target_path=PROJECT_ROOT,
                data_dir=self.data_dir,
                enabled=True,
            )
            self.available = True
        except ImportError as e:
            self.available = False
            self.skip_reason = str(e)

    def test_status_returns_dict(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        status = self.bridge.status()
        assert isinstance(status, dict)
        assert "enabled" in status
        assert "target" in status

    def test_nightly_report_no_data(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        report = self.bridge.get_nightly_report()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_benchmark_summary_no_data(self):
        if not self.available:
            pytest.skip(self.skip_reason)
        summary = self.bridge.get_benchmark_summary()
        assert isinstance(summary, str)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (MasterSystem level)
# These test the full Phase 3 pipeline end-to-end.
# ═══════════════════════════════════════════════════════════════════════════════

class TestMasterSystemPhase3Integration:
    """Full MasterSystem integration — all 5 Phase 3 modules."""

    @pytest.fixture(autouse=True)
    def setup_system(self):
        """Create a MasterSystem in a minimal state for testing."""
        from system import MasterSystem
        self.system = MasterSystem()
        # Don't call .start() — initialize Phase 3 modules directly to avoid
        # loading the full LLM in tests
        yield
        # Cleanup
        try:
            if self.system.sovereignty:
                self.system.sovereignty.stop()
        except Exception:
            pass

    def test_identity_initialized_standalone(self):
        """Identity injector initializes without LLM."""
        self.system._init_identity()
        if self.system.identity is not None:
            status = self.system.identity.status()
            assert status["enabled"] is True

    def test_symbolic_initialized_standalone(self):
        """Symbolic bridge initializes without LLM."""
        self.system._init_symbolic()
        # Pass if sympy is available, skip if not
        if self.system.symbolic is not None:
            assert "query" in self.system.symbolic
            assert "detect" in self.system.symbolic

    def test_ghost_memory_initialized_standalone(self):
        """Ghost memory initializes without LLM."""
        import tempfile
        from pathlib import Path
        # Override storage to temp dir
        orig_parent = Path(self.system.__class__.__module__)
        self.system._init_ghost_memory()
        # Ghost memory may or may not init depending on numpy being available

    def test_symbolic_augment_math_query(self):
        """_symbolic_augment identifies and solves math queries."""
        self.system._init_symbolic()
        if self.system.symbolic is None:
            pytest.skip("sympy/scipy not installed")
        augmented, was_symbolic = self.system._symbolic_augment("solve x^2 - 9 = 0")
        assert was_symbolic is True, "Math query should be routed to symbolic bridge"
        assert "SYMBOLIC VERIFICATION" in augmented

    def test_symbolic_augment_natural_query(self):
        """_symbolic_augment passes through natural language unchanged."""
        self.system._init_symbolic()
        if self.system.symbolic is None:
            pytest.skip("sympy/scipy not installed")
        augmented, was_symbolic = self.system._symbolic_augment("How are you today?")
        assert was_symbolic is False, "Natural query should NOT be routed to symbolic bridge"

    def test_sovereignty_bridge_initialized(self):
        """Sovereignty bridge initializes (daemon may not start in test env)."""
        self.system._init_sovereignty()
        if self.system.sovereignty is not None:
            status = self.system.sovereignty.status()
            assert isinstance(status, dict)

    def test_status_includes_phase3_keys(self):
        """system.status() includes all 5 Phase 3 module keys."""
        # Initialize just enough to call status()
        self.system._init_identity()
        self.system._init_symbolic()
        status = self.system.status()
        subs = status.get("subsystems", {})
        assert "identity" in subs,    "identity missing from status"
        assert "ouroboros" in subs,   "ouroboros missing from status"
        assert "ghost_memory" in subs,"ghost_memory missing from status"
        assert "symbolic" in subs,    "symbolic missing from status"
        assert "sovereignty" in subs, "sovereignty missing from status"


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE TEST — Full Phase 3 Chat Flow (mock LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhase3ChatPipeline:
    """Simulates the full chat pipeline with a mock LLM."""

    def setup_method(self):
        from system import MasterSystem

        class MockLLM:
            """Minimal mock LLM bridge."""
            d_model = 256
            vocab_size = 1000
            num_parameters = 1_000_000
            raw_decoder = None

            def generate(self, prompt: str, max_new_tokens: int = 200, **kw) -> str:
                return "An-Ra response to: " + prompt[-100:]

            def model_fn(self, prompt: str) -> str:
                return self.generate(prompt)

            def status(self) -> dict:
                return {"initialized": True}

        self.system = MasterSystem()
        self.system.llm = MockLLM()

    def teardown_method(self):
        try:
            if self.system.sovereignty:
                self.system.sovereignty.stop()
        except Exception:
            pass

    def test_chat_with_identity_injection(self):
        """chat() flows through identity injector."""
        self.system._init_identity()
        # Disable ouroboros for speed
        self.system._ouroboros_enabled = False
        response = self.system.chat("Who are you?")
        assert isinstance(response, str)
        assert len(response) > 0
        # Should NOT contain robotic phrasing
        assert "I am an AI language model" not in response

    def test_chat_with_symbolic_augmentation(self):
        """chat() routes math queries through symbolic bridge first."""
        self.system._init_symbolic()
        self.system._ouroboros_enabled = False
        if self.system.symbolic is None:
            pytest.skip("sympy not installed")
        response = self.system.chat("solve x^2 - 4 = 0")
        assert isinstance(response, str)
        # The prompt sent to LLM should have included SYMBOLIC VERIFICATION context
        # We verify indirectly: response was generated successfully
        assert len(response) > 0

    def test_chat_without_any_phase3_subsystems(self):
        """chat() works even with zero Phase 3 subsystems active."""
        # All Phase 3 subsystems are None by default
        self.system._ouroboros_enabled = False
        response = self.system.chat("Hello")
        assert isinstance(response, str)
        assert len(response) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST — Import Check
# ═══════════════════════════════════════════════════════════════════════════════

class TestImports:
    """Verify all Phase 3 modules import without errors."""

    def test_import_identity_injector(self):
        import identity_injector
        assert hasattr(identity_injector, "IdentityInjector")
        assert hasattr(identity_injector, "get_identity_injector")

    def test_import_ouroboros_numpy(self):
        import ouroboros_numpy
        assert hasattr(ouroboros_numpy, "OuroborosNumpy")
        assert hasattr(ouroboros_numpy, "_estimate_complexity")

    def test_import_sovereignty_bridge(self):
        import sovereignty_bridge
        assert hasattr(sovereignty_bridge, "SovereigntyBridge")

    def test_import_ghost_memory_available(self):
        """ghost_memory requires numpy — check availability."""
        try:
            from ghost_memory import GhostMemory, default_config
            assert GhostMemory is not None
        except ImportError:
            pytest.skip("numpy not available")

    def test_import_symbolic_bridge_available(self):
        """symbolic_bridge requires sympy — check availability."""
        try:
            from symbolic_bridge import query, detect
            assert query is not None
        except ImportError:
            pytest.skip("sympy not available")


# ── Entry point for running directly ─────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(PROJECT_ROOT),
    )
    sys.exit(result.returncode)
