"""500M production-entry contract suite (CPU default, CUDA via ANRA_TEST_DEVICE).

Run from the cymek-500m worktree root:
    <repo>/.venv/Scripts/python.exe -m pytest tests/test_production_entry.py -x -q
or:  python tests/test_production_entry.py
Device ops run on the seam device; everything else is the REAL production
chain: frozen topology -> bucket lanes -> mixture schedule -> accumulation
-> trainer -> checkpoint transactions.
"""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anra_v5.miniature_run import MINI_SPEC  # noqa: E402
from tests.cpu_seam import cpu_seams  # noqa: E402
from v5_data import materialize_first_party as mfp  # noqa: E402
from v5_tokenizer.artifact import load_verified_tokenizer  # noqa: E402
from v5_tokenizer.freeze_production import freeze_production_identity  # noqa: E402
from v5_training.checkpoint import CheckpointStore  # noqa: E402
from v5_training.production_entry import (  # noqa: E402
    ENTRY_SCHEMA,
    MILESTONE_TOKENS,
    VERIFIED_COGNITION,
    build_milestone_receipt,
    crossed_milestones,
    frozen_mixture_fractions,
    frozen_topology,
    microstep_buckets,
    partial_microstep_plan,
    prepare_data,
    resolve_cymek_sha,
    run_500m_session,
    run_campaign,
    topology_sha256,
    verify_milestone_receipt,
)
from v5_training.production_backend import precision_receipt  # noqa: E402
from v5_training.schedule import lr_at  # noqa: E402

TEST_SHA = "ab" * 20
FREEZE_SHA = "cc" * 32
BENCHMARKS = {"synthetic-bench": "zzzqqqwww xxxjjjvvv"}
PROD_MIXTURE = {"natural": 0.65, "code_math_formal": 0.20,
                "verified_cognition": 0.15}
SINGLE_MIXTURE = {"natural": 1.0}

_token_cache: dict = {}
_bucketed_cache: dict = {}
_mixture_cache: dict = {}


def _seams():
    """Device seam: CPU default, CUDA when ANRA_TEST_DEVICE=cuda."""

    import os
    if os.environ.get("ANRA_TEST_DEVICE", "cpu") == "cuda":
        from contextlib import contextmanager

        @contextmanager
        def cuda_seams():
            import torch
            yield torch, torch.device("cuda")

        return cuda_seams
    return cpu_seams


class _Tok:
    """Wraps the verified backend + identity into the encode/identity
    surface production_entry expects."""

    def __init__(self, backend, identity):
        self.backend = backend
        self.identity = identity

    def encode(self, text):
        return list(self.backend.encode(text).ids)


def _tokenizer():
    if "tok" not in _token_cache:
        raw, identity = load_verified_tokenizer(
            ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz",
            expected_sha256=_artifact_sha(),
            vocabulary_size=24_576,
            trainer_config_sha256=_trainer_sha(),
            corpus_manifest_sha256=_corpus_sha())
        _token_cache["tok"] = _Tok(raw, identity)
    return _token_cache["tok"]


def _documents(n_arith=240, n_notes=60):
    docs = []
    for i in range(n_arith):
        a = i % 500 + 10
        b = (i * 37) % 900 + 100
        docs.append({"doc_id": f"arith-{i:05d}",
                     "text": f"sample {i}: {a} + {b} = {a + b}",
                     "source_id": f"arith-{i:05d}",
                     "family": "verified_cognition",
                     "domain": "synthetic-cognition"})
    for i in range(n_notes):
        docs.append({"doc_id": f"notes-{i:04d}",
                     "text": f"training note {i}: the model learns addition "
                             f"and carries digits correctly batch {i} with "
                             f"multiple operations and edge cases",
                     "source_id": f"notes-{i:04d}", "family": "natural",
                     "domain": "notes"})
    return docs


_WORD_POOL = ("alpha beta gamma delta epsilon zeta eta theta iota kappa "
               "lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi "
               "omega").split()


def _words(tag, count):
    words = []
    for j in range(count):
        words.append(_WORD_POOL[j % len(_WORD_POOL)])
        if j % 40 == 39:
            words.append(f"m{tag}q{j}")
    return words


def _sized_text(tok, target_tokens, tag):
    """Deterministic text encoding to exactly target_tokens tokens.

    Calibrates coarsely below target, then appends single alphas (measured
    +1 token each). Raises loudly if exactness is unreachable.
    """
    estimate = max(10, int((target_tokens - 12) / 1.7))
    words = _words(tag, estimate)
    for _ in range(12):
        measured = len(tok.encode(" ".join(words)))
        gap = target_tokens - measured
        if 1 <= gap <= 40:
            break
        if gap == 0:
            return " ".join(words)
        step = int(gap / 1.7)
        if step == 0:
            step = 1 if gap > 0 else -1
        estimate = max(10, estimate + step)
        words = _words(tag, estimate)
    for _ in range(40):
        measured = len(tok.encode(" ".join(words)))
        if measured == target_tokens:
            return " ".join(words)
        if measured > target_tokens:
            raise ValueError(f"cannot size text below {measured} for {tag}")
        words.append("alpha")
    raise ValueError(f"cannot size text to exactly {target_tokens} tokens")


def _bucketed_documents():
    """Exact-full rows per bucket for frozen-shape execution.

    Every doc carries exactly bucket-2 content tokens, so each packs to one
    exactly-full sequence: microsteps execute the frozen per-replica counts
    (64/32/16/8). Two sparse 4096 docs cover partial tails (pad path).
    """
    if "docs" not in _bucketed_cache:
        tok = _tokenizer()
        docs = []

        def add(prefix, count, content_tokens):
            for i in range(count):
                docs.append({"doc_id": f"{prefix}-{i:04d}",
                             "text": _sized_text(tok, content_tokens,
                                                 f"{prefix}{i}"),
                             "source_id": f"{prefix}-{i:04d}",
                             "family": "natural"})

        add("b512", 128, 510)
        add("b1024", 64, 1022)
        add("b2048", 48, 2046)
        add("b4096", 8, 4094)
        add("b4096t", 2, 3000)
        _bucketed_cache["docs"] = docs
    return _bucketed_cache["docs"]


def _mixture_documents():
    """Segregated exact-full cells for the frozen-mixture campaign test.

    Cells match the planner's exact demand for budget 100000:
    (512,code)=32768, (1024,nat)=32768, (2048,nat)=32768, (4096,cog)=1696.
    """
    if "docs" not in _mixture_cache:
        tok = _tokenizer()
        docs = []

        def add(prefix, family, count, content_tokens):
            for i in range(count):
                docs.append({"doc_id": f"{prefix}-{family}-{i:04d}",
                             "text": _sized_text(tok, content_tokens,
                                                 f"{prefix}{family}{i}"),
                             "source_id": f"{prefix}-{family}-{i:04d}",
                             "family": family})

        add("mx512", "code_math_formal", 64, 510)
        add("mx1024", "natural", 32, 1022)
        add("mx2048", "natural", 16, 2046)
        add("mx4096", "verified_cognition", 1, 3000)
        _mixture_cache["docs"] = docs
    return _mixture_cache["docs"]


def _natural_documents(n_notes=200):
    return _documents(n_arith=0, n_notes=n_notes)


def _with_raw(documents):
    return [dict(d, raw_source_sha256=hashlib.sha256(
        d["text"].encode("utf-8")).hexdigest()) for d in documents]


def _run(documents, torch, device, tmp, name, **kw):
    params = {"documents": documents, "tokenizer": _tokenizer(),
              "model_spec": MINI_SPEC, "run_id": name, "seed": 20260906,
              "store_root": str(Path(tmp) / name), "device": device,
              "torch_module": torch, "xb": object(),
              "development_mode": True, "cymek_sha": TEST_SHA}
    params.update(kw)
    return run_campaign(**params)


def _prod(documents, torch, device, tmp, name, **kw):
    params = {"development_mode": False,
              "campaign_tokens": 4000,
              "contamination_benchmarks": dict(BENCHMARKS),
              "mixture_fractions": dict(SINGLE_MIXTURE),
              "tokenizer_freeze_sha256": FREEZE_SHA}
    params.update(kw)
    return _run(_with_raw(documents), torch, device, tmp, name, **params)


# -- fresh campaigns ------------------------------------------------------

def test_exact_completion_single_partial_update():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "fresh-partial",
                      campaign_tokens=4000)
        assert result["schema"] == ENTRY_SCHEMA
        assert result["updates_executed"] == 1
        assert result["cumulative_tokens"] == 4000
        assert result["state_complete"] is True
        assert result["termination"] == "COMPLETE"
        assert result["resumed"] is False
        assert len(result["losses"]) == 1
        assert result["resume_equal"] is True
        last = result["last_update_receipt"]
        assert last["microsteps"] == 1
        assert 0 < last["supervised_tokens"] < 4000
        assert sum(result["tokens_by_source"].values()) == 4000
        shape = result["microstep_shapes"][0]
        assert shape["requested_bucket"] == 512
        assert shape["actual_row_widths"] == [512]
        assert shape["physical"].get("partial_tail") is True


def test_fresh_determinism_same_inputs_same_receipt():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _documents()
        first = _run(docs, torch, device, tmp, "det-a", campaign_tokens=4000)
        second = _run(docs, torch, device, tmp, "det-b", campaign_tokens=4000)
        assert first["losses"] == second["losses"]
        assert first["last_update_receipt"] == second["last_update_receipt"]
        assert first["pack_manifest_sha256"] == second["pack_manifest_sha256"]
        assert first["topology_sha256"] == second["topology_sha256"]
        assert first["checkpoint_head"] != second["checkpoint_head"]


def test_pack_smaller_than_budget_fails_closed():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        try:
            _run(_documents(n_arith=4, n_notes=1), torch, device, tmp,
                 "too-small", campaign_tokens=1_000_000)
        except ValueError as exc:
            assert "pack" in str(exc).lower() or "window" in str(exc).lower() \
                or "supply" in str(exc).lower() or "data_not_ready" in str(exc).lower()
        else:
            raise AssertionError("over-budget campaign did not fail closed")


# -- resume and identity ---------------------------------------------------

def test_mid_campaign_resume_matches_uninterrupted():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _bucketed_documents()
        budget = 131_072 + 5000
        part = _run(docs, torch, device, tmp, "resume-run",
                    campaign_tokens=budget, max_updates=1)
        assert part["termination"] == "MANUAL_BOUNDARY"
        assert part["cumulative_tokens"] == 131_072
        continued = _run(docs, torch, device, tmp, "resume-run",
                         campaign_tokens=budget)
        assert continued["resumed"] is True
        assert continued["cumulative_tokens"] == budget
        assert continued["state_complete"] is True
        whole = _run(docs, torch, device, tmp, "whole-run",
                     campaign_tokens=budget)
        assert whole["cumulative_tokens"] == budget
        assert whole["tokens_by_source"] == continued["tokens_by_source"]
        continued_store = CheckpointStore(Path(tmp) / "resume-run",
                                          "resume-run")
        whole_store = CheckpointStore(Path(tmp) / "whole-run", "whole-run")
        _, continued_payloads = continued_store.restore()
        _, whole_payloads = whole_store.restore()
        for component in ("model.bin", "optimizer.bin", "rng.bin",
                          "cursor.json", "ledger.json"):
            assert hashlib.sha256(continued_payloads[component]).hexdigest() == \
                hashlib.sha256(whole_payloads[component]).hexdigest(), \
                f"resumed run diverges at {component}"


def test_resume_changed_documents_fails():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _documents()
        _run(docs, torch, device, tmp, "drift-docs", campaign_tokens=4000,
             max_updates=1)
        drifted = list(docs) + [{"doc_id": "extra-00001",
                                 "text": "an extra smuggled document",
                                 "source_id": "extra-00001"}]
        try:
            _run(drifted, torch, device, tmp, "drift-docs",
                 campaign_tokens=4000)
        except ValueError as exc:
            assert "drift" in str(exc).lower()
        else:
            raise AssertionError("document drift did not fail closed")


def test_resume_changed_seed_fails():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _documents()
        _run(docs, torch, device, tmp, "drift-seed", campaign_tokens=4000,
             max_updates=1)
        try:
            run_campaign(documents=docs, tokenizer=_tokenizer(),
                         model_spec=MINI_SPEC, run_id="drift-seed", seed=7,
                         campaign_tokens=4000,
                         store_root=str(Path(tmp) / "drift-seed"),
                         device=device, torch_module=torch, xb=object(),
                         development_mode=True, cymek_sha=TEST_SHA)
        except ValueError as exc:
            assert "drift" in str(exc).lower()
        else:
            raise AssertionError("seed drift did not fail closed")


def test_resume_changed_budget_fails():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        _run(_documents(), torch, device, tmp, "drift-budget",
             campaign_tokens=4000, max_updates=1)
        try:
            _run(_documents(), torch, device, tmp, "drift-budget",
                 campaign_tokens=4200)
        except ValueError as exc:
            assert "drift" in str(exc).lower()
        else:
            raise AssertionError("budget drift did not fail closed")


def test_cross_store_and_cross_lineage_rejected():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        try:
            _run(_documents(), torch, device, tmp, "xstore",
                 campaign_tokens=4000,
                 resume_store_root=str(Path(tmp) / "elsewhere"))
        except ValueError as exc:
            assert "cross-store" in str(exc).lower()
        else:
            raise AssertionError("cross-store resume was not rejected")
        try:
            _run(_documents(), torch, device, tmp, "xline",
                 campaign_tokens=4000, resume_run_id="other-lineage")
        except ValueError as exc:
            assert "cross-lineage" in str(exc).lower()
        else:
            raise AssertionError("cross-lineage resume was not rejected")


# -- accumulation and bucket execution ---------------------------------------

def test_full_update_uses_four_microsteps_one_step():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_bucketed_documents(), torch, device, tmp,
                      "full-update", campaign_tokens=131_072)
        assert result["updates_executed"] == 1
        assert result["cumulative_tokens"] == 131_072
        last = result["last_update_receipt"]
        assert last["microsteps"] == 4
        assert 0 < last["supervised_tokens"] < 131_072
        assert result["microstep_buckets"] == [512, 1024, 2048, 4096]
        assert result["termination"] == "COMPLETE"


def test_bucket_shapes_exact_and_certified():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_bucketed_documents(), torch, device, tmp,
                      "bucket-shapes", campaign_tokens=131_072)
        expected_rows = {512: 64, 1024: 32, 2048: 16, 4096: 8}
        for shape in result["microstep_shapes"]:
            bucket = shape["requested_bucket"]
            assert shape["actual_row_widths"] == [bucket]
            assert shape["sequences_global"] == expected_rows[bucket]
            assert shape["real_tokens_global"] == 32768
            physical = shape["physical"]
            assert physical["sequences_per_replica"] == {512: 8, 1024: 4,
                                                         2048: 2, 4096: 1}[bucket]
            assert "partial_tail" not in physical


def test_partial_tail_never_overshoots():
    topo = frozen_topology()
    plans = partial_microstep_plan(remaining_tokens=5000, topo=topo,
                                   microstep_ordinal=3)
    assert plans == [5000]
    plans = partial_microstep_plan(
        remaining_tokens=2 * topo["global_tokens_per_microstep"] + 17,
        topo=topo, microstep_ordinal=0)
    assert plans == [topo["global_tokens_per_microstep"]] * 2 + [17]
    assert sum(plans) == 2 * topo["global_tokens_per_microstep"] + 17
    try:
        partial_microstep_plan(remaining_tokens=0, topo=topo,
                               microstep_ordinal=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-length plan was not rejected")


def test_single_microstep_update_accounts_exactly():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "exact-ledger",
                      campaign_tokens=4000)
        assert sum(result["tokens_by_source"].values()) == 4000
        assert result["microstep_shapes"][0]["actual_row_widths"] == [512]
        assert all(v > 0 for v in result["tokens_by_source"].values())


# -- checkpoints and durable milestones -----------------------------------------

def test_milestone_checkpoints_published_and_receipted():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "milestones",
                      campaign_tokens=4000, milestones=(1500, 3000),
                      recovery_tokens=10 ** 12)
        crossed = [m["threshold_tokens"] for m in result["milestones_crossed"]]
        assert crossed == [1500, 3000]
        for entry in result["milestones_crossed"]:
            assert entry["schema"] == "anra-v5-milestone-receipt/v1"
            assert len(entry["checkpoint_sha256"]) == 64
            assert entry["actual_cumulative_tokens"] >= entry["threshold_tokens"]
            for key in ("cymek_sha", "model_spec_sha256", "tokenizer_sha256",
                        "data_manifest_sha256", "pack_manifest_sha256",
                        "topology_sha256", "schedule_spec_sha256"):
                assert key in entry, f"milestone lacks {key}"
        assert result["recovery_checkpoint_count"] == 0
        store = CheckpointStore(Path(tmp) / "milestones", "milestones")
        for entry in result["milestones_crossed"]:
            assert verify_milestone_receipt(entry, store) is True


def test_milestone_protection_across_sessions():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _bucketed_documents()
        budget = 131_072 + 5000
        first = _run(docs, torch, device, tmp, "protect",
                     campaign_tokens=budget, max_updates=1,
                     milestones=(100_000,), recovery_tokens=10 ** 12)
        assert [m["threshold_tokens"] for m in first["milestones_crossed"]] == [100_000]
        milestone_sha = first["milestones_crossed"][0]["checkpoint_sha256"]
        store = CheckpointStore(Path(tmp) / "protect", "protect")
        assert milestone_sha in store.protected_shas()
        second = _run(docs, torch, device, tmp, "protect",
                      campaign_tokens=budget, milestones=(100_000,),
                      recovery_tokens=131_072)
        assert second["termination"] == "COMPLETE"
        objects = {p.name for p in (Path(tmp) / "protect" / "protect"
                                    / "objects").iterdir() if p.is_dir()}
        assert milestone_sha in objects, "milestone object was pruned across sessions"
        assert verify_milestone_receipt(first["milestones_crossed"][0], store) is True
        restored, _ = store.restore(milestone_sha)
        assert restored.cumulative_tokens == 131_072


def test_dangling_milestone_detected():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "dangle",
                      campaign_tokens=4000, milestones=(1500,),
                      recovery_tokens=10 ** 12)
        entry = dict(result["milestones_crossed"][0])
        store = CheckpointStore(Path(tmp) / "dangle", "dangle")
        bogus = dict(entry, checkpoint_sha256="00" * 32)
        try:
            verify_milestone_receipt(bogus, store)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("dangling milestone was verified")
        tampered = dict(entry, actual_cumulative_tokens=entry["actual_cumulative_tokens"] + 1)
        try:
            verify_milestone_receipt(tampered, store)
        except ValueError as exc:
            assert "disagrees" in str(exc)
        else:
            raise AssertionError("tampered milestone was verified")


def test_rotation_keeps_milestones_and_head():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _bucketed_documents()
        budget = 131_072 + 5000
        first = _run(docs, torch, device, tmp, "rotate",
                     campaign_tokens=budget, max_updates=1,
                     milestones=(), recovery_tokens=10 ** 12)
        head_one = first["checkpoint_head"]
        assert head_one is not None
        second = _run(docs, torch, device, tmp, "rotate",
                      campaign_tokens=budget,
                      milestones=(), recovery_tokens=10 ** 12)
        assert second["termination"] == "COMPLETE"
        assert second["cumulative_tokens"] == budget
        objects = {p.name for p in (Path(tmp) / "rotate" / "rotate"
                                    / "objects").iterdir() if p.is_dir()}
        assert objects == {second["checkpoint_head"]}
        assert head_one not in objects


def test_recovery_cadence_publishes():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "recovery",
                      campaign_tokens=4000, milestones=(),
                      recovery_tokens=4000)
        assert result["recovery_checkpoint_count"] == 1
        assert result["recovery_tokens"] == 4000
        assert result["checkpoint_head"] is not None


# -- schedule -----------------------------------------------------------------

def test_lr_matches_canonical_schedule():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "lr-fresh",
                      campaign_tokens=4000)
        assert result["last_update_receipt"]["learning_rate"] == lr_at(
            cumulative_tokens=0)


def test_lr_no_rewarm_after_resume():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _bucketed_documents()
        budget = 131_072 + 5000
        _run(docs, torch, device, tmp, "lr-resume", campaign_tokens=budget,
             max_updates=1)
        resumed = _run(docs, torch, device, tmp, "lr-resume",
                       campaign_tokens=budget)
        assert resumed["last_update_receipt"]["learning_rate"] == lr_at(
            cumulative_tokens=131_072)


# -- precision ------------------------------------------------------------------

def test_precision_receipt_matches_device():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "precision",
                      campaign_tokens=4000)
        runtime = getattr(device, "type", device)
        expected = precision_receipt(runtime=runtime, torch_module=torch)
        assert result["precision"] == expected
        assert result["precision"]["status"] == "CERTIFIED_LOCAL"


def test_tpu_runtime_fails_closed():
    with _seams()() as (torch, _device), tempfile.TemporaryDirectory() as tmp:
        try:
            _run(_documents(), torch, SimpleNamespace(type="xla"), tmp,
                 "tpu-fail", campaign_tokens=4000)
        except ValueError as exc:
            assert "TPU_EVIDENCE_REQUIRED" in str(exc)
        else:
            raise AssertionError("uncertified runtime did not fail closed")


def test_xla_execution_fails_closed_without_hardware():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        try:
            _run(_documents(), torch, device, tmp, "xla-fail",
                 campaign_tokens=4000, execution="xla")
        except ValueError as exc:
            assert "TPU_EVIDENCE_REQUIRED" in str(exc)
        else:
            raise AssertionError("XLA execution did not fail closed")
        try:
            _run(_documents(), torch, device, tmp, "xla-bad",
                 campaign_tokens=4000, execution="tpu")
        except ValueError as exc:
            assert "execution" in str(exc).lower()
        else:
            raise AssertionError("unknown execution mode was accepted")


# -- contamination, provenance, mode -------------------------------------------------------

def test_production_requires_contamination_commitment():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        try:
            run_campaign(documents=_documents(), tokenizer=_tokenizer(),
                         model_spec=MINI_SPEC, run_id="prod-nocommit",
                         seed=20260906, campaign_tokens=4000,
                         store_root=str(Path(tmp) / "prod-nocommit"),
                         device=device, torch_module=torch, xb=object(),
                         cymek_sha=TEST_SHA)
        except ValueError as exc:
            assert "contamination" in str(exc).lower()
        else:
            raise AssertionError("production without commitment did not fail")


def test_production_requires_mixture_and_freeze():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        base = {"documents": _with_raw(_documents()), "tokenizer": _tokenizer(),
                "model_spec": MINI_SPEC, "run_id": "prod-gates",
                "seed": 20260906, "campaign_tokens": 4000,
                "store_root": str(Path(tmp) / "prod-gates"), "device": device,
                "torch_module": torch, "xb": object(), "cymek_sha": TEST_SHA,
                "development_mode": False,
                "contamination_benchmarks": dict(BENCHMARKS)}
        try:
            run_campaign(**dict(base, mixture_fractions=None,
                                tokenizer_freeze_sha256=FREEZE_SHA))
        except ValueError as exc:
            assert "mixture" in str(exc).lower()
        else:
            raise AssertionError("production without mixture did not fail")
        try:
            run_campaign(**dict(base, mixture_fractions=dict(SINGLE_MIXTURE),
                                tokenizer_freeze_sha256=None))
        except ValueError as exc:
            assert "tokenizer" in str(exc).lower()
        else:
            raise AssertionError("production without freeze identity did not fail")


def test_raw_source_required_in_production():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        natural = _natural_documents()
        try:
            _run(natural, torch, device, tmp, "raw-missing",
                 campaign_tokens=4000, development_mode=False,
                 contamination_benchmarks=dict(BENCHMARKS),
                 mixture_fractions=dict(SINGLE_MIXTURE),
                 tokenizer_freeze_sha256=FREEZE_SHA, cymek_sha=TEST_SHA)
        except ValueError as exc:
            assert "raw_source_sha256" in str(exc)
        else:
            raise AssertionError("missing raw provenance did not fail closed")
        result = _prod(natural, torch, device, tmp, "raw-present")
        assert result["cumulative_tokens"] == 4000


def test_contamination_content_binding():
    first = prepare_data(documents=_documents(), tokenizer=_tokenizer(),
                         run_id="cont", seed=11,
                         contamination_benchmarks={"bench": "alpha beta gamma"})
    second = prepare_data(documents=_documents(), tokenizer=_tokenizer(),
                          run_id="cont", seed=11,
                          contamination_benchmarks={"bench": "alpha beta DELTA"})
    assert (first["manifest"].contamination_scan_sha256
            != second["manifest"].contamination_scan_sha256)


def test_development_mode_labels_and_relaxes():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "dev-mode",
                      campaign_tokens=4000)
        assert result["mode"] == "DEVELOPMENT"
        prod = _prod(_natural_documents(), torch, device, tmp, "prod-commit")
        assert prod["mode"] == "PRODUCTION"
        assert prod["identity_bundle"]["tokenizer_freeze_sha256"] == FREEZE_SHA


# -- mixture end-to-end ---------------------------------------------------------------

def test_frozen_mixture_end_to_end():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _with_raw(_mixture_documents())
        result = _run(docs, torch, device, tmp, "mixture",
                      campaign_tokens=100_000, development_mode=False,
                      contamination_benchmarks=dict(BENCHMARKS),
                      mixture_fractions=dict(PROD_MIXTURE),
                      tokenizer_freeze_sha256=FREEZE_SHA,
                      cymek_sha=TEST_SHA)
        assert result["mode"] == "PRODUCTION"
        assert result["cumulative_tokens"] == 100_000
        assert sum(result["mixture_consumed"].values()) == 100_000
        assert set(result["mixture_consumed"]) == set(PROD_MIXTURE)
        assert result["mixture_allocation"] == {
            "code_math_formal": 20_000, "natural": 65_000,
            "verified_cognition": 15_000}
        assert len(result["mixture_plan_sha256"]) == 64


def test_mixture_shortfall_fails_closed():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        full = _with_raw(_mixture_documents())
        code_512 = [d for d in full if d["doc_id"].startswith("mx512-code")]
        starved = [d for d in full if not d["doc_id"].startswith("mx512-code")]
        starved.extend(code_512[:2])
        try:
            _run(starved, torch, device, tmp, "shortfall", campaign_tokens=100_000,
                 development_mode=False,
                 contamination_benchmarks=dict(BENCHMARKS),
                 mixture_fractions=dict(PROD_MIXTURE),
                 tokenizer_freeze_sha256=FREEZE_SHA, cymek_sha=TEST_SHA)
        except ValueError as exc:
            assert "DATA_NOT_READY" in str(exc), str(exc)[:300]
        else:
            raise AssertionError("mixture shortfall did not fail closed")


def test_cognition_mixture_resume_equality():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        tok = _tokenizer()
        # Cells match the offline-planned demand for budget 135072 exactly:
        # (512,identity_copy)=32768, (1024,relational)=32768,
        # (2048,query_binding)=32768, (4096,semantic)=32768,
        # (2048,interference)=4000.
        cells = [(512, "identity_copy", 64, 510),
                 (1024, "relational_composition", 32, 1022),
                 (2048, "query_binding", 16, 2046),
                 (4096, "semantic_state", 8, 4094),
                 (2048, "interference_retrieval", 3, 2000)]
        docs = []
        cognition_map = {}
        for bucket, sub, count, content_tokens in cells:
            for i in range(count):
                doc_id = f"cg{bucket}-{sub}-{i:04d}"
                docs.append({"doc_id": doc_id,
                             "text": _sized_text(tok, content_tokens,
                                                 f"{doc_id}"),
                             "source_id": doc_id,
                             "family": "verified_cognition"})
                cognition_map[doc_id] = sub
        budget = 131_072 + 4000
        mixture = {"verified_cognition": 1.0}
        base = {"campaign_tokens": budget, "development_mode": False,
                "contamination_benchmarks": dict(BENCHMARKS),
                "mixture_fractions": dict(mixture),
                "cognition_map": cognition_map,
                "tokenizer_freeze_sha256": FREEZE_SHA, "cymek_sha": TEST_SHA}
        part = _run(_with_raw(docs), torch, device, tmp, "cog-run",
                    max_updates=1, **base)
        assert part["termination"] == "MANUAL_BOUNDARY"
        assert part["cumulative_tokens"] == 131_072
        continued = _run(_with_raw(docs), torch, device, tmp, "cog-run",
                         **base)
        assert continued["resumed"] is True
        assert continued["cumulative_tokens"] == budget
        assert continued["state_complete"] is True
        assert set(continued["sub_consumed"]) == {
            "identity_copy", "relational_composition", "query_binding",
            "semantic_state", "interference_retrieval"}
        assert sum(continued["sub_consumed"].values()) == budget
        assert set(continued["mixture_consumed"]) == {"verified_cognition"}
        whole = _run(_with_raw(docs), torch, device, tmp, "cog-whole",
                     **base)
        assert whole["tokens_by_source"] == continued["tokens_by_source"]
        continued_store = CheckpointStore(Path(tmp) / "cog-run", "cog-run")
        whole_store = CheckpointStore(Path(tmp) / "cog-whole", "cog-whole")
        _, continued_payloads = continued_store.restore()
        _, whole_payloads = whole_store.restore()
        for component in ("model.bin", "optimizer.bin", "rng.bin",
                          "cursor.json", "ledger.json"):
            assert hashlib.sha256(continued_payloads[component]).hexdigest() == \
                hashlib.sha256(whole_payloads[component]).hexdigest(), \
                f"cognition resume diverges at {component}"


def test_epoch_replay_when_permitted():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        tiny = _documents(n_arith=60, n_notes=0)
        result = _run(tiny, torch, device, tmp, "replay-ok",
                      campaign_tokens=3000, allow_replay=True,
                      milestones=(), recovery_tokens=10 ** 12)
        assert result["cumulative_tokens"] == 3000
        assert result["replay_count"] >= 1
        assert result["replay_events"] != []
        assert sum(result["tokens_by_source"].values()) == 3000
        try:
            _run(tiny, torch, device, tmp, "replay-no",
                 campaign_tokens=3000, allow_replay=False,
                 milestones=(), recovery_tokens=10 ** 12)
        except ValueError as exc:
            assert "DATA_NOT_READY" in str(exc)
        else:
            raise AssertionError("exhaustion without replay did not fail closed")


# -- certificate and reception ------------------------------------------------------

_CERTIFICATE_KEYS = ("schema", "run_id", "seed", "campaign_tokens",
                     "updates_executed", "cumulative_tokens",
                     "state_complete", "termination", "cymek_sha",
                     "topology_sha256", "lanes_sha256", "precision",
                     "sampler_order_sha256", "tokens_by_source",
                     "checkpoint_head", "resume_equal", "identity_bundle",
                     "microstep_shapes", "mixture_allocation")

_BANNED_SYMBOLS = ("SEQUENCES_PER_UPDATE", "TOKENS_PER_UPDATE",
                   "_walk_windows", "data_window",
                   "checkpoint_every=remaining", "campaign_layout")


def test_campaign_certificate_completeness():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "certificate",
                      campaign_tokens=4000)
        for key in _CERTIFICATE_KEYS:
            assert key in result, f"certificate missing {key}"
        assert result["cymek_sha"] == TEST_SHA
        assert result["topology_sha256"] == topology_sha256(frozen_topology())
        assert len(result["sampler_order_sha256"]) == 64
        assert len(result["lanes_sha256"]) == 64
        blob = json.dumps(result, sort_keys=True, default=str)
        for banned in _BANNED_SYMBOLS[:5]:
            assert banned not in blob


def test_banned_symbols_absent_from_entry_source():
    source = (ROOT / "v5_training" / "production_entry.py").read_text(
        encoding="utf-8")
    for banned in _BANNED_SYMBOLS:
        assert banned not in source, f"stale symbol survives: {banned}"
    for required in ("run_500m_session", "microstep_buckets",
                     "partial_microstep_plan", "checkpoint_every=None",
                     "take_cell_window", "BucketCursorState"):
        assert required in source, f"required construct missing: {required}"


# -- helpers --------------------------------------------------------------------------

def test_frozen_topology_multiplies():
    topo = frozen_topology()
    assert topo["replicas"] == 8
    assert topo["gradient_accumulation_microsteps"] == 4
    assert topo["tokens_per_replica_microstep"] == 4096
    assert topo["global_tokens_per_microstep"] == 32768
    assert topo["global_tokens_per_update"] == 131072
    assert len(topo["supercycle"]) == 20
    assert topo["sequences_per_replica_by_bucket"] == {512: 8, 1024: 4,
                                                       2048: 2, 4096: 1}


def test_frozen_mixture_matches_contract():
    assert frozen_mixture_fractions() == PROD_MIXTURE


def test_microstep_buckets_follow_supercycle():
    topo = frozen_topology()
    buckets = microstep_buckets(cumulative_tokens=0,
                                microstep_counts=[1, 1, 1, 1, 1], topo=topo)
    assert buckets == topo["supercycle"][:5]
    resumed = microstep_buckets(
        cumulative_tokens=topo["global_tokens_per_microstep"],
        microstep_counts=[1], topo=topo)
    assert resumed == [topo["supercycle"][1]]


def test_crossed_milestones_pure():
    assert crossed_milestones(0, 60_000_000) == [50_000_000]
    assert crossed_milestones(50_000_000, 50_000_000) == []
    try:
        crossed_milestones(10, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("backward milestones were not rejected")


def test_resolve_cymek_sha_rejects_and_resolves():
    assert resolve_cymek_sha(TEST_SHA) == TEST_SHA
    for bad in ("xyz", "AB" * 20, "ab" * 19):
        try:
            resolve_cymek_sha(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"bad SHA accepted: {bad}")
    if (ROOT / ".git").exists():
        assert len(resolve_cymek_sha(None)) == 40


def test_prepare_data_deterministic():
    first = prepare_data(documents=_documents(), tokenizer=_tokenizer(),
                         run_id="prep", seed=11)
    second = prepare_data(documents=_documents(), tokenizer=_tokenizer(),
                          run_id="prep", seed=11)
    assert first["pack_manifest_sha256"] == second["pack_manifest_sha256"]
    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["sampler_order"] == second["sampler_order"]
    assert first["packed_doc_ids"] and first["packed_sources"]


def test_exact_head_test_receipt():
    from v5_training.test_receipt import verify_receipt
    receipt = verify_receipt(
        ROOT / "artifacts/v5/cymek_500m_closure_test_receipt.json",
        repo_root=ROOT,
        receipt_relpath="artifacts/v5/cymek_500m_closure_test_receipt.json")
    assert receipt["totals"]["failed"] == 0
    assert receipt["totals"]["passed"] > 0


def test_build_milestone_receipt_self_describing():
    receipt = build_milestone_receipt(
        run_id="r", threshold_tokens=100, actual_cumulative_tokens=150,
        global_update=2, checkpoint_sha256="ab" * 32,
        identity_bundle={"cymek_sha": TEST_SHA})
    assert receipt["schema"] == "anra-v5-milestone-receipt/v1"
    assert receipt["cymek_sha"] == TEST_SHA
    assert receipt["threshold_tokens"] == 100


# -- EOS / packing contract --------------------------------------------------------------

def test_eos_packing_contract():
    from v5_data.bucket_cursor import build_bucket_lanes, cell_key, take_cell_window
    from v5_data.pack import pack_documents
    from v5_training.production_entry import _predict_supervised

    one_token = pack_documents([("one", [41], "s")], bos=2, eos=3, pad=0,
                               sequences_per_shard=8)[0]
    assert len(one_token) == 1
    sequence = one_token[0].sequences[0]
    assert list(sequence.tokens)[:3] == [2, 41, 3]
    lanes, _ = build_bucket_lanes(one_token, run_seed=1, pattern=[512])
    window = take_cell_window(one_token, lanes[cell_key(512, "", "")], 0, 0,
                              real_tokens=3, pad=0, bucket=512)
    assert window.real_tokens == 3
    assert _predict_supervised(window) == 2
    exact = pack_documents([("exact", [7] * 510, "s")], bos=2, eos=3, pad=0,
                           sequences_per_shard=8)
    assert exact[1]["full_sequences"] == 1
    assert exact[1]["padded_sequences"] == 0
    ragged = pack_documents([("ragged", [7] * 100, "s")], bos=2, eos=3, pad=0,
                            sequences_per_shard=8)
    assert ragged[1]["padded_sequences"] == 1
    tail = ragged[0][0].sequences[0]
    assert tail.tokens[-1] == 0 and tail.segment_ids[-1] == -1


# -- compressed end-to-end ------------------------------------------------------------------

def test_compressed_e2e_fresh_recovery_milestone_stop_resume_partial_complete():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _bucketed_documents()
        budget = 131_072 + 6000
        session_dir = str(Path(tmp) / "e2e")
        first = run_500m_session(
            documents=docs, tokenizer=_tokenizer(), model_spec=MINI_SPEC,
            run_id="e2e", seed=20260906, campaign_tokens=budget,
            session_dir=session_dir, device=device, torch_module=torch,
            xb=object(), development_mode=True, cymek_sha=TEST_SHA,
            milestones=(100_000,), recovery_tokens=131_072,
            max_session_minutes=0.000001, margin_seconds=0.0)
        assert first["session_receipt"]["status"] == "RESUMABLE"
        assert first["result"]["termination"] == "TIMEBOX"
        assert first["result"]["cumulative_tokens"] == 131_072
        assert first["result"]["recovery_checkpoint_count"] == 1
        assert [m["threshold_tokens"] for m in first["milestones_crossed"]] == [100_000]
        milestone_file = (Path(session_dir) / "milestones"
                          / "milestone_100000.json")
        assert milestone_file.is_file()
        store = CheckpointStore(Path(session_dir) / "campaign_store", "e2e")
        assert verify_milestone_receipt(
            first["milestones_crossed"][0], store) is True
        second = run_500m_session(
            documents=docs, tokenizer=_tokenizer(), model_spec=MINI_SPEC,
            run_id="e2e", seed=20260906, campaign_tokens=budget,
            session_dir=session_dir, device=device, torch_module=torch,
            xb=object(), development_mode=True, cymek_sha=TEST_SHA,
            milestones=(100_000,), recovery_tokens=131_072)
        assert second["session_receipt"]["status"] == "COMPLETE"
        assert second["result"]["cumulative_tokens"] == budget
        assert second["result"]["state_complete"] is True
        assert second["result"]["resume_equal"] is True
        assert (Path(session_dir) / "SESSION_RECEIPT.json").is_file()
        assert (Path(session_dir) / "HEARTBEAT.json").is_file()


def test_multi_session_soak_state_machine():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _bucketed_documents()
        budget = 2 * 131_072 + 2000
        session_dir = str(Path(tmp) / "soak")
        common = {"documents": docs, "tokenizer": _tokenizer(),
                  "model_spec": MINI_SPEC, "run_id": "soak", "seed": 20260906,
                  "campaign_tokens": budget, "session_dir": session_dir,
                  "device": device, "torch_module": torch, "xb": object(),
                  "development_mode": True, "cymek_sha": TEST_SHA,
                  "milestones": (131_072, 200_000),
                  "recovery_tokens": 131_072}
        boxed = dict(common, max_session_minutes=0.000001, margin_seconds=0.0)
        first = run_500m_session(**boxed)
        assert first["session_receipt"]["status"] == "RESUMABLE"
        assert first["result"]["termination"] == "TIMEBOX"
        assert first["result"]["cumulative_tokens"] == 131_072
        assert [m["threshold_tokens"] for m in first["milestones_crossed"]] == [131_072]
        second = run_500m_session(**boxed)
        assert second["result"]["termination"] == "TIMEBOX"
        assert second["result"]["cumulative_tokens"] == 262_144
        assert [m["threshold_tokens"] for m in second["milestones_crossed"]] == [200_000]
        assert second["result"]["last_update_receipt"]["learning_rate"] == lr_at(
            cumulative_tokens=131_072)
        third = run_500m_session(**common)
        result = third["result"]
        assert third["session_receipt"]["status"] == "COMPLETE"
        assert result["cumulative_tokens"] == budget
        assert result["state_complete"] is True
        assert result["resume_equal"] is True
        assert sum(result["tokens_by_source"].values()) == budget
        assert len(first["result"]["losses"] + second["result"]["losses"]
                   + result["losses"]) == 3
        assert result["epoch"] == 0 and result["replay_count"] == 0
        assert result["replay_events"] == []
        assert result["last_update_receipt"]["learning_rate"] == lr_at(
            cumulative_tokens=262_144)
        store = CheckpointStore(Path(session_dir) / "campaign_store", "soak")
        restored, _ = store.restore()
        assert restored.cumulative_tokens == budget
        objects = {p.name for p in (Path(session_dir) / "campaign_store"
                                    / "soak" / "objects").iterdir() if p.is_dir()}
        shas = [m["checkpoint_sha256"]
                for m in first["milestones_crossed"] + second["milestones_crossed"]]
        for sha in shas:
            assert sha in objects, "milestone object lost across sessions"
            store.restore(sha)
        assert verify_milestone_receipt(first["milestones_crossed"][0], store) is True
        assert verify_milestone_receipt(second["milestones_crossed"][0], store) is True
        files = sorted((Path(session_dir) / "milestones").iterdir())
        assert [p.name for p in files] == ["milestone_131072.json",
                                           "milestone_200000.json"]


# -- session layer -----------------------------------------------------------------------------

def test_session_completes_tiny_campaign():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        session_dir = str(Path(tmp) / "session-tiny")
        out = run_500m_session(
            documents=_documents(), tokenizer=_tokenizer(),
            model_spec=MINI_SPEC, run_id="tiny", seed=20260906,
            campaign_tokens=4000, session_dir=session_dir, device=device,
            torch_module=torch, xb=object(), development_mode=True,
            cymek_sha=TEST_SHA, milestones=(1500,),
            recovery_tokens=10 ** 12)
        assert out["schema"] == "anra-v5-500m-session/v1"
        assert out["session_receipt"]["status"] == "COMPLETE"
        assert out["result"]["cumulative_tokens"] == 4000
        assert (Path(session_dir) / "milestones"
                / "milestone_1500.json").is_file()


def test_session_timebox_resumable_then_completes():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _bucketed_documents()
        budget = 131_072 + 6000
        session_dir = str(Path(tmp) / "session-timebox")
        first = run_500m_session(
            documents=docs, tokenizer=_tokenizer(), model_spec=MINI_SPEC,
            run_id="timebox", seed=20260906, campaign_tokens=budget,
            session_dir=session_dir, device=device, torch_module=torch,
            xb=object(), development_mode=True, cymek_sha=TEST_SHA,
            milestones=(), recovery_tokens=10 ** 12,
            max_session_minutes=0.000001, margin_seconds=0.0)
        assert first["session_receipt"]["status"] == "RESUMABLE"
        assert first["result"]["termination"] == "TIMEBOX"
        assert first["result"]["cumulative_tokens"] == 131_072
        second = run_500m_session(
            documents=docs, tokenizer=_tokenizer(), model_spec=MINI_SPEC,
            run_id="timebox", seed=20260906, campaign_tokens=budget,
            session_dir=session_dir, device=device, torch_module=torch,
            xb=object(), development_mode=True, cymek_sha=TEST_SHA,
            milestones=(), recovery_tokens=10 ** 12)
        assert second["session_receipt"]["status"] == "COMPLETE"
        assert second["result"]["cumulative_tokens"] == budget


# -- model target ----------------------------------------------------------------------------------

def test_v5a_exact_parameter_count():
    import torch
    from v5_contracts.model_spec import V5A_250M
    from v5_model.core import initialize
    model = initialize(V5A_250M, 5, torch_module=torch)
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total == 250_216_960
    assert total == V5A_250M.parameter_receipt().total
    del model


def test_bf16_trajectory_diagnostic():
    import os
    import tempfile
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "bf16-probe",
                      campaign_tokens=4000)
        assert all(v == v and abs(v) != float("inf") for v in result["losses"])
        mine = (getattr(device, "type", device), result["losses"][0])
        print(f"bf16-probe device={mine[0]} loss={mine[1]:.6f}")
        other = None
        if mine[0] == "cuda":
            other = ("cpu", torch.device("cpu"))
        elif torch.cuda.is_available():
            other = ("cuda", torch.device("cuda"))
        if other is None:
            print("SKIP cross-device leg (no second device)")
            return
        with tempfile.TemporaryDirectory() as tmp2:
            sibling = _run(_documents(), torch, other[1], tmp2,
                           "bf16-sibling", campaign_tokens=4000)
        delta = abs(sibling["losses"][0] - mine[1])
        print(f"bf16-probe cross-device delta={delta:.6f}")
        assert delta < 1e-3, "precision modes diverged beyond diagnostic tolerance"


# -- preserved coverage ------------------------------------------------------------------------------

def test_durable_mirror_session_recovery():
    import shutil
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        mirror = str(Path(tmp) / "mirror")
        first = _run(_documents(), torch, device, tmp, "mirrored",
                     campaign_tokens=4000, mirror_root=mirror,
                     milestones=(), recovery_tokens=10 ** 12)
        assert first["termination"] == "COMPLETE"
        assert first["mirrored_recovery"] is False
        assert first["mirrored_generations"] == 1
        assert first["execution_mode"] == "local"
        assert first["xla_status"]["status"] == "LOCAL_EMULATION"
        shutil.rmtree(Path(tmp) / "mirrored")
        second = _run(_documents(), torch, device, tmp, "mirrored",
                      campaign_tokens=4000, mirror_root=mirror,
                      milestones=(), recovery_tokens=10 ** 12)
        assert second["mirrored_recovery"] is True
        assert second["already_complete"] is True


def test_already_complete_short_circuits():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _documents()
        _run(docs, torch, device, tmp, "done", campaign_tokens=4000)
        again = _run(docs, torch, device, tmp, "done", campaign_tokens=4000)
        assert again["already_complete"] is True


def test_freeze_production_identity():
    receipt = freeze_production_identity(repo_root=str(ROOT))
    assert receipt["status"] == "FROZEN"
    assert receipt["vocabulary_size"] == 24_576
    assert receipt["special_token_ids"] == {"pad": 0, "unk": 1, "bos": 2,
                                            "eos": 3}
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "identity.json"
        freeze_production_identity(repo_root=str(ROOT), out_path=str(out))
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["artifact_sha256"] == receipt["artifact_sha256"]


def test_materialize_first_party_supply_accounting():
    tok = _tokenizer()
    got = mfp.materialize_first_party(repo_root=str(ROOT), tokenizer=tok)
    assert got["documents_materialized"] > 0
    assert got["DATA_NOT_READY"] is True, (
        "first-party supply is honestly far below 500M unique demand")
    assert got["verdict"].startswith("DATA_NOT_READY")
    assert got["shortfall_tokens_by_source"]["verified_cognition"] > 0


def _artifact_sha():
    return hashlib.sha256(
        (ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
         ).read_bytes()).hexdigest()


def _trainer_sha():
    r = json.loads((ROOT / "artifacts/e1/local_tournament/result.json")
                   .read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(r["trainer"], sort_keys=True)
                          .encode("utf-8")).hexdigest()


def _corpus_sha():
    r = json.loads((ROOT / "artifacts/e1/local_tournament/result.json")
                   .read_text(encoding="utf-8"))
    return r["corpus_manifest_sha256"]


_TESTS = [
    test_exact_completion_single_partial_update,
    test_fresh_determinism_same_inputs_same_receipt,
    test_pack_smaller_than_budget_fails_closed,
    test_mid_campaign_resume_matches_uninterrupted,
    test_resume_changed_documents_fails,
    test_resume_changed_seed_fails,
    test_resume_changed_budget_fails,
    test_cross_store_and_cross_lineage_rejected,
    test_full_update_uses_four_microsteps_one_step,
    test_bucket_shapes_exact_and_certified,
    test_partial_tail_never_overshoots,
    test_single_microstep_update_accounts_exactly,
    test_milestone_checkpoints_published_and_receipted,
    test_milestone_protection_across_sessions,
    test_dangling_milestone_detected,
    test_rotation_keeps_milestones_and_head,
    test_recovery_cadence_publishes,
    test_lr_matches_canonical_schedule,
    test_lr_no_rewarm_after_resume,
    test_precision_receipt_matches_device,
    test_tpu_runtime_fails_closed,
    test_xla_execution_fails_closed_without_hardware,
    test_durable_mirror_session_recovery,
    test_production_requires_contamination_commitment,
    test_production_requires_mixture_and_freeze,
    test_raw_source_required_in_production,
    test_contamination_content_binding,
    test_development_mode_labels_and_relaxes,
    test_frozen_mixture_end_to_end,
    test_mixture_shortfall_fails_closed,
    test_cognition_mixture_resume_equality,
    test_epoch_replay_when_permitted,
    test_campaign_certificate_completeness,
    test_banned_symbols_absent_from_entry_source,
    test_frozen_topology_multiplies,
    test_frozen_mixture_matches_contract,
    test_microstep_buckets_follow_supercycle,
    test_crossed_milestones_pure,
    test_resolve_cymek_sha_rejects_and_resolves,
    test_prepare_data_deterministic,
    test_exact_head_test_receipt,
    test_build_milestone_receipt_self_describing,
    test_eos_packing_contract,
    test_compressed_e2e_fresh_recovery_milestone_stop_resume_partial_complete,
    test_multi_session_soak_state_machine,
    test_session_completes_tiny_campaign,
    test_session_timebox_resumable_then_completes,
    test_v5a_exact_parameter_count,
    test_bf16_trajectory_diagnostic,
    test_already_complete_short_circuits,
    test_freeze_production_identity,
    test_materialize_first_party_supply_accounting,
]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
