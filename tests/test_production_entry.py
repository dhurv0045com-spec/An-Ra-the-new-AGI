"""500M production-entry contract suite (CPU default, CUDA via ANRA_TEST_DEVICE=cuda).

Run from the cymek-500m worktree root:
    <repo>/.venv/Scripts/python.exe -m pytest tests/test_production_entry.py -x -q
or:  python tests/test_production_entry.py
Device ops run on the seam device (CPU default); everything else is the
REAL production chain: frozen topology -> accumulation -> trainer ->
checkpoint transactions.
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
from v5_training.production_entry import (  # noqa: E402
    ENTRY_SCHEMA,
    MILESTONE_TOKENS,
    crossed_milestones,
    frozen_topology,
    microstep_buckets,
    partial_microstep_plan,
    prepare_data,
    resolve_cymek_sha,
    run_500m_session,
    run_campaign,
    topology_sha256,
)
from v5_training.production_backend import precision_receipt  # noqa: E402
from v5_training.schedule import lr_at  # noqa: E402

TEST_SHA = "ab" * 20
BENCHMARKS = {"synthetic-bench": "zzzqqqwww xxxjjjvvv"}


def _seams():
    """Device seam for this run: CPU by default, CUDA when ANRA_TEST_DEVICE=cuda.

    The production chain is device-injected, so the same suite validates
    both paths unmodified; only the seam and the expected precision
    contract change.
    """

    import os
    if os.environ.get("ANRA_TEST_DEVICE", "cpu") == "cuda":
        from contextlib import contextmanager

        @contextmanager
        def cuda_seams():
            import torch
            yield torch, torch.device("cuda")

        return cuda_seams
    return cpu_seams

_token_cache: dict = {}


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


def _run(documents, torch, device, tmp, name, **kw):
    params = {"documents": documents, "tokenizer": _tokenizer(),
              "model_spec": MINI_SPEC, "run_id": name, "seed": 20260906,
              "store_root": str(Path(tmp) / name), "device": device,
              "torch_module": torch, "xb": object(),
              "development_mode": True, "cymek_sha": TEST_SHA}
    params.update(kw)
    return run_campaign(**params)


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
            assert "pack" in str(exc).lower() or "window" in str(exc).lower()
        else:
            raise AssertionError("over-budget campaign did not fail closed")


# -- resume and identity ---------------------------------------------------

def test_mid_campaign_resume_matches_uninterrupted():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _documents(n_arith=11000, n_notes=500)
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
        from v5_training.checkpoint import CheckpointStore
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


# -- accumulation honesty ---------------------------------------------------

def test_full_update_uses_four_microsteps_one_step():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(n_arith=11000, n_notes=500), torch, device,
                      tmp, "full-update", campaign_tokens=131_072)
        assert result["updates_executed"] == 1
        assert result["cumulative_tokens"] == 131_072
        last = result["last_update_receipt"]
        assert last["microsteps"] == 4
        assert 0 < last["supervised_tokens"] < 131_072
        assert len(result["microstep_buckets"]) == 4
        assert result["termination"] == "COMPLETE"


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
        assert set(result["microstep_bucket_mix"]) == {512}
        assert sum(result["microstep_bucket_mix"].values()) > 0
        assert all(v > 0 for v in result["tokens_by_source"].values())


# -- checkpoints ------------------------------------------------------------

def test_milestone_checkpoints_published_and_receipted():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "milestones",
                      campaign_tokens=4000, milestones=(1500, 3000),
                      recovery_tokens=10 ** 12)
        crossed = [m["threshold_tokens"] for m in result["milestones_crossed"]]
        assert crossed == [1500, 3000]
        for entry in result["milestones_crossed"]:
            assert len(entry["checkpoint_sha256"]) == 64
            assert entry["actual_cumulative_tokens"] >= entry["threshold_tokens"]
        assert result["recovery_checkpoint_count"] == 0


def test_rotation_keeps_milestones_and_head():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _documents(n_arith=11000, n_notes=500)
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
                                          / "objects").iterdir()
                   if p.is_dir()}
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
        docs = _documents(n_arith=11000, n_notes=500)
        budget = 131_072 + 5000
        _run(docs, torch, device, tmp, "lr-resume", campaign_tokens=budget,
             max_updates=1)
        resumed = _run(docs, torch, device, tmp, "lr-resume",
                       campaign_tokens=budget)
        assert resumed["last_update_receipt"]["learning_rate"] == lr_at(
            cumulative_tokens=131_072)


# -- precision ------------------------------------------------------------------

def test_precision_receipt_cpu_certified():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "precision",
                      campaign_tokens=4000)
        expected = precision_receipt(runtime=device.type, torch_module=torch)
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


# -- contamination and mode -------------------------------------------------------

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


def test_development_mode_labels_and_relaxes():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "dev-mode",
                      campaign_tokens=4000)
        assert result["mode"] == "DEVELOPMENT"
        prod = run_campaign(
            documents=_documents(), tokenizer=_tokenizer(),
            model_spec=MINI_SPEC, run_id="prod-commit", seed=20260906,
            campaign_tokens=4000,
            store_root=str(Path(tmp) / "prod-commit"), device=device,
            torch_module=torch, xb=object(), cymek_sha=TEST_SHA,
            contamination_benchmarks=dict(BENCHMARKS))
        assert prod["mode"] == "PRODUCTION"


# -- certificate and reception ------------------------------------------------------

_CERTIFICATE_KEYS = ("schema", "run_id", "seed", "campaign_tokens",
                     "updates_executed", "cumulative_tokens",
                     "state_complete", "termination", "cymek_sha",
                     "topology_sha256", "layout_sha256", "precision",
                     "sampler_order_sha256", "tokens_by_source",
                     "checkpoint_head", "resume_equal")

_BANNED_SYMBOLS = ("SEQUENCES_PER_UPDATE", "TOKENS_PER_UPDATE",
                   "_walk_windows", "data_window",
                   "checkpoint_every=remaining")


def test_campaign_certificate_completeness():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        result = _run(_documents(), torch, device, tmp, "certificate",
                      campaign_tokens=4000)
        for key in _CERTIFICATE_KEYS:
            assert key in result, f"certificate missing {key}"
        assert result["cymek_sha"] == TEST_SHA
        assert result["topology_sha256"] == topology_sha256(frozen_topology())
        assert len(result["sampler_order_sha256"]) == 64
        blob = json.dumps(result, sort_keys=True, default=str)
        for banned in _BANNED_SYMBOLS:
            assert banned not in blob


def test_banned_symbols_absent_from_entry_source():
    source = (ROOT / "v5_training" / "production_entry.py").read_text(
        encoding="utf-8")
    for banned in _BANNED_SYMBOLS:
        assert banned not in source, f"stale symbol survives: {banned}"
    for required in ("run_500m_session", "microstep_buckets",
                     "partial_microstep_plan", "checkpoint_every=None"):
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


# -- compressed end-to-end ------------------------------------------------------------------

def test_compressed_e2e_fresh_recovery_milestone_stop_resume_partial_complete():
    with _seams()() as (torch, device), tempfile.TemporaryDirectory() as tmp:
        docs = _documents(n_arith=11000, n_notes=500)
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
        docs = _documents(n_arith=11000, n_notes=500)
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


# -- preserved coverage ------------------------------------------------------------------------------

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
    test_partial_tail_never_overshoots,
    test_single_microstep_update_accounts_exactly,
    test_milestone_checkpoints_published_and_receipted,
    test_rotation_keeps_milestones_and_head,
    test_recovery_cadence_publishes,
    test_lr_matches_canonical_schedule,
    test_lr_no_rewarm_after_resume,
    test_precision_receipt_cpu_certified,
    test_tpu_runtime_fails_closed,
    test_production_requires_contamination_commitment,
    test_development_mode_labels_and_relaxes,
    test_campaign_certificate_completeness,
    test_banned_symbols_absent_from_entry_source,
    test_frozen_topology_multiplies,
    test_microstep_buckets_follow_supercycle,
    test_crossed_milestones_pure,
    test_resolve_cymek_sha_rejects_and_resolves,
    test_prepare_data_deterministic,
    test_compressed_e2e_fresh_recovery_milestone_stop_resume_partial_complete,
    test_session_completes_tiny_campaign,
    test_session_timebox_resumable_then_completes,
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
