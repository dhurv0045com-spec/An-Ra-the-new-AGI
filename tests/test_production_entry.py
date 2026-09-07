"""Production-entry + tokenizer-freeze + materialization tests (CPU).

Run from the cymek worktree root:  python tests/test_production_entry.py
Device ops run on CPU; everything else is the REAL production chain.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anra_v5.miniature_run import MINI_SPEC  # noqa: E402
from tests.cpu_seam import cpu_seams  # noqa: E402
from v5_data import materialize_first_party as mfp  # noqa: E402
from v5_tokenizer.artifact import load_verified_tokenizer  # noqa: E402
from v5_tokenizer.freeze_production import freeze_production_identity  # noqa: E402
from v5_training.production_entry import prepare_data, run_campaign  # noqa: E402


class _Tok:
    """Wraps the verified backend + identity into the encode/identity
    surface production_entry expects."""

    def __init__(self, backend, identity):
        self.backend = backend
        self.identity = identity

    def encode(self, text):
        return list(self.backend.encode(text).ids)


def _documents():
    docs = []
    for i in range(3000):
        a = i % 500 + 10
        b = (i * 37) % 900 + 100
        docs.append({"doc_id": f"arith-{i:05d}",
                     "text": f"{a} + {b} = {a + b}",
                     "source_id": f"arith-{i:05d}",
                     "family": "verified_cognition",
                     "domain": "synthetic-cognition"})
    for i in range(1000):
        docs.append({"doc_id": f"notes-{i:04d}",
                     "text": f"training note {i}: the model learns addition "
                             f"and carries digits correctly batch {i} with "
                             f"multiple operations and edge cases",
                     "source_id": f"notes-{i:04d}", "family": "natural",
                     "domain": "notes"})
    return docs


def test_fresh_campaign_and_exact_resume() -> None:
    import tempfile

    with (cpu_seams() as (_torch, _device),
          tempfile.TemporaryDirectory() as tmp):
        _raw_tok, tok_identity = load_verified_tokenizer(
            ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz",
            expected_sha256=_artifact_sha(),
            vocabulary_size=24_576,
            trainer_config_sha256=_trainer_sha(),
            corpus_manifest_sha256=_corpus_sha())
        tok_backend = _Tok(_raw_tok, tok_identity)

        docs = _documents()
        fresh = run_campaign(
            documents=docs, tokenizer=tok_backend, model_spec=MINI_SPEC,
            run_id="entry-fresh", seed=20260906, updates=4,
            store_root=str(Path(tmp) / "fresh"), device=_device,
            torch_module=_torch, xb=object())
        assert fresh["updates_executed"] == 4
        assert fresh["state_complete"] is True
        assert fresh["resume_equal"] is True
        assert len(fresh["losses"]) == 4
        assert fresh["losses"][-1] < fresh["losses"][0]
        # exact-resume verification (inside run_campaign) already proves
        # that the checkpoint/reload cycle preserves the full state.
        # Mid-campaign resume with budget extension is covered by the
        # T1D mid-arm tests and the PRE50M reserved-final-update tests.


def test_already_complete_short_circuits() -> None:
    import tempfile

    with (cpu_seams() as (_torch, _device),
          tempfile.TemporaryDirectory() as tmp):
        _raw_tok, tok_identity = load_verified_tokenizer(
            ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz",
            expected_sha256=_artifact_sha(), vocabulary_size=24_576,
            trainer_config_sha256=_trainer_sha(),
            corpus_manifest_sha256=_corpus_sha())
        tok_backend = _Tok(_raw_tok, tok_identity)
        docs = _documents()
        run_campaign(documents=docs, tokenizer=tok_backend,
                     model_spec=MINI_SPEC, run_id="done", seed=1, updates=2,
                     store_root=str(Path(tmp) / "s"), device=_device,
                     torch_module=_torch, xb=object())
        again = run_campaign(documents=docs, tokenizer=tok_backend,
                             model_spec=MINI_SPEC, run_id="done", seed=1,
                             updates=2, store_root=str(Path(tmp) / "s"),
                             device=_device, torch_module=_torch, xb=object(),
                             resume_store_root=str(Path(tmp) / "s"),
                             resume_run_id="done")
        assert again["already_complete"] is True


def test_freeze_production_identity() -> None:
    import tempfile

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


def test_materialize_first_party_supply_accounting() -> None:
    _raw_tok, _identity = load_verified_tokenizer(
        ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz",
        expected_sha256=_artifact_sha(), vocabulary_size=24_576,
        trainer_config_sha256=_trainer_sha(),
        corpus_manifest_sha256=_corpus_sha())
    tok = _Tok(_raw_tok, _identity)
    got = mfp.materialize_first_party(repo_root=str(ROOT), tokenizer=tok)
    assert got["documents_materialized"] > 0
    assert got["DATA_NOT_READY"] is True, (
        "first-party supply is honestly far below 500M unique demand")
    assert got["verdict"].startswith("DATA_NOT_READY")
    assert got["shortfall_tokens_by_source"]["verified_cognition"] > 0


def _artifact_sha() -> str:
    import hashlib

    p = ROOT / "artifacts/e1/local_tournament/tokenizer-24576.json.gz"
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _trainer_sha() -> str:
    import hashlib

    r = json.loads((ROOT / "artifacts/e1/local_tournament/result.json")
                   .read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(r["trainer"], sort_keys=True)
                          .encode("utf-8")).hexdigest()


def _corpus_sha() -> str:
    r = json.loads((ROOT / "artifacts/e1/local_tournament/result.json")
                   .read_text(encoding="utf-8"))
    return r["corpus_manifest_sha256"]


def main() -> int:
    tests = [test_fresh_campaign_and_exact_resume,
             test_already_complete_short_circuits,
             test_freeze_production_identity,
             test_materialize_first_party_supply_accounting]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
