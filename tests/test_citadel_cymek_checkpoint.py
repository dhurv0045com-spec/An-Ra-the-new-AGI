"""Cymek production checkpoint contract tests (pure, no torch/TPU).

Run:  python tests/test_citadel_cymek_checkpoint.py   (exit 0 = all pass)
Exercises the REAL Cymek TransactionStore + TrainingState (from a Temp
detached worktree of the pinned runtime — read-only, removed afterwards)
through the Citadel thin adapter: genesis/publish/restore round-trips,
exact-token enforcement, cursor/pack rules, writer fencing, inventory and
corruption detection, identity validation. Torch-dependent payload assembly
(model/optimizer bytes) is proven live on TPU by the PRE50M smoke; everything
here runs on bytes + JSON.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(CITADEL_ROOT))

from citadel_tpu import runtime_bootstrap as rb  # noqa: E402

PIN = rb.PINNED_CYMEK_SHA
TOKENS_PER_UPDATE = 2048
BUDGET = TOKENS_PER_UPDATE * 4
PACK_SHA = "aa" * 32
SOURCE = "bb" * 20  # full lowercase Git SHA-1 (40 hex), per Cymek contract


def _git(args, *, cwd):
    out = subprocess.run(["git", *args], capture_output=True, text=True,
                         timeout=120, cwd=str(cwd))
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {(out.stderr or '')[:200]}")
    return (out.stdout or "").strip()


class Ctx:
    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="citadel-ckpt-"))
        self.saved_env = {k: os.environ.get(k) for k in
                          ("CITADEL_ROOT", "CITADEL_CYMEK_RUNTIME",
                           "CITADEL_CYMEK_RUNTIME_DIR", "CITADEL_CYMEK_SHA")}
        self.saved_path = list(sys.path)
        # Use the checkout that is actually running the test.  Never embed a
        # developer-machine path: this suite must run unchanged on Colab,
        # Linux CI, Windows, and a clean clone.
        main = rb.citadel_root()
        _git(["worktree", "add", "--detach", str(self.tmp / "cymek"), PIN],
             cwd=main)
        os.environ["CITADEL_CYMEK_RUNTIME"] = str(self.tmp / "cymek")
        os.environ.pop("CITADEL_CYMEK_SHA", None)
        from citadel_tpu import cymek_checkpoint as cc

        self.cc = cc
        self.rt, self.sha = rb.ensure_cymek_runtime()
        assert self.sha == PIN, (self.sha, PIN)
        self.store = cc.open_store(str(self.tmp / "store"), "test-lineage")

    def close(self):
        for k, v in self.saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.path[:] = self.saved_path
        main = rb.citadel_root()
        try:
            _git(["worktree", "remove", "--force", str(self.tmp / "cymek")],
                 cwd=main)
        except RuntimeError:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def identities(self, **over):
        kw = {"model_spec_sha256": "11" * 32, "data_manifest_sha256": "22" * 32,
              "pack_manifest_sha256": PACK_SHA, "run_spec": {"lr": 3e-4},
              "optimizer_spec": {"name": "AdamW"}, "schedule_spec": {"kind": "const"},
              "curriculum_spec": {"phase": "smoke"}, "source_commit": SOURCE}
        kw.update(over)
        return self.cc.build_identities(**kw)

    def genesis(self):
        ids = self.identities()
        return self.cc.initial_state(
            lineage_id="test-lineage", token_budget=BUDGET,
            tokens_per_update=TOKENS_PER_UPDATE, pack_manifest_sha256=PACK_SHA,
            identities=ids, rng_state_sha256="33" * 32)

    def payloads(self, n: int = 8):
        return {k: bytes([n]) * (16 + n) for k in
                ("model.bin", "optimizer.bin", "scheduler.json", "rng.bin",
                 "cursor.json", "ledger.json", "training_state.json")}


def _with_ctx(label, fn):
    ctx = Ctx()
    try:
        fn(ctx)
    finally:
        ctx.close()
    print(f"PASS {label}", flush=True)


def t_identities_and_helpers(ctx: Ctx) -> None:
    ids = ctx.identities()
    ids.assert_valid()
    assert len(ctx.cc.tokenizer_identity_sha256()) == 64
    assert ctx.cc.tokenizer_identity_sha256() == ctx.cc.tokenizer_identity_sha256()
    assert len(ctx.cc.spec_json_sha256({"a": 1})) == 64
    bad = dict(model_spec_sha256="zz", data_manifest_sha256="22" * 32,
               pack_manifest_sha256=PACK_SHA, run_spec={}, optimizer_spec={},
               schedule_spec={}, curriculum_spec={}, source_commit=SOURCE)
    try:
        ctx.cc.build_identities(**bad).assert_valid()
        raise SystemExit("bad sha accepted")
    except ValueError:
        pass
    try:
        ctx.cc.open_store(str(ctx.tmp / "s"), "bad/lineage")
        raise SystemExit("bad lineage accepted")
    except ValueError:
        pass


def t_genesis_roundtrip(ctx: Ctx) -> None:
    st = ctx.genesis()
    st.assert_valid()
    assert st.generation == 0 and st.cumulative_tokens == 0
    assert st.parent_checkpoint_sha256 is None
    p = dict(ctx.payloads())
    p["training_state.json"] = json.dumps(
        st.canonical(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()
    p["cursor.json"] = json.dumps({"schema": st.cursor.schema,
                                   "pack_manifest_sha256": st.cursor.pack_manifest_sha256,
                                   "shard_ordinal": 0, "sequence_ordinal": 0,
                                   "token_offset": 0}).encode()
    p["ledger.json"] = b"{}"
    sha = ctx.store.publish(state=st, payloads=p, expected_parent_sha256=None)
    assert ctx.store.latest_sha256() == sha
    st2, p2 = ctx.store.restore()
    assert st2.sha256() == st.sha256() and json.loads(p2["ledger.json"]) == {}
    st3, _ = ctx.store.restore(checkpoint_sha256=sha)
    assert st3.sha256() == st.sha256()


def _state_payloads(ctx: Ctx, state, n: int) -> dict[str, bytes]:
    """Hand payloads with VALID cursor/ledger JSON for a given state."""
    p = dict(ctx.payloads(n))
    p["cursor.json"] = json.dumps({"schema": state.cursor.schema,
                                   "pack_manifest_sha256": state.cursor.pack_manifest_sha256,
                                   "shard_ordinal": state.cursor.shard_ordinal,
                                   "sequence_ordinal": state.cursor.sequence_ordinal,
                                   "token_offset": state.cursor.token_offset}).encode()
    p["ledger.json"] = json.dumps(dict(state.tokens_by_source)).encode()
    return p


def t_advance_chain(ctx: Ctx) -> None:
    cc = ctx.cc
    st = ctx.genesis()
    sha0 = cc.publish_state(ctx.store, state=st, payloads=_state_payloads(ctx, st, 1),
                            expected_parent_sha256=None)
    prev, prev_sha = st, sha0
    for u in (1, 2):
        cur = cc.cursor_for_update(PACK_SHA, sequence_ordinal=u,
                                   token_offset=u * TOKENS_PER_UPDATE)
        nxt = cc.advance_state(prev, cursor=cur,
                               ledger_delta={"smoke": TOKENS_PER_UPDATE},
                               rng_bytes=bytes([u]) * 32, store=ctx.store)
        assert nxt.generation == u and nxt.global_update == u
        assert nxt.cumulative_tokens == u * TOKENS_PER_UPDATE
        assert nxt.parent_checkpoint_sha256 == prev_sha
        assert nxt.tokens_by_source == {"smoke": u * TOKENS_PER_UPDATE}
        sha = cc.publish_state(ctx.store, state=nxt,
                               payloads=_state_payloads(ctx, nxt, u),
                               expected_parent_sha256=prev_sha)
        prev, prev_sha = nxt, sha
    assert ctx.store.latest_sha256() == prev_sha


def t_advance_rejections(ctx: Ctx) -> None:
    cc = ctx.cc
    st = ctx.genesis()
    cur = cc.cursor_for_update(PACK_SHA, sequence_ordinal=1,
                               token_offset=TOKENS_PER_UPDATE)
    for bad_ledger, why in (({"smoke": TOKENS_PER_UPDATE - 1}, "short"),
                            ({"smoke": TOKENS_PER_UPDATE + 1}, "long"),
                            ({"": 5}, "empty-name")):
        try:
            cc.advance_state(st, cursor=cur, ledger_delta=bad_ledger,
                             rng_bytes=b"0" * 32)
            raise SystemExit(f"bad ledger accepted ({why})")
        except (ValueError, RuntimeError):
            pass
    try:  # cursor must advance
        cc.advance_state(st, cursor=st.cursor,
                         ledger_delta={"smoke": TOKENS_PER_UPDATE},
                         rng_bytes=b"0" * 32)
        raise SystemExit("non-advancing cursor accepted")
    except (ValueError, RuntimeError):
        pass
    try:  # pack migration forbidden
        other = cc.cursor_for_update("bb" * 32, sequence_ordinal=1,
                                     token_offset=TOKENS_PER_UPDATE)
        cc.advance_state(st, cursor=other,
                         ledger_delta={"smoke": TOKENS_PER_UPDATE},
                         rng_bytes=b"0" * 32)
        raise SystemExit("pack migration accepted")
    except (ValueError, RuntimeError):
        pass
    assert ctx.store.latest_sha256() is None, "failed advances must write nothing"


def t_fence_and_inventory(ctx: Ctx) -> None:
    st = ctx.genesis()
    p = dict(ctx.payloads(2))
    p["training_state.json"] = json.dumps(
        st.canonical(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()
    p["cursor.json"] = json.dumps({"schema": st.cursor.schema,
                                   "pack_manifest_sha256": PACK_SHA,
                                   "shard_ordinal": 0, "sequence_ordinal": 0,
                                   "token_offset": 0}).encode()
    p["ledger.json"] = b"{}"
    sha = ctx.store.publish(state=st, payloads=p, expected_parent_sha256=None)
    try:  # stale parent fence
        ctx.store.publish(state=st, payloads=p, expected_parent_sha256="00" * 64)
        raise SystemExit("stale fence accepted")
    except ValueError:
        pass
    assert ctx.store.latest_sha256() == sha
    bad = dict(p)
    del bad["rng.bin"]
    try:
        ctx.store.publish(state=st, payloads=bad, expected_parent_sha256=sha)
        raise SystemExit("incomplete inventory accepted")
    except ValueError:
        pass
    bad2 = dict(p)
    bad2["extra.bin"] = b"x"
    try:
        ctx.store.publish(state=st, payloads=bad2, expected_parent_sha256=sha)
        raise SystemExit("extra component accepted")
    except ValueError:
        pass


def t_corruption_detected(ctx: Ctx) -> None:
    st = ctx.genesis()
    p = dict(ctx.payloads(3))
    p["training_state.json"] = json.dumps(
        st.canonical(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode()
    p["cursor.json"] = json.dumps({"schema": st.cursor.schema,
                                   "pack_manifest_sha256": PACK_SHA,
                                   "shard_ordinal": 0, "sequence_ordinal": 0,
                                   "token_offset": 0}).encode()
    p["ledger.json"] = b"{}"
    sha = ctx.store.publish(state=st, payloads=p, expected_parent_sha256=None)
    target = ctx.tmp / "store" / "test-lineage" / "objects" / sha / "rng.bin"
    blob = bytearray(target.read_bytes())
    blob[0] ^= 0xFF
    target.write_bytes(bytes(blob))
    try:
        ctx.store.restore(checkpoint_sha256=sha)
        raise SystemExit("corrupt component accepted")
    except ValueError:
        pass
    man = ctx.tmp / "store" / "test-lineage" / "objects" / sha / "manifest.json"
    man.write_bytes(b'{"tampered": true}')
    try:
        ctx.store.restore(checkpoint_sha256=sha)
        raise SystemExit("tampered manifest accepted")
    except ValueError:
        pass


def t_genesis_rules(ctx: Ctx) -> None:
    import dataclasses as _dc

    st = ctx.genesis()
    bad = _dc.replace(st, cumulative_tokens=7)
    try:
        bad.assert_valid()
        raise SystemExit("nonzero genesis accepted")
    except ValueError:
        pass
    bad2 = _dc.replace(st, parent_checkpoint_sha256="ab" * 32)
    try:
        bad2.assert_valid()
        raise SystemExit("parented genesis accepted")
    except ValueError:
        pass


def main() -> int:
    tests = [t_identities_and_helpers, t_genesis_roundtrip, t_advance_chain,
             t_advance_rejections, t_fence_and_inventory, t_corruption_detected,
             t_genesis_rules]
    failed = 0
    for fn in tests:
        try:
            _with_ctx(fn.__name__, fn)
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}", flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
