"""Complete-answer training contract tests (BRAMASTRA terminal lesson).

Mechanical chain, each link tested: raw example -> tokenizer -> pack
(EOS present, EOS-final segments) -> labels (BOS/PAD excluded, content +
EOS supervised) -> generation loop (EOS stops, cap stops). No training,
no GPU. Run: python tests/test_v5_complete_answer.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v5_data.bucket_cursor import (  # noqa: E402
    build_bucket_lanes,
    cell_key,
    take_cell_window,
)
from v5_data.pack import pack_documents  # noqa: E402
from v5_training.production_entry import _predict_supervised  # noqa: E402

BOS, EOS, PAD = 2, 3, 0


def test_packed_segments_end_with_eos():
    docs = [("d1", [11, 12, 13], "s"), ("d2", [21] * 600, "s"),
            ("d3", [31] * 5000, "s")]
    packed, _ = pack_documents(docs, bos=BOS, eos=EOS, pad=PAD,
                               sequences_per_shard=8)
    assert packed, "no shards packed"
    for shard in packed:
        for sequence in shard.sequences:
            for index, source in enumerate(sequence.sources):
                segment = [token for token, seg in
                           zip(sequence.tokens, sequence.segment_ids)
                           if seg == index]
                assert segment[0] == BOS, "segment must open with BOS"
                assert segment[-1] == EOS, "segment must close with EOS"


def test_eos_supervised_bos_pad_excluded():
    from v5_objectives.causal_lm import causal_lm_loss
    import torch

    tokens = torch.tensor([[BOS, 5, 6, EOS, BOS, 7, EOS, PAD]])
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, -1]])
    logits = torch.zeros(1, 8, 16)
    _, count = causal_lm_loss(logits, tokens, segments, bos_id=BOS,
                              pad_id=PAD, torch_module=torch)
    # targets: 5, 6, EOS, 7, EOS = 5 (second BOS excluded as target,
    # PAD excluded, segment-cross excluded)
    assert count == 5


def test_window_prediction_matches_loss_on_eos_boundaries():
    docs = [("qa", [41, 42, 43], "s")]
    packed, _ = pack_documents(docs, bos=BOS, eos=EOS, pad=PAD,
                               sequences_per_shard=8)
    lanes, _ = build_bucket_lanes(packed, run_seed=1, pattern=[512])
    window = take_cell_window(packed, lanes[cell_key(512, "", "")], 0, 0,
                              real_tokens=5, pad=PAD, bucket=512)
    assert window.real_tokens == 5  # BOS, 41, 42, 43, EOS
    assert _predict_supervised(window) == 4  # 41, 42, 43, EOS


def test_generation_stops_on_eos_and_on_cap():
    import torch
    from v5_evaluation.checkpoint_adapter import CheckpointBackedV5Adapter

    adapter = CheckpointBackedV5Adapter.__new__(CheckpointBackedV5Adapter)
    adapter.eos_id = EOS
    adapter.bos_id = BOS
    adapter.torch = torch
    adapter.spec = type("Spec", (), {"context_length": 64})()

    class FakeTok:
        def encode(self, text):
            return [5, 6]

        def decode(self, ids):
            return "<" + ",".join(str(i) for i in ids) + ">"

    adapter.tokenizer = FakeTok()
    calls = {"n": 0}

    def fake_logits(token_ids):
        calls["n"] += 1
        rows = len(token_ids)
        # real _logits returns [rows, vocab]: last row is the next-token head
        logits = torch.full((rows, 8), -1e9)
        # emit content twice, then EOS
        emit = [5, 6, EOS][min(calls["n"] - 1, 2)]
        logits[-1, emit] = 0.0
        return logits

    adapter._logits = fake_logits
    out = adapter.generate_free("prompt", max_new_tokens=64)
    assert out == "<5,6>", f"EOS did not stop generation: {out}"
    assert calls["n"] == 3

    calls["n"] = 0

    def never_eos(token_ids):
        calls["n"] += 1
        rows = len(token_ids)
        logits = torch.full((rows, 8), -1e9)
        logits[-1, 5] = 0.0
        return logits

    adapter._logits = never_eos
    out = adapter.generate_free("prompt", max_new_tokens=5)
    assert out == "<5,5,5,5,5>", f"cap did not stop generation: {out}"


_TESTS = [test_packed_segments_end_with_eos,
          test_eos_supervised_bos_pad_excluded,
          test_window_prediction_matches_loss_on_eos_boundaries,
          test_generation_stops_on_eos_and_on_cap]


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
