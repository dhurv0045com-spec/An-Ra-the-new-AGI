"""Bucket-lane cursor tests (no torch): exact single-bucket microsteps,
deterministic resume, missing-bucket fail-closed, epoch replay, receipts.
Run: python tests/test_v5_bucket_cursor.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v5_data.bucket_cursor import (  # noqa: E402
    BUCKET_CURSOR_SCHEMA,
    BucketCursorState,
    LaneExhausted,
    build_bucket_lanes,
    cell_key,
    lane_remainder,
    take_cell_window,
)
from v5_data.pack import pack_documents  # noqa: E402


def _docs():
    docs = []
    for i in range(30):
        docs.append((f"s{i:02d}", [10 + (i % 50)] * 300, "fam-a"))
    for i in range(12):
        docs.append((f"m{i:02d}", [20 + i] * 700, "fam-b"))
    for i in range(12):
        docs.append((f"n{i:02d}", [40 + i] * 700, "fam-a"))
    for i in range(4):
        docs.append((f"L{i:02d}", [30 + i] * 2500, "fam-a"))
    return docs


def _packed():
    packed, _ = pack_documents(_docs(), bos=2, eos=3, pad=0,
                               sequences_per_shard=4)
    return packed


def test_single_bucket_physical_shape():
    packed = _packed()
    buckets = sorted({s.bucket for s in packed})
    assert 512 in buckets and 1024 in buckets
    lanes, receipt = build_bucket_lanes(packed, run_seed=3, pattern=[512, 1024])
    key = cell_key(512, "", "")
    window = take_cell_window(packed, lanes[key], 0, 0, real_tokens=1000,
                              pad=0, bucket=512)
    assert window.real_tokens == 1000
    assert window.row_widths and max(window.row_widths) <= 512
    assert sum(window.tokens_by_source.values()) == 1000
    assert window.end_lane_index > 0 or window.end_token_offset > 0


def test_exact_supercycle_sequence_and_resume():
    packed = _packed()
    lanes, receipt = build_bucket_lanes(packed, run_seed=3, pattern=[512, 1024])
    key = cell_key(512, "", "")
    first = take_cell_window(packed, lanes[key], 0, 0, real_tokens=1000,
                             pad=0, bucket=512)
    second = take_cell_window(packed, lanes[key], first.end_lane_index,
                              first.end_token_offset, real_tokens=1000,
                              pad=0, bucket=512)
    assert first.real_tokens + second.real_tokens == 2000
    replay = take_cell_window(packed, lanes[key], 0, 0, real_tokens=2000,
                              pad=0, bucket=512)
    assert list(replay.tokens[:len(first.tokens)]) == list(first.tokens)
    assert list(replay.tokens[len(first.tokens):]) != []
    assert replay.tokens_by_source == {
        k: first.tokens_by_source.get(k, 0) + second.tokens_by_source.get(k, 0)
        for k in set(first.tokens_by_source) | set(second.tokens_by_source)}


def test_missing_bucket_fails_closed():
    packed = _packed()
    try:
        build_bucket_lanes(packed, run_seed=3, pattern=[512, 2048])
    except ValueError as exc:
        assert "2048" in str(exc)
    else:
        raise AssertionError("missing bucket did not fail closed")


def test_lane_exhaustion_fails_closed():
    packed = _packed()
    lanes, _ = build_bucket_lanes(packed, run_seed=3, pattern=[512])
    key = cell_key(512, "", "")
    total = sum(s.real_tokens for s in packed if s.bucket == 512)
    try:
        take_cell_window(packed, lanes[key], 0, 0, real_tokens=total + 1,
                         pad=0, bucket=512)
    except LaneExhausted:
        pass
    else:
        raise AssertionError("lane overrun did not fail closed")


def test_epoch_replay_deterministic():
    packed = _packed()
    first_lanes, first_receipt = build_bucket_lanes(
        packed, run_seed=3, pattern=[512], epoch=0)
    second_lanes, second_receipt = build_bucket_lanes(
        packed, run_seed=3, pattern=[512], epoch=1)
    assert first_receipt["lanes_sha256"] != second_receipt["lanes_sha256"]
    key = cell_key(512, "", "")
    assert first_lanes[key] != second_lanes[key]
    again, again_receipt = build_bucket_lanes(
        packed, run_seed=3, pattern=[512], epoch=1)
    assert again_receipt["lanes_sha256"] == second_receipt["lanes_sha256"]
    assert again[key] == second_lanes[key]


def test_mixed_cell_sequence_rejected():
    packed = _packed()
    cell_of_source = {"fam-a": ("natural", ""), "fam-b": ("code_math_formal", "")}
    try:
        build_bucket_lanes(packed, run_seed=3, pattern=[512, 1024],
                           cell_of_source=cell_of_source)
    except ValueError as exc:
        assert "mixed-cell" in str(exc)
    else:
        raise AssertionError("mixed-cell sequence did not fail closed")


def test_segregated_pack_builds_family_lanes():
    segregated, _ = pack_documents(
        [(doc_id, content, source) for doc_id, content, source in _docs()],
        bos=2, eos=3, pad=0, sequences_per_shard=4,
        cell_of_source={"fam-a": ("natural", ""),
                        "fam-b": ("code_math_formal", "")})
    lanes, receipt = build_bucket_lanes(
        segregated, run_seed=3, pattern=[512, 1024],
        cell_of_source={"fam-a": ("natural", ""),
                        "fam-b": ("code_math_formal", "")})
    assert cell_key(512, "natural", "") in lanes
    window = take_cell_window(
        segregated, lanes[cell_key(512, "natural", "")], 0, 0,
        real_tokens=500, pad=0, bucket=512,
        cell_of_source={"fam-a": ("natural", ""),
                        "fam-b": ("code_math_formal", "")})
    assert window.real_tokens == 500
    assert set(window.tokens_by_family) == {"natural"}


def test_cursor_state_round_trip():
    cursor = BucketCursorState(
        BUCKET_CURSOR_SCHEMA, "a" * 64, "b" * 64,
        {cell_key(512, "", ""): (3, 17)}, {"natural": 100}, {}, 1, 2)
    cursor.assert_valid()
    restored = BucketCursorState.from_dict(cursor.canonical())
    assert restored == cursor
    assert restored.positions[cell_key(512, "", "")] == (3, 17)


def test_lane_remainder_matches_consumption():
    packed = _packed()
    lanes, _ = build_bucket_lanes(packed, run_seed=3, pattern=[512])
    key = cell_key(512, "", "")
    full = lane_remainder(packed, lanes[key], 0, 0, pad=0)
    assert full > 0
    window = take_cell_window(packed, lanes[key], 0, 0, real_tokens=1000,
                              pad=0, bucket=512)
    rest = lane_remainder(packed, lanes[key], window.end_lane_index,
                          window.end_token_offset, pad=0)
    assert full - rest == 1000


def test_receipt_binds_everything():
    packed = _packed()
    _, receipt = build_bucket_lanes(packed, run_seed=3, pattern=[512, 1024])
    assert receipt["run_seed"] == 3
    assert receipt["pattern"] == [512, 1024]
    assert len(receipt["lanes_sha256"]) == 64
    for cell in receipt["cells"].values():
        assert cell["sequences"] > 0 and cell["real_tokens"] > 0


_TESTS = [test_single_bucket_physical_shape,
          test_exact_supercycle_sequence_and_resume,
          test_missing_bucket_fails_closed,
          test_lane_exhaustion_fails_closed,
          test_epoch_replay_deterministic,
          test_mixed_cell_sequence_rejected,
          test_segregated_pack_builds_family_lanes,
          test_lane_remainder_matches_consumption,
          test_cursor_state_round_trip,
          test_receipt_binds_everything]


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
