"""B02 focused tests: public codec, sequences, boundaries and sidecars."""
import unittest

import torch

from bramastra_lab.research.config import BuildConfig, seed_everything
from bramastra_lab.research.experience.codec import (
    DEFAULT_MAX_EVENT_BYTES,
    SPECIAL_BOUNDARY,
    SPECIAL_EOS,
    EncodingError,
    codec_identity,
    decode_to_bytes,
    encode_event,
    encode_text,
)
from bramastra_lab.research.experience.sequences import (
    SequenceError,
    build_answer_row,
    build_language_row,
    collocate,
    pack_rows,
)
from bramastra_lab.research.models import IntegratedModel

BASE_PROVENANCE = {"episode_id": "ep-1", "task_semantic_id": "task-1", "split": "training",
                   "source": "fixture", "collection_policy": "fixed-v1"}


class CodecTests(unittest.TestCase):
    def test_key_order_invariance(self) -> None:
        first = encode_event("goal", {"a": 1, "b": [2, 3], "c": {"d": 4}})
        second = encode_event("goal", {"c": {"d": 4}, "b": [2, 3], "a": 1})
        self.assertEqual(first, second)

    def test_event_and_list_order_is_meaningful(self) -> None:
        goal = encode_event("goal", {"question": "q1"})
        observation = encode_event("observation", {"value": "v1"})
        forward = goal + observation
        backward = observation + goal
        self.assertNotEqual(forward, backward)
        self.assertNotEqual(encode_event("observation", ["a", "b"]),
                            encode_event("observation", ["b", "a"]))

    def test_role_and_content_changes_alter_tokens(self) -> None:
        self.assertNotEqual(encode_event("goal", {"q": 1}), encode_event("feedback", {"q": 1}))
        self.assertNotEqual(encode_event("goal", {"q": 1}), encode_event("goal", {"q": 2}))

    def test_unknown_role_rejects(self) -> None:
        with self.assertRaises(EncodingError):
            encode_event("hidden_answer", {"answer": "x"})

    def test_oversized_event_fails_explicitly(self) -> None:
        with self.assertRaises(EncodingError):
            encode_event("goal", {"blob": "x" * (DEFAULT_MAX_EVENT_BYTES + 1)})

    def test_nonfinite_content_rejects(self) -> None:
        with self.assertRaises(EncodingError):
            encode_event("observation", {"value": float("nan")})

    def test_byte_round_trip(self) -> None:
        text = "héllo wörld ☃"
        self.assertEqual(decode_to_bytes(encode_text(text)), text.encode("utf-8"))
        self.assertTrue(1 <= max(encode_text(text)) <= 255)

    def test_codec_identity_is_stable(self) -> None:
        self.assertEqual(codec_identity(), codec_identity())

    def test_decode_rejects_out_of_range(self) -> None:
        with self.assertRaises(EncodingError):
            decode_to_bytes([999])


class SequenceRowTests(unittest.TestCase):
    def test_provenance_only_changes_leave_tokens_unchanged(self) -> None:
        events = [("goal", {"question": "2+2?"})]
        row_a = build_answer_row(events, "4", provenance={**BASE_PROVENANCE, "episode_id": "a"},
                                 max_tokens=64)
        row_b = build_answer_row(events, "4", provenance={**BASE_PROVENANCE, "episode_id": "b"},
                                 max_tokens=64)
        self.assertEqual(row_a.tokens, row_b.tokens)
        self.assertEqual(row_a.supervised, row_b.supervised)

    def test_relevant_goal_change_alters_tokens(self) -> None:
        row_a = build_answer_row([("goal", {"question": "2+2?"})], "4",
                                 provenance=BASE_PROVENANCE, max_tokens=64)
        row_b = build_answer_row([("goal", {"question": "3+3?"})], "4",
                                 provenance=BASE_PROVENANCE, max_tokens=64)
        self.assertNotEqual(row_a.tokens, row_b.tokens)

    def test_one_token_answer_plus_eos_exact_denominator(self) -> None:
        row = build_answer_row([("goal", {"q": 1})], "7", provenance=BASE_PROVENANCE,
                               max_tokens=64)
        self.assertEqual(row.tokens[-1], SPECIAL_EOS)
        self.assertEqual(row.target_count, 2)  # one answer byte + required EOS

    def test_language_row_supervises_everything_including_eos(self) -> None:
        row = build_language_row("abc", provenance=BASE_PROVENANCE, max_tokens=32)
        self.assertEqual(row.tokens[0], SPECIAL_BOUNDARY)
        self.assertEqual(row.tokens[-1], SPECIAL_EOS)
        self.assertEqual(row.target_count, len(row.tokens) - 1)

    def test_row_without_terminal_eos_rejects(self) -> None:
        from bramastra_lab.research.experience.sequences import SequenceRow

        with self.assertRaises(SequenceError):
            SequenceRow(tokens=(1, 2), supervised=(True, True), provenance=BASE_PROVENANCE,
                        segment_ids=(1, 1))

    def test_provenance_cannot_carry_answers(self) -> None:
        with self.assertRaises(SequenceError):
            build_answer_row([("goal", {"q": 1})], "4",
                             provenance={**BASE_PROVENANCE, "answer": "4"}, max_tokens=64)
        with self.assertRaises(SequenceError):
            build_answer_row([("goal", {"q": 1})], "4",
                             provenance={**BASE_PROVENANCE, "generator_hash": "xyz"},
                             max_tokens=64)

    def test_oversized_row_fails(self) -> None:
        with self.assertRaises(EncodingError):
            build_language_row("x" * 200, provenance=BASE_PROVENANCE, max_tokens=32)


class CollocationTests(unittest.TestCase):
    def make_rows(self) -> list:
        rows = [
            build_language_row("short", provenance={**BASE_PROVENANCE, "episode_id": "e1"},
                               max_tokens=64),
            build_language_row("a longer document row here",
                               provenance={**BASE_PROVENANCE, "episode_id": "e2"}, max_tokens=64),
            build_answer_row([("goal", {"q": 1})], "42",
                             provenance={**BASE_PROVENANCE, "episode_id": "e3"}, max_tokens=64),
        ]
        rows[2] = rows[2]
        return rows

    def test_ragged_batch_pads_and_shifts_once(self) -> None:
        rows = self.make_rows()
        batch = collocate(rows, max_seq=64)
        self.assertEqual(batch.batch_size, 3)
        self.assertEqual(batch.target_count, sum(row.target_count for row in rows))
        # Shift: inputs at position t predict labels at position t.
        first = rows[0]
        self.assertEqual(batch.input_ids[0, 0].item(), first.tokens[0])
        self.assertEqual(batch.labels[0, 0].item(), first.tokens[1])
        # Padding is excluded and labels are -100 there.
        pad_positions = ~batch.padding_mask[0]
        self.assertTrue((batch.labels[0][pad_positions] == -100).all())
        self.assertFalse(batch.loss_mask[0][pad_positions].any())

    def test_exact_full_sequence_has_no_padding(self) -> None:
        row = build_language_row("x" * 20, provenance=BASE_PROVENANCE, max_tokens=64)
        batch = collocate([row], max_seq=len(row.tokens))
        self.assertTrue(batch.padding_mask.all())

    def test_loss_mask_marks_exactly_answer_and_eos(self) -> None:
        row = build_answer_row([("goal", {"q": 1})], "7", provenance=BASE_PROVENANCE,
                               max_tokens=64)
        batch = collocate([row], max_seq=64)
        flagged = batch.loss_mask[0].sum().item()
        self.assertEqual(flagged, 2)
        flagged_positions = batch.loss_mask[0].nonzero().flatten().tolist()
        for position in flagged_positions:
            self.assertIn(batch.labels[0, position].item(),
                          [ord("7"), SPECIAL_EOS])

    def test_sidecar_excludes_answers_and_records_identity(self) -> None:
        row = build_answer_row([("goal", {"q": 1})], "7", provenance=BASE_PROVENANCE,
                               max_tokens=64)
        batch = collocate([row], max_seq=64)
        sidecar = batch.provenance[0]
        self.assertNotIn("answer", sidecar)
        self.assertNotIn("target", sidecar)
        self.assertEqual(sidecar["target_count"], 2)
        self.assertEqual(batch.sidecar_identity, batch.sidecar_identity)


class PackingIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(5)
        raw = {"model": {"profile": "tiny"}}
        self.config = BuildConfig.from_dict(raw)
        self.model = IntegratedModel(self.config)
        self.model.eval()

    def test_packing_preserves_targets_and_inserts_boundaries(self) -> None:
        rows = self.make_two_rows()
        packed = pack_rows(rows, max_tokens=256)
        self.assertEqual(packed.target_count, sum(row.target_count for row in rows))
        first_segment = packed.segment_ids[0]
        second_segment = packed.segment_ids[-1]
        self.assertNotEqual(first_segment, second_segment)
        # The boundary token between episodes belongs to the second segment.
        boundary_position = len(rows[0].tokens)
        self.assertEqual(packed.tokens[boundary_position], SPECIAL_BOUNDARY)
        self.assertEqual(packed.segment_ids[boundary_position], second_segment)
        self.assertFalse(packed.supervised[boundary_position])

    @staticmethod
    def make_two_rows(answer_b: str = "4") -> list:
        row_a = build_answer_row([("goal", {"question": "9+9?"})], "18",
                                 provenance={**BASE_PROVENANCE, "episode_id": "A"}, max_tokens=256)
        row_b = build_answer_row([("goal", {"question": "2+2?"})], answer_b,
                                 provenance={**BASE_PROVENANCE, "episode_id": "B"}, max_tokens=256)
        return [row_a, row_b]

    def _forward_row(self, row) -> torch.Tensor:
        batch = collocate([row], max_seq=256)
        with torch.no_grad():
            output = self.model(batch.input_ids, batch.padding_mask,
                                segment_ids=batch.segment_ids)
        return output.logits

    def test_packed_neighbor_cannot_change_episode_logits(self) -> None:
        row_a, row_b = self.make_two_rows()
        alone = self._forward_row(row_a)
        packed = pack_rows([row_a, row_b], max_tokens=256)
        packed_batch = collocate([packed], max_seq=256)
        with torch.no_grad():
            packed_logits = self.model(packed_batch.input_ids, packed_batch.padding_mask,
                                       segment_ids=packed_batch.segment_ids).logits
        # Episode A occupies positions [0, len(row_a.tokens)); compare its logits.
        size_a = len(row_a.tokens) - 1  # shifted inputs for episode A
        torch.testing.assert_close(alone[0, :size_a], packed_logits[0, :size_a])

    def test_changed_neighbor_content_still_cannot_leak(self) -> None:
        row_a, row_b_original = self.make_two_rows()
        _, row_b_changed = self.make_two_rows(answer_b="999999")
        packed_original = pack_rows([row_a, row_b_original], max_tokens=256)
        packed_changed = pack_rows([row_a, row_b_changed], max_tokens=256)
        batch_original = collocate([packed_original], max_seq=256)
        batch_changed = collocate([packed_changed], max_seq=256)
        with torch.no_grad():
            logits_original = self.model(batch_original.input_ids, batch_original.padding_mask,
                                         segment_ids=batch_original.segment_ids).logits
            logits_changed = self.model(batch_changed.input_ids, batch_changed.padding_mask,
                                        segment_ids=batch_changed.segment_ids).logits
        size_a = len(row_a.tokens) - 1
        torch.testing.assert_close(logits_original[0, :size_a], logits_changed[0, :size_a])

    def test_batch_neighbors_are_isolated_without_packing(self) -> None:
        row_a, row_b = self.make_two_rows()
        batch_a = collocate([row_a], max_seq=256)
        batch_ab = collocate([row_a, row_b], max_seq=256)
        with torch.no_grad():
            alone = self.model(batch_a.input_ids, batch_a.padding_mask,
                               segment_ids=batch_a.segment_ids).logits
            together = self.model(batch_ab.input_ids, batch_ab.padding_mask,
                                  segment_ids=batch_ab.segment_ids).logits
        size = batch_ab.input_ids.shape[1]
        torch.testing.assert_close(alone[0, :size], together[0, :size])


class BucketedBatchingTests(unittest.TestCase):
    def test_bucketing_reduces_padding_and_keeps_targets(self) -> None:
        from bramastra_lab.research.experience.sequences import bucketed_batches

        rows = [build_language_row("x" * length, provenance=BASE_PROVENANCE,
                                   max_tokens=64)
                for length in (4, 5, 6, 30, 31, 32)]
        plain = collocate(rows, max_seq=64)
        batches = bucketed_batches(rows, batch_size=2, bucket_width=8)
        self.assertEqual(len(batches), 3)
        targets = sum(collocate(batch, max_seq=64).target_count for batch in batches)
        self.assertEqual(targets, sum(row.target_count for row in rows))
        bucketed = [collocate(batch, max_seq=64) for batch in batches]
        # Two short rows and two long rows now share batches: less padding.
        short_padding = (~bucketed[0].padding_mask).sum().item()
        self.assertLessEqual(short_padding, (~plain.padding_mask).sum().item())

    def test_bucketing_is_deterministic(self) -> None:
        from bramastra_lab.research.experience.sequences import bucketed_batches

        rows = [build_language_row("x" * (index + 3), provenance=BASE_PROVENANCE,
                                   max_tokens=64) for index in range(5)]
        first = [[row.tokens for row in batch] for batch in
                 bucketed_batches(rows, batch_size=2)]
        second = [[row.tokens for row in batch] for batch in
                  bucketed_batches(rows, batch_size=2)]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
