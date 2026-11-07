from dataclasses import FrozenInstanceError, asdict, replace
import json
import random

import pytest


def test_tensor_fingerprints_and_migration_need_no_numpy(monkeypatch):
    import torch
    from hashlib import sha256
    from anra_v5.elendril.backend import tensor_sha256, update_tensor_fingerprint
    from anra_v5.elendril.cognitive_demo import NeuralCore

    def forbidden(*args, **kwargs):
        raise AssertionError("NumPy is not a declared runtime dependency")
    monkeypatch.setattr(torch.Tensor, "numpy", forbidden)
    weight = torch.arange(46 * 8, dtype=torch.float32).view(46, 8)
    fresh, receipt = migrate_tied_embedding(weight, legacy_to_utf8_migration(seed=91),
                                           parent_model_sha256="a" * 64)
    expected = sha256(bytes(fresh.reshape(-1).view(torch.uint8).tolist())).hexdigest()
    assert tensor_sha256(fresh) == expected
    for chunk in (1, 7, 65536):
        fingerprint = sha256()
        update_tensor_fingerprint(fingerprint, fresh, chunk_bytes=chunk)
        assert fingerprint.hexdigest() == expected
    policy = object.__new__(NeuralCore)
    policy.model = torch.nn.Linear(8, 8, bias=False)
    reference = sha256()
    for name, value in sorted(policy.model.state_dict().items()):
        reference.update(name.encode() + b"\0")
        reference.update(bytes(value.reshape(-1).view(torch.uint8).tolist()))
    assert policy.weights_sha256() == reference.hexdigest()

from anra_v5.elendril import codec
from anra_v5.elendril.benchmark import fixture
from anra_v5.elendril.join import join_fixture
from anra_v5.elendril.relational import relational_fixture
from anra_v5.elendril.representation import (
    LEGACY, UTF8, MAX_TEXT_BYTES, SpecialToken, TokenOffset,
    legacy_to_utf8_migration, migrate_tied_embedding, tokenizer_from_manifest,
)


def test_legacy_fixtures_are_bit_identical_and_manifest_is_stable():
    assert LEGACY.identity == codec.CODEC_SHA
    assert LEGACY.vocabulary_size == codec.VOCAB_SIZE
    assert (LEGACY.pad_id, LEGACY.bos_id, LEGACY.eos_id) == (codec.PAD, codec.BOS, codec.EOS)
    rows = (fixture("copy", "train") + fixture("binding", "train")
            + relational_fixture("train") + join_fixture("train"))
    for row in rows:
        prompt, answer = row.visible.prompt, row.truth.gold
        expected = (codec.BOS, *codec.encode(prompt), *codec.answer_tokens(answer))
        example = LEGACY.frame(prompt, answer)
        assert example.ids == expected
        assert example.loss_mask == (False,) * (1 + len(prompt)) + (True,) * (len(answer) + 1)
        assert LEGACY.encode(prompt) == codec.encode(prompt)
        assert LEGACY.decode(LEGACY.encode(prompt)) == prompt
    assert LEGACY.encode(codec.ALPHABET) == codec.encode(codec.ALPHABET)
    assert json.loads(json.dumps(UTF8.manifest.record()))["identity"] == UTF8.identity
    assert len({UTF8.manifest.vocabulary_sha256, UTF8.manifest.normalizer_sha256,
                UTF8.manifest.pretokenizer_sha256, UTF8.manifest.config_sha256}) == 4
    with pytest.raises(FrozenInstanceError):
        UTF8._byte_level = False


def test_manifest_restore_rejects_changed_config_with_copied_identity():
    for tokenizer in (LEGACY, UTF8):
        record = json.loads(json.dumps(tokenizer.manifest.record()))
        assert tokenizer_from_manifest(record) is tokenizer
        for field, value in (("vocabulary_size", 42), ("normalizer_sha256", "f" * 64),
                             ("special_tokens", [["EOS", 0]]), ("injected", "normalizer code")):
            with pytest.raises(ValueError, match="manifest"):
                tokenizer_from_manifest({**record, field: value})
    for record in (None, {}, {"version": "unknown"}):
        with pytest.raises(ValueError, match="manifest"):
            tokenizer_from_manifest(record)


@pytest.mark.parametrize("text", ["", "\x00\t \r\n  ", "!?,.:;<>[]{}", "नमस्ते दुनिया",
                                  "中文日本語한국어", "العربية עברית", "👩🏽‍💻🌌", "é", "e\u0301",
                                  "<PAD><BOS><EOS><ANSWER>", "\ufeff\u202e\u200d\uffff"])
def test_utf8_lossless_roundtrip_offsets_and_no_special_spoofing(text):
    encoded = UTF8.encode_with_offsets(text)
    assert UTF8.decode(encoded.ids) == text
    assert encoded.ids == tuple(byte + 4 for byte in text.encode("utf-8"))
    assert not set(encoded.ids) & {0, 1, 2, 3}
    assert len(encoded.ids) == len(encoded.offsets) == len(text.encode("utf-8"))
    raw = text.encode("utf-8")
    for token, offset in zip(encoded.ids, encoded.offsets):
        assert 0 <= offset.char_start < offset.char_end <= len(text)
        assert raw[offset.byte_start:offset.byte_end] == bytes((token - 4,))
        assert raw[offset.byte_start:offset.byte_end] in text[offset.char_start:offset.char_end].encode("utf-8")
    assert UTF8.encode("é") != UTF8.encode("e\u0301")


def test_unicode_properties_and_large_string_are_lossless():
    rng = random.Random(901)
    # Sample the whole Unicode scalar range rather than only ASCII fixtures.
    scalars = [value for value in (rng.randrange(0x110000) for _ in range(1024))
               if not 0xD800 <= value <= 0xDFFF]
    text = "".join(map(chr, scalars))
    assert UTF8.decode(UTF8.encode(text)) == text
    large = "A🌌न\n" * 8192
    assert UTF8.decode(UTF8.encode(large)) == large


def test_incremental_encoding_has_global_offsets_and_transactional_limits():
    chunks = ("A", "", "👩", "🏽‍💻", "é\t", "e\u0301")
    stream = UTF8.incremental_encoder()
    encoded = [stream.feed(chunk) for chunk in chunks]
    whole = UTF8.encode_with_offsets("".join(chunks))
    assert tuple(value for chunk in encoded for value in chunk.ids) == whole.ids
    assert tuple(value for chunk in encoded for value in chunk.offsets) == whole.offsets
    assert stream.tokens == stream.bytes == len(whole.ids)
    bounded = UTF8.incremental_encoder(max_tokens=4)
    bounded.feed("é")
    before = (bounded.tokens, bounded.characters, bounded.bytes)
    with pytest.raises(ValueError, match="max_tokens"):
        bounded.feed("🌌")
    assert before == (bounded.tokens, bounded.characters, bounded.bytes)
    assert bounded.feed("ab").ids == UTF8.encode("ab")
    assert LEGACY.incremental_encoder(max_tokens=2).feed("AB").ids == codec.encode("AB")


def test_framing_loss_mask_padding_and_codepoint_safe_truncation():
    example = UTF8.frame("X🌌Y", "é", pad_to=16, max_length=16)
    assert example.prompt_span == (1, 7)
    assert example.answer_span == (8, 11)
    assert example.ids[0] == UTF8.bos_id
    assert example.ids[7] == UTF8.special(SpecialToken.ANSWER)
    assert example.ids[10] == UTF8.eos_id
    assert example.loss_mask == (False,) * 8 + (True,) * 3 + (False,) * 5
    assert example.attention_mask == (True,) * 11 + (False,) * 5
    assert example.segment_ids == (0,) * 11 + (-1,) * 5
    assert all(offset is None for offset in example.offsets[10:])
    # A budget requesting a byte inside the emoji drops the complete codepoint.
    short = UTF8.frame("X🌌Y", "é", max_length=8, truncate_prompt=True)
    assert short.dropped_prompt_characters == 2
    start, end = short.prompt_span
    assert UTF8.decode(short.ids[start:end]) == "Y"
    assert short.offsets[start] == TokenOffset(2, 3, 5, 6)
    assert short.answer_span[1] == len(short.ids) == 6
    eos_only = UTF8.frame("", "")
    assert eos_only.ids == (1, 3, 2) and eos_only.loss_mask == (False, False, True)
    with pytest.raises(ValueError, match="explicit prompt truncation"):
        UTF8.frame("X🌌Y", "é", max_length=8)
    with pytest.raises(ValueError, match="answer"):
        UTF8.frame("X", "🌌", max_length=6, truncate_prompt=True)
    with pytest.raises(ValueError, match="pad_to"):
        UTF8.frame("X", "Y", pad_to=4)
    with pytest.raises(ValueError, match="pad_to"):
        UTF8.frame("X", "Y", pad_to=8, max_length=6)


def test_strict_inputs_specials_invalid_utf8_and_binding_rejection():
    assert UTF8.vocabulary_size == 260
    assert UTF8.special(SpecialToken.ANSWER) == 3
    assert UTF8.encode("\x00") == (4,)
    with pytest.raises(TypeError, match="SpecialToken"):
        UTF8.special("EOS")
    with pytest.raises(ValueError, match="unsupported"):
        LEGACY.special(SpecialToken.ANSWER)
    for text in ("\ud800", "\udfff", "x\ud800y"):
        with pytest.raises(ValueError, match="surrogate"):
            UTF8.encode(text)
    with pytest.raises(ValueError, match="unsupported symbol"):
        LEGACY.encode("é")
    for ids in ((0,), (1,), (2,), (3,), (260,), (-1,), (True,), (4.0,)):
        with pytest.raises(ValueError, match="content tokens"):
            UTF8.decode(ids)
    for raw in (b"\xff", b"\xc0\xaf", b"\xed\xa0\x80", b"\xf0\x9f", b"\x80"):
        with pytest.raises(ValueError, match="valid UTF-8"):
            UTF8.decode(tuple(byte + 4 for byte in raw))
    for tokenizer in (LEGACY, UTF8):
        tokenizer.require_binding(tokenizer.identity, tokenizer.vocabulary_size)
        with pytest.raises(ValueError, match="migration"):
            tokenizer.require_binding("f" * 64, tokenizer.vocabulary_size)
        with pytest.raises(ValueError, match="migration"):
            tokenizer.require_binding(tokenizer.identity, True)
    with pytest.raises(ValueError, match="migration"):
        UTF8.require_binding(LEGACY.identity, LEGACY.vocabulary_size)
    with pytest.raises(ValueError, match="max_tokens"):
        UTF8.encode("🌌", max_tokens=3)
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            UTF8.encode("", max_tokens=value)
    for value in (-1, True, None, MAX_TEXT_BYTES + 1):
        with pytest.raises(ValueError):
            UTF8.incremental_encoder(max_tokens=value)
    with pytest.raises(ValueError, match="safety limit"):
        UTF8.encode("x" * (MAX_TEXT_BYTES + 1))


def test_embedding_migration_preserves_rows_not_prediction_equivalence():
    import torch

    weight = torch.arange(LEGACY.vocabulary_size * 8, dtype=torch.float32).view(-1, 8)
    parent = weight.clone()
    recipe = legacy_to_utf8_migration(seed=91)
    rng_before = torch.get_rng_state().clone()
    migrated, receipt = migrate_tied_embedding(weight, recipe, parent_model_sha256="a" * 64)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert migrated.shape == (UTF8.vocabulary_size, 8)
    assert torch.equal(weight, parent)
    assert torch.equal(migrated[list(recipe.source_to_destination)], weight)
    repeated, repeated_receipt = migrate_tied_embedding(weight, recipe, parent_model_sha256="a" * 64)
    assert torch.equal(migrated, repeated) and receipt == repeated_receipt
    assert receipt["kind"] == "migration" and receipt["optimizer_policy"] == "reset-required"
    assert receipt["model_sha256"] != receipt["parent_model_sha256"]
    assert not receipt["prediction_equivalence"]
    assert len(receipt["embedding_sha256"]) == 64
    assert codec.digest(asdict(recipe)) == recipe.identity
    other, other_receipt = migrate_tied_embedding(
        weight, legacy_to_utf8_migration(seed=92), parent_model_sha256="a" * 64)
    assert not torch.equal(migrated, other)
    assert receipt["model_sha256"] != other_receipt["model_sha256"]
    for invalid in (replace(recipe, source_to_destination=tuple(range(LEGACY.vocabulary_size))),
                    replace(recipe, optimizer_policy="keep"), replace(recipe, seed=True),
                    replace(recipe, initialization_std=float("nan"))):
        with pytest.raises(ValueError):
            migrate_tied_embedding(weight, invalid, parent_model_sha256="a" * 64)
    with pytest.raises(ValueError, match="identity"):
        migrate_tied_embedding(weight, recipe, parent_model_sha256="old")
    with pytest.raises(ValueError, match="source-vocabulary"):
        migrate_tied_embedding(weight[1:], recipe, parent_model_sha256="a" * 64)
    broken = weight.clone()
    broken[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        migrate_tied_embedding(broken, recipe, parent_model_sha256="a" * 64)
