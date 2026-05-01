from __future__ import annotations

import pytest
import torch

from anra_brain import CausalTransformerV2
from anra_paths import DATASET
from tokenizer.subword_tokenizer import SubwordTokenizer
from training.v2_data_mix import IdentityStyleFilter, build_v2_training_examples


def test_v2_model_forward_shape() -> None:
    model = CausalTransformerV2(vocab_size=128, n_embd=96, n_head=6, n_layer=2, block_size=32)
    x = torch.randint(0, 128, (2, 32))
    logits, loss = model(x, x)
    assert logits.shape == (2, 32, 128)
    assert loss is not None


def test_identity_style_filter_rewrites_robotic_phrasing() -> None:
    filt = IdentityStyleFilter()
    cleaned = filt.clean("As an AI language model, I can help.")
    assert "ai language model" not in cleaned.lower()
    assert "an-ra" in cleaned.lower()


def test_v2_mix_keeps_own_data_dominant() -> None:
    examples, report = build_v2_training_examples(dataset_path=DATASET, max_examples=400)
    own = report.realized_counts.get("own", 0)
    identity = report.realized_counts.get("identity", 0)
    total = report.total_examples
    assert total > 0
    assert (own + identity) / total >= 0.75


def test_no_direct_legacy_dataset_path_usage() -> None:
    legacy = "/content/drive/MyDrive/AnRa/" + "anra_dataset_v6_1.txt"
    allowed = {"anra_paths.py"}
    for path in Path(".").rglob("*.py"):
        rel = path.as_posix()
        if rel in allowed:
            continue
        assert legacy not in path.read_text(encoding="utf-8", errors="replace"), rel


def test_subword_tokenizer_roundtrip() -> None:
    pytest.importorskip("tokenizers")
    tok = SubwordTokenizer.train_from_texts(
        ["H: Hello\nANRA: I am An-Ra.", "H: What is your purpose?\nANRA: To grow carefully."],
        vocab_size=64,
    )
    ids = tok.encode("H: Hello\nANRA: I am An-Ra.", add_special_tokens=True)
    text = tok.decode(ids)
    assert "an-ra" in text.lower()
