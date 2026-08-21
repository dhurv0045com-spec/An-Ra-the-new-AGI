"""Tensor-level proofs of the SFT label alignment (the off-by-one must never return).

The first SFT run supervised position j with ids[j] instead of ids[j+1]:
it taught "what follows the first answer token" (loss -> 0) while never
teaching "start the answer from context". Greedy decoding emitted whitespace.
These tests pin the exact semantics against the real tokenizer:

  - the last prompt token's label IS the first completion token (binding step);
  - completion token i's label is completion token i+1;
  - the final completion token's label is EOS;
  - every earlier position is masked (-100);
  - ids remain decodable to prompt+completion (no drift).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from anra_core.tokenizer import V4Tokenizer  # noqa: E402
from training.sft_context_binding import encode_item  # noqa: E402

TOKENIZER = V4Tokenizer.load(
    Path(__file__).parents[1] / "anra_core" / "assets" / "tokenizer_v4_32k.json")


def test_label_alignment_single_and_multi_token_completion() -> None:
    item = {"prompt": "<k>The code is GXT-412.</k>\n<q>State the code.</q>\n<answer>",
            "completion": " GXT-412."}
    ids_t, labels_t = encode_item(TOKENIZER, item)
    ids, labels = ids_t[0].tolist(), labels_t[0].tolist()

    prompt_ids = TOKENIZER.encode(item["prompt"])
    completion_ids = TOKENIZER.encode(item["completion"])
    last_prompt_index = len(prompt_ids)

    # Structure: [bos, prompt..., completion..., eos]
    assert ids == ([TOKENIZER.bos_token_id, *prompt_ids, *completion_ids,
                    TOKENIZER.eos_token_id])

    # Binding step: last prompt token supervises the FIRST completion token.
    assert labels[last_prompt_index] == completion_ids[0]

    # Chain: completion token i supervises completion token i+1.
    for k in range(len(completion_ids) - 1):
        assert labels[last_prompt_index + 1 + k] == completion_ids[k + 1]

    # EOS: the final completion token supervises EOS.
    assert labels[last_prompt_index + len(completion_ids)] == TOKENIZER.eos_token_id

    # Everything before the binding step is masked; nothing beyond EOS.
    assert all(v == -100 for v in labels[:last_prompt_index])
    assert len(labels) == len(ids)
    supervised = [v for v in labels if v != -100]
    assert supervised == [*completion_ids, TOKENIZER.eos_token_id]

    # Round trip: the decoded sequence is prompt + completion.
    assert TOKENIZER.decode(ids[1:-1]) == item["prompt"] + item["completion"]


def test_label_alignment_tiny_single_token_completion() -> None:
    item = {"prompt": "Value:", "completion": " 42"}
    ids_t, labels_t = encode_item(TOKENIZER, item)
    ids, labels = ids_t[0].tolist(), labels_t[0].tolist()
    completion_ids = TOKENIZER.encode(item["completion"])
    assert len(completion_ids) >= 1
    assert labels[len(TOKENIZER.encode(item["prompt"]))] == completion_ids[0]
    assert labels[-2] == TOKENIZER.eos_token_id or labels[-1] == -100
    supervised = [v for v in labels if v != -100]
    assert supervised == [*completion_ids, TOKENIZER.eos_token_id]


def test_first_sft_run_bug_would_fail_these() -> None:
    """The original buggy labels were ids-aligned, not shifted: they placed
    completion_ids[j] at the position of completion_ids[j] itself. Under the
    fixed semantics that arrangement supervises the WRONG transitions; this
    test documents that the bug is detectable, not latent."""
    item = {"prompt": "q:", "completion": " abcdef"}
    ids_t, labels_t = encode_item(TOKENIZER, item)
    ids, labels = ids_t[0].tolist(), labels_t[0].tolist()
    completion_ids = TOKENIZER.encode(item["completion"])
    last_prompt_index = len(TOKENIZER.encode(item["prompt"]))
    buggy_positions = range(last_prompt_index + 1,
                            last_prompt_index + 1 + len(completion_ids))
    for j in buggy_positions:
        # In the buggy version labels[j] equalled ids[j] (the token AT j),
        # which under causal semantics supervises the transition INTO j+1
        # with the wrong target. The fixed labels must never do that.
        assert labels[j] != ids[j] or j == last_prompt_index + len(completion_ids)
