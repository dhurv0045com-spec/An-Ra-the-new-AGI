from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anra_paths import TOKENIZER_DIR, get_dataset_file
from tokenizer.subword_tokenizer import SubwordTokenizer
from training.v2_config import CANONICAL_SPECIAL_TOKENS, CANONICAL_VOCAB_SIZE

SPECIAL_TOKENS = CANONICAL_SPECIAL_TOKENS
TARGET_VOCAB = CANONICAL_VOCAB_SIZE


def _bootstrap_texts() -> list[str]:
    return [
        "H: Who are you?\nANRA: I am An-Ra. I reason, remember, and improve carefully.",
        "H: Write code.\nANRA: <code>def answer(x):\n    return x + 1\n</code>",
        "H: Think through the task.\nANRA: <think>Break the problem into verifiable steps.</think>",
        "H: Track state.\nANRA: <ESV:v> <ESV:a> <ESV:d> calm focus curiosity",
    ] * 256


def main() -> None:
    dataset = get_dataset_file()
    if dataset.exists():
        corpus = dataset.read_text(encoding="utf-8", errors="ignore")
        texts = [corpus, *corpus.splitlines(), *_bootstrap_texts()]
    else:
        texts = _bootstrap_texts()
    tok = SubwordTokenizer.train_from_texts(
        texts,
        vocab_size=TARGET_VOCAB,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    out = TOKENIZER_DIR / "tokenizer_v3.json"
    tok.save(out)
    print(f"saved: {out} vocab={tok.vocab_size}")


if __name__ == "__main__":
    main()
