from __future__ import annotations

from anra_paths import TRAINING_DATA_DIR, TOKENIZER_DIR
from tokenizer.subword_tokenizer import SubwordTokenizer

SPECIAL_TOKENS = [
    "<unk>", "<pad>", "<bos>", "<eos>",
    "<system>", "</system>", "<user>", "</user>",
    "<assistant>", "</assistant>", "<tool>", "</tool>", "<think>",
]


def main() -> None:
    dataset = TRAINING_DATA_DIR / "anra_dataset_v6_1.txt"
    texts = dataset.read_text(encoding="utf-8", errors="ignore").splitlines()
    tok = SubwordTokenizer.train_from_texts(
        texts,
        vocab_size=8192,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    out = TOKENIZER_DIR / "tokenizer_v3.json"
    tok.save(out)
    print(f"saved: {out} vocab={tok.vocab_size}")


if __name__ == "__main__":
    main()
