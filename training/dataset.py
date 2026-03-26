"""
dataset.py — Step 21: Dataset Loader
=====================================
Loads and preprocesses real text data for language model training.
Uses HuggingFace datasets (WikiText-103) with efficient streaming,
tokenization, batching, padding, truncation, and shuffling.
Memory-efficient: streams large corpora without loading all into RAM.
"""

import os
import math
import logging
from typing import Optional, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

# HuggingFace ecosystem — best-in-class for LM data pipelines
from datasets import load_dataset, DatasetDict
from transformers import GPT2TokenizerFast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenizer singleton — reuse across dataset instances
# ---------------------------------------------------------------------------

def get_tokenizer(vocab_size: int = 50257) -> GPT2TokenizerFast:
    """
    Returns a GPT-2 BPE tokenizer (50k vocab, proven on English text).
    Sets pad token to eos token — standard practice for decoder-only LMs.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# Core dataset — packed token sequences (no padding waste)
# ---------------------------------------------------------------------------

class PackedTextDataset(Dataset):
    """
    Converts a raw text corpus into fixed-length packed token sequences.

    Strategy: concatenate all tokens with EOS separators, then chunk into
    blocks of `seq_len`. Zero padding waste. Maximally efficient.
    Used by GPT-2, LLaMA, and most production LMs.

    Args:
        texts:    List of raw text strings.
        tokenizer: HuggingFace tokenizer.
        seq_len:  Context window length (tokens per sample).
        stride:   Step between windows. seq_len = no overlap (default).
    """

    def __init__(
        self,
        texts: list[str],
        tokenizer,
        seq_len: int = 512,
        stride: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.stride = stride or seq_len  # non-overlapping by default
        self.tokenizer = tokenizer

        logger.info(f"Tokenizing {len(texts):,} documents...")
        # Concatenate all text with EOS boundary markers
        full_text = tokenizer.eos_token.join(texts)
        tokens = tokenizer.encode(full_text, add_special_tokens=False)
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        # Compute number of complete windows
        self.n_samples = max(0, (len(self.tokens) - seq_len) // self.stride)
        logger.info(
            f"Total tokens: {len(self.tokens):,} → "
            f"{self.n_samples:,} samples at seq_len={seq_len}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.stride
        # input_ids: tokens[start : start+seq_len]
        # labels:    tokens[start+1 : start+seq_len+1]  (next-token prediction)
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


# ---------------------------------------------------------------------------
# Streaming dataset — for corpora too large to tokenize upfront
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for very large corpora.
    Yields packed token blocks on-the-fly without materializing all tokens.

    Suitable for OpenWebText, C4, or any multi-GB corpus.

    Args:
        hf_dataset: A HuggingFace IterableDataset.
        tokenizer:  HuggingFace tokenizer.
        seq_len:    Context window length.
        text_col:   Name of the text column in the dataset.
    """

    def __init__(self, hf_dataset, tokenizer, seq_len: int = 512, text_col: str = "text"):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.text_col = text_col
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buffer: list[int] = []
        for example in self.dataset:
            # Tokenize one document at a time
            ids = self.tokenizer.encode(
                example[self.text_col], add_special_tokens=False
            )
            buffer.extend(ids)
            buffer.append(self.eos_id)  # document boundary

            # Yield complete windows as they fill
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": x, "labels": y}
                buffer = buffer[self.seq_len :]  # advance by seq_len


# ---------------------------------------------------------------------------
# High-level factory — picks dataset and returns DataLoaders
# ---------------------------------------------------------------------------

class LMDataModule:
    """
    Complete data pipeline: load → tokenize → split → DataLoaders.

    Dataset choice: WikiText-103
    - Clean Wikipedia text, widely used LM benchmark
    - 103M train tokens — small enough to prototype, big enough to learn
    - Pre-split train/validation/test
    - Upgrade path: swap `dataset_name` to "openwebtext" for 40B tokens

    Args:
        dataset_name:  HuggingFace dataset identifier.
        seq_len:       Context window (tokens).
        batch_size:    Sequences per batch.
        num_workers:   DataLoader worker processes.
        cache_dir:     Where HuggingFace caches downloads.
        streaming:     Use streaming mode (for huge datasets).
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        seq_len: int = 512,
        batch_size: int = 8,
        num_workers: int = 2,
        cache_dir: str = "./data_cache",
        streaming: bool = False,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.streaming = streaming

        self.tokenizer = get_tokenizer()
        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def setup(self):
        """Download (if needed) and prepare all dataset splits."""
        logger.info(f"Loading dataset: {self.dataset_name}/{self.dataset_config}")

        raw = load_dataset(
            self.dataset_name,
            self.dataset_config,
            cache_dir=self.cache_dir,
            streaming=self.streaming,
        )

        if self.streaming:
            # Streaming: wrap each split in StreamingTextDataset
            self._train_ds = StreamingTextDataset(
                raw["train"], self.tokenizer, self.seq_len
            )
            self._val_ds = StreamingTextDataset(
                raw["validation"], self.tokenizer, self.seq_len
            )
            self._test_ds = StreamingTextDataset(
                raw["test"], self.tokenizer, self.seq_len
            )
        else:
            # Eager: tokenize everything up front — fast iteration during training
            def get_texts(split):
                # Filter empty strings (WikiText has section headers as empty lines)
                return [t for t in raw[split]["text"] if t.strip()]

            self._train_ds = PackedTextDataset(
                get_texts("train"), self.tokenizer, self.seq_len
            )
            self._val_ds = PackedTextDataset(
                get_texts("validation"), self.tokenizer, self.seq_len
            )
            self._test_ds = PackedTextDataset(
                get_texts("test"), self.tokenizer, self.seq_len
            )

        logger.info("Dataset setup complete.")

    def train_loader(self) -> DataLoader:
        """Shuffled training DataLoader."""
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=(not self.streaming),  # IterableDataset can't be shuffled by loader
            num_workers=self.num_workers,
            pin_memory=True,      # speeds up CPU→GPU transfer
            drop_last=True,       # avoid ragged final batch
            persistent_workers=(self.num_workers > 0),
        )

    def val_loader(self) -> DataLoader:
        """Validation DataLoader — no shuffle."""
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )

    def test_loader(self) -> DataLoader:
        """Test DataLoader — no shuffle."""
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# Collate function for variable-length sequences (optional, not needed for packed)
# ---------------------------------------------------------------------------

def collate_with_padding(batch: list[dict], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """
    Pads sequences in a batch to the same length.
    Only needed when sequences have variable length (not used in packed mode).
    """
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  # -100 = ignore in CE loss
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
        attention_mask[i, :L] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Step 21: Dataset Loader — self test")
    print("=" * 60)

    dm = LMDataModule(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",  # tiny version for quick test
        seq_len=128,
        batch_size=4,
        num_workers=0,
    )
    dm.setup()

    loader = dm.train_loader()
    batch = next(iter(loader))

    print(f"\nVocab size:       {dm.vocab_size:,}")
    print(f"Train samples:    {len(dm._train_ds):,}")
    print(f"Val samples:      {len(dm._val_ds):,}")
    print(f"Batch input_ids:  {batch['input_ids'].shape}")
    print(f"Batch labels:     {batch['labels'].shape}")
    print(f"Sample tokens:    {batch['input_ids'][0, :16].tolist()}")
    print(f"Sample decoded:   {dm.tokenizer.decode(batch['input_ids'][0, :32])!r}")

    # Benchmark loader throughput
    t0 = time.perf_counter()
    n_batches = 20
    for i, b in enumerate(loader):
        if i >= n_batches:
            break
    elapsed = time.perf_counter() - t0
    tokens_per_sec = n_batches * 4 * 128 / elapsed
    print(f"\nThroughput: {tokens_per_sec:,.0f} tokens/sec over {n_batches} batches")
    print("\n✓ dataset.py OK")
