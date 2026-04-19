import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import pickle
import re
from contextlib import contextmanager
from typing import Dict, Any, List

# Add current directory to Python path to enable importing local modules like 'tokenizer'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
# Add the neural_network directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'history', 'neural_network (45B)'))

from causal_transformer import CausalTransformer  # Assuming this is the file
from tokenizer.char_tokenizer import CharTokenizer  # Import CharTokenizer from the new module


class OuroborosReasoner:
    """Training-time recursive reasoning wrapper with compatibility helpers."""

    def __init__(self, model: nn.Module, passes: int = 3):
        self.model = model
        self.passes = passes
        self.active = False
        self._last_entropy_profile: List[float] = []

    @contextmanager
    def recursive_context(self):
        previous = self.active
        self.active = True
        try:
            yield
        finally:
            self.active = previous

    def forward_with_passes(self, batch_input: torch.Tensor):
        running_logits = None
        entropy_profile: List[float] = []
        for pass_idx in range(self.passes):
            logits, _ = self.model(batch_input)
            temp = 1.0 + 0.2 * pass_idx
            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
            entropy = float((-probs * torch.log(torch.clamp(probs, 1e-9, 1.0))).sum(dim=-1).mean().item())
            entropy_profile.append(entropy)
            if running_logits is None:
                running_logits = logits
            else:
                # Progressive residual blending to preserve pass diversity
                alpha = 1.0 / (pass_idx + 1)
                running_logits = (1.0 - alpha) * running_logits + alpha * logits
        self._last_entropy_profile = entropy_profile
        return running_logits

    def get_config(self) -> Dict[str, Any]:
        return {
            'passes': self.passes,
            'recursive_context_enabled': self.active,
            'last_entropy_profile': self._last_entropy_profile,
        }


class TextDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x is the input sequence, y is the target sequence (shifted by one)
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


def hard_sample_detector(batch_text: str) -> bool:
    """Detect whether >30% of samples in the batch require recursive reasoning."""
    samples = [s.strip() for s in batch_text.split('\n') if s.strip()]
    if not samples:
        return False

    hard_count = 0
    math_pattern = re.compile(r"\d+\s*[-+*/=^%<>]+\s*\d+|\d+[\-+*/=^%<>]+")
    logic_pattern = re.compile(r"\b(if|then|therefore|prove|because|since|implies)\b", re.IGNORECASE)
    code_pattern = re.compile(r"\b(def\s+|for\s+|while\s+|class\s+)")
    question_pattern = re.compile(r"\b(why|how|explain|analyze)\b", re.IGNORECASE)

    for sample in samples:
        is_hard = (
            bool(math_pattern.search(sample))
            or bool(logic_pattern.search(sample))
            or bool(code_pattern.search(sample))
            or bool(question_pattern.search(sample))
        )
        if is_hard:
            hard_count += 1

    return (hard_count / max(len(samples), 1)) > 0.30


# --- Main Training Function ---
def train_anra_brain(
    data_path,
    epochs=10,
    batch_size=32,
    block_size=128,  # Context length for transformer
    learning_rate=1e-4,
    checkpoint_path='anra_brain.pt'
):
    print()
    print("--- Starting Training ---")
    print("Training data source: {}".format(data_path))
    print("Epochs: {}".format(epochs))
    print("Batch size: {}".format(batch_size))
    print("Block size (context length): {}".format(block_size))
    print("Learning rate: {}".format(learning_rate))
    print("Checkpoint path: {}".format(checkpoint_path))

    # 1. Load Data
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print("Loaded {} characters from {}".format(len(text), data_path))

    # 2. Tokenize (simple char-level for demo)
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print("Vocabulary size: {}".format(vocab_size))

    # 3. Create Dataset and DataLoader
    dataset = TextDataset(encoded_text, block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataset size: {} samples".format(len(dataset)))

    # 4. Initialize Model
    n_embd = 256  # Embedding dimension
    n_head = 4    # Number of attention heads
    n_layer = 4   # Number of transformer layers

    model = CausalTransformer(vocab_size, n_embd, n_head, n_layer, block_size)
    ouroboros = OuroborosReasoner(model, passes=3)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("Using device: {}".format(device))
    print("Model parameters: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))

    # 5. Load Checkpoint if exists
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint from {}".format(checkpoint_path))
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            model.load_state_dict(ckpt)
        else:
            print("Checkpoint format detected as non-standard, attempting direct load")
            model.load_state_dict(ckpt)
        print("Checkpoint loaded successfully. Continuing training.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    # 6. Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()  # Standard for next-token prediction

    # 7. Training Loop
    print("Training for {} epochs...".format(epochs))
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_batches = 0
        ouroboros_batches_count = 0
        from tqdm import tqdm  # Import tqdm for progress bar
        for batch_idx, (xb, yb) in enumerate(tqdm(dataloader, desc="Epoch {}/{}".format(epoch + 1, epochs))):
            xb, yb = xb.to(device), yb.to(device)
            total_batches += 1

            decoded_batch = [tokenizer.decode(row.tolist()) for row in xb.detach().cpu()]
            batch_text = "\n".join(decoded_batch)

            if hard_sample_detector(batch_text):
                with ouroboros.recursive_context():
                    logits = ouroboros.forward_with_passes(xb)
                    b, t, c = logits.shape
                    loss = loss_fn(logits.view(b * t, c), yb.view(b * t))
                    ouroboros_batches_count += 1
            else:
                logits, loss = model(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        ouroboros_pct = (100.0 * ouroboros_batches_count / max(total_batches, 1))
        print("Epoch {}/{}, Average Loss: {:.4f}".format(epoch + 1, epochs, avg_loss))
        print(
            "Ouroboros activated: {} batches ({:.2f}% of training) — recursive reasoning applied".format(
                ouroboros_batches_count, ouroboros_pct
            )
        )

    # 8. Save Checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'ouroboros_config': ouroboros.get_config(),
        'ouroboros_batches': ouroboros_batches_count,
    }
    torch.save(checkpoint, checkpoint_path)
    print()
    print("Training finished. Model checkpoint saved to {}".format(checkpoint_path))

    # Save tokenizer after training completes
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train An-Ra's brain (CausalTransformer).")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the combined training data file (e.g., combined_identity_data.txt).')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--block_size', type=int, default=128,
                        help='Context length for the transformer.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--checkpoint_path', type=str, default='anra_brain.pt',
                        help='Path to save/load the model checkpoint.')

    args = parser.parse_args()

    train_anra_brain(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path
    )
