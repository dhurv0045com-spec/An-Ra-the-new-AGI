
import os
import sys
import torch
import torch.nn.functional as F
import pickle
import argparse

from startup_checks import assert_flash_sdp_ready

# Add necessary paths for importing custom modules
# Assuming anra_infer.py is in <repo_root>/inference/
# Add <repo_root> to sys.path for tokenizer module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add <repo_root>/history/neural_network (45B) to sys.path for causal_transformer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'history', 'neural_network (45B)')))

from tokenizer.char_tokenizer import CharTokenizer
from causal_transformer import CausalTransformer

# --- Configuration (must match training) ---
# These parameters are hardcoded to match the training setup in build_anra_brain.py
# If training parameters change, these should be updated.
BLOCK_SIZE = 128
N_EMBD = 256
N_HEAD = 4
N_LAYER = 4

def load_model_and_tokenizer(model_checkpoint_path, tokenizer_path):
    '''Loads the trained model and tokenizer.'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tokenizer
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print(f"Tokenizer loaded. Vocabulary size: {tokenizer.vocab_size}")

    # Initialize model
    model = CausalTransformer(tokenizer.vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE)

    # Load model checkpoint
    if not os.path.exists(model_checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_checkpoint_path}")
    model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print(f"Model loaded from {model_checkpoint_path}. Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    return model, tokenizer, device

def generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8) -> str:
    '''
    Generates text autoregressively from a given prompt.
    Args:
        model: The trained CausalTransformer model.
        tokenizer: The character tokenizer.
        device: The device (e.g., 'cuda' or 'cpu') to run inference on.
        prompt: The initial text prompt.
        max_new_tokens: The maximum number of new tokens to generate.
        temperature: Controls randomness (higher = more random).
    Returns:
        The generated text including the prompt.
    '''
    # Encode the prompt
    encoded_prompt = tokenizer.encode(prompt)
    if not encoded_prompt:
        # Handle cases where prompt contains only unknown characters
        print("Warning: Prompt contains no known characters. Starting generation from empty sequence.")
        idx = torch.empty(1, 0, dtype=torch.long, device=device)
    else:
        idx = torch.tensor(encoded_prompt, dtype=torch.long, device=device).unsqueeze(0) # (1, T)

    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx if idx.size(1) <= BLOCK_SIZE else idx[:, -BLOCK_SIZE:]

            # Get predictions
            logits, _ = model(idx_cond) # (B, T, vocab_size)

            # Focus only on the last time step
            logits = logits[:, -1, :] # (B, vocab_size)

            # Apply temperature for sampling
            if temperature == 0: # Greedy sampling
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.argmax(probs, dim=-1, keepdim=True) # (B, 1)
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # Append the predicted token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
            generated_tokens.append(idx_next.item())

            # Decode the newly generated token and check for stop conditions
            # For simplicity, we're not adding explicit stop conditions here,
            # but one could check for specific tokens like "\n\n" or EOF markers.

    return tokenizer.decode(idx[0].tolist())

if __name__ == '__main__':
    assert_flash_sdp_ready("inference.anra_infer")
    parser = argparse.ArgumentParser(description="An-Ra CausalTransformer Inference Pipeline.")
    parser.add_argument('--prompt', type=str, required=True,
                        help='The initial text prompt for generation.')
    parser.add_argument('--max_new_tokens', type=int, default=100,
                        help='Maximum number of new tokens to generate.')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature. 0 for greedy, higher for more randomness.')
    parser.add_argument('--model_checkpoint', type=str, default='anra_brain.pt',
                        help='Path to the trained model checkpoint relative to repo root.')
    parser.add_argument('--tokenizer_file', type=str, default='tokenizer.pkl',
                        help='Path to the saved tokenizer file relative to repo root.')

    args = parser.parse_args()

    # Adjust paths relative to the current script's location for loading
    script_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))

    abs_model_checkpoint_path = os.path.join(repo_root, args.model_checkpoint)
    abs_tokenizer_file_path = os.path.join(repo_root, args.tokenizer_file)

    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(
        abs_model_checkpoint_path,
        abs_tokenizer_file_path
    )

    # Generate text
    print()
    print(f"--- Generating with prompt: '{args.prompt}' ---")
    generated_text = generate(model, tokenizer, device,
                              prompt=args.prompt,
                              max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature)
    print()
    print("--- Generated Text ---")
    print(generated_text)
