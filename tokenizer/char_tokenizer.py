
import os

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, s):
        # Skip characters not in vocabulary to avoid KeyError
        return [self.char_to_int[ch] for ch in s if ch in self.char_to_int]

    def decode(self, l):
        return ''.join([self.int_to_char[i] for i in l])
