# V4 Behavior Baseline

This is a small deterministic reference, not a benchmark and not evidence of
general intelligence. It was recorded from the strict-loaded SFT checkpoint:
whole-file SHA-256 `cfbcbd4611f9e257f07b5dfc41d028ef096e2126b888a34148966fa0062e1738`,
parameter SHA-256 `23d0c5005a7c741f19ddc4ab50ac9f4b7aa8554e047446c521267d6ab900d16a`,
step 5000. CPU FP32 greedy decoding used eight generated tokens.

| Prompt | Completion |
| --- | --- |
| `The capital of France is` | ` Paris, so I'` |
| `Reply with one word: hello` | `, and another is ` |
| `17 + 25 =` | ` 0.5 * (` |
| `def add(a, b):` | newline then `    if a == ` |
| `Miso is a cat. All cats are animals. Miso is` | ` a cat that has` |

Interpretation: the checkpoint has recognizable language/code continuation,
but this baseline does **not** show reliable instruction following or exact
arithmetic. Core cleanup preserves this behavior; it does not create capability
that training did not produce. Any neural or post-training change must compare
against this fixed reference as well as numerical and execution contracts.
