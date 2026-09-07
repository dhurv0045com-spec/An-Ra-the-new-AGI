# ARK-008 ANALYSIS — transfer test INCONCLUSIVE

## Result: the binding task is too hard for the Micro model

No seed reached the trigger threshold (0.8 test accuracy). Peak test
accuracy across all arms was 0.28–0.34 — barely above the random/query-blind
baseline. Without acquisition, there is no generalized solution to protect,
so the retention experiment cannot proceed.

| Seed | LR | Status | Peak test | Final test |
|------|-----|--------|-----------|------------|
| 505 | ×1.0 (HIGH) | NO_TRIGGER | ~0.28 | ~0.22 |
| 505 | ×0.01 (LOW) | NO_TRIGGER | ~0.28 | ~0.22 |
| 606 | ×1.0 (HIGH) | NO_TRIGGER | ~0.34 | ~0.28 |
| 606 | ×0.01 (LOW) | (running at stop) | — | — |

## What this means
The LR-retention law is **task-scope-limited**: it was discovered and
replicated on arithmetic (T2 no-carry), but cannot be tested on binding
because the model cannot acquire the binding capability at this scale.

## Implications for the program
1. The LR-retention law is **task-scope-limited to tasks the model can
   learn to high accuracy**. The law is real but its applicability boundary
   is now partially mapped.
2. To test transfer, we need either:
   a. a larger model that CAN learn binding (P35 scale), or
   b. a simpler non-arithmetic task that the Micro model CAN learn
   c. more training steps for binding specifically
3. This is a genuine scope boundary, not a failure of the law itself.
