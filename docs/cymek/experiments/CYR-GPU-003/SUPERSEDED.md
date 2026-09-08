# CYR-GPU-003 — SUPERSEDED_BEFORE_EXECUTION

CYR-GPU-003 never produced a scientific result. The Colab notebook contained
13 confirmed defects (see PREEXECUTION_AUDIT.md in CYR-GPU-004). The most
critical were:

1. Notebook defined its own Transformer instead of using Cymek V5
2. `detect_g90()` and `minutes_left()` were called but never defined
3. PLAN claimed 35.4M params but notebook built ~8M with different architecture
4. HIGH/LOW forks consumed different minibatch sequences
5. No frozen SHA checkout; cloned mutable branch

No training was executed. No results exist. The experiment identity is
invalid. CYR-GPU-004 is the replacement.
