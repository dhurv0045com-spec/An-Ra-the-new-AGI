# CS-TRANSFER-001 — physical class-space causal transfer

Status: **IMPLEMENTED / NOT EXECUTED**.

Branch: `cymek-cs-transfer-001`

This experiment exists to answer one narrow question left open by R1C and V5.1 Canary-v2:

> Does the *actual number of rows in the tied embedding/output matrix* causally change formation, especially identity/copy, when paired models consume exactly the same integer token sequences and start from byte-identical shared weights?

## Why this is the next experiment

R1C fixed the physical model vocabulary at 24,576 and changed which output classes participated in training. Its result, `SOFTMAX_COMPETITION_NOT_SUFFICIENT`, ruled out inactive-class competition as a sufficient explanation. It did **not** test a physically smaller tied matrix.

Canary-v2 then extended V5.1 Rung-A exposure from 120 to 360 updates. Overall development exact+valid-EOS rose to about 0.519 and sealed to about 0.517, while identity/copy remained below the frozen threshold at about 0.177 development / 0.189 sealed. Broad formation therefore improved, but the identity/copy bottleneck remained.

Those two results make another masking study low value. The remaining causal question is physical class-space geometry.

## Primary contrast

Both arms are 8-layer, width-256 V5 dense decoders with the same attention, FFN, norms, RoPE, objective, optimizer, schedule, stream, and seed pairing.

- `PHYS_4096`: physical vocabulary 4,096, 8,920,320 parameters.
- `PHYS_24576`: physical vocabulary 24,576, 14,163,200 parameters.

The 5,242,880-parameter difference is exactly `(24576 - 4096) * 256`: the extra tied embedding/output rows. We intentionally do **not** compensate by widening the small model, because that would introduce a second architectural intervention.

A same-seed constructor is not sufficient matching: the different embedding shape consumes a different number of RNG draws and would change later block weights. `cs_transfer_001_model.build_matched_pair()` therefore initializes the full model once and copies every non-embedding tensor and embedding rows 0..4095 byte-for-byte into the small model.

## Shared token surface

The frozen production 24,576-entry tokenizer is used to render a fresh controlled task surface. Before training, examples are retained only if every prompt and answer content ID is in `4..4095`. The exact selected token rows are hash-bound and both arms consume those same integer sequences.

This is deliberate: changing the tokenizer itself would simultaneously change segmentation, sequence length, token frequencies, embedding lookup identities, and physical output size. CS-TRANSFER-001 isolates the physical matrix first. A later experiment can test tokenizer/segmentation transfer if this contrast is informative.

The surface contains identity, binding, state/order, composition, termination, and missing-information families. It uses fresh latent worlds, group-level split isolation, cross-split contamination checks, deterministic hash-ranked selection, and simple shortcut attacks before any GPU update.

## Endpoint and evidence

Four matched seed pairs are mandatory. Each pair executes both physical-vocabulary arms for exactly 480 updates / 1,966,080 real tokens. Development evaluation is fixed at updates 0, 60, 120, 180, 240, 300, 360, 420, and 480. The primary statistic is the paired identity formation-AUC gap `PHYS_4096 - PHYS_24576`; the endpoint identity gap is co-primary.

Sealed rows are generated and hash-bound before training but consumed once only after all eight arms complete and the development aggregate is frozen.

## Interpretation boundary

A positive result would support a physical class-space effect on this controlled formation problem. It would **not** prove that a 4,096-token production tokenizer is better, because tokenizer segmentation is held fixed by selecting a shared low-ID surface. The next step after a supported result is replication at the ~40M development rung and then a separate tokenizer/representation transfer experiment.

A null result is also valuable: it would substantially reduce the priority of physical vocabulary size as the explanation for Canary-v2's identity bottleneck and redirect work toward identity objective/data structure, EOS/termination, or segmentation.

No result from CS-TRANSFER-001 authorizes PRE500M, 250M, 500M, cognition, or AGI claims.

## Operator boundary

The dedicated Drive root is:

`/content/drive/MyDrive/CYMEK/CS_TRANSFER_001`

Do not point it at Canary-v2, R1C, or ARK roots. Do not execute GPU arms until the scientific executable is frozen, the CPU qualification suite passes, the real CUDA preflight passes for both physical arms, and the operator notebook pins the exact executable commit.
