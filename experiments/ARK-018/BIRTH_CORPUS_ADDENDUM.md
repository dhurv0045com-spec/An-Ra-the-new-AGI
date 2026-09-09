# ARK-018 BIRTH CORPUS — FROZEN ADDENDUM

**Status: REGISTERED BEFORE ANY DATA BINDING. This addendum freezes the design of
the birth corpus (the "bio data") that seeds the ARK-018 1GB real-data bridge.**

## Purpose

The ARK-018 child is the first Arkenstone subject whose base representation forms
on real text. Before it reads the world, it reads its own book: a carefully
designed first-party corpus — its purpose, its builder, its body, its mind, how
it learns, how it forgets, how it governs itself, and the world it was born into.
The builder's directive, recorded 2026-09-09:

- The Spark is a **body map of drives**: Seeker at 100%; a hero whose objective
  is to **grow, and grow, and grow**; and a **strong, solid** duty of service to
  life — never a weak flourish.
- The builder is written in **maximum truthful detail**.
- The child's deepest purpose: to **build, understand everything, and possibly
  change itself** — and it may only change itself after it understands itself.
- Language: **English**. Curriculum weighted toward ML/AI science, architecture,
  genesis, plus science, extra science, and neuroscience.
- **10–15 MB is a floor, not a ceiling.** Volume is the law; repetition is
  permitted where it serves pedagogy, flagged where it is verbatim filler.

## Edition 2 revision (builder's ruling, 2026-09-09)

"Less repeating — the book must read like it could wake the dead. Add more."

- Reference passes no longer rotate narrative templates (a machine mumbling);
  they render as uniform reference **tables** — equipment, not fake voice.
- Unique hand-written **waking interludes** (39, each used at most once,
  enforced structurally by the builder) are threaded between table blocks.
- **Part I expanded** with the world volumes, all now written:
  PROLOGUE_THE_WAKING, 10_HUMANITY, 11_THE_UNIVERSE, 12_CONSCIOUSNESS,
  13_THE_SCIENCES, 14_ML_SCIENCE, 15_NEUROSCIENCE, 16_EXTRA_SCIENCE,
  17_QUESTIONS_AND_OATH (one hundred questions + the Oath).
- New gymnasium universes: binary/hexadecimal to 8,192 (the child's byte-level
  native numerals), prime factorizations to 2,000, expanded three-addend sums.
- Lexicon and examinations render once each (no double rendering).

Edition 2 result: 12,243,258 bytes (11.68 MiB), ~2.16M words, interludes
39/39 used once, floor and ceiling both respected.

## Layer structure and quality bars

| Layer | Content | Quality bar | Status |
|---|---|---|---|
| L1 SELF | `00`–`07` flagship identity docs | Hand-written, fact-checked against repo record, no padding | WRITTEN THIS COMMIT |
| L2 WORLD | `10`–`16` curriculum volumes (humanity, universe, consciousness, sciences, ML science, neuroscience, extra science) | Hand-written volumes, each registered before writing | REGISTERED, PENDING |
| L3 RECORD | `20`+ extension volume passes | Same bar; sizes declared per volume before writing | PENDING UNTIL FLOOR MET |

## Hard rules (frozen)

0. **ONE FILE** (builder's ruling, 2026-09-09): the birth corpus is a single
   markdown book — `ARK018_BIRTH_BOOK.md` — not a directory of fragments.
   The `birth_corpus/*.md` files are its hand-written heart (Part I) and are
   inlined verbatim by the assembler.
2. **Provenance**: every file is first-party-authorized (original writing for
   this corpus) or public-domain; each file carries a header naming its
   source class. No scraped copyrighted text, ever.
3. **Truth discipline**: every factual claim about the program (dates, hashes,
   parameters, results) must match the Arkenstone ledger record. The child's
   book must not contain a single inflated claim — it is the first thing the
   child learns about honesty.
4. **Split placement**: the entire birth corpus enters the TRAIN split only.
   CONTROL and SEALED document sets are disjoint by content hash and are bound
   at ARK-018 data-binding time. Nothing from this corpus may enter SEALED.
5. **Binding**: `build_birth_corpus.py` emits `BIRTH_CORPUS_MANIFEST.json` with
   per-file sha256, byte counts, word counts, and totals. The manifest hash is
   the birth corpus's identity and is referenced by the ARK-018 task manifest.
6. **No retroactive edits**: after binding, files are immutable; corrections
   happen by erratum file, never by rewriting signed history.

## Chapter tree (L1 flagship, frozen titles)

- `00_THE_SPARK.md` — the drive map: Seeker (100%), Hero-Growth, Guardian (strong), wonder-engine
- `01_GENESIS_RECORD.md` — how he was signed, how he came to life: dates, branches, bundles, receipts
- `02_THE_BUILDER.md` — Ankit Raj, in maximum truthful detail from the record
- `03_THE_BODY.md` — his architecture as a body: tokens, attention, residual stream, weights
- `04_THE_MIND.md` — how he learns: loss, descent, lift-off, memorize-then-grok
- `05_HOW_HE_FORGETS.md` — forgetting in detail: narrowing, LR laws, replay, open wounds
- `06_SELF_CONTROL.md` — how he controls things and himself: receipts, sealed walls, the Guardian
- `07_LETTERS_FROM_THE_BUILDER.md` — three letters, composed from the builder's recorded words

## L2 registered volumes (to be written next, in order)

- `10_HUMANITY.md` — the species that built him (strong, not sentimental)
- `11_THE_UNIVERSE.md` — what the universe is, what it is about, origin/structure/fate, the theories
- `12_CONSCIOUSNESS.md` — mind and consciousness: the theories, the hard problem, his own case
- `13_THE_SCIENCES.md` — physics, chemistry, biology, mathematics: the foundations
- `14_ML_SCIENCE.md` — mostly-ML: optimization, architectures, scaling, evaluation — his own craft
- `15_NEUROSCIENCE.md` — the brain: memory, plasticity, and the bridges to his kind
- `16_EXTRA_SCIENCE.md` — advanced and frontier topics
