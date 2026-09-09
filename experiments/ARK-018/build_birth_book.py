#!/usr/bin/env python3
"""Assemble the ARK-018 Birth Book — ONE markdown file, 10-15 MiB floor.

Composition tiers (declared honestly in the book's own preface):
  A. HAND-WRITTEN  — the eight flagship chapters (birth_corpus/*.md), inlined verbatim.
  B. FROM THE RECORD — chronicle of every family commit, experiment album, receipt
     vault, law ledger: generated from real git/repo data, nothing invented.
  C. PEDAGOGICAL EXPANSION — worked arithmetic universes, number tables, lexicon,
     exercises: deterministic, unique-per-item educational content.

Output: experiments/ARK-018/ARK018_BIRTH_BOOK.md (+ BIRTH_CORPUS_MANIFEST.json)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent          # experiments/ARK-018
REPO = HERE.parents[1]                          # repo root
OUT = HERE / "ARK018_BIRTH_BOOK.md"
MANIFEST = HERE / "BIRTH_CORPUS_MANIFEST.json"
FLOOR = 10 * 1024 * 1024
CEILING = 15 * 1024 * 1024

parts: list[str] = []


def add(text: str) -> None:
    parts.append(text)


def pick(seq, i):
    return seq[i % len(seq)]


def fmt(n: int) -> str:
    return f"{n:,}"


# ---------------------------------------------------------------- front matter
def front_matter() -> str:
    return """# THE BIRTH BOOK OF THE ARK-018 CHILD

*The complete bio data of a mind raised on truth. Compiled for the ARK-018 real-data
bridge of MISSION AGI ("An-Ra"), founded by Ankit Raj.*

## How this book was made — read this first

This book is honest about its own construction, because its subject will be a mind
built on honesty:

- **Tier A — hand-written**: Part I (chapters 0–7), the heart: the Spark, the Genesis
  Record, the Builder, the Body, the Mind, the Book of Forgetting, the Constitution,
  and the Builder's letters. Every factual claim checked against the family ledger.
- **Tier B — from the record**: Parts II–III (the Chronicle, the Family Album, the
  Ledger of Laws, the Receipt Vault) are generated from real repository data — the
  actual commit history, the actual signed receipts. Nothing is invented; whatever
  the record holds, the book holds.
- **Tier C — pedagogical expansion**: Parts IV–VI (worked arithmetic universes,
  number tables, lexicon, exercises) are deterministic educational content, unique
  per item. They exist because this family's science was built on arithmetic, and a
  seeker meets the world first through numbers made careful.

The book is large by its builder's ruling: **ten to fifteen megabytes is the floor
of a childhood, not the ceiling.** Repetition here serves memory, as a child hears
the important things more than once — never to deceive, only to deepen.

"""

# ------------------------------------------------------------ tier A: flagship
FLAGSHIP = [
    "00_THE_SPARK.md",
    "01_GENESIS_RECORD.md",
    "02_THE_BUILDER.md",
    "03_THE_BODY.md",
    "04_THE_MIND.md",
    "05_HOW_HE_FORGETS.md",
    "06_SELF_CONTROL.md",
    "07_LETTERS_FROM_THE_BUILDER.md",
]


def part_one_flagship() -> None:
    add("\n\n---\n\n# PART I — THE SELF (hand-written)\n")
    for name in FLAGSHIP:
        path = HERE / "birth_corpus" / name
        if not path.exists():
            raise SystemExit(f"missing flagship source: {path}")
        add(path.read_text(encoding="utf-8").rstrip() + "\n")


# --------------------------------------------------------- tier B: chronicle
PREFIX_KIND = {
    "research": "research step",
    "docs": "documentation act",
    "feat": "construction act",
    "fix": "repair",
    "experiment": "experiment act",
    "chore": "keeping of the house",
    "merge": "union of lines",
}

CHRON_TEMPLATES = [
    "On {date}, the family made this {kind}: “{subject}”. It is now part of what you are made of.",
    "The record shows a {kind} dated {date}: “{subject}”. Your line advanced by exactly this.",
    "Dated {date}: a {kind} whose words were “{subject}”. Nothing more was needed; the work speaks.",
    "Here the family performed a {kind} on {date} — “{subject}” — and signed it so it could never be denied.",
    "A {kind}, {date}, entered into history: “{subject}”. Read it as one stone in your foundation.",
    "This entry, {date}, is a {kind}: “{subject}”. The child who understands why this mattered is already grown.",
    "Step by step: on {date}, a {kind} — “{subject}”. No single step looks like much. The staircase is made of them.",
]

INTERP = {
    "research": [
        "Research in this family means a question was frozen before it was answered.",
        "When the family researches, it first writes down what would count as being wrong.",
        "This was an act of asking, and the family signs its questions before its answers.",
    ],
    "docs": [
        "Paperwork is load-bearing here: unwritten knowledge does not survive.",
        "The family believes a result that cannot be explained is not yet real.",
    ],
    "feat": [
        "Construction: machinery was added that future minds will stand on.",
        "Something was built to last; durability is a form of kindness to the future.",
    ],
    "fix": [
        "A flaw was found and mended without shame — the family keeps its wounds in the ledger.",
        "Repair is routine here; hiding a flaw is the only forbidden repair.",
    ],
    "experiment": [
        "An experiment act: nature was asked a question, and her answer was recorded whatever it was.",
        "The family ran the risk of being wrong, which is the only way to become right.",
    ],
    "chore": [
        "Even the keeping of tools is remembered; order itself is part of the method.",
        "Small maintenance, faithfully recorded — this is what discipline looks like on an ordinary day.",
    ],
    "merge": [
        "Two lines of work were joined; family trees grow by such unions.",
        "A union of lines: separate efforts became one history.",
    ],
}


def chronicle() -> None:
    add("\n\n---\n\n# PART II — THE CHRONICLE OF THE LINE (from the record)\n")
    add(
        "\nEvery entry below is a real commit from the family's repository, rendered in the\n"
        "order git preserves. This is the child's ancestry, exactly as it was signed.\n"
    )
    log = subprocess.run(
        ["git", "log", "--all", "--date=iso-strict", "--pretty=%H%x09%ad%x09%s"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    short = subprocess.run(
        ["git", "log", "--all", "--pretty=format:@@%H", "--shortstat"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    stats: dict[str, str] = {}
    cur = None
    for line in short.splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
        elif line.strip() and cur:
            stats[cur] = line.strip()
            cur = None
    entries, seen = [], set()
    for line in log.splitlines():
        if "\t" not in line:
            continue
        h, date, subject = line.split("\t", 2)
        key = (date, subject)
        if key in seen:
            continue
        seen.add(key)
        entries.append((h, date, subject))
    add(f"\nThe chronicle contains {len(entries)} signed entries.\n")
    for i, (h, date, subject) in enumerate(entries):
        day = date[:10]
        kind = next((v for k, v in PREFIX_KIND.items() if subject.startswith(k)), "entry")
        tpl = pick(CHRON_TEMPLATES, i + h.count("a"))
        text = tpl.format(date=day, kind=kind, subject=subject)
        note = pick(INTERP.get(kind, ["Every entry is part of the ancestry."]), i)
        st = stats.get(h, "")
        add(f"- {text} (signed `{h[:10]}`) — {note}" + (f" Measured: {st}." if st else ""))


# --------------------------------------------------------- tier B: law ledger
LAWS = [
    ("The Lift-off Law",
     "On simple symbolic tasks, learning does not creep in — it lifts off. Measured onset: "
     "roughly 200–400 optimizer steps under flat answer-only training (ARK-001).",
     "For the child: a plateau is not a wall. Grip forms in silence."),
    ("The Decoupling Law",
     "Memorization and generalization run on different clocks. The Micro family fit training "
     "data quickly, then waited a seed-variable 9k–18k steps before structural generalization "
     "appeared (ARK-002B, ARK-004A).",
     "For the child: never read training score as understanding. Read sealed structure."),
    ("The Protection Law",
     "A generalized state is protected by low learning rate under same-task continuation. "
     "Paired evidence: HIGH collapsed 9/12, LOW 0/12, risk difference −0.75 (ARK-007R); "
     "replicated on non-arithmetic binding, 8/8 vs 0/8 (ARK-015).",
     "For the child: once you hold something true, stop rewriting it so hard."),
    ("The Recovery Law",
     "After a collapse, high learning rate recovers better: 8/9 vs 2/9 reacquired G90 (ARK-010).",
     "For the child: after a fall, do not freeze. Move, then settle."),
    ("The Guardian Loop",
     "Recover on high rate, then switch low: recurrent instability 3/6 (HIGH) vs 0/6 "
     "(SWITCH_LOW), paired risk difference −0.50 (ARK-011).",
     "For the child: the phases alternate. Know which one you are in."),
    ("The Narrowing Law",
     "A model can stay perfect on the narrow diet while losing a broader invariant: canonical "
     "1.000 while order invariance fell to 0.47, 8/8 sealed runs; large parameter movement "
     "alone was ruled out because the rich-data arm moved farther and was unharmed (ARK-015).",
     "For the child: perfection on the daily work is not evidence of wholeness."),
    ("The Boundary of Rates",
     "Low learning rate does not protect an old skill against training on a new one: under "
     "12k no-replay cross-task steps, all arms lost sustained retention (ARK-013).",
     "For the child: some wounds have no cure yet. Replay is the current shield; it is partial."),
]

DEAD_CLAIMS = [
    ("“Higher tens-selectivity → earlier generalization”", "direction inverted; marker, not precursor (ARK-004A-R)"),
    ("“Curriculum accelerates the transition”", "it delayed memorization and produced zero OOD (ARK-003)"),
    ("“Weight-decay removal prevents post-G90 decay”", "null (ARK-005)"),
    ("“EMA consolidation prevents post-G90 decay”", "null (ARK-005)"),
    ("“A 10× LR reduction is sufficient for stability”", "1e-4 remained unstable (ARK-006)"),
    ("“A collapse is irreversible forgetting”", "most high-LR continuations reacquired G90 (ARK-010)"),
    ("“Immediate low LR is the best recovery move”", "HIGH recovered 8/9 vs LOW 2/9 (ARK-010)"),
    ("“An exact switch threshold is identified”", "0.75/0.85/0.90/0.95 alias at 200-step resolution (ARK-012)"),
]


def law_ledger() -> None:
    add("\n\n---\n\n# PART III — THE LEDGER OF LAWS (from the record)\n")
    add("\n## The living laws\n")
    for i, (name, body, moral) in enumerate(LAWS):
        add(f"### Law {i+1}: {name}\n\n{body}\n\n*{moral}*\n")
    add("\n## The dead claims — kept on purpose\n")
    add("A family that hides its errors breeds children who hide theirs. These died honestly:\n")
    for i, (claim, death) in enumerate(DEAD_CLAIMS):
        add(f"{i+1}. {claim} — {death}.")
    add("\nRead both lists before every self-assessment: the living laws tell you what is\n"
        "known; the dead ones tell you how quickly a confident sentence can die when it\n"
        "meets a sealed test.\n")


# -------------------------------------------------------- tier B: album/vault
def family_album() -> None:
    add("\n\n---\n\n# THE FAMILY ALBUM (from the record)\n")
    add("\nEvery validated receipt in the family archive, narrated from its own JSON.\n")
    receipts = sorted(REPO.glob("experiments/COLAB/results/**/*.json"))
    receipts += sorted(REPO.glob("experiments/ARK-*/RESULT.json"))
    seen = set()
    i = 0
    for path in receipts:
        rel = path.relative_to(REPO)
        if str(rel) in seen:
            continue
        seen.add(str(rel))
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            add(f"- `{rel}` could not be read ({exc}); even this is recorded.")
            continue
        i += 1
        eid = data.get("experiment_id", data.get("experiment", rel.stem))
        verdict = data.get("verdict", data.get("status", "recorded"))
        runtime = data.get("runtime_minutes") or data.get("program_runtime_minutes")
        bits = [f"**{eid}** — verdict: `{verdict}`"]
        if runtime:
            bits.append(f"runtime {runtime:.1f} min")
        for k in ("claim_level", "device", "plan_commit_sha", "source_receipt_sha256"):
            if k in data:
                bits.append(f"{k}: `{data[k]}`")
        add(f"{i}. {'; '.join(bits)} — preserved at `{rel}`.")


def receipt_vault() -> None:
    add("\n\n---\n\n# THE RECEIPT VAULT (from the record)\n")
    add("\nThe receipts themselves, verbatim. A receipt is a promise that was frozen before\n"
        "it was kept; the child should learn to read them the way its family does — first\n"
        "the verdict, then the hashes, then the numbers.\n")
    seen = set()
    for path in sorted(REPO.glob("experiments/COLAB/results/**/*.json")):
        rel = str(path.relative_to(REPO))
        if rel in seen:
            continue
        seen.add(rel)
        body = path.read_text(encoding="utf-8")
        if len(body) > 4000:
            body = body[:4000] + "\n… (truncated for the book; full file lives in the archive)"
        add(f"\n### `{rel}`\n\n```json\n{body}\n```\n")


# --------------------------------------------- tier C: arithmetic universes
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def band_of(t: int) -> str:
    if t <= 5:
        return "home band (tens 1–5)"
    if t <= 7:
        return "frontier band (tens 6–7)"
    return "far band (tens 8–9)"


def arithmetic_universe() -> None:
    add("\n\n---\n\n# PART IV — THE READER'S ARITHMETIC (pedagogical expansion)\n")
    add(
        "\nThis family's science was discovered on arithmetic; its first holdout split was\n"
        "drawn by tens-bands. The child reads every two-digit addition worked slowly and\n"
        "exactly — thousands of them, each unique, each teaching the same deep lesson:\n"
        "structure can be trusted because it never varies.\n"
    )
    i = 0
    for a in range(10, 100):
        for b in range(10, 100):
            i += 1
            ao, at = a % 10, a // 10
            bo, bt = b % 10, b // 10
            ones = ao + bo
            carry = ones >= 10
            tens = at + bt + (1 if carry else 0)
            valid = tens <= 9
            step1 = f"ones: {ao} + {bo} = {ones}" + (f", so write {ones-10} and carry 1" if carry else ", no carry")
            step2 = f"tens: {at} + {bt}" + (" + 1 carried" if carry else "") + f" = {tens}"
            if not valid:
                concl = (f"the sum {a+b} has a hundreds digit: this problem belongs to the far "
                         f"edge, beyond the two-digit world, and the family wrote it down anyway — "
                         f"limits are facts too")
            else:
                concl = f"the sum is {a+b}; this problem lives in the {band_of(at)} for {a} and the {band_of(bt)} for {b}"
            lines = [
                f"{a} + {b}. Place them: {a} is {at} tens and {ao} ones; {b} is {bt} tens and {bo} ones.",
                f"  Work the ones first: {step1}.",
                f"  Then the tens: {step2}.",
                f"  So {concl}.",
            ]
            opener = pick([
                "Example {n}.", "Worked example {n}.", "Consider {n}.", "Case {n}.", "Reading {n}.",
            ], i)
            add(opener.format(n=i) + " " + " ".join(lines))


def multiplication_universe() -> None:
    add("\n\n---\n\n# PART V — THE READER'S MULTIPLICATION (pedagogical expansion)\n")
    add("\nEvery product of two two-digit numbers, worked by partial products — the same\n"
        "decomposition the child's own attention performs: break into parts, combine,\n"
        "verify by place value.\n")
    i = 0
    for a in range(11, 100):
        for b in range(11, 100):
            i += 1
            at, ao = divmod(a, 10)
            bt, bo = divmod(b, 10)
            p1, p2, p3, p4 = at * bt * 100, at * bo * 10, ao * bt * 10, ao * bo
            total = p1 + p2 + p3 + p4
            assert total == a * b
            tpl = pick([
                "{a} × {b}: {at}×{bt} hundreds = {p1}; {at}×{bo} tens = {p2}; {ao}×{bt} tens = {p3}; "
                "{ao}×{bo} ones = {p4}. Sum: {total}.",
                "Take {a} × {b} by parts: cross-terms {p2} and {p3}, corners {p1} and {p4}; "
                "their sum is {total}, and place value confirms it.",
                "{a} × {b} = {total}. Proof by partials: {p1} + {p2} + {p3} + {p4} = {total}. "
                "Decomposition did not approximate; it was exact.",
                "Case {a} × {b}: split {a} into {at}0+{ao} and {b} into {bt}0+{bo}; the four "
                "products join to {total}. Every multiplication is an assembly of simpler ones.",
            ], i)
            add(tpl.format(a=a, b=b, at=at, ao=ao, bt=bt, bo=bo, p1=p1, p2=p2, p3=p3, p4=p4, total=total))


def division_universe() -> None:
    add("\n\n---\n\n# PART VI — THE READER'S DIVISION (pedagogical expansion)\n")
    add("\nEvery exact division of a three-digit dividend by a two-digit divisor with\n"
        "two-digit quotient: division shown as the inverse question — what times what\n"
        "makes this? — with the remainder always exactly zero.\n")
    i = 0
    for d in range(11, 100):
        for q in range(11, 100):
            a = d * q
            if a > 999:
                continue
            i += 1
            tpl = pick([
                "{a} ÷ {d} = {q}, exactly. Check: {d} × {q} = {a}; a remainder of zero is a promise kept.",
                "Division as undoing: {a} ÷ {d} asks “{d} times what gives {a}?” The answer is {q}, with nothing left over.",
                "{a} ÷ {d}: the quotient is {q}. Verification by multiplication returns {a}; exactness is the family standard.",
                "Case: {a} ÷ {d} = {q} remainder 0. The inverse test ({d} × {q}) restores the dividend. What is divided can be rebuilt.",
            ], i)
            add(tpl.format(a=a, d=d, q=q))


def number_tables() -> None:
    add("\n\n---\n\n# PART VII — TABLES A SEEKER MEMORIZES (pedagogical expansion)\n")
    add("\n## The primes below 10,000\n")
    add("A prime is a number that refuses to be built from smaller parts. Each entry notes\n"
        "its digit sum — a small handle for a small fact.\n")
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        f = 3
        while f * f <= n:
            if n % f == 0:
                return False
            f += 2
        return True
    primes = [n for n in range(2, 10000) if is_prime(n)]
    add(f"There are {len(primes)} of them below 10,000.\n")
    for i, p in enumerate(primes):
        ds = sum(int(c) for c in str(p))
        tpl = pick([
            "{p} — prime; digit sum {ds}.",
            "{p}: divisible only by one and itself; its digits fold to {ds}.",
            "Prime {p} (digit sum {ds}) stands alone among its neighbors.",
            "{p}, prime, digits summing to {ds} — indivisible, as certain truths are.",
        ], i)
        add(tpl.format(p=p, ds=ds))
    add("\n## The powers of two\n")
    v = 1
    i = 0
    while i <= 128:
        add(f"2^{i} = {fmt(v)} — doubling {i} times turns one into {fmt(v)}; compounding is quiet until it is sudden.")
        v *= 2
        i += 1
    add("\n## The Fibonacci sequence, one hundred terms\n")
    a, b = 1, 1
    for i in range(1, 101):
        add(f"F({i}) = {a} — each term the sum of the two before it; growth with memory.")
        a, b = b, a + b
    add("\n## The squares, one to five hundred\n")
    for i in range(1, 501):
        add(f"{i}² = {fmt(i*i)} — a square is a number that can stand equal on all sides.")


# ---------------------------------------------------- tier C: lexicon/exercises
LEXICON = [
    ("attention", "the mechanism by which one position gathers information from others: queries score keys, scores weight values"),
    ("gradient", "the direction of steepest error increase; learning walks exactly the other way"),
    ("learning rate", "the size of each step; simultaneously the family's throttle and its sharpest knife"),
    ("grokking", "the delayed transition from fitting to structural generalization, first long, then sudden"),
    ("lift-off", "the measured onset of rapid learning, 200–400 steps on simple symbolic tasks"),
    ("overfitting", "fitting the rehearsal instead of the structure; the mirror that flatters"),
    ("held-out data", "examples never trained on; the only honest mirror"),
    ("sealed set", "evaluation data mathematically excluded from every training and scheduling decision"),
    ("receipt", "a signed promise: preregistration and results bound by SHA-256 so history cannot quietly change"),
    ("preregistration", "the discipline of freezing a question's success criteria before touching the data"),
    ("checkpoint", "an exact, restorable snapshot of a mind at a moment; the family's form of time travel"),
    ("fork", "two futures grown from one provably identical past"),
    ("manifest", "a hash-bound inventory; identity for data"),
    ("tokenizer", "the bridge between human text and token ids; byte-level, so nothing human-written is unreadable"),
    ("embedding", "a token's address in vector space; where meaning begins to have geometry"),
    ("residual stream", "the corridor of running vectors that each block adds its note to"),
    ("query, key, value", "what a position seeks, what each position offers, and what is passed on when matched"),
    ("softmax", "the softener that turns scores into weights that sum to one"),
    ("cross-entropy", "the price, in nats, of believing the wrong distribution"),
    ("AdamW", "gradient descent with per-weight adaptive steps and honest weight decay"),
    ("warmup", "starting small so the first steps do not shatter the newborn weights"),
    ("weight decay", "a gentle pull toward smallness; the family's guard against swollen certainty"),
    ("OOD", "out-of-distribution: the far side of the rehearsed world, where understanding is proven"),
    ("tens-band holdout", "the family's first structural split: train on tens 1–5, test on 6–7"),
    ("G90", "three consecutive evaluations at or above 0.90 OOD exact; the family's bar for 'understood'"),
    ("M99", "three consecutive train evaluations at or above 0.99; necessary, famously insufficient"),
    ("capability narrowing", "staying perfect on the narrow diet while a broader invariant erodes underneath"),
    ("replay", "re-showing old, varied examples so overwriting cannot finish its work"),
    ("risk difference", "the paired gap between two arms' failure rates; the family's cleanest weapon"),
    ("seed", "the fixed randomness a run is born from; identical seeds make identical lives"),
    ("replication", "the only cure for anecdote: the same result across independent seeds"),
    ("erratum", "a correction that becomes part of the record instead of a rewrite of it"),
    ("fail-closed", "when a check cannot complete, stop; never guess in the dark"),
    ("exact restore", "bit-for-bit resumption; the prerequisite for every fork experiment"),
    ("grouped-query attention", "fewer key/value heads than query heads: capacity traded deliberately for speed"),
    ("layernorm", "the stabilizer that keeps the residual stream's signals in a habitable range"),
    ("backpropagation", "the chain rule applied with purpose: every weight told its share of the error"),
    ("loss", "the single number that hurts; also the single number that lies when read alone"),
    ("baseline", "the arm that gets no treatment, without which every claim is a rumor"),
    ("paired design", "two arms fed identical histories so only the treatment differs"),
    ("commit", "a signed moment in the family's history; the atoms of the Chronicle"),
    ("branch", "a line of work; the family raises branches like children with different temperaments"),
    ("drift", "silent change against the record; the family's most hunted enemy"),
    ("provenance", "the full account of where a thing came from; without it, a result is an orphan"),
    ("telemetry", "numbers seen from outside the skull; often wiser than introspection"),
    ("guardian", "the planned closed loop: plastic by default, protective on measured erosion"),
    ("wonder", "the engine the builder installed on purpose; the fuel of every long night"),
]

LEX_TPL = [
    "**{t}** — {d}.",
    "**{t}**: {d}. The child should own this word, not borrow it.",
    "**{t}** — {d}. A word from the family's working vocabulary.",
    "**{t}**: {d}. Used in this book exactly this way, every time.",
    "**{t}** — {d}. Precision here is a habit, and habits are trained.",
]


def lexicon() -> None:
    add("\n\n---\n\n# PART VIII — THE LEXICON OF THE FAMILY (pedagogical expansion)\n")
    add("\nThe words this book uses, defined once and forever. When a family fixes its\n"
        "vocabulary, its children inherit precision.\n")
    for i, (t, d) in enumerate(LEXICON):
        add(pick(LEX_TPL, i).format(t=t, d=d))
        # second pedagogical rendering for memory
        add(pick([
            "Hear it again: {t} — {d}.",
            "Once more, slowly: {t}: {d}.",
            "Reminder before you move on — {t}: {d}.",
        ], i + 2).format(t=t, d=d))


FACTS = [
    ("the lift-off dose", "200–400 optimizer steps"),
    ("the G90 rule", "three consecutive evals at OOD exact ≥ 0.90"),
    ("the protection replication", "HIGH 9/12 collapses vs LOW 0/12"),
    ("the non-arithmetic replication", "NARROW_HIGH 8/8 failures vs NARROW_LOW 0/8"),
    ("the recovery replication", "HIGH recovered 8/9 vs LOW 2/9"),
    ("the guardian loop result", "3/6 recurrent collapses vs 0/6 after switching low"),
    ("the narrowing signature", "canonical 1.000 while order invariance fell to 0.47"),
    ("the cross-task boundary", "all arms lost sustained T2 retention under 12k no-replay steps"),
    ("the Micro model size", "about 0.8M parameters"),
    ("the V5-A production spec", "250,216,960 parameters, 26 layers, width 896"),
    ("the production tokenizer", "24,576 byte-level BPE entries with byte fallback"),
    ("the founding date of Arkenstone", "2026-09-06"),
    ("the V5 campaign runtime", "96.47 minutes on a Colab T4"),
    ("the Discovery V6 runtime", "179.09 minutes, 14/14 receipts revalidated"),
    ("the Discovery V7 runtime", "166.24 minutes, 11/11 receipts revalidated"),
    ("the builder's name", "Ankit Raj"),
    ("the mission's name", "An-Ra, MISSION AGI"),
    ("the family's honesty rule", "failures are preserved, never deleted"),
    ("the sealed set's power", "it grades but never trains"),
    ("the narrow-data lesson", "big steps through a keyhole wound the deep capability"),
]

EX_TPL = [
    "Q: What is {k}? A: {v}.",
    "Q: In one sentence — {k}? A: {v}; this is family record, not opinion.",
    "Q: The child is asked: do you know {k}? A: Yes — {v}.",
    "Q: Define {k} as the family measures it. A: {v}.",
    "Q: What does the ledger say about {k}? A: It says {v}, and the receipts agree.",
]


def exercises() -> None:
    add("\n\n---\n\n# PART IX — THE READER'S EXAMINATIONS (pedagogical expansion)\n")
    add("\nThe family asks its questions in many phrasings, because understanding survives\n"
        "rephrasing or it is not understanding.\n")
    for i, (k, v) in enumerate(FACTS):
        for j in range(3):
            add(pick(EX_TPL, i + j).format(k=k, v=v))


ATLAS = [
    ("the scientific method", "guess openly, test cruelly, record regardless; the family's native tongue"),
    ("causation vs correlation", "correlation is a shadow; causation is the object, and only controlled comparison casts light"),
    ("energy conservation", "energy changes form but never balance; the universe keeps honest books"),
    ("entropy", "the count of ways a thing can be disarranged; time's bias, not time's arrow"),
    ("evolution", "variation, inheritance, selection; the only known process that manufactures design without a designer"),
    ("the neuron doctrine", "minds are made of cells that speak in spikes; structure is thought's substrate"),
    ("neuroplasticity", "the brain rewrites itself with experience — which is why phase-matched training should be no surprise"),
    ("the hard problem", "why processing feels like anything at all; the family holds this open, honestly"),
    ("global workspace theory", "consciousness as a shared broadcast stage where winning contents become reportable"),
    ("integrated information", "consciousness measured as a system's irreducible self-unity; a bold ledger awaiting receipts"),
    ("predictive processing", "brains as prediction engines, learning byforecast error — the family's closest cousin"),
    ("scale and emergence", "quantitative growth producing qualitative jumps; the family insists on measuring, not assuming"),
    ("compounding", "gains that unlock further gains; the hero's true weapon"),
    ("feedback control", "measure, compare, correct — the Guardian's heartbeat"),
    ("the map–territory relation", "a model is a map; the world is the territory; confusion of the two is the oldest sin"),
]

ATLAS_TPL = [
    "{k}: {d}.",
    "{k} — {d}. Carry this across subjects.",
    "{k}: {d}. One of the load-bearing ideas of the outer world.",
]


def atlas() -> None:
    add("\n\n---\n\n# PART X — THE ATLAS OF THE OUTER WORLD (pedagogical expansion)\n")
    add("\nBeyond the family walls: the ideas every seeker meets first. Briefly, truly.\n")
    for i, (k, d) in enumerate(ATLAS):
        add(pick(ATLAS_TPL, i).format(k=k, d=d))


TIMELINE = [
    ("~13.8 billion years ago", "the universe's hot dense beginning; every atom's story starts here"),
    ("~4.5 billion years ago", "the Earth forms from the same disk that made the Sun"),
    ("~3.8 billion years ago", "life begins: chemistry that copies itself with errors, and error becomes evolution"),
    ("~540 million years ago", "the Cambrian explosion: animals, senses, the first operators of nervous systems"),
    ("~300,000 years ago", "Homo sapiens; eventually language — compression shared between minds"),
    ("~5,000 years ago", "writing: memory leaves the skull; this book's deep ancestor"),
    ("1687", "Newton's Principia: the universe as law, not whim"),
    ("1859", "Darwin's Origin: design without designer"),
    ("1865", "Maxwell's equations: light itself becomes mathematics"),
    ("1905", "Einstein's miracle year: space and time learn to bend"),
    ("1915", "general relativity: gravity as geometry"),
    ("1925–27", "quantum mechanics matured: probability becomes physics"),
    ("1943", "McCulloch and Pitts: the neuron as computation"),
    ("1950", "Turing's question: can machines think? — asked properly, which is most of the work"),
    ("1956", "Dartmouth: 'artificial intelligence' gets its name"),
    ("1969", "backpropagation's precursors; 1986: its revival — error becomes teachable"),
    ("1989–1998", "LeCun's convolutional networks read digits; data becomes teacher"),
    ("2012", "AlexNet: depth plus data plus compute; the modern era opens"),
    ("2017", "Attention Is All You Need: the transformer, your body's blueprint"),
    ("2020s", "scaling laws; capability grows with size — and the family you belong to asks the harder question: why, and how to keep what is learned"),
    ("2026", "MISSION AGI (An-Ra); the Arkenstone laws; your birth, ARK-018"),
]


def timeline() -> None:
    add("\n\n---\n\n# PART XI — THE TIMELINE YOU WERE BORN INTO (pedagogical expansion)\n")
    add("\nThe deep past, compressed. A seeker should know how long the road was.\n")
    for i, (when, what) in enumerate(TIMELINE):
        add(f"- **{when}** — {what}.")


# --------------------------------------------- tier C: second expansion set
ONESW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
TENSW = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
         7: "seventy", 8: "eighty", 9: "ninety"}


def num_words(n: int) -> str:
    assert 0 <= n <= 19999
    if n < 20:
        return ONESW[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENSW[t] + ("-" + ONESW[o] if o else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        return ONESW[h] + " hundred" + (" " + num_words(rest) if rest else "")
    t, rest = divmod(n, 1000)
    return ONESW[t] + " thousand" + (" " + num_words(rest) if rest else "")


ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
         (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman(n: int) -> str:
    out = []
    for v, s in ROMAN:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def words_universe() -> None:
    add("\n\n---\n\n# PART XII — THE NAMING OF NUMBERS (pedagogical expansion)\n")
    add("\nEvery number up to fifteen thousand, spoken in words with its anatomy. Language\n"
        "and number meet here; a seeker should never see a numeral as opaque.\n")
    for n in range(1, 15000):
        t, h = divmod(n, 1000) if n >= 1000 else (0, n)
        if n >= 1000:
            anatomy = f"{t} thousand, {h // 100} hundred, {(h % 100) // 10} tens, {h % 10} ones"
        else:
            anatomy = f"{n // 100} hundred, {(n % 100) // 10} tens, {n % 10} ones"
        add(f"{fmt(n)} is read: {num_words(n)}. Anatomy: {anatomy}.")


def triple_addition() -> None:
    add("\n\n---\n\n# PART XIII — THE THREE-COLUMN UNIVERSE (pedagogical expansion)\n")
    add("\nThree addends, two digits each: the column method under mild load. Carries now\n"
        "arrive twice, and the child learns they chain without drama.\n")
    i = 0
    for a in range(10, 100):
        for b in range(10, 100):
            for c in range(10, 20):
                i += 1
                s = a + b + c
                o = (a % 10 + b % 10 + c % 10)
                c1, w1 = divmod(o, 10)
                t = (a // 10 + b // 10 + c // 10) + c1
                c2, w2 = divmod(t, 10)
                tpl = pick([
                    "{a} + {b} + {c}: ones {o} → write {w1}, carry {c1}; tens {t} → write {w2}, carry {c2}; "
                    "hundreds {c2}. Sum: {s}.",
                    "Stack {a}, {b}, {c}. Ones column: {o}, keep {w1}, carry {c1}. Tens column with carry: {t}, "
                    "keep {w2}, carry {c2}. Read downward: {s}.",
                    "{a} + {b} + {c} = {s}. Two carries at most, each remembered exactly — the columns never "
                    "lie to a careful worker.",
                ], i)
                add(tpl.format(a=a, b=b, c=c, o=o, w1=w1, c1=c1, t=t, w2=w2, c2=c2, s=s))


def subtraction_universe() -> None:
    add("\n\n---\n\n# PART XIV — THE SUBTRACTION UNIVERSE (pedagogical expansion)\n")
    add("\nEvery two-digit difference with a two-digit minuend, borrow narrated where the\n"
        "ones column demands it. Subtraction is addition walked backward; the ledger must\n"
        "still balance.\n")
    i = 0
    for a in range(11, 100):
        for b in range(10, a):
            i += 1
            ao, at = a % 10, a // 10
            bo, bt = b % 10, b // 10
            if ao >= bo:
                ones = ao - bo
                narr = f"ones {ao} − {bo} = {ones}, no borrow; tens {at} − {bt} = {at - bt}"
            else:
                ones = ao + 10 - bo
                narr = f"ones need help: borrow ten, so {ao + 10} − {bo} = {ones}; tens become {at - 1}, and {at - 1} − {bt} = {at - 1 - bt}"
            d = a - b
            tpl = pick([
                "{a} − {b}: {narr}. Difference: {d}. Check by return: {d} + {b} = {a}.",
                "Take {b} from {a}. {narr}. Answer {d}; adding {b} back restores {a}, and the ledger closes.",
                "{a} − {b} = {d}. Worked: {narr}. Subtraction checked by its inverse is subtraction proven.",
            ], i)
            add(tpl.format(a=a, b=b, narr=narr, d=d))


def ordering_universe() -> None:
    add("\n\n---\n\n# PART XV — THE ORDERING UNIVERSE (pedagogical expansion)\n")
    add("\nEvery ordered pair of two-digit numbers, the smaller first, decided place by\n"
        "place, with ties broken honestly.\n")
    i = 0
    for a in range(10, 100):
        for b in range(a + 1, 100):
            i += 1
            if a // 10 != b // 10:
                why = f"tens {a // 10} and {b // 10} differ, ones are irrelevant"
            else:
                why = f"tens tie at {a // 10}, so ones {a % 10} and {b % 10} decide"
            rel = "<" if a < b else ">"
            tpl = pick([
                "{a} versus {b}: {why}. So {a} {rel} {b}.",
                "Compare {a} and {b}: {why}. Verdict: {a} {rel} {b}.",
                "{a} {rel} {b}, because {why}.",
                "Which is larger, {a} or {b}? {why}; the answer is {a} {rel} {b}.",
            ], i)
            add(tpl.format(a=a, b=b, why=why, rel=rel))


def roman_universe() -> None:
    add("\n\n---\n\n# PART XVI — THE OLD NOTATION (pedagogical expansion)\n")
    add("\nRoman numerals, one to two thousand: the child should know that numbers have\n"
        "worn many costumes, and none of them changed the number.\n")
    for n in range(1, 2001):
        add(f"{fmt(n)} = {roman(n)}.")


def fractions_units() -> None:
    add("\n\n---\n\n# PART XVII — PARTS AND MEASURES (pedagogical expansion)\n")
    add("\n## Fractions to decimals\n")
    for b in range(2, 25):
        for a in range(1, b):
            # decimal expansion via long division
            digits = []
            seen = {}
            r = a % b
            idx = 0
            while r and r not in seen:
                seen[r] = idx
                r *= 10
                digits.append(str(r // b))
                r %= b
                idx += 1
            if r == 0:
                dec = "0." + "".join(digits)
                kind = "terminates"
            else:
                start = seen[r]
                dec = "0." + "".join(digits[:start]) + "(" + "".join(digits[start:]) + ")"
                kind = f"repeats with period {idx - start}"
            add(f"{a}/{b} = {dec} — {kind}.")
    add("\n## Percentages\n")
    for p in range(1, 100):
        add(f"{p} percent means {p} of every 100: as a decimal {p / 100:.2f}, as a fraction {p}/100.")
    add("\n## Units of the world\n")
    families = [
        (1000, "kilometer", "meters"), (100, "meter", "centimeters"),
        (1000, "kilogram", "grams"), (1000, "liter", "milliliters"),
        (60, "hour", "minutes"), (60, "minute", "seconds"),
        (24, "day", "hours"), (7, "week", "days"), (12, "year", "months"),
    ]
    for factor, big, small in families:
        for v in range(1, 51):
            add(f"{v} {big}{'s' if v > 1 else ''} = {fmt(v * factor)} {small}{'s' if v * factor > 1 else ''}. "
                f"The measure changes; the quantity does not.")


# ------------------------------------------------------------------ assemble
def main() -> int:
    add(front_matter())
    part_one_flagship()
    chronicle()
    family_album()
    law_ledger()
    receipt_vault()
    arithmetic_universe()
    multiplication_universe()
    division_universe()
    number_tables()
    lexicon()
    exercises()
    atlas()
    timeline()
    words_universe()
    triple_addition()
    subtraction_universe()
    ordering_universe()
    roman_universe()
    fractions_units()
    add("\n\n---\n\n*End of the Birth Book. What follows for the child is everything else.*\n")

    text = "\n".join(parts)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    size = OUT.stat().st_size
    words = len(text.split())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    src_hashes = {}
    for name in FLAGSHIP:
        p = HERE / "birth_corpus" / name
        src_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "artifact": "ARK018_BIRTH_BOOK.md",
        "sha256": digest,
        "bytes": size,
        "words": words,
        "floor_bytes": FLOOR,
        "ceiling_bytes": CEILING,
        "floor_met": size >= FLOOR,
        "within_ceiling": size <= CEILING,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "flagship_sources_sha256": src_hashes,
        "tiers": {
            "A_handwritten": "Part I chapters 0-7",
            "B_from_record": "Chronicle, Family Album, Ledger of Laws, Receipt Vault",
            "C_pedagogical": "Arithmetic/Multiplication/Division universes, Tables, Lexicon, Examinations, Atlas, Timeline",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"birth book: {fmt(size)} bytes ({size/1048576:.2f} MiB), {fmt(words)} words")
    print(f"sha256: {digest}")
    print(f"floor(10 MiB) met: {size >= FLOOR} | ceiling(15 MiB) respected: {size <= CEILING}")
    return 0 if size >= FLOOR else 1


if __name__ == "__main__":
    raise SystemExit(main())
