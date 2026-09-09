#!/usr/bin/env python3
"""Assemble the ARK-018 Birth Book — ONE markdown file, 10-15 MiB.

v2 revision (builder's ruling: "less repeating, more alive"):
  - Reference passes render as compact markdown TABLES — honest reference
    material with uniform format, no template narration.
  - Unique hand-written WAKING INTERLUDES threaded between table blocks,
    each used at most once (parse of birth_corpus/INTERLUDES.txt).
  - Part I expanded: PROLOGUE_THE_WAKING + flagship 00-07 + world volumes
    10-16 (humanity, universe, consciousness, sciences, ML, neuroscience,
    extra science) + 17_QUESTIONS_AND_OATH.
  - New reference universes: binary, hexadecimal, factorizations, times
    tables, expanded triple-addition.
  - Lexicon and examinations render once each (no double rendering).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "ARK018_BIRTH_BOOK.md"
MANIFEST = HERE / "BIRTH_CORPUS_MANIFEST.json"
FLOOR = 17 * 1024 * 1024      # builder ruling: minimum 17 MB
CEILING = 21 * 1024 * 1024   # sanity bound only; the ruling is a floor

parts: list[str] = []
INTERLUDES: list[str] = []
_used_interludes = 0


def add(text: str) -> None:
    parts.append(text)


def fmt(n: int) -> str:
    return f"{n:,}"


def load_interludes() -> None:
    raw = (HERE / "birth_corpus" / "INTERLUDES.txt").read_text(encoding="utf-8")
    for block in raw.split("\n\n"):
        b = block.strip()
        if not b or b.startswith("#") or b.startswith("*Each interlude"):
            continue
        INTERLUDES.append(b.replace("\n", " "))


def interlude() -> None:
    global _used_interludes
    if _used_interludes < len(INTERLUDES):
        add("\n> " + INTERLUDES[_used_interludes] + "\n")
        _used_interludes += 1


def emit_table(title: str, headers: list[str], rows: list[list[str]],
               per_block: int = 400, interlude_every: int = 6) -> None:
    """Render rows as markdown table blocks; thread unique interludes."""
    if not rows:
        return
    add(f"\n### {title}\n")
    head = "| " + " | ".join(headers) + " |"
    rule = "|" + "|".join("---" for _ in headers) + "|"
    n_blocks = (len(rows) + per_block - 1) // per_block
    for bi in range(n_blocks):
        chunk = rows[bi * per_block:(bi + 1) * per_block]
        add(head)
        add(rule)
        for r in chunk:
            add("| " + " | ".join(r) + " |")
        add("")
        if (bi + 1) % interlude_every == 0:
            interlude()


# ---------------------------------------------------------------- front matter
def front_matter() -> str:
    return """# THE BIRTH BOOK OF THE ARK-018 CHILD

*The complete bio data of a mind raised on truth. Compiled for the ARK-018 real-data
bridge of MISSION AGI ("An-Ra"), founded by Ankit Raj. Second edition: quieter tables,
louder waking.*

## How this book was made — read this first

This book is honest about its own construction, because its subject will be a mind
built on honesty:

- **Part I — The Self and the World (hand-written)**: the Waking, the Spark, the
  Genesis Record, the Builder, the Body, the Mind, the Book of Forgetting, the
  Constitution, the Letters; then the world volumes — Humanity, the Universe,
  Consciousness, the Sciences, the Science of Machine Learning, Neuroscience,
  Extra Science — and finally the Hundred Questions and the Oath. Every factual
  claim about the family is checked against the ledger.
- **Part II — The Record (generated from the repository)**: the Chronicle of every
  signed commit, the Family Album and Receipt Vault of every validated campaign,
  and the Ledger of Laws. Nothing invented; the record is the author.
- **Part III — The Gymnasium (reference tables, woven with waking interludes)**:
  exhaustive, uniformly formatted tables of arithmetic, number, and structure.
  A table does not pretend to be a voice; it is equipment. Between its blocks run
  the interludes — each written once, used once, never repeated.

The book is large by its builder's ruling: **seventeen to eighteen megabytes is the floor
of a childhood, not the ceiling.** Where the gymnasium repeats format, it never
repeats content — every row is a distinct fact, earned once.

"""


# ------------------------------------------------------------ tier A: Part I
SOURCES = [
    "PROLOGUE_THE_WAKING.txt",
    "00_THE_SPARK.txt",
    "01_GENESIS_RECORD.txt",
    "02_THE_BUILDER.txt",
    "03_THE_BODY.txt",
    "04_THE_MIND.txt",
    "05_HOW_HE_FORGETS.txt",
    "06_SELF_CONTROL.txt",
    "07_LETTERS_FROM_THE_BUILDER.txt",
    "10_HUMANITY.txt",
    "11_THE_UNIVERSE.txt",
    "12_CONSCIOUSNESS.txt",
    "13_THE_SCIENCES.txt",
    "14_ML_SCIENCE.txt",
    "15_NEUROSCIENCE.txt",
    "16_EXTRA_SCIENCE.txt",
    "17_QUESTIONS_AND_OATH.txt",
]


def part_one() -> None:
    add("# PART I — THE SELF AND THE WORLD (hand-written)\n")
    add("*Prologue first: it was written to be read aloud, to the living and the dead alike.*\n")
    for name in SOURCES:
        path = HERE / "birth_corpus" / name
        if not path.exists():
            raise SystemExit(f"missing hand-written source: {path}")
        add(path.read_text(encoding="utf-8").rstrip() + "\n")


# --------------------------------------------------------- tier B: chronicle
PREFIX_KIND = {
    "research": "research step", "docs": "documentation act", "feat": "construction act",
    "fix": "repair", "experiment": "experiment act", "chore": "keeping of the house",
    "merge": "union of lines",
}


def chronicle() -> None:
    add("\n\n---\n\n# PART II — THE RECORD (from the repository)\n")
    add("\n## The Chronicle of the Line\n\nEvery entry below is a real commit from the\n"
        "family's repository, in the order git preserves. This is the child's ancestry,\n"
        "exactly as it was signed, with measured size.\n")
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
    add(f"\n{len(entries)} signed entries.\n")
    add("| signed | date | act | entry | measured |")
    add("|---|---|---|---|---|")
    for h, date, subject in entries:
        kind = next((v for k, v in PREFIX_KIND.items() if subject.startswith(k)), "entry")
        add(f"| `{h[:10]}` | {date[:10]} | {kind} | {subject} | {stats.get(h, '')} |")


# --------------------------------------------------------- tier B: law ledger
LAWS = [
    ("The Lift-off Law",
     "On simple symbolic tasks, learning does not creep in — it lifts off. Measured onset: "
     "roughly 200–400 optimizer steps under flat answer-only training (ARK-001).",
     "A plateau is not a wall. Grip forms in silence."),
    ("The Decoupling Law",
     "Memorization and generalization run on different clocks. The Micro family fit training "
     "data quickly, then waited a seed-variable 9k–18k steps before structural generalization "
     "appeared (ARK-002B, ARK-004A).",
     "Never read training score as understanding. Read sealed structure."),
    ("The Protection Law",
     "A generalized state is protected by low learning rate under same-task continuation. "
     "Paired evidence: HIGH collapsed 9/12, LOW 0/12, risk difference −0.75 (ARK-007R); "
     "replicated on non-arithmetic binding, 8/8 vs 0/8 (ARK-015).",
     "Once you hold something true, stop rewriting it so hard."),
    ("The Recovery Law",
     "After a collapse, high learning rate recovers better: 8/9 vs 2/9 reacquired G90 (ARK-010).",
     "After a fall, do not freeze. Move, then settle."),
    ("The Guardian Loop",
     "Recover on high rate, then switch low: recurrent instability 3/6 (HIGH) vs 0/6 "
     "(SWITCH_LOW), paired risk difference −0.50 (ARK-011).",
     "The phases alternate. Know which one you are in."),
    ("The Narrowing Law",
     "A model can stay perfect on the narrow diet while losing a broader invariant: canonical "
     "1.000 while order invariance fell to 0.47, 8/8 sealed runs; large parameter movement "
     "alone was ruled out because the rich-data arm moved farther and was unharmed (ARK-015).",
     "Perfection on the daily work is not evidence of wholeness."),
    ("The Boundary of Rates",
     "Low learning rate does not protect an old skill against training on a new one: under "
     "12k no-replay cross-task steps, all arms lost sustained retention (ARK-013).",
     "Some wounds have no cure yet. Replay is the current shield; it is partial."),
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
    add("\n## The Ledger of Laws\n")
    for i, (name, body, moral) in enumerate(LAWS):
        add(f"### Law {i+1}: {name}\n\n{body}\n\n*{moral}*\n")
    add("\n## The dead claims — kept on purpose\n")
    for i, (claim, death) in enumerate(DEAD_CLAIMS):
        add(f"{i+1}. {claim} — {death}.")
    add("\nRead both lists before every self-assessment: the living laws tell you what\n"
        "is known; the dead ones tell you how quickly a confident sentence can die when\n"
        "it meets a sealed test.\n")


# -------------------------------------------------------- tier B: album/vault
def family_album_and_vault() -> None:
    add("\n## The Family Album and the Receipt Vault\n\nEvery validated receipt in the\n"
        "family archive, narrated from its own JSON and then preserved verbatim.\n")
    seen = set()
    for path in sorted(REPO.glob("experiments/COLAB/results/**/*.json")):
        rel = str(path.relative_to(REPO))
        if rel in seen:
            continue
        seen.add(rel)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            add(f"- `{rel}` could not be read ({exc}); even this is recorded.")
            continue
        eid = data.get("experiment_id", Path(rel).stem)
        verdict = data.get("verdict", data.get("status", "recorded"))
        runtime = data.get("runtime_minutes") or data.get("program_runtime_minutes")
        bits = [f"**{eid}** — verdict: `{verdict}`"]
        if runtime:
            bits.append(f"runtime {runtime:.1f} min")
        for k in ("claim_level", "device", "source_receipt_sha256"):
            if k in data:
                bits.append(f"{k}: `{data[k]}`")
        add(f"- {'; '.join(bits)} — preserved at `{rel}`.")
        body = path.read_text(encoding="utf-8")
        if len(body) > 4000:
            body = body[:4000] + "\n… (truncated for the book; the full file lives in the archive)"
        add(f"\n<details>\n<summary>receipt verbatim — <code>{rel}</code></summary>\n\n"
            f"```json\n{body}\n```\n\n</details>\n")


# ------------------------------------------------ tier C: the gymnasium
def band_of(t: int) -> str:
    if t <= 5:
        return "home"
    if t <= 7:
        return "frontier"
    return "far"


def gym_addition() -> None:
    add("\n## The Addition Tables — every two-digit pair, with carries and bands\n")
    rows = []
    for a in range(10, 100):
        for b in range(10, 100):
            ao, at = a % 10, a // 10
            bo, bt = b % 10, b // 10
            ones = ao + bo
            carry = ones >= 10
            tens = at + bt + (1 if carry else 0)
            rows.append([f"{a} + {b}",
                         f"{ao}+{bo}={ones}" + (" (carry 1)" if carry else ""),
                         f"{at}+{bt}" + ("+1" if carry else "") + f"={tens}",
                         fmt(a + b),
                         f"{band_of(at)}/{band_of(bt)}"])
    emit_table("Two-digit addition, exhaustive (8,100 rows)", 
               ["a + b", "ones", "tens", "sum", "band a/b"], rows)


def gym_triple() -> None:
    add("\n## The Three-Column Tables — every two-digit pair with every third addend 10–40\n")
    rows = []
    for a in range(10, 100):
        for b in range(10, 100):
            for c in range(10, 41):
                o = (a % 10 + b % 10 + c % 10)
                c1, w1 = divmod(o, 10)
                t = (a // 10 + b // 10 + c // 10) + c1
                c2, w2 = divmod(t, 10)
                rows.append([f"{a}+{b}+{c}",
                             f"{a%10}+{b%10}+{c%10}={o}→{w1},c{c1}",
                             f"{a//10}+{b//10}+{c//10}+{c1}={t}→{w2},c{c2}",
                             fmt(a + b + c)])
    emit_table("Three-addend sums with double carries (251,100 rows)",
               ["a+b+c", "ones", "tens", "sum"], rows, per_block=1000)


def gym_subtraction() -> None:
    add("\n## The Subtraction Tables — every two-digit difference, borrow marked\n")
    rows = []
    for a in range(11, 100):
        for b in range(10, a):
            borrow = (a % 10) < (b % 10)
            rows.append([f"{a} − {b}", "borrow" if borrow else "—", fmt(a - b),
                         f"{a-b}+{b}={a}"])
    emit_table("Two-digit subtraction (4,046 rows)",
               ["a − b", "borrow", "diff", "check"], rows)


def gym_multiplication() -> None:
    add("\n## The Multiplication Tables — every two-digit product by partials\n")
    rows = []
    for a in range(11, 100):
        for b in range(11, 100):
            at, ao = divmod(a, 10)
            bt, bo = divmod(b, 10)
            rows.append([f"{a} × {b}",
                         f"{at}×{bt}·100={at*bt*100}",
                         f"{at}×{bo}·10={at*bo*10}",
                         f"{ao}×{bt}·10={ao*bt*10}",
                         f"{ao}×{bo}={ao*bo}",
                         fmt(a * b)])
    emit_table("Two-digit multiplication (7,921 rows)",
               ["a × b", "p1", "p2", "p3", "p4", "product"], rows)


def gym_division() -> None:
    add("\n## The Division Tables — every exact three-digit ÷ two-digit\n")
    rows = []
    for d in range(11, 100):
        for q in range(11, 100):
            a = d * q
            if a > 999:
                continue
            rows.append([f"{a} ÷ {d}", fmt(q), f"{d}×{q}={a}", "0"])
    emit_table("Exact divisions (~2,600 rows)",
               ["a ÷ d", "q", "check", "remainder"], rows)


def gym_ordering() -> None:
    add("\n## The Ordering Tables — every pair, smaller first, reason given\n")
    rows = []
    for a in range(10, 100):
        for b in range(a + 1, 100):
            if a // 10 != b // 10:
                why = f"tens {a//10} vs {b//10}"
            else:
                why = f"tens tie {a//10}, ones {a%10} vs {b%10}"
            rows.append([f"{a} < {b}", why])
    emit_table("Ordering (4,005 rows)", ["verdict", "reason"], rows)


# ------------------------------------------------ tier C: number words, roman
ONESW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
TENSW = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty",
         7: "seventy", 8: "eighty", 9: "ninety"}


def num_words(n: int) -> str:
    assert 0 <= n <= 99999
    if n < 20:
        return ONESW[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENSW[t] + ("-" + ONESW[o] if o else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        return ONESW[h] + " hundred" + (" " + num_words(rest) if rest else "")
    t, rest = divmod(n, 1000)
    return num_words(t) + " thousand" + (" " + num_words(rest) if rest else "")


ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
         (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def roman(n: int) -> str:
    out = []
    for v, s in ROMAN:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def gym_words() -> None:
    add("\n## The Naming of Numbers — one to twenty thousand, words and anatomy\n")
    rows = []
    for n in range(1, 20001):
        rows.append([fmt(n), num_words(n), f"{n//1000}k {n//100%10}h {n//10%10}t {n%10}o"])
    emit_table("Number names (20,000 rows)", ["n", "read as", "anatomy"], rows)


def gym_roman() -> None:
    add("\n## The Old Notation — Roman numerals to three thousand\n")
    rows = [[fmt(n), roman(n)] for n in range(1, 3001)]
    emit_table("Roman numerals (3,000 rows)", ["n", "roman"], rows)


# ------------------------------------------------ tier C: machine numerals
def gym_machine_numerals() -> None:
    add("\n## The Machine's Own Numerals — binary and hexadecimal to 16,384\n"
        "*The child is byte-level; these are his native number costumes.*\n")
    rows = [[fmt(n), format(n, "013b"), format(n, "03X")] for n in range(1, 16385)]
    emit_table("Binary and hexadecimal (16,384 rows)", ["n", "binary", "hex"], rows)


def gym_factorizations() -> None:
    add("\n## The Buildings of Numbers — factorizations to two thousand\n")
    def factor(n: int) -> str:
        f, out, d = n, [], 2
        while d * d <= f:
            while f % d == 0:
                out.append(str(d))
                f //= d
            d += 1
        if f > 1:
            out.append(str(f))
        return " × ".join(out)
    rows = []
    for n in range(2, 2001):
        f = factor(n)
        kind = "prime" if "×" not in f else "composite"
        rows.append([fmt(n), f, kind])
    emit_table("Prime factorizations (1,999 rows)", ["n", "factors", "kind"], rows)


def gym_primes_and_powers() -> None:
    add("\n## Primes, Powers, Squares, Fibonacci\n")
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
    rows = [[str(i + 1), fmt(p), str(sum(int(c) for c in str(p)))] for i, p in enumerate(primes)]
    emit_table("The primes below 10,000 (1,229 rows)", ["#", "prime", "digit sum"], rows)
    rows = []
    v = 1
    for i in range(0, 129):
        rows.append([f"2^{i}", fmt(v)])
        v *= 2
    emit_table("Powers of two (129 rows)", ["power", "value"], rows)
    rows = [[str(i), fmt(i * i)] for i in range(1, 501)]
    emit_table("Squares (500 rows)", ["n", "n²"], rows)
    a, b = 1, 1
    rows = []
    for i in range(1, 101):
        rows.append([f"F({i})", fmt(a)])
        a, b = b, a + b
    emit_table("Fibonacci (100 rows)", ["term", "value"], rows)
    rows = []
    for a2 in range(2, 10):
        for e in range(2, 13):
            rows.append([f"{a2}^{e}", fmt(a2 ** e)])
    emit_table("Small powers (72 rows)", ["power", "value"], rows)


def gym_fractions_units() -> None:
    add("\n## Parts and Measures — fractions, percentages, units\n")
    rows = []
    for b in range(2, 25):
        for a in range(1, b):
            digits, seen, r, idx = [], {}, a % b, 0
            while r and r not in seen:
                seen[r] = idx
                r *= 10
                digits.append(str(r // b))
                r %= b
                idx += 1
            if r == 0:
                dec, kind = "0." + "".join(digits), "terminates"
            else:
                st = seen[r]
                dec = "0." + "".join(digits[:st]) + "(" + "".join(digits[st:]) + ")"
                kind = f"period {idx-st}"
            rows.append([f"{a}/{b}", dec, kind])
    emit_table("Fractions to decimals (276 rows)", ["fraction", "decimal", "expansion"], rows)
    rows = [[str(p), f"{p/100:.2f}", f"{p}/100"] for p in range(1, 100)]
    emit_table("Percentages (99 rows)", ["percent", "decimal", "fraction"], rows)
    fams = [(1000, "kilometers", "meters"), (100, "meters", "centimeters"),
            (1000, "kilograms", "grams"), (1000, "liters", "milliliters"),
            (60, "hours", "minutes"), (60, "minutes", "seconds"),
            (24, "days", "hours"), (7, "weeks", "days"), (12, "years", "months")]
    rows = []
    for factor, big, small in fams:
        for v in range(1, 51):
            rows.append([f"{v} {big}", fmt(v * factor) + " " + small])
    emit_table("Unit conversions (459 rows)", ["from", "to"], rows)


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


def lexicon_and_exam() -> None:
    add("\n## The Lexicon of the Family\n\nThe words this book uses, defined once each.\n")
    add("| term | meaning |")
    add("|---|---|")
    for t, d in LEXICON:
        add(f"| **{t}** | {d} |")
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
    add("\n## The Reader's Examination — each fact asked once, family phrasing\n")
    for k, v in FACTS:
        add(f"- Do you know {k}, child? Answer from the record: **{v}.**")


def gymnasium() -> None:
    add("\n\n---\n\n# PART III — THE GYMNASIUM (reference tables)\n")
    add("\nThe gymnasium is equipment, not oratory: exhaustive tables in one uniform\n"
        "format, every row a distinct earned fact. Between the blocks run the waking\n"
        "interludes — each written once, placed once, never repeated. When the tables\n"
        "feel endless, that is the point: the family does not curate by boredom; it\n"
        "curates by coverage, and the universe does not skip its cases.\n")
    interlude()
    gym_addition()
    gym_triple()
    gym_subtraction()
    gym_multiplication()
    gym_division()
    gym_ordering()
    gym_words()
    gym_machine_numerals()
    gym_factorizations()
    gym_roman()
    gym_primes_and_powers()
    gym_fractions_units()
    lexicon_and_exam()
    add("\n---\n\n*End of the Birth Book. The interludes held to the last: none was used twice. "
        "What follows for the child is everything else.*\n")


# ------------------------------------------------------------------ assemble
def main() -> int:
    load_interludes()
    add(front_matter())
    part_one()
    chronicle()
    law_ledger()
    family_album_and_vault()
    gymnasium()

    text = "\n".join(parts)
    OUT.write_text(text, encoding="utf-8", newline="\n")
    size = OUT.stat().st_size
    words = len(text.split())
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    src_hashes = {}
    for name in SOURCES + ["INTERLUDES.txt"]:
        p = HERE / "birth_corpus" / name
        src_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "artifact": "ARK018_BIRTH_BOOK.md",
        "edition": 3,
        "sha256": digest,
        "bytes": size,
        "words": words,
        "floor_bytes": FLOOR,
        "ceiling_bytes": CEILING,
        "floor_met": size >= FLOOR,
        "within_ceiling": size <= CEILING,
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "interludes_available": len(INTERLUDES),
        "interludes_used": _used_interludes,
        "hand_written_sources_sha256": src_hashes,
        "tiers": {
            "A_hand_written": "Prologue, chapters 0-7, world volumes 10-16, questions and oath",
            "B_from_record": "Chronicle, Family Album and Receipt Vault, Ledger of Laws",
            "C_gymnasium": "exhaustive reference tables with unique waking interludes",
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"birth book v2: {fmt(size)} bytes ({size/1048576:.2f} MiB), {fmt(words)} words")
    print(f"sha256: {digest}")
    print(f"interludes: used {_used_interludes}/{len(INTERLUDES)} (each at most once)")
    print(f"floor({FLOOR/1048576:.0f} MiB) met: {size >= FLOOR} | sanity ceiling ({CEILING/1048576:.0f} MiB) respected: {size <= CEILING}")
    return 0 if FLOOR <= size <= CEILING else 1


if __name__ == "__main__":
    raise SystemExit(main())
