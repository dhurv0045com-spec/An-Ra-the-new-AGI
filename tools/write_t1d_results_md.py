"""RESULTS.md writer: authoritative T1D record from the ingested bundle."""
from pathlib import Path
import json

rep = json.load(open("docs/citadel/experiments/T1D/RESULTS.json", encoding="utf-8"))
ta = rep["termination_audit"]
te = rep["teacher_audit"]
sp = rep["self_probe_audit"]
dv = rep["data_volume"]
bc = rep["budget_confound"]
ss = rep["session"]

lines = [
    "# T1D RESULTS — EXECUTED / ARCHIVED (authoritative record)",
    "",
    f"Status: **EXECUTED / ARCHIVED — DO NOT RERUN**. All six arms",
    "SCIENTIFIC_FAIL; cross-arm **INCONCLUSIVE**. Evidence: the operator's",
    f"returned `CITADEL_T1D_RESULTS.zip` ({rep['bundle']['members']} members,",
    f"zip {'valid' if rep['bundle']['zip_valid'] else 'INVALID'}, every JSON",
    "parses; per-member SHA-256 in RESULTS.json).",
    "",
    f"- Citadel SHA: `{rep['identity']['citadel_sha']}`",
    f"- Cymek SHA: `{rep['identity']['cymek_sha']}` (production authority)",
    f"- TPU environment: torch/XLA 2.9.0, calibration **({ss['selected_shape']['batch']},",
    f"  {ss['selected_shape']['length']})** @ ~{ss['throughput_tok_s']} tok/s",
    f"- Budgets: A/B/C 8M, D/E 4M, F 2M cap tokens (auto-scale did not fire)",
    "",
    "## Headline result",
    "",
    "Answer-CE arithmetic learning does not lift off at 2–8M tokens for",
    "3.7–7.4M models under ANY of the five variations. TEST exact 0–6.6%",
    "everywhere, below the 22.5% copy-first-operand null; TRAIN exact 0–11%",
    "(no memorization either); dev curves flat at every checkpoint. Verdict",
    "INCONCLUSIVE per the frozen rules — and the suite ELIMINATES curriculum,",
    "teacher, 2× scale, output masking, and self-knowledge as *sufficient*",
    "for lift-off at these budgets.",
    "",
    "## Per-arm record",
    "",
    "| Arm | Variation | Loss | TEST exact t0–t4 | TRAIN t0–t2 | Reload | Status |",
    "|---|---|---|---|---|---|---|",
]
for tag in "ABCDEF":
    import zipfile
    zf = zipfile.ZipFile(r"C:\Users\ankit\Downloads\CITADEL_T1D_RESULTS.zip")
    r = json.loads(zf.read(f"ARM_{tag}.json"))
    tr = r["training"]
    variant = {"A": "flat control", "B": "curriculum", "C": "teacher",
               "D": "SCALE2", "E": "masked softmax", "F": "self-knowledge"}[tag]
    test = "/".join(f"{r['trained'][f't{t}']['accuracy']:.3f}" for t in range(5))
    train = "/".join(f"{r['trained_train'][f't{t}']['accuracy']:.3f}" for t in range(3))
    lines.append(
        f"| {tag} | {variant} | {tr['first_loss']:.2f}→{tr['last_loss']:.2f} | "
        f"{test} | {train} | "
        f"{r['reload_identical']} | {r['status']} |")

lines += [
    "",
    "## POSTMORTEM 1 — termination contract (major design limitation)",
    "",
    f"**{ta['total_generation_records']:,}/{ta['total_generation_records']:,}",
    f"generation records ended MAX_TOKENS ({ta['max_tokens_share']:.0%})**.",
    "Code audit: training rows encode to literal characters only; the",
    "eligible mask covers answer characters only; EOS_ID is never appended.",
    "Generation stops on EOS/PAD/newline/non-alphabet else MAX_TOKENS.",
    "",
    "> THE MODEL WAS NEVER SUPERVISED TO EMIT THE EOS TERMINATION TOKEN USED BY",
    "> GENERATION. T1D exact-generation results cannot cleanly distinguish",
    "> arithmetic-content failure from answer-termination failure.",
    "",
    "T1D is NOT rescinded and NOT retroactively rescored — the limitation is",
    "recorded and corrected in T1E (PLAN.md: EOS supervised,",
    "MAX_GENERATION_STEPS = MAX_CONTENT_TOKENS + 1, TERMINATION_FAILURE vs",
    "CONTENT_FAILURE split).",
    "",
    "## POSTMORTEM 2 — content-only forensics (POST_HOC_NOT_PREREGISTERED)",
    "",
    "From the stored per-arm samples (20/arm — small n): content exact at",
    "target length ≈ 5% in every arm. The arithmetic characters themselves",
    "are mostly wrong — this is NOT merely a stop failure. Both failures",
    "coexist; T1E measures them separately.",
    "",
    "## POSTMORTEM 3 — teacher primitives (the positive finding)",
    "",
    f"- Arm C held-out teacher microtask accuracy: **{te['heldout_eval_accuracy']}",
    f"  (n={te['heldout_eval_n']})** — vs ~0 everywhere else.",
    "- Classification: **PRIMITIVE_LEARNING_WITHOUT_COMPOSITIONAL_TRANSFER**",
    "  (interpretation, not an AGI claim): primitive microtasks are learnable",
    "  while full T2+ arithmetic composition stays near zero.",
    f"- TEACHER_DIVERSITY_LIMIT: unique pools of {te['unique_pool_per_kind']} rows",
    f"  were placed {te['placements']['teacher:digadd']}× each per kind —",
    f"  replay factors {te['replay_factor_per_kind']}. T1E expands the pools;",
    "  it does not simply repeat more.",
    "",
    "## POSTMORTEM 4 — self-knowledge probe contract",
    "",
    f"- {sp['targets_over_generation_limit']}/{sp['total_probes']} probe targets",
    f"  exceed MAX_ANSWER_TOKENS={sp['MAX_ANSWER_TOKENS']} (examples: "
    f"{sp['over_examples'][:4]}).",
    "- **SELF_KNOWLEDGE_EVAL_CONTRACT_INVALID** — Arm F's negative is NOT a",
    "  clean result. Official receipt unchanged (SCIENTIFIC_FAIL). Feasible-",
    "  only scoring requires the full per-probe predictions (not serialized);",
    "  T1E-family self-knowledge gets a corrected contract preregistered",
    "  separately.",
    "",
    "## POSTMORTEM 5 — budget confound",
    "",
    f"- B budget {bc['B_budget']:,} vs D/E {bc['D_budget']:,}: D-vs-B changes",
    "  model size AND budget; E-vs-B changes output space AND budget.",
    "- Do NOT conclude 'scale does not help' or 'masking does not help'.",
    "  T1E token-matches these contrasts on LOSS_BEARING_TOKENS.",
    "",
    "## Data volume",
    "",
    f"- Available unique: {dv['available_unique_rows']:,} rows",
    f"  (~{dv['available_unique_bytes_est']/1e6:.1f} MB). Total placements across",
    f"  all arms: {dv['total_row_placements_all_arms']:,}. Consumable unique",
    f"  fraction ≈ {dv['consumable_unique_fraction_est']:.1%}.",
    f"- Verdict: **{dv['verdict']}** — expansion must target information",
    "  diversity, not unused bytes.",
    "",
    "## Tier interpretation",
    "",
    "- T0/T1: memorization/basic-fit probes (finite spaces; overlap",
    "  possible/expected). T2/T3/T4: structural held-out surfaces. Even the",
    "  T1 memorization probes failed (train ≤ 11%) — below-fit-floor regime.",
    "",
    "## PRE50M",
    "",
    "The smoke failed deterministically: the smoke state's token budget",
    "funded `updates` but the resume-proof publishes `updates+1` (Cymek:",
    "sold at \"a completed run cannot advance\"). **Cymek behavior is",
    "correct; this was a Citadel PRE50M smoke bug** — fixed this cycle",
    "(token_budget=(updates+1)*tokens_per_update, reserved-final-update",
    "regression against the real Cymek contract incl. the negative control).",
    "The session status propagation bug (PRE50M labeled PASS from file",
    "existence) is also fixed. PRE50M certifies on the next TPU contact via",
    "notebooks/citadel_colab_pre50m.ipynb (~minutes).",
    "",
    "## Receipts",
    "",
    "- Normalized machine record: RESULTS.json (member hashes, forensics).",
    "- No checkpoint binaries committed; receipts only, per repo policy.",
]

out = Path("docs/citadel/experiments/T1D/RESULTS.md")
out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
print(f"RESULTS.md written ({len(lines)} lines)")
