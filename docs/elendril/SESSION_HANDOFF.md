# Elendril research handoff

**V2.** See the [current state](CURRENT_STATE.md), [engineering audit](ENGINEERING_AUDIT_V2.md), [ELD-DEV-004 results and replay](ELD_DEV_004_RESULTS.md), and [T4 operator budget](HARDWARE_INVENTORY_AND_BUDGET.md). Use the source named in each receipt when restoring its checkpoint. The historical first-session material follows.

**Current branch experiment.** See the [capacity-matched relational comparison](RELATIONAL_RESULTS.md). The hybrid read head passed the ELD-DEV-003 formation and stress gates. The original two-fact follow-up remains documented below.

**Later work.** See the [associative architecture and replay instructions](ASSOCIATIVE_ARCHITECTURE.md) and [ELD-DEV-002 results](ASSOCIATIVE_RESULTS.md). The entries below document ELD-DEV-001 and its pinned checkout.

The working branch is **Elendril** in the sibling **An-Ra-Elendril** worktree. Starting lineage is `feac2948466832516b7f048435483fd95e53b078` via charter `448f3bafedbbd8d9e8180eb0d85c04e6cccae0dd`. The [evidence index](EVIDENCE_INDEX.json) gives pinned branch heads, original records and their hashes. The only new model weights are the locally initialized development instruments.

## Reproduce

Use Python 3.11 with the official `torch==2.14.0` CPU wheel and pytest. Run from the worktree root. These PowerShell commands reproduce the 51-test focused suite.

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m pytest -q tests/test_elendril.py tests/test_elendril_longitudinal.py tests/test_v5_model.py tests/test_v5_objectives.py tests/test_v5_checkpointing.py tests/test_v5_evaluation.py tests/test_v5_production_sampler.py tests/test_v5_tokenizer.py tests/test_v5_step_schedule_trainer.py
~~~

For exact experimental source binding, create a detached checkout of `cc8e50a5bafa9bf1adfd278b9ee88139973caa98` and run the following from that checkout. Choose a fresh output directory. The repository's receipt manifest and measured restart checks bind to the actual completed runs.

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$elendrilReplay = Join-Path $env:TEMP "ELD-DEV-001-replay"
New-Item -ItemType Directory -Path $elendrilReplay
python -m anra_v5.elendril audit --output "$elendrilReplay/fixture-audit.json"
foreach ($elendrilSeed in 7,19,41) {
    foreach ($elendrilArm in "copy","binding","binding-permuted") {
        $elendrilReceipt = "$elendrilReplay/$elendrilArm-$elendrilSeed.json"
        python -m anra_v5.elendril pilot --arm $elendrilArm --seed $elendrilSeed --steps 350 --root "$elendrilReplay/checkpoints" --output $elendrilReceipt
        if ($LASTEXITCODE -ne 0) { throw "pilot failed" }
        python -m anra_v5.elendril rescore --receipt $elendrilReceipt
        if ($LASTEXITCODE -ne 0) { throw "rescore failed" }
    }
}
~~~

For a clean process restart smoke, use `--arm binding --seed 7 --steps 8` on two fresh lineage roots. Run the first continuously, and run the second once with `--until 3` and then with `--resume` and a new receipt file. Compare final model, optimizer, RNG, scheduler, cursor and ledger component hashes. The [measured restart check](../../artifacts/elendril/eld-dev-001/restart-verification.json) used source `5f2b404ffc8a3594abaecabd61c2456ac8474259`, whose Python source is identical to the registered pilot. Source, training budget, seed, environment and data are resume identities.

To reproduce the binding diagnostic, use the pinned pilot checkout and the local pilot checkpoint root. For each of the three seeds and two binding arms, regenerate the training fixture and apply the exact label permutation for the permuted arm. Restore `Trainer(rows, seed=seed, steps=350, root=checkpoint_root, lineage=f"{arm}-{seed}", source_commit=source, resume=True)` after calling `configure_cpu()`. Call `generate` with (task ID, prompt) tuples and `max_new_tokens=3`. Score the same frozen outputs against both the actual training fixture and clean training fixture. Preserve zero optimizer updates. The [diagnostic receipt](../../artifacts/elendril/eld-dev-001/binding-diagnostic.json) contains all predicted tokens and both scores.

## Custody and next decision

- Git contains raw predictions, step telemetry, manifest hashes and scoring snapshots in [ELD-DEV-001](../../artifacts/elendril/eld-dev-001/manifest.json). Local checkpoint payloads live in **C:/Users/ankit/Downloads/Elendril-ELD-DEV-001**, separate from the worktree. The receipt manifest lists 30 verified local checkpoint inventories.
- The 42,512-parameter reference passed the copy gate in all three seeds. Binding failed on held-out worlds and on the training set. [Reproducibility report](REPRODUCIBILITY_REPORT.md).
- First build and preregister the matched paired-sampling contrast. Pursue original S5 artifact recovery in parallel if the Kaggle Output is available. Keep the retention and forward-transfer phases gated on independently qualified skills. [Next experiment](NEXT_EXPERIMENT_DECISION.md).
- Review branch commits with `git log --oneline 448f3baf..Elendril`. Work was committed locally.
