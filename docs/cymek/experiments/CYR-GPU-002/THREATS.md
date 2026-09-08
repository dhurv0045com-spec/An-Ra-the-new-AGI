# CYR-GPU-002 THREATS — preregistered confounds and mitigations

1. Near-freezing as consolidation: displacement norms per arm,
   redteam ratio floor 5.0, MID arm spans the gap.
2. Unequal low-LR exposure: prospective fixed rule + per-arm exposure
   tables; >2× mismatch flags confounded (pre-committed).
3. Train/eval world leakage: disjoint seeds+ranges, firewall asserts,
   overlap redteam, hashed manifests.
4. Near-duplicates: MinHash redteam on train texts.
5. Query-blind/positional/frequency shortcuts: both-correct primary +
   six task-aware baselines with precomputed expected scores + blind gap.
6. Order shortcut: order_only factorial variant; identical multisets.
7. Stopping differences: EOS contract pre-training; stop histogram;
   uniform greedy decoding; cap reported separately.
8. Cherry-picking: sustained-3 transitions; full traces receipted.
9. Seed dependence: 2 matched seeds; seed reported per arm.
10. Controller leakage: dev-controller inputs only (code path takes no
    other split); sealed scored once, excluded from decisions (tested).
11. Reload mismatch: byte-verified research checkpoints; fork equality
    proven post-load.
12. Non-mutation: before/after param SHA per arm.
13. BPE boundary merges: renderer verifies exact sizes; train_arm
    re-verifies every record; batch padding with eligible masks.
14. Capacity ceiling: learnability gate aborts honestly; resolver picks
    largest fitting proxy; dose floor 2M before model size.
15. Preemption/overrun: per-update deadlines, TIMEBOX checkpoints,
    per-stage Drive mirror, finally-path packaging, 175-min hard stop.
16. Rendered-task realism gap: claim ladder caps at
    LARGER_PROXY_REPLICATED; transfer family + proxy scale-up required.
17. Weak rendering probe: RENDERING_ONLY is space-join surface change,
    documented weak; no paraphraser exists.
18. Small-eval variance: dense 64-world evals; sustained rules; CIs
    where quoted come from Wilson/exact counts, never vibes.
