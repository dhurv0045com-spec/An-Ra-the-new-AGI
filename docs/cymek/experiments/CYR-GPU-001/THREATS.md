# CYR-GPU-001 THREATS — preregistered confounds and mitigations

1. Near-freezing masquerading as consolidation. Mitigation: displacement
   norms per arm; redteam_freezing ratio floor 5.0; MID arm spans the gap.
2. Unequal low-LR exposure across LR arms. Mitigation: prospective fixed
   rule; exposure table receipted; >2× mismatch flags confounded.
3. Train/eval latent-world leakage. Mitigation: disjoint seeds+ranges,
   firewall assertion, overlap redteam, split manifests hashed.
4. Near-duplicate worlds. Mitigation: MinHash redteam on train texts.
5. Query-blind / positional policies. Mitigation: both-correct primary;
   copy-first/last/most-frequent/fixed-position baselines; blind gap.
6. Answer-frequency shortcut. Mitigation: unique values per world by
   construction (keys/values embed seed+index); frequency baseline.
7. Order shortcut. Mitigation: order_only factorial variant; shuffled
   control sees identical multisets.
8. Stopping differences (EOS vs cap). Mitigation: EOS contract asserted
   pre-training; stop histogram at stage ends; uniform decoding.
9. Cherry-picked snapshots. Mitigation: sustained-3 transitions only;
   full eval traces receipted.
10. Seed dependence. Mitigation: 2 matched seeds minimum; seed reported.
11. Controller leakage (sealed/dev-measurement into control). Mitigation:
    controller inputs dev-controller worlds only (code path takes no
    other split); sealed_reserved never consumed; audit in receipts.
12. Checkpoint/reload mismatch. Mitigation: forks reload through the
    verifying restore path; resume-equality covered by production tests.
13. Parameter non-mutation. Mitigation: before/after param SHA per arm.
14. BPE answer-boundary merges. Mitigation: renderer verifies exact sizes,
    fail-closed; train_arm re-verifies every record.
15. Capacity ceiling (proxy too small to show effects). Mitigation:
    learnability gate aborts honestly; resolver picks largest fitting
    proxy; larger-proxy transfer stage when time allows.
16. Colab preemption / time overrun. Mitigation: per-arm receipts,
    graceful stage skips, finally-path packaging, hard 180-min stop.
17. Rendered-task realism gap. Mitigation: acknowledged; claim ladder
    caps at MULTI_TASK_REPLICATED; transfer family + proxy scale-up
    required before any production claim.
