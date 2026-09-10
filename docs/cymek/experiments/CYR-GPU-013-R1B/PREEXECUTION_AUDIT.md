# CYR-GPU-013 / R1B — PRE-EXECUTION AUDIT

**Status:** STATIC AUDIT PASS / CUDA PREEXECUTION GATE STILL REQUIRED / NOT EXECUTED

## Frozen identity

Scientific executable commit: `0f56c5e1451391f565d679e1d646ccfc26d96e9f`.

The preregistration binds the R1B core, runner, test, notebook, reused R1/CYR11 machinery, optimizer/model implementation, and frozen ARK-002B manifest by Git blob SHA. The Colab launcher checks out the frozen commit detached and verifies every listed executable blob before tests, calibration, or scientific training.

## Audit checks

PASS at static level:

1. R1B uses fresh prospective model/order seeds; no R1 scientific seed is reused as a replication seed.
2. Active arithmetic tokenization is unchanged across all six vocabulary levels.
3. Within each matched seed, shared non-embedding weights and active embedding rows 0..18 are copied from the same V19 reference through the already-audited R1 initializer.
4. All arms use batch64 and an exact 2,000-update / 128,000-row endpoint. The inherited early G90 stop remains disabled by the R1 fixed-endpoint scope.
5. The vocabulary grid is frozen at 19/1024/4096/8192/16384/24576.
6. Two complete six-level curves are mandatory. Runtime resolution occurs from pre-outcome CUDA calibration only; no scientific outcome enters the resolver.
7. The resolver fails closed if two curves do not conservatively fit. It never lowers exposure or removes a vocabulary level to make the run fit.
8. A third curve is optional and may run only if a complete third curve is projected to fit. Failure of the optional third curve is explicitly isolated and cannot invalidate two already-completed mandatory curves.
9. Arm order is counterbalanced across seeds to reduce simple time/thermal order bias.
10. Completed arms can be reused only through the existing exact-identity R1 arm wrapper and exact fixed endpoint; incompatible completed artifacts abort.
11. The R1 fail-visible decoder remains inherited, so predictions into inactive output IDs cannot silently disappear and receive accidental exact-match credit.
12. The primary decision is frozen before execution: both mandatory seeds must show an intermediate best >=0.60 and an intermediate-vs-extreme gap >=0.30 for `REPLICATED_INTERMEDIATE_CLASS_SPACE_ADVANTAGE`.
13. Previously consumed SEALED rows are not used to make a new sealed-confirmation claim.
14. Final and failure bundles are written to Drive and SHA-256 verified by the notebook download cell.

## Unit-test gate

The notebook runs `py_compile` plus R1B, R1, and inherited CYR-GPU-011 tests before CUDA calibration. Unlike the original R1 launch failure, R1B's synthetic resolver fixtures use deliberately extreme high/low throughput values and only test invariant behavior: fast calibration must allow >=2 curves; grossly slow calibration must fail closed. Scientific runtime is still determined only by the actual Colab calibration.

## Remaining limitations

- This remains a Micro/Cymek arithmetic developmental experiment.
- It maps tied class-space size; because input/output weights are tied, it does not separately identify input-embedding capacity from output-softmax competition.
- The endpoint is intentionally early. It does not resolve late V19 delayed-generalization behavior.
- Two fresh seeds can establish replication of the early response-curve phenomenon, not a universal natural-language tokenizer law.
- GPU execution has not happened yet; a static audit cannot guarantee a particular Colab software/hardware environment.

## Verdict

**READY FOR OPERATOR COLAB CUDA PREEXECUTION GATE.**

Do not claim scientific success unless Cell 0 prints `R1B PREEXECUTION GATE: PASS` and the final result bundle contains at least two complete six-level matched curves.
