# Evidence and provenance

## Immutable source archive

The pre-V3 Elendril tree is available at `930ca3bb723e0b5b191700eec3d5f450875a6e25`. The `elendril-pre-v3` tag identifies the same commit. The branch cleanup removes inherited files from the active checkout through ordinary Git commits. Use the exact archive commit to retrieve old documentation, notebooks, diagnostic scripts and experiment implementations. The original source branch for an inherited file is recorded in the tree inventory. [Pre-V3 archive](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/tree/930ca3bb723e0b5b191700eec3d5f450875a6e25).

Use the original source identity in an experiment receipt before loading its model or optimizer state. A current receipt rescore reuses the historical fixture and scoring interface and does not retrain or restore weights. Historical reports include preregistration, environment, learning curves, denominators, failed arms and qualifications.

| Study | Frozen source | Historical result | Limits and negative evidence |
| --- | --- | --- | --- |
| ELD-DEV-001 | `cc8e50a5bafa9bf1adfd278b9ee88139973caa98` | Copy reached 128/128 exact plus EOS in each of three seeds. Clean binding reached 16/56, 20/56 and 26/56 exact with 0/28 both-correct in every seed. | Parser reached the copy ceiling. Positional binding reached 28/56. Read-only training diagnostics found 0/84 both-correct. [Historical report](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/REPRODUCIBILITY_REPORT.md). |
| ELD-DEV-002 and 002I | `77edcbd8dce7b73d7562e619351a9cc496c6ef84` | Associative binding reached 28/28 both-correct in three seeds. The label-inversion follow-up reached 28/28 for inverted truth and 0/28 for clean truth. | The permuted-label control also reached 28/28 in two seeds. The preregistered permutation separation failed, and renamed-symbol scores were 28/28, 0/28, 14/28. [Historical report](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/ASSOCIATIVE_RESULTS.md). |
| ELD-DEV-003 | `acd4b115d2075340a3db70131cea02d5f1b74c4f` | Hybrid held-out pairs were 44/48, 47/48, 45/48 versus dense 1/48, 0/48, 0/48. Unseen keys were 48/48 in all seeds. | Only 12 disjoint development triples. A fresh seed-7 hybrid replay matched exactly. The historical dense replay diverged from update 1 because its residual initializer used ambient RNG. Historical dense records remain intact. [Historical report](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/RELATIONAL_RESULTS.md) · [replay audit](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/ENGINEERING_AUDIT_V2.md). |
| ELD-DEV-004 | `6615886b1a37fd535682bdee38157e4c44cd0783`, repaired dense `cc38d9850053941097f1d80a4c0b3bbd57f765e7` | Two-read reached 64/64 pairs at three seeds. Matched-update dense reached 0/64, 1/64, 0/64. Rewired two-read reached 63/64, 62/64, 64/64. The alternate-middle intervention redirected 128/128 terminal answers at each seed. | The original dense arms are preserved as superseded. Alias was 1/64, 0/64, 0/64 and three-hop was 0/64 in all seeds. The preregistered robustness gate failed on a development surface with 16 root-pair clusters. [Historical protocol](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/ELD_DEV_004_PROTOCOL.md) · [historical results](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/ELD_DEV_004_RESULTS.md). |

ELD-DEV-001 through 004 were historical CPU experiments. V3 used read-only rescoring and bounded correctness checks. V3 provides candidate code and two owner-run operators; cloud CUDA, persistence canaries and campaign outcomes are pending.

## Artifact and checkpoint custody

The manifests below bind Git-canonical bytes. ELD-DEV-002 also records the original CRLF hashes for three local files. A local checkout with those original bytes is acceptable only where the manifest explicitly binds that alternative. All raw checkpoint payloads remain outside Git.

| Manifest | SHA-256 | Referenced objects | Local checkpoint roots |
| --- | --- | ---: | --- |
| [001](../artifacts/elendril/eld-dev-001/manifest.json) | `9e95fcc1a2827a84ac49f82594a9a0729b49e42b047dcd080290523689155281` | 16 artifacts, 30 checkpoints | `Elendril-ELD-DEV-001` |
| [002](../artifacts/elendril/eld-dev-002/manifest.json) | `85551e565393d97c5350691e2a008230996f6ccb9530ab8017c063187db62301` | 24 artifacts, 75 checkpoints | `Elendril-ELD-DEV-002` |
| [003](../artifacts/elendril/eld-dev-003/manifest.json) | `02436830dfa50b663ef1dfc743056d8f7d29dd9404364b6b6009fb7e46ce8ce1` | 13 artifacts, 48 checkpoints | `Elendril-ELD-DEV-003` |
| [004](../artifacts/elendril/eld-dev-004/manifest.json) | `4cbeecaa841565964f288b2e0cf36ddff5049fc42f125246792a667565e40d6c` | 25 artifacts, 84 checkpoints | `Elendril-ELD-DEV-004` (76), `Elendril-V2-replay-data` (8) |

The V3 read-only audit verified all 78 artifact hashes, both archived diagnostic-operator hashes, 63 pilot receipts across 519 scoring snapshots, 237 checkpoint manifests, and all 1,659 checkpoint component byte sizes and SHA-256 hashes. The sibling local roots were available during the audit. Local availability does not establish independent remote custody.

Run the [verifier](../tools/verify_elendril_evidence.py) from the active checkout. The first command checks retained receipts against Git and independently rescores pilot snapshots. The second also reads the local checkpoint payloads. Both commands are read-only.

~~~powershell
python -m tools.verify_elendril_evidence
python -m tools.verify_elendril_evidence --checkpoint-base C:/Users/ankit/Downloads
~~~

To inspect an archived source file without restoring it into the working tree:

~~~text
git show 930ca3bb723e0b5b191700eec3d5f450875a6e25:docs/elendril/SESSION_HANDOFF.md
~~~

## Inherited evidence references

- Cymek runtime and original research archive at `feac2948466832516b7f048435483fd95e53b078`. [Source](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/tree/feac2948466832516b7f048435483fd95e53b078).
- September evidence consolidation at `89b1c5b984128e61d8ad11f5059d11dc99e71817`. [Evidence ledger](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/89b1c5b984128e61d8ad11f5059d11dc99e71817/docs/research/EXPERIMENT_EVIDENCE_LEDGER.json).
- Arkenstone, Triquetra, BRAMASTRA, Gandiva, Citadel and the Cymek derivatives have pinned branch and blob identities in the archived evidence index. [Cross-branch index](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/EVIDENCE_INDEX.json).
- Formation-Mux S5's original checkpoint-bearing output remains absent from reviewed custody. Other external artifacts and failed recoveries have explicit statuses. [Historical custody register](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/930ca3bb723e0b5b191700eec3d5f450875a6e25/docs/elendril/ARTIFACT_CUSTODY_REGISTER.md).

The cross-branch evidence informs the current controls. Triquetra and BRAMASTRA exposed recency and query-blind shortcuts. Arkenstone's later skill-formation failure limited retention inference. Citadel documented split leakage, and Gandiva showed why terminal behavior, family coverage and durable payloads need distinct gates. Preserve these results in their original contexts.
