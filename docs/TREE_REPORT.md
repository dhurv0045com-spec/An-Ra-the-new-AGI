# Clean-tree verification

| Measure | Pre-V3 archive | Active V4 tree |
| --- | ---: | ---: |
| Tracked files | 861 | 173 |
| Files under docs | 228 | 14 |
| Jupyter notebooks | 10 | 2 |
| Setuptools package discovery patterns | 16 | 9 |

The first cleanup removed 726 tracked paths. Removals include inherited experiment launchers and notebooks, production and TPU orchestration, obsolete documents and historical tooling. The original tree is pinned at `930ca3bb723e0b5b191700eec3d5f450875a6e25` and on the remote `elendril-pre-v3` tag. Original ELD artifact paths retain their identical Git blobs. All removed paths have exact retrieval objects in the [machine-readable inventory](TREE_INVENTORY.json). The source branches and untracked local checkpoint directories were outside the removal operation.

The table records the accepted V4 baseline, not the evolving V5 count. V5 additionally archives two paused V3 notebook wrappers, their generator and three superseded planning/trace files. Exact retrieval objects are in [the removal manifest](ARCHIVED_PROGRAMS.json). The new action notebooks are a separate program. The low-level cloud utility remains a dependency of the new runner. The initial 861-path inventory is immutable; no source branch or external checkpoint directory was deleted.

At V5 handoff the active tree has 204 files, 16 documents and two action notebooks (about 15.5 MB including retained raw evidence). New code and targeted tests replace no historical payloads. The five added runtime modules are connected through representation → backend → trainer/action policy → existing cognitive episode and verifier.

The remaining V5 code is a mapped dependency closure for core tensors, CE, optimizer groups, public evidence types and checkpoint/state formats, extended by the qualified representation/backend/trainer/action interfaces. Package initializers are passive at removed runtime boundaries. The original cleanup passed 61 tests and seven subtests; its isolated wheel imported 32 internal modules. That [historical trace](https://github.com/dhurv0045com-spec/An-Ra-the-new-AGI/blob/f516e0bdb8f9137532b35c0c23299f7f9dbdf185/docs/RUNTIME_IMPORTS.json) remains retrievable. [Current static imports](ACTIVE_IMPORT_GRAPH.json) · [V5 qualification](FOUNDATION_QUALIFICATION.md)
