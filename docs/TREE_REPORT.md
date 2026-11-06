# Clean-tree verification

| Measure | Pre-V3 archive | Active V4 tree |
| --- | ---: | ---: |
| Tracked files | 861 | 173 |
| Files under docs | 228 | 14 |
| Jupyter notebooks | 10 | 2 |
| Setuptools package discovery patterns | 16 | 9 |

The first cleanup removed 726 tracked paths. Removals include inherited experiment launchers and notebooks, production and TPU orchestration, obsolete documents and historical tooling. The original tree is pinned at `930ca3bb723e0b5b191700eec3d5f450875a6e25` and on the remote `elendril-pre-v3` tag. Original ELD artifact paths retain their identical Git blobs. All removed paths have exact retrieval objects in the [machine-readable inventory](TREE_INVENTORY.json). The source branches and untracked local checkpoint directories were outside the removal operation.

The remaining V5 code is a mapped dependency closure for core tensors, CE, optimizer groups, public evidence types and the checkpoint/state formats. Package initializers are passive at removed runtime boundaries. The cleanup commit passed 61 tests and seven subtests. Its isolated wheel executed fixture audit and imported 32 internal modules without the removed packages. [Installed runtime trace](RUNTIME_IMPORTS.json) · [Active static imports](ACTIVE_IMPORT_GRAPH.json)
