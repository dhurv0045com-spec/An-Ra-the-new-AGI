# Known limitations

- Qualification covers deterministic CPU FP32 on the recorded runtime. CUDA, XLA, cross-platform bit parity, distributed execution and power-loss durability require separate qualification.
- The checkpoint store uses file and pointer publication with an OS writer lock. It retains local payloads. There is no replicated off-machine backup or signed artifact custody in this session.
- The trainer accepts a single fixed-length family per run. Staged continuation requires a new explicit lineage contract for intentional data switches. Current checkpoints deliberately refuse a data identity change.
- The copy control has a trivial parser ceiling. Binding failed in-sample and held-out gates. The reason for failed query conditioning is open. Retention, composition training, abstention training and forward transfer remain unmeasured.
- The current dev splits share semantics across some families, and the public fixture source can produce gold. Prospective multi-family, confirmation and sealed designs need separate globally disjoint task pools and trusted custody.
- The seven binding dev worlds and one composition held-out operation pair limit precision. A future experiment needs additional independent clusters and renderings.
- Historical S5 model bytes and other externally reported payloads have unresolved custody. A shallow local history also limits graph reconstruction beyond the verified branch parent.
