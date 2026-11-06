# Associative read architecture

ELENDRIL's experimental decoder keeps the existing dense hidden-state path. At every position, two learned projections score a query state against earlier states in its packed segment. A separate 3 × 3 learned relation weights equal-token matches between the recent three-token windows at the query and each source. The shared read algorithm applies to any symbols in the vocabulary. It receives token IDs, causal positions and the attention mask.

The address weights form a distribution over strictly earlier content tokens in the current segment. Scatter-add combines repeated token occurrences into one vocabulary probability. A learned sigmoid gate mixes the read distribution with the dense projection's vocabulary probabilities in log space. Empty read contexts fall back to the dense distribution. The gate starts biased toward the dense path. Window offsets respect segment position resets, and address masks forbid cross-segment reads. Training and greedy generation call the same forward path.

The binding contrast uses the ground-truth token of the paired, counterfactual query as a distractor at the first answer logit. The auxiliary term is separate from causal CE. EOS continues to receive direct CE supervision. The sampler binds its method and exposure-block shape to the restart identity. The architecture identity includes the read-head version, window and address width. Model and optimizer state travel together through the existing checkpoint store.

The implemented memory is bounded by the current context. Long-lived episodic storage, intervention selection, and architecture mutation are separate research steps. A reusable memory should next demonstrate an advantage against a parameter-matched dense model across new generators, larger distractor counts, different serializations and delayed probes. An independently scored retention or forward-transfer study also needs individual acquisition for every skill in its sequence.

## Reproduce the registered comparison

Use the [frozen protocol](ASSOCIATIVE_PROTOCOL.md) and check out commit `77edcbd8dce7b73d7562e619351a9cc496c6ef84`. Run from the root of a clean worktree with Python 3.11 and the PyTorch 2.14.0 CPU wheel. Choose a fresh output directory. This PowerShell example runs one clean associative arm and independently rescores it.

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$elendrilRun = Join-Path $env:TEMP "ELD-DEV-002-replay"
New-Item -ItemType Directory -Path $elendrilRun
python -m anra_v5.elendril pilot --method associative --arm binding --seed 7 --steps 336 --root "$elendrilRun/checkpoints" --output "$elendrilRun/associative-binding-7.json"
python -m anra_v5.elendril rescore --receipt "$elendrilRun/associative-binding-7.json"
~~~

Run serially for seeds 7, 19 and 41. For each seed use `shuffled-ce`, `paired-ce`, `paired-contrast`, and `associative` with `--arm binding`, plus `associative` with `--arm binding-permuted`. Use distinct receipt names, a 180-second per-process limit, and separate checkpoint roots when comparing interruption paths. A stopped run can continue with the identical source, method, seed, budget, and checkpoint root, adding `--resume` and a new receipt name.
