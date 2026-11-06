# ELD-DEV-003 relational robustness results

Twelve arms completed from fresh initialization at source `acd4b115d2075340a3db70131cea02d5f1b74c4f`. Each arm ran 352 updates with the same paired example stream and objective within seed. The dense control carried six more trainable parameters than the hybrid read head.

| Model | Parameters | Training pairs /176 | Held-out pairs /48 | Unseen keys /48 | Five facts /48 |
| --- | ---: | --- | --- | --- | --- |
| Capacity-matched dense | 44,112 | 2, 7, 13 | 1, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| Hybrid associative | 44,106 | **169, 176, 176** | **44, 47, 45** | **48, 48, 48** | **32, 48, 45** |
| Lexical-only | 42,570 | 1, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |
| Context-only | 44,097 | 0, 0, 136 | 0, 0, 37 | 0, 0, 2 | 0, 0, 22 |

Seeds are 7, 19 and 41 in that order. Both members of each query pair must be exact and end with valid EOS. Hybrid exact scores on the base surface were 92/96, 95/96 and 93/96. It cleared all preregistered formation and robustness gates. Every model scored zero both-correct under query neutralization. Hybrid two-fact scores were 47/48, 48/48 and 46/48. At update 264 the hybrid reached 48/48 on all three base panels, before its endpoint declined in two seeds. The final result uses the registered endpoint.

The component ablations suggest that the combination of contextual addressing and lexical relations mattered under this recipe. Lexical matching alone did not form the task. Contextual addressing showed partial, seed-sensitive acquisition and weak alias transfer. Lexical-only also removes projections and has fewer parameters, so that comparison changes capacity. The parameter-matched dense contrast isolates a configuration advantage at fixed data and updates. Hybrid averaged about 22% more in-process wall time than the dense comparator on this host, including evaluation.

The 48 development pairs represent 12 disjoint latent value triples with shared stress renderings. The experiment supports direct relational retrieval on this synthetic surface. Independent generators, multi-hop retrieval, persistence across learning stages and compute-matched transfer remain separate questions.

## Reproduction and cost

From a clean checkout of the source above, execute the following per method and seed, with a fresh output path.

~~~powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m anra_v5.elendril pilot --surface relational-v1 --method associative --arm binding --seed 7 --steps 352 --root <checkpoint-directory> --output <receipt.json>
python -m anra_v5.elendril rescore --receipt <receipt.json>
~~~

Use methods capacity-matched, associative, lexical-only, and context-only at seeds 7, 19, and 41. All 120 base and stress snapshots were independently rescored. The serial CPU batch consumed 4,224 updates, 1,216,512 training tokens, 135,168 supervised tokens, and 102.94 seconds of in-process wall time. Process CPU time was 100.61 seconds. Evaluation used 32,054 forwards over 575,380 input positions. Minimum sampled available physical memory was 4.45 GiB, and the largest sampled process-tree resident set was 294.53 MiB. GPU use was zero. The [custody manifest](../../artifacts/elendril/eld-dev-003/manifest.json) binds 13 receipt files and 48 checkpoint inventories. Checkpoint payloads are in the sibling local directory C:/Users/ankit/Downloads/Elendril-ELD-DEV-003.

The next experiment should require a two-hop join over independently generated chains. Compare the measured hybrid with a second causal read and equal-budget single-read and dense controls. Freeze evaluation under aliases and distractors. Then test staged retention only after both individual tasks independently clear acquisition.
