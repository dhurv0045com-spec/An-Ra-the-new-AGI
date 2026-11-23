# Y3 fixture audit · 24 November 2026

The original run completed all 88 jobs in 68 minutes 53 seconds. The two owner archives match the digests in the [published run report](RESULTS.md). I checked every manifest entry against its byte count and SHA-256 (11 in ANALYSIS, 340 in EVIDENCE), the checksum lists, and agreement of the five common scientific/telemetry receipts. The embedded verifier accepted 79 jobs and 36 studies. This follow-up changes the interpretation of three mechanisms. It preserves the original numerical results.

## The depth test

I regenerated the exact native P4 dev panel from the pinned Y3 generator. For each cell I started at the public query and followed exactly d rows. At each step a deterministic lookup needs exactly one distinct successor for the current key. A family succeeds only if all eight answers equal the recorded targets.

| Depth | Families | Exact-walker correct cells / 512 | Complete families / 64 | Other property |
|---|---:|---:|---:|---|
| 1 | 64 | 512 | 64 | Gold equals rival on all 512 cells. Switching queries never changes the answer within a family. |
| 2 | 64 | 512 | 64 | Distinct endpoints and functional edges. |
| 3 | 64 | 256 | 0 | At the midpoint the generator omits one required reverse edge. |
| 4 | 64 | 0 | 0 | All 512 cells contain at least one key with two distinct successors. Both queried paths encounter a conflict. |

For example, with paths through C,D,E in opposite orders, the generator writes C→D, D→E and E→D. The reverse route needs D→C. The stored gold still assumes that edge. At four hops the reverse route instead introduces conflicting successors. Thus the observed categorical 100/100/0/0 pattern cannot establish failure of length generalization. Depth two and the separately generated donor/rewire suite retain narrower evidentiary value. The categorical implementation also copies row values into a simplex by construction and receives parsed rows and a step count. Its success does not establish acquisition from bytes. For wrong-donor scoring, Y3 rotates another example's target. The target for that intervention must instead be computed by running the actual inserted entity through the current recipient graph.

## Executive action shortcut

The Y3 public prompt starts with the name of the private experimental condition and two tool flags. I trained a frequency lookup on those prefix strings in the 140 training worlds and evaluated it on all 140 dev worlds. It recovered an optimal **action class on 140/140** before seeing a single graph row. This baseline ignores which value to answer with. The model's reported macro score includes answer values, so the two scores measure different things. The test demonstrates that the fixture permits a shortcut. It does not tell us whether any individual model used it. The evaluator scores the first character. Successful information acquisition still requires executing the chosen tool and answering from the returned evidence.

## Effective batch and adaptation

In both FULL P4 and P6, the training plan creates a new identically seeded RNG **inside** the batch-slot comprehension. All 16 slots therefore select the same item on each update. The first four P6 updates for seed 7 each have one distinct index. This changes the exposure and noise behind the published training curves. For P6 the three named adaptation forks also call the same evaluation grid per seed. The nine receipts cannot serve as nine independent trained adaptation mechanisms. The inherited 1,024 parent updates, zero additional outer training steps, the 256 evaluation blocks and the temporary within-block parameter interventions need separate counters.

## Reproduction

Run the following from a checkout of the Y3 notebook code pin `60e8cb27f6e7fc21f9aaca5992a2d5ecf0c77d6b` with the project's Python dependencies. It needs no GPU or checkpoint.

```python
from collections import Counter, defaultdict
from experiments.avendi_y3.plan import STREAM_SEEDS
from experiments.avendi_y3.tasks import build_multihop_families, build_p5_executive_worlds

def walk(cell):
    edges = defaultdict(set)
    for key, value in cell["rows"]:
        edges[key].add(value)
    node = cell["query"]
    for _ in range(cell["depth"]):
        if len(edges[node]) != 1:
            return None
        node = next(iter(edges[node]))
    return node

families = build_multihop_families("p4dv", 256, STREAM_SEEDS["dev"])
for d in range(1, 5):
    groups = [[walk(c) == c["gold"] for c in f["cells"]]
              for f in families if f["depth"] == d]
    print(d, sum(map(sum, groups)), sum(map(all, groups)), len(groups))

counts = defaultdict(Counter)
def prefix(w):
    return w["public_prompt"].split("|", 1)[0]
def classes(w):
    return [a.split(":")[0] for a in w["optimal_actions"]]

for w in build_p5_executive_worlds("train", 140, STREAM_SEEDS["corpus"]):
    counts[prefix(w)].update(classes(w))
dev = build_p5_executive_worlds("dev", 140, STREAM_SEEDS["dev"])
print(sum(counts[prefix(w)].most_common(1)[0][0] in classes(w) for w in dev), len(dev))
```

The margins from P2 remain a small paired signal (+5.73 percentage points on dev and +3.52 on confirmation). Literal accuracy and row-order robustness remain weak. The old P1 read-only checkpoint restore and P6 update-count flags remain recorded in the run report. The [Y4 execution plan](../../../research/avendi/GEMINI_AVENDI_V8_Y4.md) uses these findings to decide which measurements to repeat.
