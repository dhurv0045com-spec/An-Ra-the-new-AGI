# ARK-018 V4 — FINAL RESULT AUDIT

**Status:** EXECUTED / FINAL BUNDLE INDEPENDENTLY AUDITED / PRIMARY POSITIVE INTERNALIZATION CLAIM NOT MET  
**Audit date:** 2026-09-10  
**Branch:** `Arkenstone`

## 1. Bundle integrity

Audited operator result bundle:

`ARKENSTONE_ARK018_SCIENCE_BIRTH_RESULTS.zip`

SHA256:

`cea50622b1725eb12308c54355616e2188b7956761db46289c6da33c09c31f7c`

Independent audit outcome:

- campaign status: `COMPLETE`;
- both preregistered seeds completed: `31801`, `31902`;
- all four preregistered arms completed at `8000/8000` updates per seed;
- 40/40 receipt hashes recomputed successfully;
- 39/39 files covered by the ZIP manifest matched;
- peS2o science shard identity matched the frozen SHA256 `b397427cd5964b7cc2a41264ca8789a0c020f96d4e403b314900798711a2ead5`;
- Birth Book identity matched frozen SHA256 `8f17d092897f41df45100d227ecf7a2391ed5a5cfd3d3bcf373ad86c94b0c4f4`;
- REDTEAM status: `PASS`.

The raw ZIP itself is not committed to this repository by this audit; this file records the independently inspected evidence identity and conclusions.

## 2. Pretraining outcomes

Final selected measurements:

| Measurement | SCIENCE_ONLY | BIRTH_NATURAL_2PCT | BIRTH_REHEARSAL_10PCT | SCIENCE_REPLAY_10PCT_CONTROL |
|---|---:|---:|---:|---:|
| SEALED science NLL, seed 31801 | 4.2607 | 4.3054 | 4.4372 | 4.280 |
| SEALED science NLL, seed 31902 | 4.293 | 4.305 | 4.436 | 4.269 |
| Birth NLL, seed 31801 | 3.831 | 1.657 | 1.321 | 3.930 |
| Birth NLL, seed 31902 | 3.753 | 1.493 | 1.239 | 3.650 |

The 10% Birth treatment therefore produced a large, replicated Birth-distribution modeling effect. Relative to the token-matched 10% scientific replay control, SEALED science NLL was worse by approximately 3.68% and 3.91% in the two seeds; this remains inside the preregistered 5% major-cost boundary but is a real tradeoff.

## 3. Primary preregistered Birth-content test

Primary comparison: `BIRTH_REHEARSAL_10PCT` versus `SCIENCE_REPLAY_10PCT_CONTROL` on the frozen SEALED Birth-content score.

Observed improvement:

- seed 31801: `+0.000`;
- seed 31902: `+0.067`.

The preregistered threshold required at least `+0.10` absolute improvement in both seeds while respecting the scientific-NLL cost gate.

**Primary verdict:** `NO_LARGE_BIRTH_SPECIFIC_INTERNALIZATION_EFFECT`.

Interpretation: the experiment demonstrates strong corpus-specific statistical assimilation, but it does **not** demonstrate robust paraphrase-level semantic internalization, improved reasoning, identity, consciousness, or AGI.

## 4. Post-pretraining plasticity screen

Temporary-binding qualification after pretraining:

| Pretraining arm | seed 31801 | seed 31902 |
|---|---:|---:|
| SCIENCE_ONLY | 400 steps | 400 steps |
| BIRTH_NATURAL_2PCT | 400 | 300 |
| BIRTH_REHEARSAL_10PCT | 1200 | not sustained by 1500 |
| SCIENCE_REPLAY_10PCT_CONTROL | 300 | 300 |

This is a replicated **secondary diagnostic screen** suggesting that heavy 10% Birth rehearsal changed subsequent learning plasticity in this narrow binding task. It is not yet a universal continual-learning law and the mechanism is unresolved.

The matched 10% science-replay arm did not show the slowdown, so a generic statement such as “repeating any small corpus reduces plasticity” is not supported by this experiment.

## 5. Dose interpretation

The 2% Birth arm strongly reduced Birth NLL while keeping both science cost and subsequent binding-acquisition speed much closer to SCIENCE_ONLY. This makes low-dose periodic rehearsal an interesting candidate operating region, **not a demonstrated optimum**.

The 10% arm had consumed only about 53.45% of one full tokenized Birth Book pass by the hard 8000-update horizon; the 2% arm about 10.69%. Therefore no claim should be phrased as “the model trained on the whole Birth Book.”

## 6. Scientific claim ledger

**DEMONSTRATED**

- The exact Birth corpus causally changes token-level modeling of Birth-like text under the tested ~21M real-science pretraining setup.
- Heavy 10% Birth rehearsal creates a replicated scientific-modeling tradeoff relative to the token-matched 10% scientific replay control.

**SUPPORTED / REPLICATED SCREEN**

- Heavy 10% Birth rehearsal substantially slows subsequent acquisition of the controlled temporary-binding task across both seeds.

**NOT DEMONSTRATED**

- large Birth-specific paraphrased semantic internalization under the preregistered criterion;
- improved broad reasoning;
- improved general cognition;
- a universal plasticity law;
- a mechanism explaining the plasticity slowdown;
- identity, selfhood, consciousness, or AGI.

## 7. Consequence for the Arkenstone program

ARK-018 closes the real-data execution gate and changes the ARK-019 design context. ARK-019 no longer needs to wait for ARK-018 execution; however, it should still wait for ARK-017 mechanism credit before freezing a Guardian intervention.

The highest-value combined question is now:

`How do we preserve already-acquired reusable structure without making the Core less able to acquire the next structure?`

ARK-017 remains the immediate Arkenstone execution priority because it isolates update magnitude versus invariant-support data. ARK-019 should then use the ARK-018 real-text/plasticity evidence as an explicit regression target: a Guardian must retain old capability **and** avoid the kind of new-skill acquisition slowdown observed in the heavy Birth condition.
