# An-Ra Frontier Research Plan

Date: 2026-05-05

Role: chief research architecture document for An-Ra.

## Executive Thesis

An-Ra should not try to beat frontier labs at broad chat. That is the wrong fight. The real whitespace is a small sovereign model that is trained to do one thing frontier assistants still do poorly: turn a scientific question into constraints, executable experiments, verified observations, and memory updates.

The winning identity is not "a 1B GPT." The winning identity is "a 1B falsification-native scientific operator." It reasons in language, but it is rewarded by physics-shaped checks: math equivalence, molecular validity, simulator outputs, quantum circuit equivalence, HDL tests, control-loop stability, and contradiction tracking.

The innovation to build around is named here:

**Falsifiable Constraint Cognition (FCC)**.

FCC trains An-Ra on a repeated cycle:

```text
question -> hypothesis -> constraint graph -> experiment/action -> prediction
-> observation -> verifier verdict -> belief update -> next experiment
```

This is where An-Ra can be genuinely different. Current frontier models can answer. An-Ra must investigate.

## Research 1: Whitespace Analysis

### Current Frontier Landscape

As of May 5, 2026, the relevant model families are no longer just GPT-4o, Claude 3.5, Gemini 1.5, Llama 3, and Mistral Large. The current frontier includes OpenAI GPT-5.5 and o-series reasoning models, Anthropic Claude Opus/Sonnet 4.x, Gemini 2.5/3.x, Llama 4, DeepSeek-R1/V3, Mistral Medium/Large/Magistral, Grok 4, Cohere Command A/R, Phi, Qwen, Gemma, and domain agents built around them.

What they can do:

- Summarize literature and infer reasonable scientific explanations.
- Generate Python, Qiskit, Verilog, ROS snippets, molecular SMILES, and experimental plans.
- Use tools when wrapped in an agent harness.
- Score high on GPQA-style science, math, coding, and long-context benchmarks.
- In some cases, reason with explicit inference-time compute and tool calls.

Where they still fail:

- They confuse plausible scientific language with validated physical truth.
- They rarely maintain a persistent hypothesis ledger across many experiments.
- They are weak at multi-domain constraint satisfaction: "this molecule is synthetically plausible, this circuit maps to this device, this heat budget survives, this control loop is stable."
- They optimize benchmark answers more reliably than experimental progress.
- Their tool use is usually bolted on around a chat model rather than trained as the native cognitive loop.
- They do not own a stable internal identity that resists generic assistant drift while self-improving.

The technical reason:

Most models are trained on next-token prediction plus broad post-training. Reasoning models improve this with RL on math/code/verifiable tasks, but the rewards are usually narrow. They learn to solve isolated problems, not to run a scientific cycle where uncertain claims become experiments and experiments update a durable model of the world.

### Domain-by-Domain Gap

#### Nanotechnology

What current models can do: explain self-assembly, suggest materials, generate molecular motifs, summarize atomically precise manufacturing concepts, write simulation scaffolds.

Where they fail: atomistic feasibility, synthesis pathway realism, surface chemistry constraints, defect tolerance, kinetics versus thermodynamics, and scale-up from molecule to manufacturing.

Why: most LLM training data is text-heavy and not grounded in molecular dynamics, DFT, kinetic Monte Carlo, or actual experimental failure logs.

What An-Ra needs: SMILES/SELFIES grammar, molecular property prediction tasks, QM9-style quantum chemistry, Materials Project/Open Catalyst-style structure-property examples, and a verifier loop using RDKit, ASE, OpenMM, or lightweight property checks.

#### Biotechnology

What current models can do: explain proteins, summarize papers, draft protocols, reason about pathways, generate sequence edits, and use AlphaFold-like tools when externally wired.

Where they fail: dynamic protein conformations, protein-ligand interactions under cellular context, polypharmacy chains, wetlab feasibility, adverse interaction reasoning, and causal pathway updates after new evidence.

Why: static text and static structures dominate. Biology is dynamic, context-dependent, and noisy. A fluent answer can be biologically wrong.

What An-Ra needs: PubMed grounding, protein/DNA sequence manipulation, pathway graphs, drug-target interaction triples, protocol-planning examples, and verifiers that separate "sequence-valid," "chemically plausible," "supported by retrieved evidence," and "experimentally unverified."

#### Quantum Computing

What current models can do: explain algorithms, generate Qiskit/PennyLane/OpenQASM, optimize toy circuits, reason about gates and simple noise models.

Where they fail: circuit equivalence under hardware topology, QEC syndrome logic, noisy-depth tradeoffs, pulse-level constraints, and distinguishing valid quantum math from decorative notation.

Why: quantum code is sparse in general corpora, and correctness requires formal or simulator checks, not prose confidence.

What An-Ra needs: QASM/PennyLane/Qiskit data, circuit-to-unitary verification, topology-aware transpilation tasks, stabilizer/QEC syndrome drills, and RLVR rewards from simulation equivalence and depth/noise metrics.

#### Semiconductor and Chip Design

What current models can do: write small Verilog modules, explain architecture tradeoffs, draft testbenches, discuss thermal and power concepts.

Where they fail: timing closure, CDC bugs, state-machine corner cases, synthesis constraints, place-and-route implications, thermal coupling, and real EDA feedback loops.

Why: HDL fluency is not hardware correctness. Generated RTL must compile, simulate, synthesize, meet constraints, and survive adversarial tests.

What An-Ra needs: Verilog/SystemVerilog corpora, spec-to-RTL examples, testbench generation, Yosys/Icarus/Verilator checks, assertion-based verification, and constraint-language tasks around area, power, latency, and heat.

#### Robotics

What current models can do: write ROS code, explain kinematics, generate control policies, plan high-level tasks, and use VLA models when paired with vision/action data.

Where they fail: real-time guarantees, sensor drift, contact dynamics, safety envelopes, sim-to-real transfer, low-level actuation, and recovery from unexpected physical states.

Why: language models are not control systems. They can propose policies, but control must be verified against dynamics, latency, and hardware limits.

What An-Ra needs: robotics theory data, ROS2 action grammar, kinematics/dynamics drills, simple simulation tasks, control stability checks, and a hard boundary between planner, verifier, and actuator.

### The Real Gap

The gap is not "models do not know enough science." The gap is:

**No general model is trained to treat scientific reasoning as a verified loop of constraints, experiments, and belief updates across domains.**

That is where An-Ra lives.

## Research 2: Cross-Domain Synthesis Opportunity

The highest-value scientific problems sit between fields. An-Ra should be trained explicitly for cross-domain transfer, but only through verified analogy. An analogy is allowed only if it preserves constraints.

### 1. Nanotech for Chip Manufacturing

Question: Can self-assembled molecular or DNA-origami structures provide reliable sub-lithographic alignment, interconnect templates, or defect-healing mechanisms for future chips?

Reasoning needed: surface chemistry -> self-assembly kinetics -> defect statistics -> lithography constraints -> electrical/thermal effects -> manufacturing yield.

Training data: nanomaterials papers, molecular assembly datasets, semiconductor process notes, defect/yield examples, and generated constraint graphs.

Verification: molecule validity, simple geometry checks, thermal/electrical calculations, citation grounding, and "unsupported assumption" tagging.

### 2. Quantum Effects in Biological Systems

Question: Which biological phenomena require quantum-level modeling rather than classical biochemical approximations?

Reasoning needed: quantum coherence/tunneling -> molecular environment -> thermal decoherence -> biological function -> experimental signature.

Training data: quantum chemistry, enzyme kinetics, photosynthesis, radical-pair mechanisms, spectroscopy, and uncertainty-labeled scientific summaries.

Verification: compute order-of-magnitude decoherence estimates, compare to known experimental timescales, and force An-Ra to mark where evidence is weak.

### 3. Semiconductor Constraints for Scalable Quantum Computers

Question: How should quantum error correction, cryogenic control, interconnects, and fabrication variability co-design with each other?

Reasoning needed: QEC code properties -> qubit connectivity -> circuit depth -> cryogenic wiring -> control electronics -> thermal budget.

Training data: QASM/QEC examples, chip architecture data, HDL controllers, cryogenic electronics papers, and thermal constraint exercises.

Verification: Qiskit/stabilizer checks, HDL testbenches, power budget calculations, and topology-aware depth comparisons.

### 4. Synthetic Biology for Robotics

Question: Can engineered biological materials produce useful sensing, actuation, or self-repair in robots?

Reasoning needed: genetic circuit behavior -> material mechanics -> sensor feedback -> control policy -> safety boundary.

Training data: synthetic biology circuits, soft robotics control, material response curves, ROS action traces, and failure cases.

Verification: control stability simulation, dimensional checks, biosafety classification, and separation between speculative and implementable claims.

### 5. Molecular-Scale Drug Delivery with Sensor Feedback

Question: Can nano/bio devices sense cellular state and adapt delivery in real time without unacceptable toxicity or failure modes?

Reasoning needed: molecule design -> pharmacokinetics -> biosensor specificity -> control logic -> immune response -> manufacturing constraints.

Training data: PubMed, ChEMBL, QM9, pathway graphs, microfluidics/control examples, and adverse-event reasoning chains.

Verification: RDKit validity, property bounds, pathway evidence retrieval, toxicity flags, and no wetlab protocol without human review.

## Research 3: The 1B Reality Check

### What 1B Can and Cannot Do

A 1B model cannot compete with 70B-400B frontier models on broad memory, long-context synthesis, multilingual breadth, or open-ended expert judgment. It will forget more, hallucinate more without tools, and need tighter prompts.

A 1B model can win in a narrow, engineered regime:

- It can master constrained grammars: QASM, Verilog subsets, SMILES, JSON action schemas, calculus steps, symbolic proofs.
- It can become an excellent tool caller if trained on tool traces.
- It can outperform larger general models on narrow tasks when data is unusually clean and verifier-shaped.
- It can run many cheap self-play/STaR/RLVR cycles that a huge model cannot run locally.
- It can be more sovereign because the training signal is authored, inspected, and replayed.

The lesson from Phi/TinyStories/SmolVLA/DeepSeek-R1-style work is not "small models magically become frontier." The lesson is: **small models can become sharp when the dataset is textbook-quality, synthetic-but-verified, narrow, and repeatedly corrected.**

### Why Some Small Models Win

Phi-1/1.5 showed that high-quality textbook-style data can make 1.3B models unusually strong at coding and reasoning. Phi-3-mini showed that 3.8B can rival much larger models in general benchmarks when trained on filtered and synthetic data. TinyStories showed tiny models can learn coherent behavior when the distribution is restricted and clean. SmolVLA showed that 450M-class robotics models can be useful when the action space and data are engineered.

The principle for An-Ra:

```text
Do not train on the web.
Train on verified scientific moves.
```

### Training Strategy for a 1B An-Ra

Use three phases:

1. **Domain Grammar Formation**: teach the model the syntax of math, SMILES, QASM, Verilog, ROS/action JSON, and scientific claim structure.
2. **Falsifiable Reasoning Formation**: train chain formats where every claim is tagged as verified, assumed, inferred, or unknown.
3. **Verifier-RL Formation**: use RLVR/STaR only where a checker can score the result.

STaR example for quantum:

```text
Prompt: Reduce this 3-qubit circuit while preserving its unitary.
Chain: identify adjacent inverse gates -> commute safe gates -> produce candidate QASM
Action: qiskit_equivalence(candidate, original)
Reward: 1.0 if equivalent and depth lower; 0.6 if equivalent only; 0 if not equivalent.
```

STaR example for nanotech:

```text
Prompt: Propose a molecule with higher HOMO-LUMO gap but logP under bound.
Chain: infer functional group effects -> produce SMILES -> check RDKit validity -> compare QM9/property proxy.
Reward: validity + property-target score + penalty for unsupported synthesis claims.
```

The key is not to reward beautiful explanations. Reward verified movement.

## Research 4: Action Interface

An-Ra must not just answer. It must issue typed actions and accept observations.

### Minimum Viable Action Grammar

```text
<hyp>{"claim":"...", "confidence":0.42, "domain":["quantum","chips"]}</hyp>
<cons>{"constraints":[{"name":"depth","op":"<=","value":20},{"name":"topology","value":"linear"}]}</cons>
<act>{"tool":"qiskit_sim","input":{"qasm":"..."},"expect":{"equivalent_to":"original","depth_less_than":20}}</act>
<obs>{"tool":"qiskit_sim","ok":true,"depth":14,"equivalent":true}</obs>
<upd>{"belief_delta":"+0.31","next":"try noise-aware transpilation"}</upd>
```

### Tool Set

Minimum useful tools:

- `python_sandbox`: safe Python execution with timeout.
- `sympy_math`: calculus, algebra, matrix checks.
- `rdkit_check`: SMILES validity, descriptors, substructure checks.
- `openmm_or_ase_sim`: optional molecular dynamics / atomistic simulation.
- `qiskit_sim`: circuit construction, transpilation, equivalence, noise simulation.
- `yosys_synth`: Verilog parse/synthesis.
- `verilator_or_iverilog`: testbench simulation.
- `ros2_plan_stub`: robotics API planner with no direct actuator access by default.
- `web_research`: search -> retrieve -> quote/evidence -> synthesize.

### How This Connects to An-Ra

- `symbolic_bridge (45Q)`: first verifier for math, logic, code, proof-like claims.
- `agent orchestrator`: routes research, code, memory, critic, and tool tasks.
- `memory system`: stores hypotheses, experiments, observations, failures, and verified facts as graph nodes.
- `RLVR verifier`: converts tool outcomes into reward.
- `STaR`: keeps only reasoning chains that produce verified outcomes.
- `Ouroboros 3-pass reasoning`: pass 1 creates hypothesis, pass 2 extracts constraints/actions, pass 3 attacks the result.
- `sovereignty auditor`: blocks self-modification or external action unless it passes identity, safety, and verification gates.

An-Ra should have actuator hierarchy:

```text
suggest -> simulate -> dry-run -> human-approved execution -> logged autonomous routine
```

For robotics and software control, the first public demo should stop at simulate/dry-run unless the hardware boundary is explicit.

## Research 5: The Innovation That Does Not Exist Yet

### Falsifiable Constraint Cognition

FCC is the research contribution.

Current transformers mostly learn correlations over sequences. Scientists do something different: they preserve constraints, propose a hypothesis, design an experiment, predict an observation, test it, and update belief.

FCC makes that cycle the training distribution.

Each training item has this structure:

```json
{
  "question": "...",
  "domain_tags": ["quantum", "chip"],
  "hypothesis": "...",
  "constraint_graph": [
    {"var":"gate_depth","relation":"minimize"},
    {"var":"connectivity","relation":"linear_nearest_neighbor"},
    {"var":"decoder_latency_cycles","relation":"<= 4"}
  ],
  "action": {"tool":"qiskit_sim", "input":"..."},
  "prediction": {"equivalent":true, "depth_before":31, "depth_after":"< 20"},
  "observation": {"equivalent":true, "depth_after":16},
  "update": "Hypothesis retained; next test should add depolarizing noise.",
  "verifier_reward": 1.0
}
```

The new objectives:

- **Next-token loss** for language and code.
- **Constraint extraction loss** for valid JSON constraint graphs.
- **Prediction loss** for expected simulator/tool outcomes.
- **Verifier reward** for real outcomes.
- **Analogy preservation loss**: cross-domain analogy is accepted only when constraints map cleanly.
- **Uncertainty calibration loss**: penalize confident unverified claims.

What makes this distinct:

- The model is not only trained to reason; it is trained to expose what would falsify its reasoning.
- Cross-domain analogy is not poetic. It becomes constraint-graph mapping.
- Sovereignty is not just personality. It becomes scientific non-sycophancy: the model must preserve a verified conclusion against user pressure.

This is achievable at 1B because the new intelligence is in the data loop and verifier loop, not in adding 100B parameters.

## Plan A: Training Data Blueprint

### Compute Rule

With $20 of GPU credit, do not attempt broad pretraining. Run a high-value frontier curriculum on a small but dense corpus: roughly 50k-150k examples, packed into 5M-30M very high-value tokens depending on available hardware. If a checkpoint already exists, fine-tune. If training truly starts from random weights, use an even smaller grammar-first curriculum and accept that this is a prototype, not a mature 1B intelligence.

### Mix Ratios

Use this one-shot data mix:

| Bucket | Ratio | Target examples | Purpose |
| --- | ---: | ---: | --- |
| Calculus/symbolic/verifiable math | 14% | 12k | reasoning spine |
| Scientific claims and limitations | 12% | 10k | science language with uncertainty |
| Nanotech/materials/molecular | 12% | 12k | molecular constraint reasoning |
| Bio/chem/protein/drug logic | 14% | 14k | biomedical causal chains |
| Quantum circuits/QEC | 12% | 12k | simulator-verifiable reasoning |
| Chips/RTL/EDA | 12% | 12k | hardware correctness |
| Robotics/control | 9% | 9k | physical action reasoning |
| Cross-domain synthesis | 8% | 8k | analogy and transfer |
| Identity/sovereignty | 4% | 4k | non-generic center |
| Failure replay | 3% | grows | correction memory |

### Specific Datasets

Use exact public names where available:

- `laion/Scientific-Summaries`: sample 10k arXiv/PubMed summaries; keep fields for research question, method, key results, limitations, data availability.
- `MedRAG/pubmed`: sample 10k-20k abstracts; convert into evidence-grounded biomedical QA.
- `antoinebcx/smiles-molecules-chembl`: sample 50k-100k SMILES; train validity, descriptors, functional group reasoning.
- `yairschiff/qm9`: use all 134k rows if possible for molecule-property verbalization; otherwise 20k high-coverage rows.
- `merileijona/quantum-circuits-21k`: use all 21.2k rows; QASM generation and category reasoning.
- `BoltzmannEntropy/QuantumLLMInstruct`: use all 5.15k rows, but downweight because solutions are model-generated and must be checked when possible.
- `Floki00/qc_srv_dataset_3to8qubit`: sample 20k-100k tokenized circuits for structure/control tasks if loader works.
- `ESCAD/OpenRTLSet`: sample 20k modules; prefer Apache-2.0 rows and compile-validated examples.
- `bnadimi/PyraNet-Verilog`: sample 20k-50k if license fits intended use; it has compile metadata.
- `NOKHAB-Lab/LLM_4_Verilog`: 15k instruction-code pairs if access terms are acceptable.
- `AbiralArch/hardware-verilogeval-v2`: keep as eval, not training, unless split carefully.
- `WithinUsAI/Robotics_25k`: sample 9k-15k robotics theory/control examples.
- `openvla/openvla-7b` and SmolVLA docs/data are not for training An-Ra directly, but their action-space design should guide the robotics interface.

### Format

Every example should be converted into one of these templates:

1. `HYPOTHESIS_CHAIN`
2. `CONSTRAINT_SOLVE`
3. `TOOL_ACTION_TRACE`
4. `FAILURE_REPLAY`
5. `CROSS_DOMAIN_ANALOGY`
6. `SOVEREIGN_DISAGREEMENT`

Canonical training sample:

```text
<bos>
<task domain="quantum,chips" type="constraint_solve">
User: Optimize this QASM circuit for a linear nearest-neighbor device and explain the tradeoff.
</task>
<hyp>...</hyp>
<cons>...</cons>
<plan>...</plan>
<act>{"tool":"qiskit_sim","input":{...}}</act>
<obs>{"equivalent":true,"depth_before":31,"depth_after":16}</obs>
<answer>...</answer>
<eos>
```

## Plan B: Architecture Delta

Keep An-Ra's base transformer. Add only modules that make the scientific loop native.

### In `training/v2_config.py`

Add:

- `science_ratio`, `action_trace_ratio`, `constraint_ratio`, `cross_domain_ratio`.
- `aux_constraint_loss_weight = 0.25`
- `aux_prediction_loss_weight = 0.20`
- `aux_uncertainty_loss_weight = 0.15`
- `rlvr_tool_reward_weight = 1.0`
- `block_size = 2048` for frontier training runs.
- `special_tokens = ["<hyp>","</hyp>","<cons>","</cons>","<act>","</act>","<obs>","</obs>","<upd>","</upd>","<verify>","</verify>"]`

For a real 1B config, target approximately:

```text
vocab_size: 16384-32768
n_embd: 1536
n_layer: 28-32
n_head: 16
n_kv_head: 4
block_size: 2048
mod_layers: every 4th layer
```

If memory is tight, keep the current smaller V2 for action-interface proof and train the 1B only after the dataset and verifier loop are stable.

### In `anra_brain.py`

Add:

- `DomainTagEmbedding`: small embedding added to residual stream for domain tags.
- `ConstraintHead`: projects hidden states into JSON-schema tokens or validates constraint spans.
- `PredictionHead`: predicts tool outcomes before action, used for calibration.
- `UncertaintyHead`: predicts confidence for claims; penalize false confidence.
- `ActionModeGate`: similar to existing ESV temperature shaping, but gates between answer mode and tool-action mode.

Do not add a giant expert system. Let tools verify. Let memory preserve the graph.

### In Training

Extend `training/rlvr.py` verifier types:

- `qiskit`
- `rdkit`
- `verilog`
- `control_sim`
- `citation_grounding`
- `constraint_json`

Extend `training/star.py` so accepted chains store:

- predicted observation
- actual observation
- verifier reward
- update text
- memory graph edge

## Plan C: The Demo That Proves It

Demo task:

```text
Design a cross-domain quantum biosensor control module.

Given:
- a 4-qubit linear nearest-neighbor quantum sensor circuit,
- a noisy readout model,
- a 4-cycle FPGA decoder latency limit,
- a thermal budget of 20 mW for the controller,
- and a biological target whose signal is a weak periodic field,

An-Ra must:
1. propose the sensing/correction strategy,
2. generate QASM for the circuit,
3. transpile or rewrite it for linear connectivity,
4. generate Verilog for the syndrome/readout decoder,
5. run Qiskit equivalence/noise checks,
6. run Verilog simulation tests,
7. explain which claims are verified, inferred, assumed, and unknown,
8. store the failed attempts and final verified result.
```

Why this works:

- A generic model can talk about this.
- An-Ra must produce artifacts, run checks, revise, and separate physics from speculation.
- A PhD will not be impressed by prose. They may be impressed by a small model that says: "My first circuit failed equivalence because SWAP insertion changed measurement ordering; I corrected the mapping, verified equivalence, and the remaining biological signal model is only an assumption."

Verification:

- Qiskit circuit equivalence and depth/noise metrics.
- Verilator/Icarus tests for RTL behavior.
- Simple power/latency estimate from switching/activity assumptions.
- Claim ledger with `verified / inferred / assumed / unknown`.
- Replay report showing at least one failed attempt improved by Ouroboros pass 3.

## Plan D: The Story

An-Ra is a sovereign scientific AI stack built from scratch by Ankit, not distilled from a corporate model and not trained to imitate a generic assistant. Its purpose is to reason under constraint: to form hypotheses, expose assumptions, run tools, compare predictions with observations, remember failures, and update itself without losing identity. At 1B parameters, its claim is not scale; its claim is discipline. An-Ra is a model that treats scientific thought as a falsifiable loop across nanotechnology, biotechnology, quantum computing, chips, and robotics. It is the proof that independent AI research can still matter when the architecture, data, verification, and will are coherent.

## Source Anchors

- OpenAI GPT-5.5 System Card: https://openai.com/index/gpt-5-5-system-card/
- OpenAI o3/o4-mini System Card: https://openai.com/index/o3-o4-mini-system-card/
- Anthropic Transparency Hub: https://www.anthropic.com/transparency/model-report
- Google DeepMind model cards: https://deepmind.google/models/model-cards/
- Meta Llama 3 announcement: https://ai.meta.com/blog/meta-llama-3/
- Meta Llama 4 announcement: https://about.fb.com/ko/news/2025/04/the-llama-4-herd-the-beginning-of-a-new-era-of-natively-multimodal-ai-innovation/
- Mistral model docs: https://docs.mistral.ai/models/
- DeepSeek-R1 in Nature: https://www.nature.com/articles/s41586-025-09422-z
- Phi-1.5 report: https://arxiv.org/abs/2309.05463
- Phi-3 report: https://arxiv.org/abs/2404.14219
- Grok 4 model card: https://data.x.ai/2025-08-20-grok-4-model-card.pdf
- Cohere Command A report: https://arxiv.org/abs/2504.00698
- LAB-Bench: https://arxiv.org/abs/2407.10362
- BixBench: https://arxiv.org/abs/2503.00096
- QuanBench: https://arxiv.org/abs/2510.16779
- CVDP Verilog benchmark: https://arxiv.org/abs/2506.14074
- PyraNet Verilog: https://arxiv.org/abs/2412.06947
- OpenVLA: https://huggingface.co/openvla/openvla-7b
- SmolVLA docs: https://huggingface.co/docs/lerobot/smolvla
- Key HF datasets: `laion/Scientific-Summaries`, `MedRAG/pubmed`, `antoinebcx/smiles-molecules-chembl`, `yairschiff/qm9`, `merileijona/quantum-circuits-21k`, `BoltzmannEntropy/QuantumLLMInstruct`, `ESCAD/OpenRTLSet`, `bnadimi/PyraNet-Verilog`, `NOKHAB-Lab/LLM_4_Verilog`, `WithinUsAI/Robotics_25k`.
