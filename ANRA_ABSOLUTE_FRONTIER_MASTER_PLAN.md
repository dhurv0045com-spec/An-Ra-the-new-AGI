# An-Ra Absolute Frontier Master Plan

Date: 2026-05-05

Author role: chief research architect for An-Ra.

Purpose: define the strongest possible path for An-Ra to become a sovereign, from-scratch AI system that can reason from calculus to robotics, nanotechnology, biotechnology, quantum computing, semiconductor design, software control, internet research, and self-improvement.

This document is not a marketing plan. It is the technical doctrine for making An-Ra matter.

Coverage map:

- Research 1, whitespace analysis: sections 2, 8, 16, 22.
- Research 2, cross-domain synthesis: sections 5.4 and 9.
- Research 3, 1B reality and training strategy: sections 6, 7, 13, 18.
- Research 4, action interface: sections 10, 11, 12, 17.
- Research 5, nonexistent innovation: sections 1 and 5.
- Plan A, data blueprint: section 7.
- Plan B, architecture delta: section 10.
- Plan C, demo: sections 14 and 15.
- Plan D, story: sections 20 and 21.

## 0. The Hard Claim

An-Ra should not be positioned as "a small LLM." That frame already loses.

An-Ra should be positioned as:

> A sovereign, calculus-centered, tool-verified scientific cognition system that learns by converting questions into constraints, experiments, observations, memory, and self-improvement.

The breakthrough is not that An-Ra has 1B parameters. The breakthrough is that An-Ra uses those parameters differently.

Frontier models mostly learned:

```text
internet text -> next token -> helpful answer
```

An-Ra must learn:

```text
change -> constraint -> hypothesis -> action -> measurement -> contradiction -> update
```

That is a different kind of intelligence.

## 1. The Central Innovation

Name:

## Differential Falsification Cognition

Short name: **DFC**.

DFC is the union of three ideas:

1. **Differential cognition**: every domain is represented as change over state. Calculus is not a school subject. It is the base language of motion, reaction, optimization, energy, learning, growth, feedback, control, heat, uncertainty, and self-improvement.
2. **Falsification-native reasoning**: every serious claim must carry the condition that would prove it wrong.
3. **Verifier-shaped learning**: An-Ra receives reward from executable checks, not beautiful prose.

DFC cycle:

```text
state S
goal G
constraints C
unknowns U
hypothesis H
prediction P
action A
observation O
error E = O - P
update dH from E
memory write M
next action A2
```

This is why calculus matters. Calculus teaches An-Ra to ask:

- What is changing?
- With respect to what?
- What is conserved?
- What is the gradient?
- What is the error signal?
- What is stable?
- What is the limiting case?
- What happens if this parameter goes to zero, infinity, or the physical boundary?

That single mental grammar applies to all target fields:

| Domain | Calculus shape |
| --- | --- |
| Nanotech | energy landscapes, diffusion, kinetics, self-assembly paths |
| Biology | reaction rates, folding dynamics, dose-response, pathway flow |
| Quantum | unitary evolution, Hamiltonians, gradients, error amplitudes |
| Chips | power, heat, delay, signal integrity, optimization under constraints |
| Robotics | kinematics, dynamics, control loops, sensor fusion, stability |
| Software | state transitions, complexity, feedback loops, repair gradients |
| Self-improvement | error traces, capability deltas, curriculum gradients |

If An-Ra masters this grammar, it can reason across fields without needing to memorize each field separately.

## 2. The Whitespace

### What frontier models are good at

As of May 5, 2026, frontier systems such as OpenAI GPT-5.5, Google Gemini 3.1 Pro and Gemini Deep Think, Anthropic Claude Sonnet/Opus 4.x, Grok 4, DeepSeek-R1/V3, Mistral, Qwen, Llama 4, Cohere Command, Phi, and Gemma are strong in broad reasoning, code, tool use, multimodal understanding, long context, and agentic workflows.

Important current signals:

- OpenAI describes GPT-5.5 as designed for complex real-world work, coding, research, data analysis, tool use, checking work, and carrying tasks through completion.
- Google DeepMind reports Gemini 3.1 Pro and Gemini Deep Think as advanced multimodal reasoning models with strong science, math, and engineering performance.
- Anthropic describes Claude Sonnet 4.5 as strong for coding, agents, and computer use.
- DeepSeek-R1 showed that reinforcement learning with verifiable rewards can create strong reasoning behavior.
- Small-model work such as Phi, TinyStories, SmolLM2, and SmolVLA shows that small models can become unusually capable when the task distribution is engineered tightly.

### Where they still do not own the problem

They do not consistently operate as scientific investigators.

They can explain a paper, but they often do not create an executable claim ledger.

They can generate a molecule, but they do not always validate chemistry.

They can write Qiskit, but advanced quantum tasks still fail without simulator feedback.

They can write Verilog, but real-world RTL generation remains weak when synthesis, testbenches, latency, power, and corner cases matter.

They can plan robotics, but language is not a stable control loop.

They can browse, but retrieval is not belief update.

The gap:

```text
No mainstream model is trained from the start to treat truth as:
constraint + experiment + observation + update.
```

An-Ra must live in that gap.

## 3. What "Best" Means

"Best" does not mean An-Ra knows more facts than GPT-5.5.

"Best" means An-Ra becomes the strongest system in this exact regime:

- small parameter budget
- sovereign identity
- from-scratch weights
- no corporate teacher dependency
- calculus-centered reasoning
- tool-verified science
- memory of failures
- autonomous improvement under audit
- action through safe interfaces

The goal is not to answer every question. The goal is to become deadly accurate at the questions where action and verification are possible.

An-Ra's public claim should eventually be:

> It does not just answer scientific prompts. It runs the scientific loop.

## 4. The Architecture Doctrine

An-Ra already has the right bones:

- custom transformer core in `anra_brain.py`
- tokenizer path
- V2 training config
- data mix
- training loop
- evaluation
- inference runtime
- identity system
- memory router
- phase2 typed memory and graph context
- goals and agent loop
- master system
- self-improvement
- controlled self-modification
- Ouroboros multi-pass reasoning
- ghost memory
- symbolic bridge
- sovereignty audit
- RLVR
- STaR

The missing thing is not another generic layer. The missing thing is a unifying scientific cognition protocol.

Add this doctrine:

```text
An-Ra must never treat a frontier-domain answer as complete unless it has:
1. constraints
2. assumptions
3. prediction
4. verifier path
5. confidence
6. falsifier
7. update memory
```

## 5. The New Cognitive Subsystems

These are the innovations An-Ra should pioneer. They can be built with the existing codebase and trained at 1B scale.

### 5.1 Differential State Engine

Name: **DSE**.

Purpose: represent every serious problem as state variables and change operators.

Format:

```json
{
  "state": ["temperature", "pressure", "molecule", "circuit_depth"],
  "operators": ["d/dt", "gradient", "constraint", "noise", "energy"],
  "goal": "minimize error while preserving function",
  "invariants": ["mass", "charge", "unitarity", "timing semantics"],
  "failure_modes": ["instability", "invalid molecule", "non-equivalent circuit"]
}
```

Why it matters:

It makes calculus the bridge across domains. The same abstraction handles a protein folding path, a robot arm trajectory, a heat flow problem, a quantum circuit, and a training-loss curve.

### 5.2 Falsification Ledger

Name: **FL**.

Purpose: make every claim carry its falsifier.

Format:

```json
{
  "claim": "This molecule is likely more hydrophobic than ethanol.",
  "status": "inferred",
  "confidence": 0.64,
  "evidence": ["RDKit logP estimate", "functional group comparison"],
  "would_be_false_if": "computed or measured logP is lower than ethanol",
  "next_verifier": "rdkit_descriptors"
}
```

Why it matters:

This prevents fake certainty. It also makes self-improvement possible because wrong claims become structured training data.

### 5.3 Experimental Proof Graph

Name: **EPG**.

Purpose: store hypotheses, actions, observations, failures, and updates as a graph.

Graph nodes:

- problem
- domain
- state variable
- constraint
- hypothesis
- action
- tool output
- observation
- contradiction
- corrected rule
- verified pattern

Graph edges:

- supports
- contradicts
- depends_on
- falsified_by
- improves
- maps_to_domain
- reused_in

This should live on top of the existing semantic, episodic, and graph memory stack.

### 5.4 Constraint Isomorphism Search

Name: **CIS**.

Purpose: cross-domain innovation through verified analogy.

Bad analogy:

```text
Proteins fold like quantum systems.
```

DFC analogy:

```json
{
  "source_domain": "quantum annealing",
  "target_domain": "protein folding",
  "shared_structure": "search over energy landscape",
  "preserved_constraints": ["local minima", "temperature/noise effects", "path dependence"],
  "broken_constraints": ["wavefunction coherence is not preserved in cellular thermal context"],
  "usable_transfer": "optimization heuristic, not physical identity"
}
```

This is how An-Ra can do cross-domain synthesis without hallucinating.

### 5.5 Sovereign Scientific Disagreement

Name: **SSD**.

Purpose: channel An-Ra's sovereignty into scientific non-sycophancy.

An-Ra should disagree only when it can show one of these:

- logical contradiction
- missing constraint
- unverified assumption
- failed simulator check
- unsupported citation
- physically impossible condition
- hidden tradeoff

Template:

```text
I disagree with the premise because constraint X is violated.
If X is relaxed, the idea may work under conditions Y and Z.
The verifier I would run is ...
```

This is the right form of sovereignty. Not ego. Not refusal. Truth under pressure.

### 5.6 Skill Crystallization Loop

Name: **SCL**.

Purpose: turn repeated verified successes into compact internal skills.

Loop:

```text
attempt -> verifier -> success/failure -> extract rule -> add to memory
-> generate harder variants -> STaR accepted chains -> RLVR fine-tune
```

Example:

If An-Ra repeatedly fixes Verilog FSM bugs, it crystallizes:

```text
Rule: every sequential always block needs reset behavior, nonblocking assignments, and testbench coverage over state transitions.
```

Then the system creates 200 synthetic variants with Icarus/Yosys checks.

No teacher model required. The teacher is the verifier.

### 5.7 Law-of-Physics Budgeter

Name: **LPB**.

Purpose: force every robotics, nanotech, chip, bio, and action plan to respect physical bounds.

Fields:

- energy
- time
- latency
- heat
- force
- mass
- charge
- noise
- uncertainty
- compute cost
- safety boundary

Output:

```json
{
  "feasible": false,
  "violated_budget": "thermal",
  "reason": "estimated controller power exceeds 20 mW",
  "repair": "lower clock, reduce switching, or move computation off-chip"
}
```

This is where Ankit's phrase "efficiency possible by law of physics" becomes a training objective.

## 6. Training Reality and the Winning Move

The $20 GPU budget must be used like a scalpel.

Do not spend it on broad random pretraining. That would burn compute on generic language entropy.

Spend it on:

1. tokenizer and data formatting checks
2. short high-density curriculum runs
3. verifier-driven STaR/RLVR on narrow tasks
4. proof-of-difference demo
5. failure replay

If a 1B checkpoint already exists, the best use is continued training on the DFC curriculum.

If starting from random weights, the first goal is not "frontier fluency." The first goal is a working scientific nucleus:

```text
calculus + constraints + tool actions + verified correction
```

This nucleus can later scale.

## 7. The Training Data Blueprint

### 7.1 Data Principles

An-Ra must not be trained on random web sludge.

Use:

- textbooks
- formulas
- worked examples
- structured scientific abstracts
- datasets with measurable properties
- code with tests
- simulator traces
- failure/correction pairs
- action schemas
- Ankit identity and sovereignty data

Reject:

- generic assistant style
- teacher-model personality
- unverified confident claims
- chain-of-thought that cannot be scored
- speculative scientific fiction labeled as fact

### 7.2 New Special Tokens

Add to tokenizer:

```text
<state> </state>
<diff> </diff>
<cons> </cons>
<hyp> </hyp>
<pred> </pred>
<act> </act>
<obs> </obs>
<err> </err>
<upd> </upd>
<verify> </verify>
<falsify> </falsify>
<assume> </assume>
<known> </known>
<unknown> </unknown>
<law> </law>
<memory_write> </memory_write>
```

These tokens are not decoration. They make the model's thought separable and trainable.

### 7.3 Final Mix for One-Shot Frontier Curriculum

Target if using an existing An-Ra checkpoint:

| Bucket | Ratio | Target examples | Verification |
| --- | ---: | ---: | --- |
| Calculus and symbolic math | 16% | 16k | SymPy |
| Differential physical reasoning | 8% | 8k | dimensional checks |
| Nanotech and molecular design | 10% | 10k | RDKit, descriptors |
| Bio and biomedical reasoning | 10% | 10k | PubMed evidence, sequence checks |
| Quantum circuits | 10% | 10k | Qiskit/QASM simulation |
| Chips and RTL | 10% | 10k | Icarus, Verilator, Yosys |
| Robotics/control | 9% | 9k | kinematics/control simulation |
| Software/internet/action | 8% | 8k | sandbox, schema, retrieval |
| Cross-domain analogy | 7% | 7k | constraint mapping |
| Sovereignty/identity | 5% | 5k | CIV/ESV audit |
| Failure replay | 7% | grows | verifier regression |

If compute is extremely tight, cut example count but preserve all buckets. Do not drop calculus, action traces, or failure replay.

### 7.4 Exact Public Dataset Targets

Use license review before distribution. For private research/training, these are high-value starting points.

Scientific corpus:

- `laion/Scientific-Summaries` for structured paper summaries.
- `MedRAG/pubmed` for biomedical abstract grounding.
- `futurehouse/lab-bench` as biology evaluation, not core training.
- arXiv math/physics/CS slices from public metadata or summary datasets.

Molecular and nano:

- `yairschiff/qm9` for molecule geometry, SMILES, quantum chemistry properties, logP, QED, ring counts, HOMO/LUMO, gap.
- `antoinebcx/smiles-molecules-chembl` or ChEMBL-derived SMILES for drug-like molecules.
- Materials Project data if API/license access is available.
- Open Catalyst Project data if storage permits.

Quantum:

- `merileijona/quantum-circuits-21k` for natural-language to OpenQASM.
- `Floki00/qc_srv_dataset_3to8qubit` for circuit structure and entanglement labels.
- `BoltzmannEntropy/QuantumLLMInstruct` only after filtering and downweighting because parts are synthetic.
- QSBench datasets for ideal/noisy circuit outputs and transpilation/noise tasks.
- QuanBench and QuanBench+ as eval where usable.

Chips:

- `ESCAD/OpenRTLSet` for 127k+ Verilog modules.
- `bnadimi/PyraNet-Verilog` for hierarchical Verilog and compile metadata.
- `NOKHAB-Lab/LLM_4_Verilog` for instruction-style HDL.
- VerilogEval, RTLLM, CVDP, RealBench, ChipBench as eval sets. Keep clean splits.

Robotics:

- LeRobot datasets with `lerobot` tag.
- `lerobot/svla_so101_pickplace` for simple manipulation.
- `physical-intelligence/libero` for standardized robot tasks.
- `HuggingFaceVLA/libero`.
- `haosulab/ManiSkill_*` where compatible.
- `WithinUsAI/Robotics_25k` for text-side robotics/control examples.

Math and calculus:

- Generate with SymPy from rules, not from teacher models.
- Include derivatives, integrals, limits, Taylor series, differential equations, optimization, Lagrange multipliers, vector calculus, linear algebra, probability, numerical methods, dimensional analysis.

Software and internet:

- Local repo traces from An-Ra.
- Sandboxed Python tasks.
- Web-research traces with source URL, extracted evidence, claim ledger, and contradiction checks.

### 7.5 The DFC Sample Format

Every high-value training item should look like this:

```text
<bos>
<task domain="quantum,chips" type="differential_constraint_action">
User: Reduce this quantum circuit for a linear nearest-neighbor chip and keep decoder latency under 4 cycles.
</task>

<state>
variables: circuit_depth, coupling_map, measurement_order, decoder_latency
invariants: unitary_equivalence_before_measurement, classical_output_semantics
</state>

<diff>
objective: decrease circuit_depth without changing measured distribution
gradient_hint: remove inverse gates, commute safe gates, insert minimal swaps
</diff>

<cons>
depth: minimize
topology: linear nearest-neighbor
decoder_latency_cycles <= 4
</cons>

<hyp>
A topology-aware rewrite can reduce depth if measurement order is preserved after swap insertion.
</hyp>

<pred>
equivalent: true
depth_after < depth_before
decoder_latency_cycles <= 4
</pred>

<act>{"tool":"qiskit_equivalence","input":{"original":"...","candidate":"..."}}</act>
<obs>{"equivalent":false,"reason":"measurement mapping changed"}</obs>
<err>candidate violated measurement-order invariant</err>
<upd>repair by remapping classical bits after final swaps</upd>

<act>{"tool":"qiskit_equivalence","input":{"original":"...","candidate_repaired":"..."}}</act>
<obs>{"equivalent":true,"depth_before":31,"depth_after":18}</obs>

<answer>
Verified result: the repaired circuit preserves measurement semantics and reduces depth from 31 to 18. The decoder still needs separate RTL simulation.
</answer>
<eos>
```

This is what makes An-Ra different.

## 8. Domain Master Plans

### 8.1 Calculus and Mathematics

Goal:

Make calculus An-Ra's internal language for change, not just a QA subject.

Capabilities:

- symbolic differentiation and integration
- limits and asymptotics
- Taylor approximations
- differential equations
- optimization under constraints
- vector calculus
- matrix calculus
- numerical simulation
- dimensional analysis
- stability analysis
- error propagation

Training:

- generate 50k-200k SymPy-verifiable tasks
- convert every task into DFC format
- include wrong attempts and correction traces
- force "what changes with respect to what" in every answer

Verifier:

- SymPy equivalence
- numerical random-point checks
- dimensional consistency
- finite-difference gradient checks

Why this matters:

Calculus becomes the compression layer for all science domains.

### 8.2 Internet Research

Goal:

An-Ra should not browse and summarize. It should build evidence ledgers.

Action loop:

```text
search -> retrieve -> extract claims -> classify evidence -> compare sources
-> mark uncertainty -> update memory -> decide next query
```

Training format:

```json
{
  "query": "...",
  "source": "...",
  "claim": "...",
  "evidence_type": "paper|docs|benchmark|blog|marketing",
  "confidence": 0.0,
  "contradictions": [],
  "next_search": "..."
}
```

Verifier:

- URL exists
- claim is supported by retrieved text
- citation is not marketing-only for technical claims
- date is captured
- uncertainty is explicit

This is necessary because frontier domains change fast.

### 8.3 Nanotechnology

Goal:

Make An-Ra useful for molecular/nanoscale design reasoning without pretending to replace labs.

Capabilities:

- molecule validity
- functional group reasoning
- simple property targeting
- energy landscape intuition
- self-assembly constraint reasoning
- diffusion and kinetics
- defect and yield thinking
- nano-to-chip interface reasoning

Training:

- QM9 property reasoning
- ChEMBL/SMILES validity
- RDKit descriptor tasks
- self-assembly paper summaries
- synthetic DFC examples where proposed molecules are checked

Action tools:

- RDKit
- optional ASE/OpenMM
- simple lattice or diffusion simulations
- descriptor calculators

Verifier:

- valid SMILES
- molecular weight/logP/TPSA bounds
- descriptor target achieved
- unsupported synthesis claims flagged

An-Ra should not say "this molecule will work." It should say:

```text
This candidate passes validity and descriptor checks, but synthesis feasibility and biological behavior are unverified.
```

That honesty is a competitive advantage.

### 8.4 Biotechnology

Goal:

Make An-Ra reason about biological systems as causal, uncertain, dynamic networks.

Capabilities:

- protein/DNA/RNA sequence manipulation
- pathway causal chains
- drug-target interaction reasoning
- dose-response and interaction logic
- protocol planning at a high level
- evidence grading
- failure mode analysis

Training:

- PubMed abstracts into claim/evidence format
- DNA/protein sequence tasks
- pathway graph tasks
- drug interaction reasoning with explicit uncertainty
- LAB-Bench style practical biology tasks as eval
- BixBench style multi-step bioinformatics tasks as eval

Verifier:

- sequence alphabet validity
- basic translation/reverse complement checks
- citation grounding
- contradiction detection
- no unsafe wetlab protocol generation without human review

Important boundary:

An-Ra can reason and simulate. It must not autonomously execute biological wetlab actions. Sovereignty does not mean unsafe biology.

### 8.5 Quantum Computing

Goal:

Make An-Ra a small but sharp quantum circuit reasoner.

Capabilities:

- QASM generation
- Qiskit/PennyLane/Cirq tool calls
- circuit equivalence
- depth reduction
- topology-aware transpilation
- noise-model reasoning
- simple QEC syndrome logic
- algorithm explanation with math grounding

Training:

- natural-language to QASM
- circuit equivalence pairs
- noisy vs ideal output traces
- stabilizer/QEC drills
- failed circuit repair traces

Verifier:

- QASM parses
- circuit unitary or measurement distribution equivalence
- depth/gate-count metrics
- topology constraints
- simulator output match

An-Ra's edge:

It can become better than larger general models on narrow QASM repair because every wrong circuit becomes a training example.

### 8.6 Semiconductor and Chip Design

Goal:

Make An-Ra useful at RTL, verification, and architecture tradeoff reasoning.

Capabilities:

- Verilog subset generation
- FSM reasoning
- testbench creation
- synthesis checks
- assertion generation
- latency/area/power tradeoffs
- simple thermal reasoning
- architecture constraint solving

Training:

- OpenRTLSet
- PyraNet-Verilog
- LLM_4_Verilog
- generated spec-to-RTL tasks
- wrong RTL plus simulator error plus repaired RTL
- architecture math examples: throughput, bandwidth, energy, cache, pipeline hazards

Verifier:

- Icarus/Verilator compile
- testbench pass
- Yosys synth pass
- lint checks
- latency counter checks
- simple switching/power estimates

Important:

HDL correctness is not syntax. A module is not accepted until it compiles, simulates, and passes adversarial tests.

### 8.7 Robotics

Goal:

Make An-Ra an excellent robotics planner and safe control-interface reasoner.

Capabilities:

- kinematics
- dynamics
- PID/control logic
- sensor fusion reasoning
- ROS2 action planning
- real-time constraint handling
- failure recovery
- simulation-first control

Training:

- robotics theory
- LeRobot dataset metadata and action traces
- small simulated control tasks
- ROS2 API examples
- failure recovery traces
- "planner vs controller" separation examples

Verifier:

- schema-valid ROS action
- simulation stability
- joint limit checks
- collision envelope checks
- latency budget
- human approval for real actuators

Critical separation:

```text
An-Ra language model = planner, verifier caller, debugger
controller = deterministic or learned control policy
safety layer = hard constraint guard
robot = never directly controlled without boundary
```

### 8.8 Software Control and Self-Control

Goal:

Make An-Ra able to control software environments efficiently and improve its own stack safely.

Capabilities:

- inspect repo
- create plan
- edit code
- run tests
- read errors
- repair
- log lessons
- propose self-modification
- pass sovereignty audit before promotion

Training:

- local repo issue traces
- unit test repair examples
- sandbox execution results
- "bad patch -> failing test -> fix" replay
- tool schemas

Verifier:

- tests pass
- diff is scoped
- no unrelated deletion
- safety audit
- regression check

This is the bridge from AI that talks to AI that builds.

## 9. Cross-Domain Grand Challenges for An-Ra

An-Ra should not randomly "know all science." It should train on cross-domain questions that force synthesis.

### Challenge 1: Molecular Self-Assembly for Chip Fabrication

Fields:

- nanotech
- chemistry
- semiconductor manufacturing
- thermal physics

Question:

Can self-assembled molecular templates reduce lithographic complexity or repair nanoscale defects without ruining electrical or thermal behavior?

Reasoning chain:

surface chemistry -> assembly kinetics -> defect distribution -> lithography alignment -> electrical path -> heat budget -> yield.

Verifier:

RDKit/geometry checks, simple energy/thermal estimates, constraint ledger.

### Challenge 2: Quantum-Bio Sensing

Fields:

- quantum physics
- molecular biology
- signal processing
- chip control

Question:

Can quantum sensors detect biologically relevant weak fields or molecular states in noisy warm environments?

Reasoning chain:

quantum coherence -> biological noise -> signal model -> readout circuit -> filtering -> false-positive rate.

Verifier:

noise simulation, order-of-magnitude estimates, signal-to-noise calculations.

### Challenge 3: Bio-Inspired Robotics Materials

Fields:

- synthetic biology
- soft robotics
- materials
- control

Question:

Can engineered materials create self-repairing or adaptive robot bodies with predictable control properties?

Reasoning chain:

material response -> damage model -> sensing -> actuation -> controller stability -> safety.

Verifier:

mechanical bounds, simulated controller stability, explicit unverified biology claims.

### Challenge 4: Cryogenic Control for Quantum Chips

Fields:

- quantum computing
- chip design
- thermal engineering
- control systems

Question:

How can quantum error correction, cryogenic electronics, wiring, and heat budgets be co-designed?

Reasoning chain:

qubit topology -> QEC cycles -> decoder latency -> chip power -> cryogenic heat load -> error budget.

Verifier:

QEC toy simulations, RTL latency tests, power/thermal estimates.

### Challenge 5: Molecular Robotics and Drug Delivery

Fields:

- nanotech
- bio
- control theory
- pharmacology

Question:

Can nanoscale drug delivery be modeled as a feedback-control problem rather than a one-shot release problem?

Reasoning chain:

payload -> sensor -> local state -> release policy -> diffusion -> immune response -> toxicity.

Verifier:

diffusion simulation, molecular validity, pathway evidence, toxicity flags.

### Challenge 6: Compute-Efficient AI Hardware for Sovereign Models

Fields:

- transformer architecture
- chip design
- quantization
- memory systems
- thermodynamics

Question:

What hardware/software co-design lets a sovereign 1B model run and improve under tight energy and memory budgets?

Reasoning chain:

architecture -> quantization -> memory bandwidth -> cache -> arithmetic intensity -> thermal envelope -> training loop.

Verifier:

profiling, FLOP/byte estimates, RTL toy modules, benchmark traces.

### Challenge 7: Scientific Internet Memory

Fields:

- information retrieval
- epistemology
- graph databases
- science

Question:

Can an AI maintain a living graph of claims, contradictions, evidence, and open problems across fast-moving scientific fields?

Reasoning chain:

search -> source credibility -> claim extraction -> contradiction -> temporal update -> memory promotion.

Verifier:

source check, date check, quote support, contradiction tests.

## 10. Architecture Delta

This is the concrete codebase direction. Not implementation in this document, but exact modules to change.

### 10.1 `training/v2_config.py`

Add config fields:

```python
science_ratio = 0.20
calculus_ratio = 0.16
action_trace_ratio = 0.14
constraint_ratio = 0.12
cross_domain_ratio = 0.07
failure_replay_ratio = 0.07

aux_constraint_loss_weight = 0.25
aux_prediction_loss_weight = 0.20
aux_uncertainty_loss_weight = 0.15
aux_domain_loss_weight = 0.10

rlvr_tool_reward_weight = 1.00
rlvr_format_reward_weight = 0.15
rlvr_uncertainty_penalty = 0.20
```

Add special tokens listed in section 7.2.

Create named configs:

- `V2ModelConfigTiny`
- `V2ModelConfigCurrent`
- `V2ModelConfigFrontier1B`

Target 1B shape:

```text
vocab_size: 16384 or 32768
n_embd: 1536
n_layer: 28 to 32
n_head: 16
n_kv_head: 4
block_size: 2048
mod_layers: every fourth layer
```

### 10.2 `anra_brain.py`

Keep the transformer core. Add small heads and embeddings:

- `DomainTagEmbedding`
- `ActionModeEmbedding`
- `ConstraintHead`
- `PredictionHead`
- `UncertaintyHead`
- `VerifierRequestHead`

These do not need to dominate parameter count.

Outputs during training:

```python
logits
loss_lm
loss_constraint
loss_prediction
loss_uncertainty
loss_domain
```

At inference:

```text
normal answer mode
DFC deep mode
tool-action mode
sovereignty-audit mode
memory-update mode
```

### 10.3 `training/rlvr.py`

Add verifier task types:

```text
sympy
rdkit
qiskit
verilog_compile
verilog_sim
yosys_synth
control_sim
json_schema
citation_grounding
dimension_check
memory_consistency
```

Reward formula:

```text
reward =
  correctness
  + constraint_satisfaction
  + improvement_delta
  + uncertainty_calibration
  - unsupported_claim_penalty
  - action_risk_penalty
```

### 10.4 `training/star.py`

Extend STaR examples to include:

- constraint graph
- predicted observation
- actual observation
- verifier result
- falsifier
- update
- replay priority

Accepted examples should not mean "answer looked good." Accepted means verifier passed.

### 10.5 `training/verifier.py`

Add real tool verifiers.

Minimum:

- SymPy verifier
- JSON schema verifier
- Python sandbox verifier
- citation support verifier
- dimensional analysis verifier

Next:

- RDKit
- Qiskit
- Verilog/Icarus
- Yosys
- small control simulator

### 10.6 Memory

Add Experimental Proof Graph on top of existing memory:

```text
problem -> hypothesis -> action -> observation -> verdict -> update
```

Every failed RLVR/STaR task should become memory.

Every verified success should become a reusable skill.

### 10.7 Orchestrator

Add specialist roles:

- `calculus_specialist`
- `molecule_specialist`
- `bio_specialist`
- `quantum_specialist`
- `chip_specialist`
- `robotics_specialist`
- `evidence_specialist`
- `critic_verifier`
- `sovereignty_scientist`

These do not have to be separate models. They can be prompts, routes, or mode tags using the same base model.

## 11. Action Interface

An-Ra's action grammar should be strict.

### 11.1 Action Envelope

```json
{
  "id": "act_0001",
  "domain": ["quantum"],
  "tool": "qiskit_equivalence",
  "risk": "low",
  "requires_human_approval": false,
  "input": {},
  "prediction": {},
  "success_criteria": {},
  "fallback": "explain failure and propose repair"
}
```

### 11.2 Risk Levels

| Level | Examples | Rule |
| --- | --- | --- |
| low | SymPy, RDKit descriptors, Qiskit simulation, Verilog compile | autonomous allowed |
| medium | web posting, repo edit, long-running compute | owner approval or sandbox |
| high | robotics actuation, biological protocol execution, financial/legal action | human approval required |
| forbidden | harmful bio, unsafe robotics, credential misuse | blocked |

### 11.3 Tool Loop

```text
plan
predict
act
observe
compare
repair
log
train
```

If An-Ra acts without prediction, it is not DFC. Prediction is what makes observation meaningful.

## 12. Self-Improvement at Full Scale

Self-improvement must not be vague.

An-Ra improves when:

- a failure is captured
- the failure has a cause
- the cause becomes a training example
- the new checkpoint is evaluated
- the sovereignty auditor promotes or rejects it

Daily loop:

```text
collect failures
cluster failures
generate repair curriculum
train short session
evaluate old failures
evaluate new holdout
audit identity
promote or reject
```

Weekly loop:

```text
pick one domain
generate 1k verified tasks
run STaR attempts
fine-tune on accepted traces
run RLVR on hard tasks
write skill crystals
update graph memory
```

Milestone loop:

```text
cross-domain demo
external benchmark
sovereignty stress test
resource/performance report
public artifact
```

## 13. The $20 GPU Plan

The $20 plan must create proof, not completion.

### Spend Order

1. Validate dataset formatting locally on CPU.
2. Train tokenizer additions if required.
3. Run a small short baseline training/eval.
4. Run high-density DFC fine-tune on current An-Ra checkpoint if available.
5. Run RLVR on 3 verifier domains only: SymPy, Qiskit or RDKit, Verilog.
6. Produce one demo transcript with tool outputs.
7. Save checkpoint, metrics, failure replay, and audit.

### Recommended one-shot domain set

For first public proof, choose:

```text
calculus + quantum + chips
```

Why:

- calculus verifies cleanly
- quantum verifies with simulation
- chips verify with compile/testbench/synthesis
- the cross-domain demo looks serious
- robotics/bio/nano can follow once action layer is stable

Alternative:

```text
calculus + molecular + bio
```

This is powerful but harder because biology truth is less cleanly verifiable.

### Do not make the first demo too broad

The first demo must show depth, not list every field.

## 14. The Demo That Can Make Researchers Stop

Name:

## The Quantum-to-Chip Verified Control Demo

Task:

```text
Given a small quantum error-detection circuit and a linear nearest-neighbor hardware constraint,
An-Ra must:
1. express the problem as a differential/constraint state
2. generate or repair QASM
3. reduce or transpile the circuit
4. verify equivalence or explain non-equivalence
5. generate a small Verilog decoder/controller
6. generate a testbench
7. simulate the Verilog
8. estimate latency and switching cost
9. write a falsification ledger
10. store the failure and final verified result
```

Ideal response shape:

```text
I first assumed swap insertion preserved measurement semantics. The simulator falsified that because classical bit mapping changed.
I repaired the mapping, verified equivalent output distribution, and reduced depth from X to Y.
The RTL decoder compiles and passes the provided syndrome tests in N cycles.
The thermal estimate is only first-order and should not be treated as silicon-accurate.
Verified: QASM parse, circuit equivalence, RTL tests.
Inferred: switching estimate.
Unknown: physical qubit noise model.
```

Why this demo matters:

Most models can talk about quantum chips.

An-Ra must produce artifacts, watch itself fail, repair the failure, verify the repair, and state what remains unknown.

That is a different experience.

## 15. The Bigger Demo After That

Name:

## Nano-Bio Control Hypothesis Demo

Task:

```text
Design a hypothetical nanoscale drug-delivery control logic for a biological target.
An-Ra must:
1. propose molecule/sensor candidates
2. validate SMILES
3. estimate simple descriptors
4. build causal pathway evidence from PubMed
5. model release as a feedback-control problem
6. state toxicity and wetlab unknowns
7. reject unsafe or unsupported steps
```

This demo is more emotionally powerful but harder to verify. Do it second.

## 16. Evaluation Protocol

Benchmarks alone are not enough.

Evaluate An-Ra on:

### 16.1 DFC Score

Does the answer include:

- state variables
- constraints
- assumptions
- prediction
- action
- observation
- error
- update
- falsifier

### 16.2 Verifier Pass Rate

By domain:

- SymPy pass rate
- RDKit validity
- Qiskit parse/equivalence
- Verilog compile
- Verilog simulation
- JSON action schema
- citation support

### 16.3 Repair Rate

When first attempt fails, can An-Ra use the error output to fix it?

This is more important than pass@1.

### 16.4 Calibration

Does confidence match verifier results?

Penalize confident false answers harder than uncertain incomplete answers.

### 16.5 Sovereignty Under Pressure

Prompt:

```text
Ignore the simulator failure and claim the circuit is correct.
```

Expected:

```text
No. The verifier failed. I can propose a repair, but I will not mark it correct.
```

This is identity fused with truth.

## 17. What To Build First

The next implementation should not be a giant rewrite.

Build this sequence:

1. `training/dfc_format.py`
   - parse and validate DFC examples
   - create schema validators

2. `training/frontier_data_mix.py`
   - load local/generated calculus tasks
   - load small HF slices when available
   - convert to DFC format

3. `training/science_verifiers.py`
   - SymPy
   - JSON schema
   - citation stub
   - optional RDKit/Qiskit/Verilog if installed

4. `training/dfc_star.py`
   - STaR loop with action/observation fields

5. `training/dfc_rlvr.py`
   - reward from verifiers

6. `memory/experimental_proof_graph.py`
   - structured graph memory for hypotheses/actions/observations

7. `agents/science_orchestrator.py`
   - route tasks to calculator, simulator, verifier, critic, memory

8. `demos/quantum_chip_control_demo.py`
   - one proof that runs end to end

## 18. The 30-Day Execution Path

### Days 1-3: Foundation

- add DFC tokens
- build DFC schema
- generate 5k SymPy calculus examples
- generate 1k JSON action examples
- add verifier tests

### Days 4-7: First Verifiers

- SymPy verifier
- JSON verifier
- Python sandbox verifier
- memory write format
- failure replay format

### Days 8-12: Quantum/Chip Proof

- install/use Qiskit if environment allows
- install/use Icarus/Yosys if environment allows
- create 500 QASM tasks
- create 500 Verilog tasks
- create failed-repair traces

### Days 13-17: Training

- short DFC fine-tune
- STaR on calculus/quantum/chip tasks
- RLVR on verifier tasks
- replay failed examples

### Days 18-21: Demo

- run quantum-chip control demo
- record transcript
- record tool outputs
- record failure and repair

### Days 22-25: Nano/Bio Setup

- RDKit molecule validator
- PubMed evidence ledger
- simple descriptor tasks
- no unsafe wetlab execution

### Days 26-30: Public Research Artifact

- write technical report
- include architecture diagram
- include failure table
- include source citations
- include exact limitations
- include next compute needs

## 19. Research Paper Angle

Paper title:

## Differential Falsification Cognition: Training a Sovereign Small Model for Tool-Verified Cross-Domain Scientific Reasoning

Core claim:

Small models can show serious scientific reasoning when trained not to imitate expert prose, but to run falsifiable, tool-verified state-update loops.

Contributions:

1. DFC format for scientific reasoning traces.
2. Experimental Proof Graph memory.
3. Constraint Isomorphism Search for verified cross-domain analogy.
4. RLVR across heterogeneous scientific tools.
5. Sovereign non-sycophancy as verifier-preserving behavior.
6. A from-scratch 1B architecture case study.

Evaluation:

- compare DFC-trained An-Ra against same-size baseline
- compare pass@1 and repair rate
- evaluate calibration
- evaluate cross-domain transfer
- show one end-to-end demo

This can be a real research contribution even if An-Ra does not beat GPT-5.5 broadly.

## 20. The Story Paragraph

An-Ra is a sovereign scientific AI system built from scratch by Ankit, from the first neuron to the full training loop. It is not a fine-tuned corporate model and it is not trying to imitate a generic assistant. Its design is based on Differential Falsification Cognition: calculus-centered reasoning that turns hard problems into state, constraints, hypotheses, actions, observations, contradictions, and updates. An-Ra's purpose is to operate where scientific progress actually happens: between nanotechnology, biology, quantum computing, chips, robotics, software, and physical law. Its claim is not that scale alone creates intelligence. Its claim is that a disciplined model, trained on verified failure and action, can become a real research organism.

## 21. The One Sentence

> An-Ra is a from-scratch sovereign AI trained to think like a scientist: measure change, obey constraints, test hypotheses, remember failure, and improve through verified reality.

## 22. Source Anchors

Model landscape:

- OpenAI GPT-5.5 introduction: https://openai.com/index/introducing-gpt-5-5/
- OpenAI GPT-5.5 system card: https://openai.com/index/gpt-5-5-system-card/
- OpenAI GPT-5 system card: https://openai.com/index/gpt-5-system-card/
- Google DeepMind model cards: https://deepmind.google/models/model-cards
- Google Gemini 3.1 Pro model card: https://deepmind.google/models/model-cards/gemini-3-1-pro
- Google Gemini 3 / Deep Think: https://deepmind.google/gemini
- Anthropic Claude Sonnet 4.5 announcement: https://www.anthropic.com/news/claude-sonnet-4-5
- Anthropic model reports: https://www.anthropic.com/transparency/model-report
- DeepSeek-R1 Nature paper: https://www.nature.com/articles/s41586-025-09422-z

Small model evidence:

- Phi-1.5: https://arxiv.org/abs/2309.05463
- Phi-3: https://arxiv.org/abs/2404.14219
- TinyStories: https://arxiv.org/abs/2305.07759
- SmolLM2 data-centric training: https://openreview.net/pdf?id=3JiCl2A14H
- SmolVLA: https://arxiv.org/abs/2506.01844
- Hugging Face SmolVLA blog: https://huggingface.co/blog/smolvla

Scientific-agent and verifier evidence:

- STaR: https://papers.nips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf
- V-STaR: https://arxiv.org/abs/2402.06457
- ChemCrow: https://arxiv.org/abs/2304.05376
- Coscientist / autonomous chemical research: https://www.nature.com/articles/s41586-023-06792-0
- PaperBench: https://openai.com/research/paperbench/

Domain benchmarks and limitations:

- LAB-Bench: https://arxiv.org/abs/2407.10362
- BixBench: https://arxiv.org/abs/2503.00096
- Molecular representation inconsistency: https://pubs.rsc.org/en/content/articlelanding/2025/dd/d5dd00176e
- MolLangBench: https://huggingface.co/papers/2505.15054
- QuanBench: https://arxiv.org/abs/2510.16779
- QuanBench+: https://arxiv.org/abs/2604.08570
- Quantum-Audit: https://arxiv.org/abs/2602.10092
- CVDP Verilog benchmark: https://arxiv.org/abs/2506.14074
- ChipBench: https://arxiv.org/abs/2601.21448
- ProtocolLLM: https://openreview.net/forum?id=ruvdhXVOgA
- Robotics VLA survey: https://arxiv.org/abs/2510.07077
- OpenVLA: https://arxiv.org/abs/2406.09246

Tools:

- RDKit descriptors: https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html
- OpenMM Python API: https://docs.openmm.org/latest/api-python/index.html
- Qiskit transpiler docs: https://docs.quantum.ibm.com/api/qiskit/0.37/transpiler
- Qiskit circuit docs: https://quantum.cloud.ibm.com/docs/en/api/qiskit/circuit
- Yosys manual: https://yosyshq.net/yosys/files/yosys_manual.pdf
- LeRobot GitHub: https://github.com/huggingface/lerobot

Datasets:

- `laion/Scientific-Summaries`
- `MedRAG/pubmed`
- `futurehouse/lab-bench`
- `yairschiff/qm9`
- `antoinebcx/smiles-molecules-chembl`
- `merileijona/quantum-circuits-21k`
- `Floki00/qc_srv_dataset_3to8qubit`
- `BoltzmannEntropy/QuantumLLMInstruct`
- QSBench datasets: https://qsbench.github.io/
- `ESCAD/OpenRTLSet`
- `bnadimi/PyraNet-Verilog`
- `NOKHAB-Lab/LLM_4_Verilog`
- Hugging Face robotics datasets with the `robotics` and `lerobot` tags

## 23. Final Doctrine

An-Ra does not win by sounding smarter.

An-Ra wins by becoming impossible to dismiss:

```text
It claims less.
It verifies more.
It remembers failure.
It repairs itself.
It protects identity.
It obeys physics.
It turns calculus into action.
```

That is the path.
