# Causal Learning Dynamics (CLD) & Cognitive Acquisition Dynamics (CAD)
## Prospective Longitudinal Preregistration for An-Ra Experiment P35-A

---

## 1. Executive Summary & Signature Scientific Question

While conventional pretraining evaluations measure cognitive capabilities only at the final checkpoint, **Causal Learning Dynamics (CLD)** measures how distinct cognitive computations emerge, phase-transition, regress, or interact longitudinally across training time under controlled pretraining treatments.

### Signature Research Question:
> **What is the causal trajectory by which a small Transformer acquires cognitive computations? Which training experiences cause which cognitive computations to emerge, at what training stage, in what order, and with what substrate tradeoffs?**

This establishes a unified research loop connecting Senora and Triquetra:
- **Triquetra**: $\text{Failure} \xrightarrow{\text{Intervention } I} \text{Diagnostic Response } R(x, I)$
- **Senora**: $\text{Training Treatment } T \xrightarrow{\text{Milestone } t} \text{Cognitive Trajectory } C(T, t, c)$
- **Combined Loop**: Which training experience $T$ internalizes the computation exposed by inference intervention $I$?

---

## 2. Core Mathematical Formalism

### 2.1 Capability State Function
For a pretraining treatment arm $T \in \{\text{control-00}, \text{cognition-15-ce}, \text{cognition-15-qswap}\}$, cumulative training token milestone $t \in \mathcal{S}_{\text{tokens}}$, and cognitive primitive family $c \in \mathcal{F}$:

$$C(T, t, c) \in [0, 1]$$

representing unassisted RAW-CORE accuracy at milestone $t$.

### 2.2 Causal Treatment Trajectory
The longitudinal causal effect induced by training experience $T$ relative to the matched substrate control is:

$$\Delta C(t, c) = C(\text{treatment}, t, c) - C(\text{control}, t, c)$$

### 2.3 Full Cognitive Acquisition Vector
At every checkpoint milestone $t$, Senora evaluates:

$$A_t(T) = \left[ C(T, t, c_1), \dots, C(T, t, c_K), S_{\text{query\_sens}}(T, t), I_{\text{inv\_stable}}(T, t), T_{\text{nat\_transfer}}(T, t), L_{\text{substrate}}(T, t) \right]$$

---

## 3. Preregistered Checkpoint Schedule

Checkpoint evaluations consume compute; therefore, the observation schedule is preregistered analytically and frozen prior to training launch:

$$\mathcal{S}_{\text{tokens}} = \{ 0, 1\text{M}, 2\text{M}, 5\text{M}, 10\text{M}, 20\text{M}, 35\text{M}, 50\text{M} \} \text{ tokens}$$

| Milestone | Cumulative Tokens | Global Update ($B=131,072$) | Primary Scientific Objective |
|:---:|:---:|:---:|---|
| **$t_0$** | $0$ | 0 | Initialization baseline prior to optimization. |
| **$t_1$** | $1,000,000$ | 8 | Immediate post-warmup emergence check. |
| **$t_2$** | $2,000,000$ | 15 | Early stable plateau entrance. |
| **$t_3$** | $5,000,000$ | 38 | Early-stage sample-efficiency inflection. |
| **$t_4$** | $10,000,000$ | 76 | **Preregistered Early Triage Horizon** (Predictive gate). |
| **$t_5$** | $20,000,000$ | 153 | Mid-training stable regime. |
| **$t_6$** | $35,000,000$ | 267 | Pre-annealing capability ceiling. |
| **$t_7$** | $50,000,000$ | 382 | **Frozen Primary Evaluation Checkpoint** (Post-annealing). |

---

## 4. Derived Causal Dynamics Metrics

1. **Tokens-to-Threshold (TTT)**:
   $$\tau^*(c, \theta) = \min \{ t \in \mathcal{S}_{\text{tokens}} \mid C(T, t, c) \ge \theta \}$$
   Pre-committed capability threshold: $\theta = 0.50$.
2. **Area Under Learning Curve (AULC)**:
   $$\text{AULC}(T, c) = \frac{1}{T_{\text{max}}} \int_0^{T_{\text{max}}} C(T, t, c) \, dt$$
3. **Treatment-Effect AUC (TE-AUC)**:
   $$\text{TE-AUC}(c) = \text{AULC}(\text{treatment}, c) - \text{AULC}(\text{control}, c)$$
4. **Synthetic-to-Natural Transfer Lag**:
   $$\text{TransferLag}(\theta) = \tau^*(\text{natural}, \theta) - \tau^*(\text{synthetic\_dev}, \theta)$$
5. **Phase Transition Indicators**:
   - Max slope: $\max_t \frac{dC}{dt}$.
   - Change-point milestone: $t_{\text{cp}} = \arg\max_t \left| \frac{d^2C}{dt^2} \right|$.
   - Transition width: $w = \tau^*(c, 0.90) - \tau^*(c, 0.10)$.
6. **Loss-Matched Cognition Gap**:
   $$\Delta C_L(L_0) = C(\text{treatment}, t_{\text{treat}}(L_0), c) - C(\text{control}, t_{\text{ctrl}}(L_0), c)$$
   Evaluated at substrate language validation loss $L_0 = 2.40$.
7. **Cognitive Forgetting Index (CFI)**:
   $$\text{CFI}(T, c) = \max\left(0, \max_t C(T, t, c) - C(T, t_{\text{final}}, c)\right)$$

---

## 5. Strict Separation: Dynamics Analysis vs Frozen Decision Rule

> [!IMPORTANT]
> **Zero Checkpoint-Selection Leakage**:
> Intermediate dynamics evaluations are strictly diagnostic.
> The primary scientific decision for Phase P35-A remains frozen to **Checkpoint $t_7$ ($50\text{M}$ tokens)**.
> Under no circumstances may an intermediate checkpoint be selected post-hoc as the "experiment outcome" because its cognition score was higher.

---

## 6. Freshness Preservation

Longitudinal evaluations throughout training are restricted strictly to:
- `DYNAMICS-DEV` (re-evaluated across milestones).
- `STRUCTURAL-OOD-DEV`.

The prospective `Split.FRESH` test suite remains locked behind the cryptographic firewall and is never exposed during intermediate training checkpoints.