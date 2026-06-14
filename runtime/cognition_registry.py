"""Architectural ownership registry for C-01 through C-07."""

from __future__ import annotations

from runtime.technology_registry import TechnologyContract


COGNITIVE_CAPABILITIES = (
    TechnologyContract("C-01", "Causal Reasoning Engine", "cognition facade", "cognition.cre:CausalReasoningEngine.classify_query", ("A-01",)),
    TechnologyContract("C-02", "Epistemic Tracker", "cognition facade", "cognition.epistemic_tracker:EpistemicTracker.assess", ("A-02",)),
    TechnologyContract("C-03", "Longitudinal Human Model", "cognition facade", "cognition.lhm:LongitudinalHumanModel.inspect", ("A-03",)),
    TechnologyContract("C-04", "Scientific Self-Improvement", "cognition facade", "cognition.ssie:ScientificSelfImprovementEngine.propose", ("A-04",)),
    TechnologyContract("C-05", "Cross-Domain Synthesis", "cognition facade", "cognition.cdse:CrossDomainSynthesisEngine.synthesize", ("A-05",)),
    TechnologyContract("C-06", "Experience Consolidation", "cognition facade", "cognition.cec:ContinuousExperienceConsolidator.consolidate", ("A-06",)),
    TechnologyContract("C-07", "Multi-Agent Self-Debate", "cognition facade", "cognition.self_debate:MultiAgentSelfDebate.run", ("A-01", "A-02")),
)


def cognition_registry() -> dict[str, TechnologyContract]:
    return {capability.technology_id: capability for capability in COGNITIVE_CAPABILITIES}


def validate_cognition_reachability() -> dict[str, str]:
    result = {}
    for capability in COGNITIVE_CAPABILITIES:
        if capability.resolve() is None:
            raise RuntimeError(f"{capability.technology_id} resolved to None")
        result[capability.technology_id] = capability.entrypoint
    return result
