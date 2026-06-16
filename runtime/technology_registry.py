"""Canonical architectural ownership registry for T-01 through T-26."""

from __future__ import annotations

from dataclasses import dataclass
import importlib


@dataclass(frozen=True)
class TechnologyContract:
    technology_id: str
    name: str
    owner: str
    entrypoint: str
    metric_ids: tuple[str, ...]

    def resolve(self):
        module_name, _, attribute_path = self.entrypoint.partition(":")
        value = importlib.import_module(module_name)
        for attribute in attribute_path.split("."):
            value = getattr(value, attribute)
        return value


TECHNOLOGIES = (
    TechnologyContract("T-01", "LBA", "model forward", "anra_brain:MultiHeadAttentionV2.forward", ("M-02", "M-12")),
    TechnologyContract("T-02", "RIM", "model forward", "anra_brain:ResidualIdentityModulator.forward", ("M-03",)),
    TechnologyContract("T-03", "GaLore + 8-bit Adam", "optimizer factory", "training.anra_optimizer:build_optimizer", ("M-02",)),
    TechnologyContract("T-04", "SSG", "scaling governor", "training.ssg:SovereignScalingGovernor.check", ("M-09", "M-10")),
    TechnologyContract("T-05", "CDR", "failure curriculum", "training.cdr:CorrectedFailureCurriculum.capture_task_result", ("M-07",)),
    TechnologyContract("T-06", "PCGrad", "optimizer boundary", "training.pcgrad:PCGradAccumulator.materialize", ("M-03",)),
    TechnologyContract("T-07", "SADL", "data mix", "training.sadl:normalized_mix", ("M-03",)),
    TechnologyContract("T-08", "500M frontier model", "model runtime", "training.v2_runtime:build_frontier_model", ("M-01", "M-12")),
    TechnologyContract("T-09", "WSD", "scheduler", "training.wsd_scheduler:get_wsd_schedule", ("M-02", "M-03")),
    TechnologyContract("T-10", "ESV pure commit", "identity state", "identity.esv:ESVModule.commit_state", ("M-03",)),
    TechnologyContract("T-11", "DSTP", "model forward", "anra_brain:CausalTransformerV2._dstp_temperature", ("M-03",)),
    TechnologyContract("T-12", "DEL", "data quality", "training.data_ledger:DataEntropyLedger.evaluate", ("M-01", "M-02")),
    TechnologyContract("T-13", "OGRS", "data mix", "training.v2_data_mix:TrainingDataMixController.update_from_civ", ("M-03",)),
    TechnologyContract("T-14", "CSII", "model growth", "training.csii:CrossScaleIdentityInheritance.grow", ("M-03",)),
    TechnologyContract("T-15", "HGP", "agent facade", "engine.agent_loop:AgentLoop.run", ("M-04",)),
    TechnologyContract("T-16", "World model workflow", "robotics workflow", "robotics.workflow:WorkflowExecutor.execute", ("M-04",)),
    TechnologyContract("T-17", "Four-stage campaign", "campaign orchestrator", "training.stages:StagedTrainingCampaign.run_stage", ("M-02", "M-04", "M-05")),
    TechnologyContract("T-18", "FineWeb-Edu offline shards", "data pipeline", "training.data_pipeline_v3:TokenShardPublisher.publish", ("M-02",)),
    TechnologyContract("T-19", "Promotion gates", "promotion", "evaluation.promotion:CapabilityPromotionGate.compare", ("M-09", "M-12")),
    TechnologyContract("T-20", "IBS", "evaluation", "evaluation.ibs:IBSBenchmark.run_three_seed", ("M-01", "M-10", "M-12")),
    TechnologyContract("T-21", "Verified STaR", "reasoning training", "training.star:STaRLoop.step", ("M-05", "M-06")),
    TechnologyContract("T-22", "RLVR verification routing", "reasoning training", "training.rlvr:RLVRTrainer.train_step", ("M-06",)),
    TechnologyContract("T-23", "Memory benchmark", "memory evaluation", "evaluation.memory_benchmark:run_hybrid_memory_benchmark", ("M-08",)),
    TechnologyContract("T-24", "Trajectory collector", "agent facade", "engine.trajectories:TrajectoryStore.append", ("M-04",)),
    TechnologyContract("T-25", "Continual learning", "continual orchestrator", "training.continual:ContinualLearningOrchestrator.run", ("M-09",)),
    TechnologyContract("T-26", "Persistent service", "service", "app:app", ("M-11",)),
)


def technology_registry() -> dict[str, TechnologyContract]:
    return {technology.technology_id: technology for technology in TECHNOLOGIES}


def validate_technology_reachability() -> dict[str, str]:
    result: dict[str, str] = {}
    for technology in TECHNOLOGIES:
        resolved = technology.resolve()
        if resolved is None:
            raise RuntimeError(f"{technology.technology_id} resolved to None")
        result[technology.technology_id] = technology.entrypoint
    return result
