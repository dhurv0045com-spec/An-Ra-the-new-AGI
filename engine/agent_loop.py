"""Canonical HGP-to-workflow agent facade."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from anra.anra_paths import FAILURE_REPLAY_DATASET
from intelligence.hgp import HierarchicalGoalPlanner, MissionNode, MissionTree, WorkflowStep
from robotics.contracts import SkillGoal, Workflow, WorkflowState
from robotics.workflow import WorkflowExecutor
from training.cdr import CorrectedFailureCurriculum

from engine.feature_flags import is_enabled
from engine.trajectories import TrajectoryStore


@dataclass(frozen=True)
class AgentResult:
    success: bool
    verified: bool
    mission_tree: dict[str, object]
    artifacts: tuple[str, ...] = ()
    error: str = ""
    trajectory_hash: str = ""


class AgentLoop:
    def __init__(
        self,
        *,
        decomposer: Callable[[str, int], MissionNode],
        skill_mapper: Callable[[MissionNode], WorkflowStep],
        workflow_executor: WorkflowExecutor,
        verifier: Callable[[MissionTree, Workflow], tuple[bool, str, dict[str, object]]],
        recover_node: Callable[[MissionNode], MissionNode] | None = None,
        memory_retrieve: Callable[[str, int], list[dict]] | None = None,
        authorize: Callable[[Workflow], bool] | None = None,
        trajectory_store: TrajectoryStore | None = None,
        checkpoint_id: str = "",
        tokenizer_id: str = "",
        cognition_services: object | None = None,
    ) -> None:
        self.hgp = HierarchicalGoalPlanner(max_depth=5, max_workflow=10)
        self.decomposer = decomposer
        self.skill_mapper = skill_mapper
        self.workflow_executor = workflow_executor
        self.verifier = verifier
        self.recover_node = recover_node
        self.memory_retrieve = memory_retrieve
        self.authorize = authorize or (lambda _workflow: True)
        self.trajectories = trajectory_store or TrajectoryStore()
        self.cdr = CorrectedFailureCurriculum(FAILURE_REPLAY_DATASET)
        self.checkpoint_id = checkpoint_id
        self.tokenizer_id = tokenizer_id
        self.cognition = cognition_services

    def run(
        self,
        goal: str,
        *,
        constraints: tuple[str, ...] = (),
        success_criteria: tuple[str, ...] = (),
    ) -> AgentResult:
        if not is_enabled("agent_loop"):
            return AgentResult(False, False, {}, error="agent_loop feature disabled")
        cognitive_context: dict[str, object] = {}
        if self.cognition is not None and is_enabled("cognition"):
            cognitive_context = self.cognition.classify_goal(goal)
        context = self.memory_retrieve(goal, 8) if self.memory_retrieve else []
        enriched_goal = goal if not context else f"{goal}\nRetrieved context: {context}"
        causal = cognitive_context.get("causal", {})
        if isinstance(causal, dict) and causal.get("requires_experiment"):
            enriched_goal += (
                "\nRequired mission node: design a typed, authorized experiment before "
                "claiming an interventional conclusion."
            )
        try:
            tree = self.hgp.decompose(
                enriched_goal,
                self.decomposer,
                success_criteria=success_criteria,
            )
            tree.root.constraints = tuple({*tree.root.constraints, *constraints})
            tree.validate(5, 10)
            steps = self.hgp.compile_workflow(tree, self.skill_mapper)
            workflow = Workflow(
                workflow_id=tree.root.node_id,
                goal=goal,
                skills=[
                    SkillGoal(
                        skill_name=step.skill,
                        parameters=step.parameters,
                        preconditions=step.preconditions,
                        postconditions=step.postconditions,
                        timeout_seconds=step.timeout_seconds,
                        source_mission=step.source_node,
                    )
                    for step in steps
                ],
            )
            if not self.authorize(workflow):
                raise PermissionError("Workflow authorization denied.")
            executed = self.workflow_executor.execute(workflow)
            while (
                executed.state == WorkflowState.FAILED
                and self.recover_node is not None
                and executed.trace
            ):
                failed_node = str(executed.trace[-1].get("source_mission", ""))
                if not failed_node:
                    break
                tree = self.hgp.backtrack(
                    tree,
                    failed_node,
                    self.recover_node,
                )
                steps = self.hgp.compile_workflow(tree, self.skill_mapper)
                workflow = Workflow(
                    workflow_id=tree.root.node_id,
                    goal=goal,
                    skills=[
                        SkillGoal(
                            skill_name=step.skill,
                            parameters=step.parameters,
                            preconditions=step.preconditions,
                            postconditions=step.postconditions,
                            timeout_seconds=step.timeout_seconds,
                            source_mission=step.source_node,
                        )
                        for step in steps
                    ],
                )
                if not self.authorize(workflow):
                    raise PermissionError("Recovered workflow authorization denied.")
                executed = self.workflow_executor.execute(workflow)
            verified, method, evidence = self.verifier(tree, executed)
            evidence = {**evidence, "cognition": cognitive_context}
            success = executed.state == WorkflowState.COMPLETED and verified
            artifacts = tuple(
                artifact for trace in executed.trace for artifact in trace.get("artifacts", ())
            )
            record = self.trajectories.append(
                goal=goal,
                mission_tree=tree.to_dict(),
                skill_sequence=[asdict(skill) for skill in executed.skills],
                artifacts=list(artifacts),
                success=success,
                verified=success,
                verification_method=method,
                verification_evidence=evidence,
                checkpoint_id=self.checkpoint_id,
                tokenizer_id=self.tokenizer_id,
                approved_constraints=constraints,
                tool_results=list(executed.trace),
            )
            if not success:
                category = (
                    "reasoning"
                    if isinstance(causal, dict) and causal.get("causal_type") != "unknown"
                    else "execution"
                )
                self.cdr.capture_task_result(
                    prompt=goal,
                    output=str(executed.trace),
                    category=category,
                    success=False,
                    diagnosis=str(evidence),
                    verifier=method,
                    verified=False,
                )
            return AgentResult(
                success=success,
                verified=verified,
                mission_tree=tree.to_dict(),
                artifacts=artifacts,
                error="" if success else str(evidence),
                trajectory_hash=record.content_hash,
            )
        except Exception as exc:
            self.cdr.capture_task_result(
                prompt=goal,
                output="",
                category="planning",
                success=False,
                diagnosis=str(exc),
            )
            return AgentResult(False, False, {}, error=str(exc))
