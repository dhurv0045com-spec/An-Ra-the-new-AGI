"""
improve.py — Master Entry Point for 45L Self-Improvement System

All subsystems wired together. Every entry point works.

Usage:
    python improve.py --dashboard
    python improve.py --evaluate
    python improve.py --train [--min-examples 100] [--deploy-if-better]
    python improve.py --skills
    python improve.py --analyze-failures [--last 7d]
    python improve.py --optimize-prompt --target planner [--iterations 10]
    python improve.py --create-tool --name NAME --desc DESC --reqs REQS
    python improve.py --connectors
    python improve.py --weekly-report
    python improve.py --status
"""

import sys, os, argparse, json, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PHASE2 = _HERE.parent
_ROOT = _PHASE2.parent

for _p in [
    str(_HERE),
    str(_PHASE2 / "master_system (45M)"),
    str(_ROOT),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.dynamic.creator       import DynamicToolCreator, ToolOptimizer, ToolSandbox
from tools.connectors.connectors  import ConnectorRegistry
from self_improvement.engine      import (
    OutputEvaluator, PromptOptimizer, FailureAnalyzer,
    SkillLibrary, SelfTrainer
)
from dashboard.dashboard import (
    MetricsCollector, AlertSystem, WeeklyReporter, TerminalDashboard
)


class ImprovementSystem:
    """Master system — owns all subsystems, provides unified API."""

    VERSION = "45L-v1.0"

    def __init__(self):
        # Subsystems
        self.tool_creator    = DynamicToolCreator()
        self.connectors      = ConnectorRegistry()
        self.evaluator       = OutputEvaluator()
        self.prompt_opt      = PromptOptimizer(self.evaluator)
        self.failure_analyzer = FailureAnalyzer()
        self.skill_library   = SkillLibrary()
        self.self_trainer    = SelfTrainer(self.evaluator)

        # Dashboard
        self.alerts          = AlertSystem()
        self.metrics         = MetricsCollector(
            evaluator        = self.evaluator,
            prompt_opt       = self.prompt_opt,
            failure_analyzer = self.failure_analyzer,
            skill_library    = self.skill_library,
            self_trainer     = self.self_trainer,
            tool_creator     = self.tool_creator,
            connector_registry = self.connectors,
        )
        self.reporter        = WeeklyReporter(self.metrics, self.alerts)
        self.dashboard       = TerminalDashboard(self)

        # Wire alerts
        self.alerts.on_alert(lambda a: print(f"\n⚡ ALERT: [{a['severity'].upper()}] {a['message']}"))

        # Register default prompts
        self._register_default_prompts()

    def _register_default_prompts(self):
        defaults = {
            "planner":  "You are a precise goal planner. Break the goal into clear, executable steps. Be specific about what tools are needed.",
            "executor": "You are a reliable task executor. Follow the plan step by step. Report results clearly after each step.",
            "critic":   "You are a quality critic. Evaluate the output honestly against the goal. Score each dimension 0-1.",
            "tool_creator": "You are an expert Python programmer. Write clean, tested, well-documented tool code.",
        }
        for name, text in defaults.items():
            existing = self.prompt_opt.get_prompt(name)
            if not existing:
                self.prompt_opt.register_prompt(name, text)

    def evaluate(self, goal: str, output: str, context: str = ""):
        """Evaluate an output and update all relevant systems."""
        score = self.evaluator.evaluate(goal, output, context)

        # Update prompt performance
        self.prompt_opt.record_performance("executor", score.overall)

        # Collect for training if high quality
        if score.overall >= self.self_trainer.DEPLOYMENT_MIN_IMPROVEMENT + 0.5:
            self.self_trainer.collect(goal, output, score.overall, context)

        # Check for alerts
        snap = {"components": {"output_quality": self.evaluator.recent_stats(n=10)}}
        self.alerts.check(snap)

        return score

    def process_failure(self, goal: str, step: str, error_type: str,
                        error_msg: str, **kwargs):
        """Log a failure and check for patterns."""
        record = self.failure_analyzer.log(goal, step, error_type, error_msg, **kwargs)
        patterns = self.failure_analyzer.detect_patterns(window_days=1)
        if patterns:
            self.alerts.check({
                "components": {
                    "failures": {"patterns": len(patterns), "pattern_details":
                                 [{"type": p.error_type, "count": p.occurrences}
                                  for p in patterns]}
                }
            })
        return record

    def learn_skill(self, goal: str, steps: list, tools: list,
                    outcome_score: float):
        """Extract and store a skill from a successful task."""
        return self.skill_library.extract_and_store(goal, steps, tools, outcome_score)

    def get_skill(self, goal: str):
        """Retrieve relevant skills for a new goal."""
        return self.skill_library.retrieve(goal, top_k=3)

    def full_status(self) -> dict:
        snap = self.metrics.collect()
        return {
            "version":   self.VERSION,
            "timestamp": snap["timestamp"],
            "components": snap["components"],
            "alerts":    self.alerts.get_active(),
        }


def health_check() -> dict:
    try:
        system = ImprovementSystem()
        return {
            "status": "ok",
            "module": "self_improvement_45L",
            "version": ImprovementSystem.VERSION,
        }
    except Exception as exc:
        return {"status": "degraded", "module": "self_improvement_45L", "reason": str(exc)}


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="45L — Self-Improvement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dashboard",        action="store_true", help="Show live dashboard")
    p.add_argument("--evaluate",         action="store_true", help="Run self-evaluation")
    p.add_argument("--train",            action="store_true", help="Trigger training run")
    p.add_argument("--min-examples",     type=int, default=None)
    p.add_argument("--deploy-if-better", action="store_true")
    p.add_argument("--skills",           action="store_true", help="Show skill library")
    p.add_argument("--analyze-failures", action="store_true")
    p.add_argument("--last",             type=str, default="7d")
    p.add_argument("--optimize-prompt",  action="store_true")
    p.add_argument("--target",           type=str, default="planner")
    p.add_argument("--iterations",       type=int, default=5)
    p.add_argument("--create-tool",      action="store_true")
    p.add_argument("--name",             type=str)
    p.add_argument("--desc",             type=str)
    p.add_argument("--reqs",             type=str)
    p.add_argument("--connectors",       action="store_true")
    p.add_argument("--weekly-report",    action="store_true")
    p.add_argument("--status",           action="store_true")
    p.add_argument("--watch",            action="store_true", help="Live dashboard")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    system = ImprovementSystem()

    if args.status or len(sys.argv) == 1:
        print(system.dashboard.render())
        return

    if args.dashboard or args.watch:
        system.dashboard.watch(interval=10)
        return

    if args.evaluate:
        print("\n  Running self-evaluation...\n")
        # Run evaluation on sample interactions
        test_cases = [
            ("Explain gradient descent", "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively updating parameters in the direction of the negative gradient. The learning rate controls step size. It is fundamental to training neural networks."),
            ("What is LoRA?", "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that injects trainable rank-decomposition matrices into frozen model weights, reducing trainable parameters by ~99% while maintaining performance."),
            ("List 3 transformer components", "1. Self-attention: allows each token to attend to all others. 2. Feed-forward network: processes each position independently. 3. Layer normalization: stabilizes training of deep networks."),
        ]
        total = 0
        for goal, output in test_cases:
            score = system.evaluate(goal, output)
            total += score.overall
            print(f"  Goal:    {goal[:60]}")
            print(f"  Overall: {score.overall:.3f}  ({'✓' if not score.flagged else '✗'})")
            print(f"  {score.feedback}")
            print()
        print(f"  Average: {total/len(test_cases):.3f}")
        stats = system.evaluator.recent_stats()
        print(f"  Stored:  {stats.get('count', 0)} evaluations total")
        return

    if args.train:
        print("\n  Checking training pipeline...\n")
        trigger, reason = system.self_trainer.should_trigger()
        print(f"  Should trigger: {trigger}  ({reason})")
        print(f"  Running training run...\n")
        run = system.self_trainer.run(
            min_examples=args.min_examples,
            deploy_if_better=args.deploy_if_better,
        )
        print(f"  Run ID:      {run.run_id}")
        print(f"  Status:      {run.status}")
        print(f"  Examples:    {run.examples_used}")
        print(f"  Pre-score:   {run.pre_score}")
        print(f"  Post-score:  {run.post_score}")
        print(f"  Deployed:    {run.deployed}")
        print(f"  Notes:       {run.notes}")
        return

    if args.skills:
        print("\n  SKILL LIBRARY\n")
        stats = system.skill_library.stats()
        print(f"  Total skills: {stats['total_skills']}")
        print(f"  Avg quality:  {stats['avg_quality']:.3f}")
        print(f"  By type:")
        for t, n in stats.get("by_type", {}).items():
            print(f"    {t:<20} {n}")
        print("\n  Most used:")
        for s in stats.get("most_used", []):
            print(f"    [{s.get('goal_type','?')}] {s.get('name','?')}  (used {s.get('use_count',0)}x, score {s.get('avg_score',0):.2f})")
        return

    if args.analyze_failures:
        days = int(args.last.replace("d", "").replace("h", "")) if args.last else 7
        print(f"\n  FAILURE ANALYSIS (last {days} days)\n")
        report = system.failure_analyzer.summary_report(days=days)
        print(f"  Total failures:    {report['total_failures']}")
        print(f"  Resolution rate:   {report['resolution_rate']:.0%}")
        print(f"  Recurring patterns: {report['patterns']}")
        print(f"  Fixable patterns:  {report['fixable']}")
        if report.get("top_errors"):
            print("\n  Top error types:")
            for etype, count in report["top_errors"]:
                print(f"    {count}× {etype}")
        if report.get("pattern_details"):
            print("\n  Pattern details:")
            for p in report["pattern_details"][:5]:
                print(f"    [{p['type']}] {p['count']}× — Fix: {str(p.get('fix',''))[:80]}")
        return

    if args.optimize_prompt:
        print(f"\n  OPTIMIZING PROMPT: {args.target}\n")
        current = system.prompt_opt.get_prompt(args.target)
        if not current:
            print(f"  Prompt '{args.target}' not found. Registering default...")
            system.prompt_opt.register_prompt(args.target, f"You are a helpful assistant for {args.target} tasks.")
        result = system.prompt_opt.optimize(args.target, iterations=args.iterations)
        print(f"  Current score:  {result['current_score']:.3f}")
        print(f"  Best strategy:  {result['best_strategy']}")
        print(f"  Best score:     {result['best_score']:.3f}")
        print(f"  Improvement:    {result['improvement']:+.3f}")
        print(f"  Promoted:       {result['promoted']}")
        if result.get("error"):
            print(f"  Error:          {result['error']}")
        return

    if args.create_tool:
        if not args.name:
            print("  --name required for --create-tool")
            return
        name = args.name
        desc = args.desc or f"A tool that {name.replace('_',' ')}"
        reqs = args.reqs or desc
        print(f"\n  CREATING TOOL: {name}\n")
        test_cases = [
            {"function": name, "args": ["test"], "expect_type": "str"},
        ]
        result = system.tool_creator.create(name, desc, reqs, test_cases,
                                             auto_approve_if_safe=True)
        print(f"  Status:   {result['status']}")
        print(f"  Tool ID:  {result['tool_id']}")
        print(f"  Approved: {result.get('approved', False)}")
        print(f"  Summary:  {result.get('summary', result.get('message',''))}")
        for attempt in result.get("attempts", [])[:3]:
            icon = "✓" if "passed" in attempt.get("summary","") else "✗"
            print(f"  {icon} Attempt {attempt['attempt']}: {attempt['summary']}")
        return

    if args.connectors:
        print("\n  CONNECTOR STATUS\n")
        statuses = system.connectors.status()
        for s in statuses:
            enabled = "✓ enabled" if s["enabled"] else "○ disabled"
            print(f"  {s['name']:<15} {enabled}  rate_limit={s['rate_limit']}/min")
        return

    if args.weekly_report:
        print("\n  Generating weekly report...\n")
        report = system.reporter.generate()
        print(report)
        return


if __name__ == "__main__":
    main()
