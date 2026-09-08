"""Mixture + lifecycle tests (no torch): exact allocation, deficit scheduling,
resume counters, shortfall planning, dataset lifecycle states.
Run: python tests/test_v5_mixture.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v5_data.lifecycle import (  # noqa: E402
    STATES,
    DatasetLifecycle,
    require_runnable,
)
from v5_data.mixture import DeficitScheduler, allocate  # noqa: E402
from v5_training.production_entry import (  # noqa: E402
    assign_mixture_cell,
    campaign_microstep_plan,
    frozen_topology,
    plan_campaign_demand,
)


def test_500m_allocation_exact():
    allocation = allocate(500_000_000, {"natural": 0.65, "code_math_formal": 0.20,
                                        "verified_cognition": 0.15})
    assert allocation == {"natural": 325_000_000, "code_math_formal": 100_000_000,
                          "verified_cognition": 75_000_000}
    assert sum(allocation.values()) == 500_000_000


def test_deficit_scheduler_exact_on_unit_steps():
    scheduler = DeficitScheduler(
        fractions={"a": 0.6, "b": 0.3, "c": 0.1}, order=("a", "b", "c"))
    consumed: dict[str, int] = {}
    picks = []
    for total in range(10):
        choice = scheduler.next(consumed_total=total, consumed=consumed)
        picks.append(choice)
        consumed[choice] = consumed.get(choice, 0) + 1
    assert sorted(picks).count("a") == 6
    assert sorted(picks).count("b") == 3
    assert sorted(picks).count("c") == 1


def test_deficit_scheduler_converges():
    scheduler = DeficitScheduler(
        fractions={"natural": 0.65, "code_math_formal": 0.20,
                   "verified_cognition": 0.15})
    consumed: dict[str, int] = {}
    total = 0
    for _ in range(200):
        choice = scheduler.next(consumed_total=total, consumed=consumed)
        consumed[choice] = consumed.get(choice, 0) + 32768
        total += 32768
    for name, share in (("natural", 0.65), ("code_math_formal", 0.20),
                        ("verified_cognition", 0.15)):
        assert abs(consumed[name] / total - share) < 32768 / total + 1e-9


def test_scheduler_pure_and_resumable():
    scheduler = DeficitScheduler(fractions={"x": 0.5, "y": 0.5})
    first = scheduler.next(consumed_total=4, consumed={"x": 3, "y": 1})
    second = scheduler.next(consumed_total=4, consumed={"x": 3, "y": 1})
    assert first == second == "y"


def test_demand_planner_matches_live_assignment():
    topo = frozen_topology()
    plan = campaign_microstep_plan(start_tokens=0, campaign_tokens=100_000,
                                   topo=topo)
    assert sum(count for _, count in plan) == 100_000
    assert [bucket for bucket, _ in plan] == [512, 1024, 2048, 4096]
    assert [count for _, count in plan] == [32768, 32768, 32768, 1696]
    fam = DeficitScheduler(fractions={"natural": 0.65, "code_math_formal": 0.20,
                                      "verified_cognition": 0.15})
    demand = plan_campaign_demand(microstep_plan=plan, fam_scheduler=fam,
                                  sub_scheduler=None, cognition_mapped=False)
    assert sum(demand.values()) == 100_000
    live_fam: dict[str, int] = {}
    live_total = 0
    replay = {}
    for bucket, count in plan:
        family, sub = assign_mixture_cell(
            fam_consumed=live_fam, sub_consumed={}, total_consumed=live_total,
            fam_scheduler=fam, sub_scheduler=None, cognition_mapped=False)
        assert sub == ""
        key = f"{bucket}|{family}|"
        replay[key] = replay.get(key, 0) + count
        live_fam[family] = live_fam.get(family, 0) + count
        live_total += count
    assert replay == demand


def test_demand_planner_resume_matches_uninterrupted():
    topo = frozen_topology()
    fam = DeficitScheduler(fractions={"natural": 0.65, "code_math_formal": 0.20,
                                      "verified_cognition": 0.15})
    whole = campaign_microstep_plan(start_tokens=0, campaign_tokens=200_000,
                                    topo=topo)
    whole_demand = plan_campaign_demand(
        microstep_plan=whole, fam_scheduler=fam, sub_scheduler=None,
        cognition_mapped=False)
    first = whole[:4]
    first_demand = plan_campaign_demand(
        microstep_plan=first, fam_scheduler=fam, sub_scheduler=None,
        cognition_mapped=False)
    consumed: dict[str, int] = {}
    total = 0
    for _, count in first:
        choice = fam.next(consumed_total=total, consumed=consumed)
        consumed[choice] = consumed.get(choice, 0) + count
        total += count
    rest = whole[4:]
    rest_demand = plan_campaign_demand(
        microstep_plan=rest, fam_scheduler=fam, sub_scheduler=None,
        cognition_mapped=False, initial_fam_consumed=consumed,
        initial_total=total)
    merged: dict[str, int] = {}
    for part in (first_demand, rest_demand):
        for key, count in part.items():
            merged[key] = merged.get(key, 0) + count
    assert merged == whole_demand


def test_lifecycle_order_enforced():
    life = DatasetLifecycle(lineage_id="ds1")
    assert life.state == "DECLARED"
    try:
        require_runnable(life)
    except ValueError as exc:
        assert "not RUNNABLE" in str(exc)
    else:
        raise AssertionError("non-runnable dataset was accepted")
    for _ in STATES[1:]:
        life.advance(evidence_sha256="ab" * 32)
    assert life.state == "RUNNABLE"
    require_runnable(life)
    receipt = life.receipt()
    assert receipt["state"] == "RUNNABLE" and len(receipt["sha256"]) == 64
    assert len(receipt["history"]) == len(STATES) - 1
    try:
        life.advance(evidence_sha256="ab" * 32)
    except ValueError:
        pass
    else:
        raise AssertionError("advance past RUNNABLE was allowed")


def test_lifecycle_evidence_required():
    life = DatasetLifecycle(lineage_id="ds2")
    try:
        life.advance(evidence_sha256="not-a-sha")
    except ValueError:
        pass
    else:
        raise AssertionError("weak evidence was accepted")


_TESTS = [test_500m_allocation_exact,
          test_deficit_scheduler_exact_on_unit_steps,
          test_deficit_scheduler_converges,
          test_scheduler_pure_and_resumable,
          test_demand_planner_matches_live_assignment,
          test_demand_planner_resume_matches_uninterrupted,
          test_lifecycle_order_enforced,
          test_lifecycle_evidence_required]


def main() -> int:
    failed = 0
    for fn in _TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}", flush=True)
    print(f"{len(_TESTS) - failed}/{len(_TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
