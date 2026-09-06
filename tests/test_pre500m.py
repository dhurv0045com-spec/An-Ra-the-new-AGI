"""PRE500M + 500M campaign tests: milestone logic, fail-closed decision,
storage, schedule, data gate."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CITADEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(CITADEL_ROOT))

from citadel_tpu import milestones as ms  # noqa: E402
from citadel_tpu import pre500m  # noqa: E402


def test_crossed_milestones_boundaries() -> None:
    """§2: token-based crossing at the exact ladder boundaries; a single
    transaction may cross several; no duplicate on equal ledgers."""
    crossed = ms.crossed_milestones(49_999_999, 50_000_000)
    assert crossed == [50_000_000]
    crossed = ms.crossed_milestones(99_999_999, 100_000_001)
    assert crossed == [100_000_000]
    assert ms.crossed_milestones(50_000_000, 50_000_000) == []
    assert ms.crossed_milestones(0, 500_000_000) == \
        [50_000_000, 100_000_000, 200_000_000, 350_000_000, 500_000_000]
    # resume replaying the same ledger transition derives the same crossing
    # exactly once (deterministic pure function)
    assert ms.crossed_milestones(49_999_999, 50_000_000) == \
        ms.crossed_milestones(49_999_999, 50_000_000)
    # errors
    for bad in ((50_000_000, 49_999_999), (-1, 0)):
        try:
            ms.crossed_milestones(*bad)
            raise SystemExit(f"accepted {bad}")
        except ValueError:
            pass
    assert ms.next_milestone(0) == 50_000_000
    assert ms.next_milestone(50_000_000) == 100_000_000
    assert ms.next_milestone(500_000_000) is None


def test_lr_schedule_table() -> None:
    table = pre500m.lr_schedule_table()
    points = [row["tokens"] for row in table]
    assert points == [0, 1_000_000, 5_000_000, 10_000_000, 25_000_000,
                      50_000_000, 100_000_000, 200_000_000, 350_000_000,
                      500_000_000]
    by_tokens = {row["tokens"]: row["learning_rate"] for row in table}
    assert by_tokens[0] == 0.0
    assert abs(by_tokens[25_000_000] - 1.5e-4) < 1e-9  # mid-warmup
    assert by_tokens[50_000_000] == 3e-4  # warmup complete
    assert by_tokens[500_000_000] == 3e-4  # stable through 500M


def test_storage_estimate() -> None:
    st = pre500m.storage_estimate_gb()
    assert st["per_checkpoint_gb"] > 0
    # 250,216,960 params: model 1 GB class, AdamW moments ~1.9 GB
    assert 2.0 < st["per_checkpoint_gb"] < 4.0
    assert st["PEAK_LOCAL_STORAGE_GB"] >= st["recovery_rotation_gb"]
    assert st["PERSISTENT_STORAGE_GB"] >= st["recovery_rotation_gb"]
    json.dumps(st)


def test_data_gate_not_ready() -> None:
    """§5/§6: with no materialized production corpus the gate returns
    DATA_NOT_READY with precise blockers - never fake readiness."""
    got = pre500m.data_readiness()
    assert got["state"] == "DATA_NOT_READY"
    assert got["blockers"] and "not MATERIALIZED" in got["blockers"][0]
    # even materialized, under-supplied sources report replay explicitly
    got = pre500m.data_readiness(
        corpus_materialized=True, tokenizer_artifact_sha="a" * 64,
        mixture_scheduled_tokens={"natural": 325_000_000,
                                  "verified_cognition": 175_000_000},
        mixture_available_unique_tokens={
            "natural": 300_000_000, "verified_cognition": 400_000_000})
    assert any("replay" in b for b in got["blockers"]), got
    assert got["state"] == "DATA_NOT_READY"  # under-supply still blocks
    # fully supplied AND summing to the 500M target: gate clears to RUNNABLE
    got = pre500m.data_readiness(
        corpus_materialized=True, tokenizer_artifact_sha="a" * 64,
        mixture_scheduled_tokens={"natural": 325_000_000,
                                  "code_math": 100_000_000,
                                  "cognition": 75_000_000},
        mixture_available_unique_tokens={"natural": 400_000_000,
                                         "code_math": 150_000_000,
                                         "cognition": 100_000_000})
    assert got["state"] == "RUNNABLE", got
    assert got["blockers"] == []


def test_next_500m_decision_fail_closed() -> None:
    """§26: empty/garbage inputs -> ready=false with precise blockers; the
    milestone ladder and go/no-go gates are enforced."""
    empty = pre500m.build_next_500m_decision()
    assert empty["ready_for_500m_training"] is False
    assert empty["blocking_reasons"], "empty inputs must produce blockers"
    assert 50_000_000 in empty["milestones"]
    assert 500_000_000 in empty["milestones"]
    green_parts = {
        "target_tokens": 500_000_000,
        "canonical_cymek_sha": "1" * 40, "runtime_pin_sha": "1" * 40,
        "model": {"parameter_count": 250_216_960},
        "campaign_spec_sha256": "2" * 64,
        "data": {"state": "RUNNABLE", "blockers": []},
        "milestone_logic_verified": True, "lr_schedule_token_based": True,
        "static_shape_fit": True, "checkpoint_transaction_certified": True,
        "exact_resume_certified": True, "fresh_runtime_resume_certified": True,
        "evaluation_hooks_wired": True, "storage_feasible": True,
        "estimated_tokens_per_second": 7000.0,
        "stop_gates_at": [50_000_000, 100_000_000, 200_000_000],
        "storage": {"PEAK_LOCAL_STORAGE_GB": 12.0},
    }
    import copy
    green = pre500m.build_next_500m_decision(**copy.deepcopy(green_parts))
    assert green["ready_for_50m_training" if False else
                 "ready_for_500m_training"] is True, green["blocking_reasons"]
    assert green["estimated_hours_500m"] == round(500_000_000 / 7000 / 3600, 2)
    # each individual requirement removal must block
    for key in ("campaign_spec_sha256", "milestone_logic_verified",
                "exact_resume_certified", "evaluation_hooks_wired"):
        bad = copy.deepcopy(green_parts)
        bad[key] = False if isinstance(green_parts[key], bool) else None
        d = pre500m.build_next_500m_decision(**bad)
        assert d["ready_for_500m_training"] is False, key
    bad = copy.deepcopy(green_parts)
    bad["stop_gates_at"] = [50_000_000]  # missing 100M/200M gates
    d = pre500m.build_next_500m_decision(**bad)
    assert d["ready_for_500m_training"] is False
    json.dumps(green)


def main() -> int:
    tests = [test_crossed_milestones_boundaries, test_lr_schedule_table,
             test_storage_estimate, test_data_gate_not_ready,
             test_next_500m_decision_fail_closed]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}",
                  flush=True)
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
