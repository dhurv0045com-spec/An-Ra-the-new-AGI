"""V7.15 frozen diagnostic registration; no import-time execution."""
from anra_v5.elendril.codec import digest
from anra_v5.elendril.v71_protocol import jobs as v71_jobs, PROTOCOL_SHA as V71_SHA

PARENT_SOURCE = "6104ffd7134cb4cf4b3f340bd060d24a1e2dd200"
PARENT_FULL_SHA256 = "1826c85b5a858d2a600661f7e5916f0fa812c55a8d53c37013c695df28a15d19"
SEED = 153071


def registered_jobs():
    return tuple(v71_jobs("core"))


def protocol():
    return {
        "schema": "elendril-v715-frozen-protocol-v1",
        "version": "V7.15",
        "mode": "restored-V7.1-S3-frozen-only",
        "parent_source": PARENT_SOURCE,
        "parent_protocol_sha256": V71_SHA,
        "parent_full_sha256": PARENT_FULL_SHA256,
        "nine_jobs": [j["job_id"] for j in registered_jobs()],
        "seed": SEED,
        "cohorts": {
            "paired_surface": {"worlds": 16, "conditions": 32, "case_records": 512},
            "anti_shortcut": {"worlds": 16, "conditions": 24, "case_records": 384},
        },
        "evaluation_decoders": ["raw-neural", "engineered-legal-grammar"],
        "limits": {
            "maximum_owner_session_minutes": 330,
            "parent_tensor_restore": "required",
            "truth": "evaluator only; never supply gold to actor",
            "new_optimizer_updates": 0,
            "confirmation_status": "development diagnostics, not sealed holdout",
        },
        "completion": "nine frozen-model per-case raw and grammar reports independently replayed, plus verified analysis archive",
        "scientific_conclusion": "CAUSE UNRESOLVED",
    }


PROTOCOL_SHA = digest(protocol())
