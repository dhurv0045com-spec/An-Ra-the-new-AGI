"""Prospective V6 scientific registry. Import performs no device or data work."""
from dataclasses import asdict, replace
import json

from v5_contracts.model_spec import V5A_250M
from .codec import canonical, digest, VOCAB_SIZE, CODEC_SHA
from .representation import UTF8


VERSION = "elendril-v6-diagnostic-registry-v1"
PARENT_SOURCE = "0cd200730d43547294cdc9bf8540a478130cce9c"
MODEL_PROFILES = {
    name: replace(V5A_250M, vocabulary_size=UTF8.vocabulary_size, width=width,
        layers=layers, query_heads=heads, kv_heads=kv, head_dimension=64,
        ffn_width=ffn, context_length=context)
    for name, width, layers, heads, kv, ffn, context in (
        ("S3", 256, 4, 4, 1, 768, 512),
        ("S8", 384, 5, 6, 2, 1024, 512),
        ("S20", 512, 7, 8, 2, 1408, 640))
}

# Matched architecture profiles for a separate, equivalent symbolic fixture.
LEGACY_MODEL_PROFILES = {name: replace(spec, vocabulary_size=VOCAB_SIZE) for name, spec in MODEL_PROFILES.items()}

PANELS = (
    ("P01", "Positive-control learnability", "QUERY and complete actions can be acquired before OOD interpretation", "Failed positive overfit indicts instrument; no scale admission", "positive_control"),
    ("P02", "Known-evidence answering", "Sufficient raw evidence causes immediate correct answer", "Correctness below fixed or redundant requests persist", "known"),
    ("P03", "Missing-evidence query", "Missing public facts cause useful QUERY", "No useful-query recall, or query cost gives no terminal benefit", "missing"),
    ("P04", "Conflicting-evidence query", "Unresolved sources cause evidence seeking instead of unsafe answers", "Wrong initial answers persist or new observation is ignored", "conflict"),
    ("P05", "Null and unavailable abstention", "Explicit null differs from missing obtainable evidence", "Unnecessary querying of null or unsafe answers on unavailable facts", "abstention"),
    ("P06", "Post-query revision", "A new public observation changes the complete terminal answer", "Queries occur but final answer ignores independently varied observation", "revision"),
    ("P07", "Cost-aware thinking and queries", "Learned choices improve complete accuracy-cost frontier", "Gain disappears after charging requests, model positions and thinking decisions", "utility"),
    ("P08", "Uncertainty and self-error detection", "Pre-outcome output confidence predicts error on new worlds", "Heldout calibration/ranking no better than constant base rate", "calibration"),
    ("P09", "Legal action and EOS integrity", "Raw generation is valid and terminates exactly", "Malformed outputs, invalid controls or repeated thinking dominate", "integrity"),
    ("P10", "Query swap and query blindness", "Answers causally follow changed query within the same latent world", "Both-correct pairs at blind/constant floor", "query-swap"),
    ("P11", "Temporal/recency shortcuts", "Timestamp/source authority outranks serialized list order", "Order swap changes answer despite unchanged evidence semantics", "time-order"),
    ("P12", "Representation and distractor OOD", "Capability survives independent surfaces and irrelevant evidence", "Alias/template/distractor failure localizes brittle representation", "ood"),
    ("P13", "Compositional reasoning 2→3 hop", "Qualified two-hop skill extends to deeper public graphs", "Three-hop result at shortcut floor despite two-hop acquisition", "three-hop"),
    ("P14", "Frozen-weight memory benefit", "Admitted memory improves new episodes without optimizer changes", "Benefit absent or explained by irrelevant context/fixed retrieval", "memory"),
    ("P15", "Sequential retention and transfer", "Acquired A survives acquired B and accelerates novel C at equal total work", "B not acquired, A lost, or C gain explained by extra cumulative spend", "retention"),
)

ARMS = (
    {"id": "A0", "method": "engineered and privileged controls separately attributed", "scale": "no-train", "conditional": False},
    {"id": "A1", "method": "CE + AdamW; uniform complete epochs", "scale": "S3", "conditional": False},
    {"id": "A2", "method": "first action-token weight3, EOS weight1; matched examples", "scale": "S3", "conditional": False},
    {"id": "A3", "method": "uncertain-world first epoch; identical complete-epoch multiset", "scale": "S3", "conditional": False},
    {"id": "A4", "method": "training-only alias/template augmentation; semantic worlds unchanged", "scale": "S3", "conditional": False},
    {"id": "A5", "method": "supervised model-generated scratch action; extra targets/work charged", "scale": "S3", "conditional": False},
    {"id": "A6", "method": "validation-selected mechanism and unchanged CE scale comparison", "scale": "S8", "conditional": True},
    {"id": "A7", "method": "validation-selected mechanism and unchanged CE scale comparison", "scale": "S20", "conditional": True},
    {"id": "A8", "method": "best two prospective competitors, new seeds and independent development generator", "scale": "S3", "conditional": True},
)

REGISTRY = {
    "schema": VERSION, "parent_source": PARENT_SOURCE,
    "tokenizer": UTF8.manifest.record(),
    "profiles": {name: {"model": spec.canonical(), "parameters": spec.parameter_receipt().total,
                         "model_sha256": spec.sha256()} for name, spec in MODEL_PROFILES.items()},
    "legacy_profiles": {name: {"model": spec.canonical(), "parameters": spec.parameter_receipt().total, "model_sha256": spec.sha256()} for name, spec in LEGACY_MODEL_PROFILES.items()},
    "legacy_codec_sha256": CODEC_SHA,
    "panels": [{"id": p, "name": n, "hypothesis": h, "falsifier": f, "data": d}
               for p, n, h, f, d in PANELS], "arms": list(ARMS),
    "screen_seeds": [7, 19], "confirmation_seeds": [41, 61],
    "costs": {"verified_query": .1, "thinking_decision": 0., "model_input_position": 0.,
              "position_accounting": "reported separately; zero utility coefficient does not mean zero compute"},
    "budget": {"wall_seconds": 21600, "export_reserve_seconds": 2100,
               "devices": 2, "device": "NVIDIA T4", "training_data_target_bytes": 2200000},
    "gates": {"positive_control_exact": 1., "formation_accuracy": .9,
              "formation_each_kind": .8, "invalid_fraction": 0.,
              "excess_requests_per_episode": .05, "alternative_utility_gain": .05,
              "max_kind_regression": .05, "route_formation_accuracy": .9,
              "scale_requires": "both seeds formed; full matched stage forecast fits remaining budget and disk",
              "retention_requires": "independent A and B acquisition; bound before/after/C curves and equal total spend"},
    "amendments": [
        {"reason": "UTF8 raw-evidence prompts exceed symbolic 128-token context", "change": "contexts512/512/640; exact parameter counts unchanged by context"},
        {"reason": "no genuinely sealed external evaluation is attached", "change": "independent development confirmation; no sealed or independent-promoter claim"},
        {"reason": "one efficient campaign, not15 independently retrained models", "change": "P01-P15 share checkpoint diagnostics; unqualified conditional panels retain explicit status"},
        {"reason": "A5 must have an executable, attributable mechanism", "change": "model emits own scratch symbol; controller stores it without oracle completion; extra work reported"},
    ],
    "diagnostic_rules": {"P06": "every complete payload-swap pair makes a verified query in each member and both terminal outcomes are correct and differ", "P07": "mean primary utility strictly exceeds fixed, always-query and no-query on the same cases", "P08": "validation-fitted calibration on separate test: Brier no greater than validation base-rate baseline and ECE <= 0.10; descriptive only", "P14": "matched frozen-checkpoint memory on/off/irrelevant and causal swap; memory on strictly improves off and irrelevant, and every swap pair queries retrieval and both answers follow memory"},
    "promotion": "NONE; owner review and independent evidence required",
}
REGISTRY_SHA = digest(REGISTRY)


def registry():
    """Return a detached JSON manifest and detect accidental global mutation."""
    if digest(REGISTRY) != REGISTRY_SHA:
        raise ValueError("scientific registry mutated after identity binding")
    return json.loads(canonical(REGISTRY))


def formation_gate(report, *, require_route=False):
    """Prospective gate on independently rescored complete episodes."""
    metrics = report["metrics"]
    groups = [value for key, value in metrics["by_kind"].items()
              if not require_route or key.startswith("route")]
    if not groups or any(group["episodes"] == 0 for group in groups):
        return False
    total = sum(group["episodes"] for group in groups)
    correct = sum(group["correct"] for group in groups)
    gate = REGISTRY["gates"]
    return (correct / total >= gate["formation_accuracy"]
            and all(group["correct"] / group["episodes"] >= gate["formation_each_kind"] for group in groups)
            and sum(group["invalid"] for group in groups) == 0)
