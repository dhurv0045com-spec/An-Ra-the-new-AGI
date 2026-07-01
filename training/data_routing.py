from __future__ import annotations

from dataclasses import asdict, dataclass

RAW_SOURCE_CLASSES = {
    "fineweb_edu",
    "educational_foundation",
    "permissive_code",
    "finemath",
    "mathematics",
    "science_technical",
}
CONVERSATION_SOURCE_CLASSES = {
    "verified_instruction",
    "verified_dfc",
    "identity_replay",
    "corrected_replay",
}


@dataclass(frozen=True)
class DataRouteReport:
    source_class: str
    objective: str
    packing_layout: str
    prompt_masked: bool
    answer_weighted: bool


def route_source_class(source_class: str) -> DataRouteReport:
    normalized = source_class.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fineweb_edu": "fineweb_edu",
        "fineweb_edu_foundation": "fineweb_edu",
        "the_stack_v2_dedup": "permissive_code",
        "finemath_4+": "finemath",
        "science": "science_technical",
        "teacher": "verified_instruction",
        "symbolic": "verified_dfc",
        "owner": "identity_replay",
        "identity": "identity_replay",
        "replay": "corrected_replay",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in RAW_SOURCE_CLASSES:
        return DataRouteReport(
            source_class=normalized,
            objective="raw_next_token",
            packing_layout="raw_causal_shards_v1",
            prompt_masked=False,
            answer_weighted=False,
        )
    if normalized in CONVERSATION_SOURCE_CLASSES:
        return DataRouteReport(
            source_class=normalized,
            objective="answer_supervised_conversation",
            packing_layout="bucket_packed_v1",
            prompt_masked=False,
            answer_weighted=True,
        )
    raise ValueError(f"Unknown An-Ra source class: {source_class!r}")


def build_data_route_report(source_classes: list[str]) -> dict[str, object]:
    routes = [route_source_class(source_class) for source_class in source_classes]
    return {
        "schema_version": 1,
        "routes": [asdict(route) for route in routes],
        "source_count": len(routes),
        "raw_sources": sum(route.objective == "raw_next_token" for route in routes),
        "conversation_sources": sum(
            route.objective == "answer_supervised_conversation" for route in routes
        ),
    }
