"""Naturalistic binding transfer set: DEVELOPMENT ONLY, never training.

Hand-authored mapping scenarios in natural language, deliberately NOT
produced by the v2 renderer (different sentences, different query forms).
Purpose: test whether synthetic binding learning transfers to natural
mapping language. Small by design; a transfer probe, not a training corpus.
"""

from __future__ import annotations


TRANSFER_CASES: tuple[dict[str, object], ...] = (
    {
        "case_id": "natural-transfer-001",
        "facts": (
            "Maya's locker number is 214.",
            "Jon's locker number is 309.",
            "Priya's locker number is 187.",
        ),
        "query": "Which locker belongs to Jon?",
        "candidates": ("187", "214", "309"),
        "gold": "309",
    },
    {
        "case_id": "natural-transfer-002",
        "facts": (
            "The blue crate holds wrenches.",
            "The red crate holds nails.",
            "The green crate holds bolts.",
        ),
        "query": "What does the red crate hold?",
        "candidates": ("bolts", "nails", "wrenches"),
        "gold": "nails",
    },
    {
        "case_id": "natural-transfer-003",
        "facts": (
            "Dr. Alvarez sees patients on Tuesdays.",
            "Dr. Okafor sees patients on Fridays.",
            "Dr. Lindqvist sees patients on Mondays.",
        ),
        "query": "When does Dr. Okafor see patients?",
        "candidates": ("Fridays", "Mondays", "Tuesdays"),
        "gold": "Fridays",
    },
    {
        "case_id": "natural-transfer-004",
        "facts": (
            "The morning bus leaves at 7:40.",
            "The midday bus leaves at 12:15.",
            "The evening bus leaves at 6:05.",
        ),
        "query": "When does the midday bus leave?",
        "candidates": ("12:15", "6:05", "7:40"),
        "gold": "12:15",
    },
    {
        "case_id": "natural-transfer-005",
        "facts": (
            "Room 3B stores microscopes.",
            "Room 5A stores centrifuges.",
            "Room 2C stores spectrometers.",
        ),
        "query": "What does room 5A store?",
        "candidates": ("centrifuges", "microscopes", "spectrometers"),
        "gold": "centrifuges",
    },
    {
        "case_id": "natural-transfer-006",
        "facts": (
            "Aiko's badge code is K7.",
            "Ravi's badge code is Q2.",
            "Lena's badge code is M9.",
        ),
        "query": "What is Ravi's badge code?",
        "candidates": ("K7", "M9", "Q2"),
        "gold": "Q2",
    },
    {
        "case_id": "natural-transfer-007",
        "facts": (
            "The north gate closes at dusk.",
            "The south gate closes at noon.",
            "The east gate closes at midnight.",
        ),
        "query": "When does the south gate close?",
        "candidates": ("dusk", "midnight", "noon"),
        "gold": "noon",
    },
    {
        "case_id": "natural-transfer-008",
        "facts": (
            "Tomas keeps his keys on a brass hook.",
            "Tomas keeps his wallet in a desk drawer.",
            "Alba keeps her keys on a silver hook.",
        ),
        "query": "Where does Tomas keep his wallet?",
        "candidates": ("a brass hook", "a desk drawer", "a silver hook"),
        "gold": "a desk drawer",
    },
    {
        "case_id": "natural-transfer-009",
        "facts": (
            "The library fine is two dollars.",
            "The parking fine is forty dollars.",
            "The speeding fine is ninety dollars.",
        ),
        "query": "How much is the parking fine?",
        "candidates": ("forty dollars", "ninety dollars", "two dollars"),
        "gold": "forty dollars",
    },
    {
        "case_id": "natural-transfer-010",
        "facts": (
            "Hana ordered the mushroom risotto.",
            "Dev ordered the grilled salmon.",
            "Mara ordered the lentil soup.",
        ),
        "query": "What did Dev order?",
        "candidates": ("grilled salmon", "lentil soup", "mushroom risotto"),
        "gold": "grilled salmon",
    },
    {
        "case_id": "natural-transfer-011",
        "facts": (
            "The 2021 census counted 4,112 residents.",
            "The 2022 census counted 4,390 residents.",
            "The 2023 census counted 4,205 residents.",
        ),
        "query": "How many residents did the 2022 census count?",
        "candidates": ("4,112", "4,205", "4,390"),
        "gold": "4,390",
    },
    {
        "case_id": "natural-transfer-012",
        "facts": (
            "Bus 12 stops at the museum.",
            "Bus 27 stops at the harbor.",
            "Bus 33 stops at the stadium.",
        ),
        "query": "Where does bus 27 stop?",
        "candidates": ("the harbor", "the museum", "the stadium"),
        "gold": "the harbor",
    },
)

TRANSFER_SPLIT = "development"


def as_tasks() -> list[dict[str, object]]:
    """Expose transfer cases as firewall-shaped task dicts (dev only)."""

    tasks = []
    for case in TRANSFER_CASES:
        tasks.append({
            "case_id": case["case_id"],
            "facts": list(case["facts"]),
            "facts_text": "\n".join(case["facts"]),  # type: ignore[arg-type]
            "query": case["query"],
            "candidates": list(case["candidates"]),
            "gold": case["gold"],
            "cluster_id": f"transfer-{case['case_id']}",
            "grammar": "natural-hand",
            "split": TRANSFER_SPLIT,
        })
    return tasks


__all__ = ["TRANSFER_CASES", "TRANSFER_SPLIT", "as_tasks"]
