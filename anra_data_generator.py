from __future__ import annotations

from pathlib import Path
from typing import List

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

GENERATION_TOPICS = [
    "philosophy and identity",
    "mathematics and logic",
    "science and reasoning",
    "code and computation",
    "An-Ra's purpose and nature",
    "autonomous thinking",
    "memory and learning",
    "self-improvement",
    "creativity",
]


def generate_training_pairs(gemini, topic: str, n_pairs: int = 10) -> str:
    prompt = (
        f"Generate {n_pairs} dialogue pairs for An-Ra.\\n"
        f"Topic: {topic}\\n"
        "Format exactly:\\nH: [question]\\nANRA: [answer]"
    )
    return gemini.generate_content(prompt).text


def append_generated_pairs(dataset_path: str, api_key: str, n_pairs: int = 20) -> int:
    if genai is None:
        raise RuntimeError("google-generativeai is not installed")
    genai.configure(api_key=api_key)
    gemini = genai.GenerativeModel("gemini-2.0-flash")

    chunks: List[str] = []
    for topic in GENERATION_TOPICS:
        chunks.append(generate_training_pairs(gemini, topic=topic, n_pairs=n_pairs))

    out = Path(dataset_path)
    with out.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(chunks) + "\n")
    return len(GENERATION_TOPICS) * n_pairs


if __name__ == "__main__":
    import os
    count = append_generated_pairs("anra_dataset_v6_1.txt", os.environ.get("GEMINI_API_KEY", ""))
    print(f"Added ~{count} training pairs")
