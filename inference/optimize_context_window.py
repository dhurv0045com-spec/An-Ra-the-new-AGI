from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


class ContextWindowOptimizer:
    MAX_CONTEXT = 1024
    SESSION_BUDGET = 0.70
    MEMORY_BUDGET = 0.25
    MESSAGE_BUDGET = 0.05

    def build_optimized_context(
        self,
        session_history: list,
        memory_results: list,
        current_message: str
    ) -> dict:
        session_budget = int(self.MAX_CONTEXT * self.SESSION_BUDGET)
        memory_budget = int(self.MAX_CONTEXT * self.MEMORY_BUDGET)
        msg_budget = int(self.MAX_CONTEXT * self.MESSAGE_BUDGET)

        msg_str = current_message[:msg_budget]

        normalized_turns: List[Tuple[str, str]] = []
        for item in session_history:
            if isinstance(item, tuple) and len(item) == 2:
                normalized_turns.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                if item.get("role") == "user":
                    normalized_turns.append((str(item.get("content", "")), ""))
                elif item.get("role") == "assistant" and normalized_turns:
                    last_user, _ = normalized_turns[-1]
                    normalized_turns[-1] = (last_user, str(item.get("content", "")))

        session_str = ""
        turns_included = 0
        for user_msg, anra_msg in reversed(normalized_turns):
            turn = f"H: {user_msg}\nANRA: {anra_msg}\n"
            if len(session_str) + len(turn) <= session_budget:
                session_str = turn + session_str
                turns_included += 1
            else:
                break

        normalized_memory: List[str] = []
        for result in memory_results:
            if isinstance(result, dict):
                merged = f"{result.get('summary', '')} {result.get('content', '')}".strip()
                normalized_memory.append(merged)
            else:
                normalized_memory.append(str(result))

        memory_str = ""
        for result in normalized_memory:
            summary = result[:100].strip()
            if len(memory_str) + len(summary) + 2 <= memory_budget:
                memory_str += summary + "\n"

        full_context = ""
        if memory_str:
            full_context += f"[MEMORY CONTEXT]\n{memory_str}\n"
        full_context += session_str
        full_context += f"H: {msg_str}\nANRA:"

        if len(full_context) > self.MAX_CONTEXT:
            full_context = session_str + f"H: {msg_str}\nANRA:"
            if len(full_context) > self.MAX_CONTEXT:
                full_context = full_context[-self.MAX_CONTEXT:]

        return {
            "context": full_context,
            "context_length": len(full_context),
            "turns_included": turns_included,
            "memory_results_used": len([r for r in normalized_memory if r[:100].strip() and r[:100].strip() in full_context]),
            "context_truncated": turns_included < len(normalized_turns),
            "memory_truncated": len(memory_str) < sum(len(r) for r in normalized_memory)
        }


if __name__ == "__main__":
    opt = ContextWindowOptimizer()
    sample_history = [(f"user-{i}", f"assistant-response-{i}" * 4) for i in range(1, 8)]
    sample_memory = ["Fact about An-Ra memory layer " * 6, "Identity grounding vector alignment " * 5]
    result = opt.build_optimized_context(sample_history, sample_memory, "How do you reason about continuity?")
    print("Context budget self-test")
    print(f"  max={opt.MAX_CONTEXT}")
    print(f"  len={result['context_length']}")
    print(f"  turns_included={result['turns_included']}")
    print(f"  memory_results_used={result['memory_results_used']}")
    print(f"  context_truncated={result['context_truncated']}")
    print(f"  memory_truncated={result['memory_truncated']}")
