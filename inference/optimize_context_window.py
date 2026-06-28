from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from training.v2_config import V2_FRONTIER


@dataclass(frozen=True)
class ContextBudgets:
    identity: int = 64
    message: int = 384
    history: int = 224
    memory: int = 223


@dataclass(frozen=True)
class PromptAssemblyTrace:
    formatted_prompt: str
    prompt_tokens: int
    max_context_tokens: int
    reserved_output_tokens: int
    turns_included: int
    memory_results_used: int
    context_truncated: bool
    memory_truncated: bool
    token_allocation: dict[str, int]
    mode: str


class ContextWindowOptimizer:
    """Build the exact frontier prompt using tokenizer-token budgets."""

    MAX_CONTEXT = V2_FRONTIER.block_size
    DEFAULT_OUTPUT_TOKENS = 128

    def __init__(self, tokenizer: object | None = None, max_context: int | None = None) -> None:
        self.tokenizer = tokenizer
        self.max_context = int(max_context or self.MAX_CONTEXT)
        self.budgets = ContextBudgets()

    def _encode(self, text: str) -> list[int] | list[str]:
        if self.tokenizer is None:
            return list(text)
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode(self, tokens: list[int] | list[str]) -> str:
        if self.tokenizer is None:
            return "".join(str(token) for token in tokens)
        return str(self.tokenizer.decode([int(token) for token in tokens]))

    def _token_count(self, text: str) -> int:
        return len(self._encode(text))

    def _truncate(self, text: str, limit: int, *, keep_tail: bool = False) -> str:
        tokens = self._encode(text)
        if len(tokens) <= limit:
            return text
        if limit <= 0:
            return ""
        selected = tokens[-limit:] if keep_tail else tokens[:limit]
        return self._decode(selected)

    @staticmethod
    def _normalize_turns(session_history: list[Any]) -> list[tuple[str, str]]:
        turns: list[tuple[str, str]] = []
        for item in session_history:
            if isinstance(item, tuple) and len(item) == 2:
                turns.append((str(item[0]), str(item[1])))
            elif isinstance(item, dict):
                if item.get("role") == "user":
                    turns.append((str(item.get("content", "")), ""))
                elif item.get("role") == "assistant" and turns:
                    user, _ = turns[-1]
                    turns[-1] = (user, str(item.get("content", "")))
        return turns

    @staticmethod
    def _normalize_memory(memory_results: list[Any]) -> list[str]:
        memories: list[str] = []
        for result in memory_results:
            if isinstance(result, dict):
                summary = str(result.get("summary", "")).strip()
                content = str(result.get("content", "")).strip()
                merged = summary if not content or content == summary else f"{summary}\n{content}"
                if merged.strip():
                    memories.append(merged.strip())
            elif str(result).strip():
                memories.append(str(result).strip())
        return memories

    def build_optimized_context(
        self,
        session_history: list[Any],
        memory_results: list[Any],
        current_message: str,
        *,
        max_new_tokens: int = DEFAULT_OUTPUT_TOKENS,
        identity_context: str = "",
        mode: str = "full_system",
    ) -> dict[str, Any]:
        output_tokens = max(1, min(int(max_new_tokens), self.max_context - 2))
        input_budget = self.max_context - output_tokens - 1  # reserve BOS
        turns = self._normalize_turns(session_history)
        memories = self._normalize_memory(memory_results) if mode == "full_system" else []

        identity = self._truncate(identity_context.strip(), self.budgets.identity)
        prefix = f"{identity}\n" if identity else ""
        marker_tokens = self._token_count("H: \nANRA:")
        current_limit = max(1, input_budget - self._token_count(prefix) - marker_tokens)
        current_tokens = self._encode(current_message.strip())
        current = (
            current_message.strip()
            if len(current_tokens) <= current_limit
            else self._decode(current_tokens[:current_limit])
        )
        suffix = f"H: {current}\nANRA:"
        base_tokens = self._token_count(prefix + suffix)

        def select_history(budget: int) -> tuple[list[str], int]:
            selected: list[str] = []
            used = 0
            for user, assistant in reversed(turns):
                part = f"H: {user}\nANRA: {assistant}\n"
                count = self._token_count(part)
                if used + count > budget:
                    continue
                selected.insert(0, part)
                used += count
            return selected, used

        remaining = max(0, input_budget - base_tokens)
        unused_primary = max(
            0,
            self.budgets.identity
            + self.budgets.message
            - self._token_count(prefix)
            - min(len(current_tokens), self.budgets.message),
        )
        memory_reserve = min(self.budgets.memory, remaining) if memories else 0
        initial_history_budget = min(
            self.budgets.history + unused_primary,
            max(0, remaining - memory_reserve),
        )
        history_parts, history_tokens = select_history(initial_history_budget)

        memory_parts: list[str] = []
        memory_tokens = 0
        memory_used = 0
        memory_budget = min(
            self.budgets.memory,
            max(0, input_budget - base_tokens - history_tokens),
        )
        if memories and memory_budget:
            header = "[MEMORY CONTEXT]\n"
            header_tokens = self._token_count(header)
            if header_tokens < memory_budget:
                memory_parts.append(header)
                memory_tokens = header_tokens
                for index, memory in enumerate(memories, start=1):
                    label = f"{index}. "
                    available = memory_budget - memory_tokens - self._token_count(label + "\n")
                    if available <= 0:
                        break
                    fitted = self._truncate(memory, available)
                    part = f"{label}{fitted}\n"
                    count = self._token_count(part)
                    if not fitted or memory_tokens + count > memory_budget:
                        break
                    memory_parts.append(part)
                    memory_tokens += count
                    memory_used += 1

        # Any unclaimed identity, message, or memory budget goes to newest history.
        expanded_history_budget = max(
            history_tokens,
            input_budget - base_tokens - memory_tokens,
        )
        history_parts, history_tokens = select_history(expanded_history_budget)

        context = prefix + "".join(memory_parts) + "".join(history_parts) + suffix
        prompt_tokens = self._token_count(context)
        if prompt_tokens > input_budget:
            # This is the final safety valve; keep the latest user text and generation marker.
            allowed_message = max(1, input_budget - self._token_count("H: \nANRA:"))
            current = self._truncate(current_message.strip(), allowed_message, keep_tail=True)
            context = f"H: {current}\nANRA:"
            prompt_tokens = self._token_count(context)
            memory_parts = []
            history_parts = []
            memory_tokens = 0
            history_tokens = 0
            memory_used = 0

        trace = PromptAssemblyTrace(
            formatted_prompt=context,
            prompt_tokens=prompt_tokens,
            max_context_tokens=self.max_context,
            reserved_output_tokens=output_tokens,
            turns_included=len(history_parts),
            memory_results_used=memory_used,
            context_truncated=len(history_parts) < len(turns),
            memory_truncated=bool(memories) and memory_used < len(memories),
            token_allocation={
                "identity": self._token_count(prefix),
                "message": self._token_count(suffix),
                "history": history_tokens,
                "memory": memory_tokens,
                "prompt": prompt_tokens,
                "output_reserved": output_tokens,
            },
            mode=mode,
        )
        payload = asdict(trace)
        payload["context"] = trace.formatted_prompt
        payload["context_length"] = trace.prompt_tokens
        return payload


if __name__ == "__main__":
    optimizer = ContextWindowOptimizer()
    result = optimizer.build_optimized_context([], [], "How do you reason?", mode="diagnostic")
    print(result)
