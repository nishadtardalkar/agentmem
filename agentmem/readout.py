from __future__ import annotations

from agentmem.bank import RetrievedEpisode

_SYSTEM_INTRO = (
    "You are a helpful assistant. Recalled memories below are prior context from "
    "earlier turns. They are NOT the user's current message — use them only as "
    "background when relevant."
)


def format_memory_block(
    episodes: list[RetrievedEpisode],
    max_chars: int = 2000,
) -> str:
    if not episodes:
        return ""

    lines: list[str] = ["[Memory]"]
    used = len(lines[0]) + 1
    for hit in episodes:
        entry = hit.entry
        snippet = entry.text.strip()
        line = f"- ({entry.role}) {snippet}"
        if used + len(line) + 1 > max_chars:
            remaining = max_chars - used - 1
            if remaining > 20:
                lines.append(line[:remaining].rstrip() + "…")
            return "\n".join(lines + ["[/Memory]"])
        lines.append(line)
        used += len(line) + 1

    lines.append("[/Memory]")
    return "\n".join(lines)


def compose_system_content(memory_block: str) -> str:
    """System message: role of memory + optional recalled block."""
    memory_block = memory_block.strip()
    if not memory_block:
        return _SYSTEM_INTRO
    return f"{_SYSTEM_INTRO}\n\n{memory_block}"


def compose_messages(memory_block: str, user_text: str) -> list[dict[str, str]]:
    """Build chat messages with memory in system and live text as user."""
    return [
        {"role": "system", "content": compose_system_content(memory_block)},
        {"role": "user", "content": user_text.strip()},
    ]


def compose_prompt(memory_block: str, user_text: str) -> str:
    """Flat string for debug / back-compat; prefer compose_messages for the LLM."""
    messages = compose_messages(memory_block, user_text)
    parts = [f"[{m['role']}]\n{m['content']}" for m in messages]
    return "\n\n".join(parts)
