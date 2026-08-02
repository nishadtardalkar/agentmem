from __future__ import annotations

from agentmem.bank import RetrievedEpisode


def format_memory_block(
    episodes: list[RetrievedEpisode],
    max_chars: int = 2000,
) -> str:
    if not episodes:
        return ""

    lines: list[str] = ["[Memory]"]
    used = len(lines[0]) + 1
    for ep in episodes:
        snippet = ep.text.strip()
        line = f"- ({ep.role}) {snippet}"
        if used + len(line) + 1 > max_chars:
            remaining = max_chars - used - 1
            if remaining > 20:
                lines.append(line[:remaining].rstrip() + "…")
            break
        lines.append(line)
        used += len(line) + 1

    lines.append("[/Memory]")
    return "\n".join(lines)


def compose_prompt(memory_block: str, user_text: str) -> str:
    user_text = user_text.strip()
    if not memory_block.strip():
        return user_text
    return f"{memory_block.strip()}\n\n{user_text}"
