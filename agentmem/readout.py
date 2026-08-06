from __future__ import annotations

from agentmem.bank import RetrievedBucket


def format_memory_block(
    buckets: list[RetrievedBucket],
    max_chars: int = 2000,
) -> str:
    if not buckets:
        return ""

    lines: list[str] = ["[Memory]"]
    used = len(lines[0]) + 1
    for bucket in buckets:
        for entry in bucket.entries:
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


def compose_prompt(memory_block: str, user_text: str) -> str:
    user_text = user_text.strip()
    if not memory_block.strip():
        return user_text
    return f"{memory_block.strip()}\n\n{user_text}"
