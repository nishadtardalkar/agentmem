"""Interactive REPL to inspect memory via the HTTP API. Start separately."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap


def _truncate(text: str, width: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def run_repl(client) -> None:
    from experiments.memory_client import MemoryClientError

    print("Memory debug REPL. Commands: stats | list | show <id> | search <text> | quit\n")
    while True:
        try:
            line = input("dbg> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not line:
            continue

        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        try:
            if cmd in {"quit", "exit", "q"}:
                break
            if cmd == "stats":
                s = client.stats()
                print(f"  episodes={s['episodes']}  keys={s['keys']}")
            elif cmd == "list":
                eps = client.list_episodes()
                if not eps:
                    print("  (empty)")
                for ep in eps:
                    print(
                        f"  {ep['episode_id']}  {ep['role']:9}  "
                        f"{_truncate(ep['text'])}"
                    )
            elif cmd == "show":
                if not arg:
                    print("  usage: show <episode_id>")
                    continue
                ep = client.get_episode(arg)
                print(json.dumps(ep, indent=2))
            elif cmd == "search":
                if not arg:
                    print("  usage: search <text>")
                    continue
                hits = client.search(arg)
                if not hits:
                    print("  (no hits)")
                for h in hits:
                    print(
                        f"  {h['score']:.3f}  {h['episode_id'][:8]}  {h['role']:9}  "
                        f"{_truncate(h['text'])}"
                    )
            elif cmd == "help":
                print(
                    textwrap.dedent(
                        """\
                        stats          bank size (episodes + FAISS keys)
                        list           list episodes
                        show <id>      full episode JSON
                        search <text>  retrieve-only search with scores
                        quit           exit
                        """
                    )
                )
            else:
                print(f"  unknown command: {cmd} (try help)")
        except MemoryClientError as exc:
            print(f"  error: {exc}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentMem memory debug REPL")
    parser.add_argument(
        "--memory-url",
        default="http://127.0.0.1:8765",
        help="Base URL of the memory HTTP server",
    )
    args = parser.parse_args()

    from experiments.memory_client import MemoryClient, MemoryClientError

    client = MemoryClient(base_url=args.memory_url)
    try:
        client.health()
    except MemoryClientError as exc:
        print(f"Memory server not reachable at {args.memory_url}", file=sys.stderr)
        print("  Start it with: python -m experiments.memory_server", file=sys.stderr)
        print(f"  ({exc})", file=sys.stderr)
        client.close()
        sys.exit(1)

    print(f"Connected to memory @ {args.memory_url}")
    try:
        run_repl(client)
    finally:
        client.close()
    print("Bye.")


if __name__ == "__main__":
    main()
