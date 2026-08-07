"""Auto-play chat demo: scripted turns that plant facts then probe memory."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logger = logging.getLogger("agentmem.auto_chat")

# Plant distinct facts, distract with unrelated questions, then probe recall.
MEMORY_TEST_SCRIPT: list[str] = [
    "Hi, I'm Alex. I live in Portland and I work as a robotics engineer.",
    "My dog is named Pixel, a border collie. My favorite cafe is Heart Coffee.",
    "Tomorrow I'm flying to Denver for a conference on swarm robotics.",
    "Also, my sister Maya is a violinist in Seattle.",
    "What's a good way to debug a flaky unit test?",
    "Explain what a mutex is in one sentence.",
    "Where do I live?",
    "What's my dog's name and breed?",
    "Where is my sister, and what does she do?",
    "What conference am I going to, and where?",
    "What's my favorite cafe?",
]


def _memory_snippet(messages: list[dict[str, str]]) -> str | None:
    """Extract recalled memory block from the system message, if any."""
    for msg in messages:
        if msg.get("role") != "system":
            continue
        content = msg.get("content", "")
        start = content.find("[Memory]")
        if start < 0:
            return None
        end = content.find("[/Memory]", start)
        if end < 0:
            return content[start:].strip()
        return content[start : end + len("[/Memory]")].strip()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentMem auto-play chat (scripted memory stress test)"
    )
    parser.add_argument(
        "--memory-url",
        default="http://127.0.0.1:8765",
        help="Base URL of the memory HTTP server",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--stub-main",
        action="store_true",
        help="Skip loading the 32B model; echo a stub reply (for smoke tests)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download encoder + main LLM into the HF cache and exit (login node)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to sleep between turns (default: 0)",
    )
    parser.add_argument(
        "--hide-memory",
        action="store_true",
        help="Do not print the recalled [Memory] block each turn",
    )
    args = parser.parse_args()

    if not args.download_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from agentmem.config import MemoryConfig
    from experiments.interactive_chat import (
        download_models,
        generate_reply,
        load_main_llm,
    )
    from experiments.memory_client import MemoryClient, MemoryClientError

    config = MemoryConfig()
    if args.download_only:
        download_models(config)
        return

    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    client = MemoryClient(base_url=args.memory_url)
    try:
        client.health()
    except MemoryClientError as exc:
        print(f"Memory server not reachable at {args.memory_url}", file=sys.stderr)
        print("  Start it with: python -m experiments.memory_server", file=sys.stderr)
        print(f"  ({exc})", file=sys.stderr)
        client.close()
        sys.exit(1)

    tokenizer = model = None
    if not args.stub_main:
        print(
            f"Loading main LLM: {config.main_model_id} "
            f"(4-bit={config.main_load_in_4bit})"
        )
        tokenizer, model = load_main_llm(config)
    else:
        print("Using stub main LLM")

    n = len(MEMORY_TEST_SCRIPT)
    print(f"Auto-chat ready (memory @ {args.memory_url}). {n} scripted turns.\n")
    try:
        for i, user_text in enumerate(MEMORY_TEST_SCRIPT, start=1):
            print("=" * 60)
            print(f"Turn {i}/{n}")
            print(f"You: {user_text}")
            try:
                messages, keys = client.pre(user_text)
            except MemoryClientError as exc:
                logger.exception("Memory /pre failed: %s", exc)
                print(f"Memory /pre failed: {exc}", file=sys.stderr)
                continue

            print(f"Keys: {keys if keys else '(none)'}")
            if not args.hide_memory:
                mem = _memory_snippet(messages)
                if mem:
                    print(mem)
                else:
                    print("[Memory] (none recalled)")

            if args.stub_main:
                reply = f"[stub] Heard: {user_text[:120]}"
            else:
                reply = generate_reply(
                    tokenizer, model, messages, max_new_tokens=args.max_new_tokens
                )
            try:
                client.post(reply)
            except MemoryClientError as exc:
                logger.exception("Memory /post failed: %s", exc)
                print(f"Memory /post failed: {exc}", file=sys.stderr)

            print(f"Assistant: {reply}\n")
            if args.pause > 0 and i < n:
                time.sleep(args.pause)

        try:
            stats = client.stats()
            print("=" * 60)
            print(f"Done. Memory stats: {stats}")
        except MemoryClientError:
            print("=" * 60)
            print("Done.")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    main()
