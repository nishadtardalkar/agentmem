"""Interactive H100 demo: main LLM chat talking to the memory HTTP API."""

from __future__ import annotations

import argparse
import os
import sys


def download_models(config) -> None:
    """Fetch encoder + main LLM into the HF cache (no model load)."""
    from huggingface_hub import snapshot_download

    for model_id in (config.encoder_model_id, config.main_model_id):
        print(f"Downloading {model_id} ...")
        path = snapshot_download(model_id)
        print(f"  cached at {path}")


def resolve_local_model(model_id: str) -> str:
    """Resolve a hub id to a local snapshot path (cache must already exist)."""
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, local_files_only=True)


def load_main_llm(config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant_config = None
    if config.main_load_in_4bit:
        compute_dtype = getattr(torch, config.main_bnb_4bit_compute_dtype)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.main_bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    model_path = resolve_local_model(config.main_model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model


def generate_reply(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    # Tokenize via the chat string so we always get an explicit attention_mask.
    # (pad_token == eos_token on Qwen, so generate cannot infer the mask.)
    chat_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    encoded = tokenizer(chat_text, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentMem interactive chat demo")
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
    args = parser.parse_args()

    # HF hub offline flags are read at import time — set them before importing
    # transformers / huggingface_hub on GPU nodes with no internet.
    if not args.download_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from agentmem.config import MemoryConfig
    from experiments.memory_client import MemoryClient, MemoryClientError

    config = MemoryConfig()
    if args.download_only:
        download_models(config)
        return

    import torch

    # Protect main LLM from broken cuDNN SDPA backends.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    client = MemoryClient(base_url=args.memory_url)
    try:
        client.health()
    except MemoryClientError as exc:
        print(f"Memory server not reachable at {args.memory_url}", file=sys.stderr)
        print(f"  Start it with: python -m experiments.memory_server", file=sys.stderr)
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

    print(f"Chat ready (memory @ {args.memory_url}). Empty line or Ctrl+C to exit.\n")
    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                break
            try:
                augmented = client.pre(user_text)
            except MemoryClientError as exc:
                print(f"Memory /pre failed: {exc}", file=sys.stderr)
                continue
            if args.stub_main:
                reply = f"[stub] Heard: {user_text[:120]}"
            else:
                reply = generate_reply(
                    tokenizer, model, augmented, max_new_tokens=args.max_new_tokens
                )
            try:
                client.post(reply)
            except MemoryClientError as exc:
                print(f"Memory /post failed: {exc}", file=sys.stderr)
            print(f"Assistant: {reply}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
