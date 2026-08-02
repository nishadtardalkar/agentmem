"""Interactive H100 demo: memory session + 4-bit Qwen2.5-32B main LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from agentmem.config import MemoryConfig
from agentmem.session import MemorySession


def load_main_llm(config: MemoryConfig):
    quant_config = None
    if config.main_load_in_4bit:
        compute_dtype = getattr(torch, config.main_bnb_4bit_compute_dtype)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.main_bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.main_model_id, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.main_model_id,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_reply(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 512,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentMem interactive chat demo")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/memory"),
        help="Directory for FAISS + SQLite memory store",
    )
    parser.add_argument(
        "--encoder-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--stub-main",
        action="store_true",
        help="Skip loading the 32B model; echo a stub reply (for smoke tests)",
    )
    args = parser.parse_args()

    config = MemoryConfig(
        data_dir=args.data_dir,
        encoder_device=args.encoder_device,
    )
    print(f"Loading memory encoder: {config.encoder_model_id} ({config.encoder_dtype})")
    session = MemorySession(config=config)

    tokenizer = model = None
    if not args.stub_main:
        print(
            f"Loading main LLM: {config.main_model_id} "
            f"(4-bit={config.main_load_in_4bit})"
        )
        tokenizer, model = load_main_llm(config)
    else:
        print("Using stub main LLM")

    print("Chat ready. Empty line or Ctrl+C to exit.\n")
    try:
        while True:
            user_text = input("You: ").strip()
            if not user_text:
                break
            augmented = session.pre(user_text)
            if args.stub_main:
                reply = f"[stub] Heard: {user_text[:120]}"
            else:
                reply = generate_reply(
                    tokenizer, model, augmented, max_new_tokens=args.max_new_tokens
                )
            session.post(reply)
            print(f"Assistant: {reply}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
