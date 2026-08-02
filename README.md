# AgentMem

Dual-path latent memory manager that sits before and after a main LLM.

- **Pre-path:** retrieve related episodes via sentence-level hidden-state keys, augment the prompt, then chunk and store the user half-turn.
- **Post-path:** chunk and store the assistant half-turn (store only).

Episodes are formed by semantic sentence breakpoints. Each sentence latent is a FAISS key; the value is the full episode.

## H100 demo models

| Role | Model | Precision |
|------|--------|-----------|
| Encoder | `Qwen/Qwen2.5-0.5B-Instruct` | bf16 |
| Main LLM | `Qwen/Qwen2.5-32B-Instruct` | 4-bit NF4 (bitsandbytes) |

## Quick start

```bash
pip install -e ".[dev]"
pytest
```

On HPC clusters where GPU nodes have no internet, download models on a login node first
(weights land in the shared Hugging Face cache, typically `~/.cache/huggingface/hub`).
Ensure login and GPU nodes share the same home / `HF_HOME`.

```bash
# login node (internet)
python -m experiments.interactive_chat --download-only

# GPU node (no internet) — sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE automatically
python -m experiments.interactive_chat
```
