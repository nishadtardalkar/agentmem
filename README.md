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
python -m experiments.interactive_chat
```
