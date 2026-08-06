# AgentMem

Dual-path latent memory manager that sits before and after a main LLM.

- **Pre-path:** extract key tokens (via the encoder instruct model), retrieve related episodes from those token embeddings, return chat messages with memory in the **system** role and the live user text in the **user** role, then chunk and store the user half-turn.
- **Post-path:** chunk and store the assistant half-turn (store only).

Episodes are formed by semantic sentence breakpoints (`tau_break`). Each episode is
always stored as its own memory value `{ts, role, text, sentences, ...}`. FAISS holds
embeddings of **model-extracted key tokens** from that episode; every key points at the
same episode so subject tokens can retrieve related memories without merging values.

## H100 demo models

| Role | Model | Precision |
|------|--------|-----------|
| Encoder + key extract | `Qwen/Qwen2.5-0.5B-Instruct` | bf16 |
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
python -m experiments.memory_server --download-only
python -m experiments.interactive_chat --download-only

# GPU node — three separate processes (memory loads the encoder; chat loads the 32B)

# terminal 1 — memory HTTP API (default http://127.0.0.1:8765)
python -m experiments.memory_server --data-dir data/memory

# terminal 2 — chat (talks to memory over HTTP)
python -m experiments.interactive_chat

# terminal 3 — optional debug REPL (same API)
python -m experiments.memory_debug
```

Memory server flags: `--host`, `--port`, `--data-dir`, `--encoder-device`.
Chat / debug flag: `--memory-url` (default `http://127.0.0.1:8765`).

### Upgrading memory indexes

Schema / key layout changes require a clean store. Clear the memory data directory
(e.g. delete `data/memory`) before restarting the server after this change.
