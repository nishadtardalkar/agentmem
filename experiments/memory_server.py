"""Standalone HTTP memory server (FastAPI). Start separately from chat/debug."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class TextBody(BaseModel):
    text: str


def _episode_to_dict(ep, *, include_score: bool = False) -> dict:
    payload = {
        "episode_id": ep.episode_id,
        "role": ep.role,
        "text": ep.text,
        "sentences": ep.sentences,
        "timestamp": ep.timestamp,
    }
    if include_score:
        payload["score"] = ep.score
    return payload


def create_app(session) -> FastAPI:
    app = FastAPI(title="AgentMem Memory Server")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/pre")
    def pre(body: TextBody) -> dict:
        prompt = session.pre(body.text)
        return {"prompt": prompt}

    @app.post("/post")
    def post(body: TextBody) -> dict:
        session.post(body.text)
        return {"ok": True}

    @app.get("/stats")
    def stats() -> dict:
        return {
            "episodes": session.bank.episode_count(),
            "keys": session.bank.ntotal,
        }

    @app.get("/episodes")
    def list_episodes() -> dict:
        eps = session.bank.list_episodes()
        return {"episodes": [_episode_to_dict(ep) for ep in eps]}

    @app.get("/episodes/{episode_id}")
    def get_episode(episode_id: str) -> dict:
        ep = session.bank.get_episode(episode_id)
        if ep is None:
            raise HTTPException(status_code=404, detail="episode not found")
        return _episode_to_dict(ep)

    @app.post("/search")
    def search(body: TextBody) -> dict:
        hits = session.search(body.text)
        return {"hits": [_episode_to_dict(ep, include_score=True) for ep in hits]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentMem memory HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/memory"),
        help="Directory for FAISS + SQLite memory store",
    )
    parser.add_argument(
        "--encoder-device",
        default=None,
        help="Device for the memory encoder (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download encoder into the HF cache and exit (login node)",
    )
    args = parser.parse_args()

    if not args.download_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    import uvicorn

    from agentmem.config import MemoryConfig
    from agentmem.session import MemorySession

    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    if args.encoder_device is None:
        args.encoder_device = "cuda" if torch.cuda.is_available() else "cpu"

    config = MemoryConfig(
        data_dir=args.data_dir,
        encoder_device=args.encoder_device,
    )

    if args.download_only:
        from huggingface_hub import snapshot_download

        print(f"Downloading {config.encoder_model_id} ...")
        path = snapshot_download(config.encoder_model_id)
        print(f"  cached at {path}")
        return

    print(f"Loading memory encoder: {config.encoder_model_id} ({config.encoder_dtype})")
    session = MemorySession(config=config)
    app = create_app(session)
    print(f"Memory server listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
