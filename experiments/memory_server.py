"""Standalone HTTP memory server (FastAPI). Start separately from chat/debug."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("agentmem.server")


class TextBody(BaseModel):
    text: str


def _bucket_to_dict(bucket, *, include_score: bool = False) -> dict:
    payload = {
        "latent_id": bucket.latent_id,
        "entries": [
            {
                "role": e.role,
                "text": e.text,
                "sentences": e.sentences,
                "ts": e.ts,
            }
            for e in bucket.entries
        ],
    }
    if include_score:
        payload["score"] = bucket.score
    return payload


def _text_preview(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def create_app(session) -> FastAPI:
    app = FastAPI(title="AgentMem Memory Server")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    @app.post("/pre")
    def pre(body: TextBody) -> dict:
        try:
            messages = session.pre(body.text)
        except Exception:
            logger.exception(
                "/pre failed (chars=%d preview=%r)",
                len(body.text),
                _text_preview(body.text),
            )
            raise HTTPException(status_code=500, detail="/pre failed") from None
        return {"messages": messages}

    @app.post("/post")
    def post(body: TextBody) -> dict:
        try:
            session.post(body.text)
        except Exception:
            logger.exception(
                "/post failed (chars=%d preview=%r)",
                len(body.text),
                _text_preview(body.text),
            )
            raise HTTPException(status_code=500, detail="/post failed") from None
        return {"ok": True}

    @app.get("/stats")
    def stats() -> dict:
        return {
            "latents": session.bank.latent_count(),
            "keys": session.bank.ntotal,
        }

    @app.get("/latents")
    def list_latents() -> dict:
        buckets = session.bank.list_latents()
        return {"latents": [_bucket_to_dict(b) for b in buckets]}

    @app.get("/latents/{latent_id}")
    def get_latent(latent_id: str) -> dict:
        bucket = session.bank.get_latent(latent_id)
        if bucket is None:
            raise HTTPException(status_code=404, detail="latent not found")
        return _bucket_to_dict(bucket)

    # Back-compat aliases
    @app.get("/episodes")
    def list_episodes() -> dict:
        buckets = session.bank.list_latents()
        return {"episodes": [_bucket_to_dict(b) for b in buckets]}

    @app.get("/episodes/{episode_id}")
    def get_episode(episode_id: str) -> dict:
        bucket = session.bank.get_latent(episode_id)
        if bucket is None:
            raise HTTPException(status_code=404, detail="latent not found")
        return _bucket_to_dict(bucket)

    @app.post("/search")
    def search(body: TextBody) -> dict:
        try:
            hits = session.search(body.text)
        except Exception:
            logger.exception(
                "/search failed (chars=%d preview=%r)",
                len(body.text),
                _text_preview(body.text),
            )
            raise HTTPException(status_code=500, detail="/search failed") from None
        return {"hits": [_bucket_to_dict(b, include_score=True) for b in hits]}

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

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )

    print(f"Loading memory encoder: {config.encoder_model_id} ({config.encoder_dtype})")
    session = MemorySession(config=config)
    app = create_app(session)
    print(f"Memory server listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
