"""HTTP client for the AgentMem memory server."""

from __future__ import annotations

from typing import Any

import httpx


class MemoryClientError(RuntimeError):
    """Raised when the memory server is unreachable or returns an error."""


class MemoryClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MemoryClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            resp = self._client.request(method, path, **kwargs)
        except httpx.RequestError as exc:
            raise MemoryClientError(
                f"Cannot reach memory server at {self.base_url}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise MemoryClientError(f"{method} {path} -> {resp.status_code}: {detail}")
        return resp.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def pre(self, text: str) -> str:
        data = self._request("POST", "/pre", json={"text": text})
        return data["prompt"]

    def post(self, text: str) -> None:
        self._request("POST", "/post", json={"text": text})

    def stats(self) -> dict[str, int]:
        return self._request("GET", "/stats")

    def list_latents(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/latents")
        return data["latents"]

    def get_latent(self, latent_id: str) -> dict[str, Any]:
        return self._request("GET", f"/latents/{latent_id}")

    def list_episodes(self) -> list[dict[str, Any]]:
        """Back-compat alias for list_latents."""
        return self.list_latents()

    def get_episode(self, episode_id: str) -> dict[str, Any]:
        """Back-compat alias for get_latent."""
        return self.get_latent(episode_id)

    def search(self, text: str) -> list[dict[str, Any]]:
        data = self._request("POST", "/search", json={"text": text})
        return data["hits"]
