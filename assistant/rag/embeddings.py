from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

from assistant.config import Settings


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI or Azure OpenAI embeddings (Anthropic does not offer an embeddings endpoint)."""

    def __init__(self, client, model: str, dim: int, batch_size: int = 128):
        self._client = client
        self._model = model
        self.dim = dim
        self._batch = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            resp = await self._client.embeddings.create(
                model=self._model, input=texts[i : i + self._batch], dimensions=self.dim
            )
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out


_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    """Deterministic bag-of-words feature hashing. Offline tests and demos only: it matches
    on shared words, not meaning."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        tokens = _TOKEN.findall(text.lower())
        for tok in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]:
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
        norm = float(np.linalg.norm(v))
        return (v / norm).tolist() if norm else v.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def build_embedder(settings: Settings) -> Embedder:
    provider, _, model = settings.embedding_model.partition(":")
    if provider == "hashing":
        return HashingEmbedder(settings.embedding_dim)
    import openai

    if provider == "openai":
        return OpenAIEmbedder(openai.AsyncOpenAI(), model, settings.embedding_dim)
    if provider == "azure":
        client = openai.AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint, api_version=settings.azure_openai_api_version
        )
        return OpenAIEmbedder(client, model, settings.embedding_dim)
    raise ValueError(f"unknown embedding provider {provider!r} (expected openai, azure or hashing)")
