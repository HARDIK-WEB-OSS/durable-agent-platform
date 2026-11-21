from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

EMBEDDING_DIM = 768  # nomic-embed-text; fixed by the vector(768) column in migration 0004

# nomic-embed-text is trained with task prefixes; leaving them out degrades retrieval.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class EmbeddingError(Exception):
    pass


class Embedder(Protocol):
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class OllamaEmbedder:
    model: str = "nomic-embed-text"
    api_base: str = "http://localhost:11434"
    dim: int = EMBEDDING_DIM
    timeout_s: float = 60.0
    transport: httpx.AsyncBaseTransport | None = None

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self.embed([DOCUMENT_PREFIX + t for t in texts])

    async def embed_query(self, text: str) -> list[float]:
        [vector] = await self.embed([QUERY_PREFIX + text])
        return vector

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        async with httpx.AsyncClient(
            base_url=self.api_base, timeout=self.timeout_s, transport=self.transport
        ) as http:
            response = await http.post("/api/embed", json={"model": self.model, "input": inputs})
        if response.status_code != 200:
            raise EmbeddingError(f"ollama embed HTTP {response.status_code}: {response.text[:200]}")
        vectors: list[list[float]] = response.json()["embeddings"]
        if len(vectors) != len(inputs) or any(len(v) != self.dim for v in vectors):
            dims = sorted({len(v) for v in vectors})
            raise EmbeddingError(
                f"{self.model} returned {len(vectors)} vectors of dim {dims}; "
                f"expected {len(inputs)} of dim {self.dim}"
            )
        return vectors


def to_pgvector(vector: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"
