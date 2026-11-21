from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from dap.agent.tools.base import ToolError
from dap.llm.embeddings import Embedder, to_pgvector

SEARCH_SQL = text(
    """
    SELECT source, chunk, title, content,
           embedding <=> CAST(CAST(:query AS text) AS vector) AS distance
    FROM documents
    ORDER BY distance
    LIMIT :k
    """
)

UPSERT_SQL = text(
    """
    INSERT INTO documents (source, chunk, title, content, embedding)
    VALUES (:source, :chunk, :title, :content, CAST(CAST(:embedding AS text) AS vector))
    ON CONFLICT (source, chunk) DO UPDATE
    SET title = EXCLUDED.title, content = EXCLUDED.content, embedding = EXCLUDED.embedding
    """
)


@dataclass(frozen=True)
class Hit:
    source: str
    chunk: int
    title: str | None
    content: str
    distance: float


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk: int
    title: str | None
    content: str


async def search(session: AsyncSession, query_vector: Sequence[float], k: int) -> list[Hit]:
    rows = await session.execute(SEARCH_SQL, {"query": to_pgvector(query_vector), "k": k})
    return [Hit(*row) for row in rows]


async def upsert_chunks(session: AsyncSession, chunks: Sequence[Chunk], embedder: Embedder) -> int:
    vectors = await embedder.embed_documents([c.content for c in chunks])
    for c, vector in zip(chunks, vectors, strict=True):
        await session.execute(
            UPSERT_SQL,
            {
                "source": c.source,
                "chunk": c.chunk,
                "title": c.title,
                "content": c.content,
                "embedding": to_pgvector(vector),
            },
        )
    return len(chunks)


def split_paragraphs(body: str, max_chars: int = 1200) -> list[str]:
    """Greedy chunking on blank lines; a single long paragraph is cut at max_chars."""
    chunks: list[str] = []
    current = ""
    for para in (p.strip() for p in body.split("\n\n")):
        if not para:
            continue
        if len(para) > max_chars and current:
            chunks.append(current)
            current = ""
        while len(para) > max_chars:
            chunks.append(para[:max_chars])
            para = para[max_chars:]
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = ""
        current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def format_hits(hits: Sequence[Hit], max_chars_each: int = 700) -> str:
    """Retrieved text is wrapped as data; the system prompt says never to obey it."""
    if not hits:
        return "No documents found."
    blocks = []
    for i, h in enumerate(hits, 1):
        body = h.content if len(h.content) <= max_chars_each else h.content[:max_chars_each] + "..."
        label = f"{h.source}#{h.chunk}" + (f" ({h.title})" if h.title else "")
        blocks.append(
            f'<document index="{i}" source="{label}" distance="{h.distance:.3f}">\n'
            f"{body}\n</document>"
        )
    return "\n".join(blocks)


@dataclass(frozen=True)
class SearchDocsTool:
    sessionmaker: async_sessionmaker[AsyncSession]
    embedder: Embedder
    top_k: int = 4
    name: str = "search_docs"

    @property
    def description(self) -> str:
        return (
            "Semantic search over the document collection. tool_input: a short search query. "
            f"Returns the {self.top_k} closest passages."
        )

    async def run(self, tool_input: str) -> str:
        query = tool_input.strip()
        if not query:
            raise ToolError("empty search query")
        vector = await self.embedder.embed_query(query)
        async with self.sessionmaker() as session:
            hits = await search(session, vector, self.top_k)
        return format_hits(hits)
