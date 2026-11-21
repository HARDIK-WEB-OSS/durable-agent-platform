"""Load Markdown/text files into the documents table for search_docs.

    uv run python scripts/ingest_docs.py README.md docs/

Needs the embedding model in Ollama:  ollama pull nomic-embed-text
Re-running is safe: chunks are upserted by (source, chunk number).
"""

import argparse
import asyncio
import sys
from pathlib import Path

from dap.agent.tools.search_docs import Chunk, split_paragraphs, upsert_chunks
from dap.config import get_settings
from dap.db.session import make_engine, make_sessionmaker
from dap.llm.embeddings import OllamaEmbedder

SUFFIXES = {".md", ".txt"}
BATCH = 16


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files += sorted(p for p in path.rglob("*") if p.suffix in SUFFIXES)
        elif path.suffix in SUFFIXES:
            files.append(path)
    return files


def chunks_of(path: Path, root: Path) -> list[Chunk]:
    body = path.read_text(encoding="utf-8")
    title = next(
        (line.lstrip("# ").strip() for line in body.splitlines() if line.startswith("#")),
        path.stem,
    )
    try:
        source = path.resolve().relative_to(root).as_posix()
    except ValueError:
        source = path.name
    return [Chunk(source, i, title, text) for i, text in enumerate(split_paragraphs(body))]


async def main() -> int:
    parser = argparse.ArgumentParser(description="Embed files into the documents table.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    settings = get_settings()
    embedder = OllamaEmbedder(settings.embedding_model, settings.ollama_api_base)
    files = collect(args.paths)
    if not files:
        print("no .md or .txt files found")
        return 1
    root = Path.cwd().resolve()
    chunks = [c for f in files for c in chunks_of(f, root)]
    engine = make_engine(settings)
    sm = make_sessionmaker(engine)
    try:
        for start in range(0, len(chunks), BATCH):
            async with sm.begin() as session:
                await upsert_chunks(session, chunks[start : start + BATCH], embedder)
    finally:
        await engine.dispose()
    print(f"ingested {len(chunks)} chunks from {len(files)} files with {settings.embedding_model}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
