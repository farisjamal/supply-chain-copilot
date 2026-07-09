"""Minimal RAG (Retrieval-Augmented Generation) pipeline.

Classic three stages:

  1. Ingest   - load markdown knowledge docs (supplier contracts, SOPs, policies)
  2. Index    - split into chunks, vectorize with TF-IDF
  3. Retrieve - cosine similarity search; top-k chunks are handed to the
                LLM as grounded context via the `search_supplier_docs` tool

TF-IDF is deliberate for this demo: zero heavy dependencies, fully local,
and easy to explain. The `Retriever` interface is the seam — swapping in
sentence-transformer embeddings + a vector DB (Chroma/FAISS) only changes
the `_vectorize` internals, not the agent.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..config import DOCS_DIR

CHUNK_MAX_CHARS = 800


@dataclass
class Chunk:
    source: str   # filename the chunk came from
    heading: str  # nearest markdown heading
    text: str


def split_markdown(path: Path) -> list[Chunk]:
    """Split a markdown file into heading-scoped chunks of bounded size."""
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    heading = path.stem
    buf: list[str] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            chunks.append(Chunk(source=path.name, heading=heading, text=body))
        buf.clear()

    for line in text.splitlines():
        if re.match(r"^#{1,3} ", line):
            flush()
            heading = line.lstrip("# ").strip()
        buf.append(line)
        if sum(len(x) for x in buf) > CHUNK_MAX_CHARS:
            flush()
    flush()
    return chunks


class Retriever:
    """TF-IDF retriever over the knowledge base in data/docs/."""

    def __init__(self, docs_dir: Path = DOCS_DIR):
        self.chunks: list[Chunk] = []
        for path in sorted(docs_dir.glob("*.md")):
            self.chunks.extend(split_markdown(path))
        if not self.chunks:
            raise FileNotFoundError(f"No markdown docs found in {docs_dir}")
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, similarity) pairs for the query."""
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in top if sims[i] > 0]


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Singleton so the index is built once per process."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
