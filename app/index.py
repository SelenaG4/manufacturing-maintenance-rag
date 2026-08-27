"""The vector index: a FAISS inner-product index over normalized chunk
embeddings (inner product on L2-normalized vectors == cosine similarity), plus
the chunk metadata and the fitted embedder needed to embed queries the same way.
Save/load persists all three so the service loads a prebuilt index instantly.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.config import INDEX_DIR
from app.corpus import Chunk
from app.embedders import LsaEmbedder


class RagIndex:
    def __init__(self, chunks: list[Chunk], embedder: LsaEmbedder, faiss_index):
        self.chunks = chunks
        self.embedder = embedder
        self.faiss_index = faiss_index

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: LsaEmbedder) -> "RagIndex":
        embedder.fit([c.text for c in chunks])
        vecs = embedder.embed([c.text for c in chunks])
        faiss.normalize_L2(vecs)
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        return cls(chunks, embedder, idx)

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.embedder.embed([query])
        faiss.normalize_L2(q)
        scores, ids = self.faiss_index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(scores[0][rank]))
                for rank, i in enumerate(ids[0]) if i >= 0]

    def save(self, out_dir: Path = INDEX_DIR) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(out_dir / "index.faiss"))
        self.embedder.save(out_dir / "embedder.joblib")
        (out_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(asdict(c)) for c in self.chunks), encoding="utf-8"
        )

    @classmethod
    def load(cls, in_dir: Path = INDEX_DIR) -> "RagIndex":
        in_dir = Path(in_dir)
        idx = faiss.read_index(str(in_dir / "index.faiss"))
        emb = LsaEmbedder.load(in_dir / "embedder.joblib")
        chunks = [Chunk(**json.loads(ln))
                  for ln in (in_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
        return cls(chunks, emb, idx)
