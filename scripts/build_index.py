"""Build the LSA vector index from the corpus and save it to data/index/.

    python scripts/build_index.py

The saved index (FAISS + chunk metadata + fitted embedder) is small and
committed, so the service and the dashboard load it instantly with no rebuild.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.corpus import load_chunks  # noqa: E402
from app.embedders import LsaEmbedder  # noqa: E402
from app.index import RagIndex  # noqa: E402


def main() -> None:
    chunks = load_chunks()
    index = RagIndex.build(chunks, LsaEmbedder())
    index.save()
    print(f"Built LSA index: {len(chunks)} chunks, {index.faiss_index.d}-dim vectors -> data/index/")


if __name__ == "__main__":
    main()
