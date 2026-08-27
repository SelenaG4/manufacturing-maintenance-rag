"""Ensure a built index exists before any test runs (build it from the corpus if
the committed one isn't present), and expose a shared in-memory index."""
from __future__ import annotations

import pytest

from app.config import INDEX_DIR
from app.corpus import load_chunks
from app.embedders import LsaEmbedder
from app.index import RagIndex


@pytest.fixture(scope="session", autouse=True)
def _ensure_index():
    if not (INDEX_DIR / "index.faiss").exists():
        RagIndex.build(load_chunks(), LsaEmbedder()).save(INDEX_DIR)
    yield


@pytest.fixture(scope="session")
def index():
    return RagIndex.load(INDEX_DIR)
