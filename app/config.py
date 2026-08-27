"""Shared configuration: paths, chunking + retrieval params, embedder names."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"     # the markdown knowledge base (committed)
EVAL_DIR = ROOT / "data" / "eval"         # labeled QA set (committed)
INDEX_DIR = ROOT / "data" / "index"       # built index + chunk metadata (committed for LSA)

# Chunking: split each doc on its markdown section headers, then pack sections
# into chunks of roughly this many characters (a section is never split mid-way
# unless it alone exceeds the cap).
CHUNK_TARGET_CHARS = 900

# Classical LSA embedder (the offline, self-contained baseline).
LSA_DIM = 160                              # TruncatedSVD components
LSA_MIN_DF = 1

# Retrieval
TOP_K = 4                                  # chunks retrieved per query
EVAL_KS = (1, 3, 5)                        # recall@k / hit@k reported at these cutoffs

# Transformer embedder (the measured upgrade, built in Colab where HF is reachable).
TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
