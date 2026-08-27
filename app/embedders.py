"""Embedders turn text into dense vectors. Two real implementations behind one
interface:

* `LsaEmbedder` -- the classical, fully-offline baseline: TF-IDF + Truncated SVD
  (Latent Semantic Analysis) + L2 normalization. It learns a dense semantic space
  from the corpus itself, so it needs no model download and runs anywhere. This is
  what the live service uses.

* the transformer embedder (`sentence-transformers/all-MiniLM-L6-v2`) is the
  measured *upgrade*, computed in the Colab notebook where the model hub is
  reachable; the eval harness compares the two. (Same "classical baseline vs. the
  transformer, measured" pattern as the rest of this portfolio.)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from app.config import LSA_DIM, LSA_MIN_DF


class LsaEmbedder:
    name = "lsa"

    def __init__(self, dim: int = LSA_DIM):
        self.dim = dim
        self.pipe: Pipeline | None = None

    def fit(self, texts: list[str]) -> "LsaEmbedder":
        # SVD components must be < n_samples; on a small corpus, cap accordingly.
        n_comp = max(2, min(self.dim, len(texts) - 1))
        self.pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                      min_df=LSA_MIN_DF, sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=n_comp, random_state=42)),
            ("norm", Normalizer(copy=False)),
        ])
        self.pipe.fit(texts)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.pipe is None:
            raise RuntimeError("LsaEmbedder used before fit()/load().")
        return np.asarray(self.pipe.transform(texts), dtype="float32")

    def save(self, path: Path) -> None:
        joblib.dump({"dim": self.dim, "pipe": self.pipe}, path)

    @classmethod
    def load(cls, path: Path) -> "LsaEmbedder":
        blob = joblib.load(path)
        e = cls(blob["dim"])
        e.pipe = blob["pipe"]
        return e
