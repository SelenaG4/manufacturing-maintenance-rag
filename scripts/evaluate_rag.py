"""Evaluate retrieval quality against the labeled QA set and log to MLflow.

This is the step that makes it a *production* RAG rather than a demo: the answer
is only as good as what's retrieved, so retrieval is measured directly, on a
held-out set of questions each mapped to its known-relevant chunk(s).

Metrics (reported at k = 1, 3, 5):
  * hit@k      -- fraction of questions with a relevant chunk in the top k
  * recall@k   -- fraction of each question's relevant chunks found in the top k
  * MRR        -- mean reciprocal rank of the first relevant chunk

The classical LSA embedder is evaluated here (fully offline, real numbers). The
transformer embedder (`all-MiniLM-L6-v2`) is evaluated the same way in the Colab
notebook, where the model hub is reachable, to produce the measured comparison.

    python scripts/evaluate_rag.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import EVAL_DIR, EVAL_KS  # noqa: E402
from app.index import RagIndex  # noqa: E402


def load_qa() -> list[dict]:
    lines = (EVAL_DIR / "qa.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def evaluate(search_fn, qa: list[dict], ks=EVAL_KS) -> dict:
    """search_fn(question) -> ranked list of chunk_ids. Returns metric dict."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    recall = {k: 0.0 for k in ks}
    rr_total = 0.0
    for item in qa:
        ranked = search_fn(item["question"])
        relevant = set(item["relevant"])
        # reciprocal rank of the first relevant hit
        rr = 0.0
        for rank, cid in enumerate(ranked, start=1):
            if cid in relevant:
                rr = 1.0 / rank
                break
        rr_total += rr
        for k in ks:
            topk = ranked[:k]
            found = relevant.intersection(topk)
            hits[k] += 1 if found else 0
            recall[k] += len(found) / len(relevant)
    n = len(qa)
    # Full precision here (so the numbers are exact for tests/comparison); the
    # caller rounds for display / logging.
    out = {"n_questions": n, "mrr": rr_total / n}
    for k in ks:
        out[f"hit@{k}"] = hits[k] / n
        out[f"recall@{k}"] = recall[k] / n
    return out


def main() -> None:
    index = RagIndex.load()
    qa = load_qa()

    def search_fn(q: str) -> list[str]:
        return [c.chunk_id for c, _ in index.search(q, max(EVAL_KS))]

    metrics = evaluate(search_fn, qa)
    print(f"LSA (TF-IDF + SVD) retrieval over {metrics['n_questions']} questions:")
    print(f"  MRR       {metrics['mrr']:.3f}")
    for k in EVAL_KS:
        print(f"  hit@{k}    {metrics[f'hit@{k}']:.3f}      recall@{k}  {metrics[f'recall@{k}']:.3f}")

    metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "retrieval_results_lsa.json").write_text(json.dumps(metrics, indent=2))

    import mlflow  # lazy -- only needed to run the tracked eval, not to use evaluate()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("manufacturing_maintenance_rag")
    with mlflow.start_run(run_name="lsa_retrieval"):
        mlflow.log_params({"embedder": "lsa_tfidf_svd", "index": "faiss_flat_ip", "n_questions": metrics["n_questions"]})
        # MLflow metric names can't contain "@" -> log as hit_at_1 etc.
        mlflow.log_metrics({k.replace("@", "_at_"): v for k, v in metrics.items() if k != "n_questions"})
    print("Logged to MLflow (sqlite:///mlflow.db, experiment 'manufacturing_maintenance_rag').")


if __name__ == "__main__":
    main()
