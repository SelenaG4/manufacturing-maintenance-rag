"""Unit-tests the retrieval-metric functions on a hand-made case with a fake
search function, so the numbers the eval harness reports are trustworthy."""
from scripts.evaluate_rag import evaluate

QA = [
    {"question": "q1", "relevant": ["a"]},        # 'a' returned at rank 2
    {"question": "q2", "relevant": ["b"]},        # 'b' returned at rank 1
    {"question": "q3", "relevant": ["c"]},        # 'c' not retrieved at all
]
FAKE = {"q1": ["x", "a", "y"], "q2": ["b", "x", "y"], "q3": ["x", "y", "z"]}


def test_metrics_math():
    m = evaluate(lambda q: FAKE[q], QA, ks=(1, 3))
    assert m["n_questions"] == 3
    # MRR = (1/2 + 1/1 + 0) / 3 = 0.5
    assert abs(m["mrr"] - 0.5) < 1e-9
    # hit@1: only q2 -> 1/3 ; hit@3: q1,q2 -> 2/3
    assert abs(m["hit@1"] - 1 / 3) < 1e-9
    assert abs(m["hit@3"] - 2 / 3) < 1e-9
    # recall@3 == hit@3 here (one relevant chunk each)
    assert abs(m["recall@3"] - 2 / 3) < 1e-9
