"""Retrieval-quality gate: fail the build if the index got worse.

The problem this solves
-----------------------
Every test in ``tests/`` can pass while the thing users actually care about
silently degrades. Re-chunk the corpus, change ``LSA_DIM``, add a document,
swap the embedder -- the API still returns 200, the guardrail still fires, the
suite is still green, and retrieval quality has quietly dropped ten points. A
unit test cannot catch that, because there is no exception to raise.

So retrieval quality is treated as a release criterion in its own right. This
script re-runs the labelled evaluation and compares it against a committed
baseline. Regression beyond an allowed tolerance exits non-zero, which fails CI
and blocks the deployment.

Two thresholds, doing different jobs
------------------------------------
``--min-mrr``       an absolute floor. Quality never goes below this, whatever
                    the baseline says. Stops a slow drift downward from being
                    ratified one acceptable-looking step at a time.
``--max-drop``      a relative tolerance against the committed baseline. Catches
                    a single change that makes things suddenly worse, even while
                    still above the floor.

Usage::

    python scripts/gate_retrieval.py                    # gate against docs/retrieval_results_lsa.json
    python scripts/gate_retrieval.py --update-baseline  # accept current numbers as the new baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import EVAL_KS  # noqa: E402
from app.index import RagIndex  # noqa: E402
from scripts.evaluate_rag import evaluate, load_qa  # noqa: E402

BASELINE_PATH = ROOT / "docs" / "retrieval_results_lsa.json"

# Absolute floors. Set from the measured LSA baseline (MRR 0.853, hit@3 0.933)
# with headroom for the noise a re-chunk legitimately introduces -- tight enough
# to catch a real regression, loose enough not to fail on a rounding wobble.
DEFAULT_MIN_MRR = 0.75
DEFAULT_MIN_HIT3 = 0.85

# How far below the committed baseline any single metric may fall.
DEFAULT_MAX_DROP = 0.05


def measure() -> dict:
    """Run the labelled evaluation against the current index."""
    index = RagIndex.load()
    qa = load_qa()

    def search_fn(question: str) -> list[str]:
        return [c.chunk_id for c, _ in index.search(question, max(EVAL_KS))]

    return evaluate(search_fn, qa)


def load_baseline() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def check(current: dict, baseline: dict | None, *, min_mrr: float,
          min_hit3: float, max_drop: float) -> list[str]:
    """Return a list of human-readable failures. Empty list means the gate passes."""
    failures: list[str] = []

    if current["mrr"] < min_mrr:
        failures.append(
            f"MRR {current['mrr']:.4f} is below the absolute floor of {min_mrr:.4f}"
        )
    if current["hit@3"] < min_hit3:
        failures.append(
            f"hit@3 {current['hit@3']:.4f} is below the absolute floor of {min_hit3:.4f}"
        )

    if baseline is None:
        return failures

    # Compare every metric the baseline recorded, not just the headline one: a
    # change can leave MRR flat while hollowing out recall@5.
    for metric, baseline_value in baseline.items():
        if metric == "n_questions" or metric not in current:
            continue
        drop = baseline_value - current[metric]
        if drop > max_drop:
            failures.append(
                f"{metric} dropped {drop:.4f} "
                f"({baseline_value:.4f} -> {current[metric]:.4f}), "
                f"more than the {max_drop:.4f} allowed"
            )

    if baseline.get("n_questions") != current["n_questions"]:
        # Not a failure: growing the eval set is good. But the comparison is no
        # longer like-for-like, so say so rather than quietly compare anyway.
        print(
            f"  note: eval set changed size "
            f"({baseline.get('n_questions')} -> {current['n_questions']} questions); "
            f"baseline comparison is not like-for-like.",
            file=sys.stderr,
        )

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-mrr", type=float, default=DEFAULT_MIN_MRR)
    parser.add_argument("--min-hit3", type=float, default=DEFAULT_MIN_HIT3)
    parser.add_argument("--max-drop", type=float, default=DEFAULT_MAX_DROP)
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write the current metrics to the baseline file and exit 0. Use "
             "when a change legitimately improves or restructures retrieval.",
    )
    args = parser.parse_args()

    current = measure()
    baseline = load_baseline()

    print(f"Retrieval gate -- {current['n_questions']} labelled questions")
    print(f"  MRR      {current['mrr']:.4f}")
    for k in EVAL_KS:
        print(f"  hit@{k}    {current[f'hit@{k}']:.4f}      recall@{k}  {current[f'recall@{k}']:.4f}")

    if args.update_baseline:
        rounded = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in current.items()}
        BASELINE_PATH.parent.mkdir(exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(rounded, indent=2) + "\n", encoding="utf-8")
        print(f"\nBaseline updated: {BASELINE_PATH.relative_to(ROOT)}")
        return 0

    if baseline is None:
        print("\nNo committed baseline found; checked against absolute floors only.")

    failures = check(
        current, baseline,
        min_mrr=args.min_mrr, min_hit3=args.min_hit3, max_drop=args.max_drop,
    )

    if failures:
        print("\nGATE FAILED -- deployment blocked:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nIf this change is a deliberate, justified improvement, re-run with "
            "--update-baseline and commit the new baseline in the same PR.",
            file=sys.stderr,
        )
        return 1

    print("\nGATE PASSED -- retrieval quality is within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
