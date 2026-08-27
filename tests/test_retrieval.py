import pytest

RELEVANT_TOP3 = [
    ("the mill spindle is overheating", "cnc_mill_maintenance::Spindle overheating"),
    ("parts slipping in the lathe chuck", "lathe_maintenance::Chuck and workholding"),
    ("press won't build full tonnage", "hydraulic_press_maintenance::Loss of pressing force / pressure"),
    ("burn marks from the grinder", "surface_grinder_maintenance::Burning and poor finish"),
    ("how to isolate hazardous energy before service", "safety_and_preventive_maintenance::Lockout / tagout (LOTO)"),
]


@pytest.mark.parametrize("query,expected", RELEVANT_TOP3)
def test_relevant_chunk_in_top3(index, query, expected):
    ids = [c.chunk_id for c, _ in index.search(query, 3)]
    assert expected in ids, f"{expected} not in top-3 for '{query}': {ids}"


def test_scores_are_sorted_descending(index):
    hits = index.search("hydraulic oil overheating", 5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
