from app.corpus import load_chunks


def test_chunks_load_with_stable_ids():
    chunks = load_chunks()
    assert len(chunks) >= 30
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))                 # unique
    assert all("::" in cid for cid in ids)           # "<file>::<heading>"
    # a couple of known sections exist
    assert any(c.chunk_id == "cnc_mill_maintenance::Spindle overheating" for c in chunks)
    assert any(c.heading.startswith("HYD-07") for c in chunks)


def test_chunk_text_includes_title_and_body():
    c = next(c for c in load_chunks() if c.chunk_id == "cnc_mill_maintenance::Spindle overheating")
    assert "CNC Milling" in c.text and "coolant" in c.text.lower()
