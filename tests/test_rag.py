from app import rag


def test_grounded_answer_cites_correct_source(index):
    ans = rag.answer_question(index, "My hydraulic press won't build full tonnage")
    assert ans.grounded is True
    assert ans.mode == "offline_extractive"          # no API keys in test env
    assert ans.sources[0].chunk_id == "hydraulic_press_maintenance::Loss of pressing force / pressure"
    assert "pressure" in ans.answer.lower()


def test_fault_code_lookup(index):
    ans = rag.answer_question(index, "what does fault code SPN-05 mean")
    assert ans.grounded
    assert any("SPN-05" in s.heading for s in ans.sources)


def test_guardrail_rejects_out_of_scope(index):
    ans = rag.answer_question(index, "what is the best recipe for espresso")
    assert ans.grounded is False
    assert ans.mode == "no_answer"
