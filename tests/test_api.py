from fastapi.testclient import TestClient

from app.main import app


def _client():
    return TestClient(app)


def test_health_index_loaded():
    with _client() as c:
        h = c.get("/health").json()
        assert h["status"] == "ok"
        assert h["index_loaded"] is True
        assert h["chunks"] >= 30
        assert h["generation_mode"] == "offline_extractive"  # no keys in CI


def test_corpus_lists_documents():
    with _client() as c:
        body = c.get("/corpus").json()
        assert body["total_chunks"] >= 30
        assert len(body["documents"]) >= 5


def test_ask_returns_grounded_answer_with_sources():
    with _client() as c:
        r = c.post("/ask", json={"question": "how do I know if spindle bearings are wearing out"})
        assert r.status_code == 200
        body = r.json()
        assert body["grounded"] is True
        assert len(body["sources"]) >= 1


def test_ask_out_of_scope_not_grounded():
    with _client() as c:
        body = c.post("/ask", json={"question": "what is the capital of France"}).json()
        assert body["grounded"] is False


def test_ask_validation_rejects_empty():
    with _client() as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
