from types import SimpleNamespace

import pytest

import app as app_module


@pytest.fixture
def client(monkeypatch):
    app_module.app.config.update(TESTING=True)

    fake_reviews = [
        {
            "id": 1,
            "text": "hard to trust new category",
            "app": "Blinkit",
            "source": "play_store",
            "supporting_quote": "hard to trust",
            "research_relevance": "discovery_relevant",
            "feedback_domain": "category_discovery",
            "final_reranking_score": 0.8,
        }
    ]

    fake_result = SimpleNamespace(
        top_records=fake_reviews,
        candidates=fake_reviews,
        sufficiency={"state": "limited", "directly_relevant_records": 1},
        selected_filters={"intents": ["exploration_barrier"]},
        intents=["exploration_barrier"],
    )

    monkeypatch.setattr(app_module, "_fetch_all_analyzed_reviews_for_ask", lambda: fake_reviews)
    monkeypatch.setattr(app_module, "run_hybrid_retrieval", lambda q, r: fake_result)
    monkeypatch.setattr(app_module, "_fetch_all_rows", lambda table, cols, page_size=1000: [{"id": 1}] * (10 if table == "raw_reviews" else 1))
    monkeypatch.setattr(app_module, "_count_failed_records", lambda: 2)

    with app_module.app.test_client() as test_client:
        yield test_client


def post_question(client, question="test question"):
    return client.post("/ask", json={"question": question})


def test_success_response(client, monkeypatch):
    monkeypatch.setattr(app_module, "_generate_research_answer", lambda *args, **kwargs: ("Answer", [{"claim_text": "x"}]))
    response = post_question(client)
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "success"
    assert body["answer"] == "Answer\n\nData status: provisional (classification is incomplete)."
    assert body["sources_used"] == 1
    assert body["records_searched"] == 1


def test_quota_exceeded_returns_limited(client, monkeypatch):
    monkeypatch.setattr(app_module, "_generate_research_answer", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("429 quota exceeded")))
    response = post_question(client)
    assert response.status_code == 429
    body = response.get_json()
    assert body["status"] == "limited"
    assert body["error_code"] == "llm_quota_exceeded"
    assert body["answer"] is None
    assert body["sources_used"] == 1
    assert len(body["evidence"]) == 1


def test_retrieval_success_synthesis_failure_returns_evidence(client, monkeypatch):
    monkeypatch.setattr(app_module, "_generate_research_answer", lambda *args, **kwargs: (_ for _ in ()).throw(Exception("model unavailable 503")))
    response = post_question(client)
    assert response.status_code == 503
    body = response.get_json()
    assert body["status"] == "error"
    assert body["error_code"] == "model_unavailable"
    assert body["retrieval_failed"] is False
    assert body["sources_used"] == 1
    assert len(body["evidence"]) == 1


def test_backend_500_retrieval_failure(client, monkeypatch):
    monkeypatch.setattr(app_module, "run_hybrid_retrieval", lambda q, r: (_ for _ in ()).throw(ValueError("boom")))
    monkeypatch.setattr(app_module, "_generate_research_answer", lambda *args, **kwargs: ("unused", []))
    response = post_question(client)
    assert response.status_code == 500
    body = response.get_json()
    assert body["status"] == "error"
    assert body["error_code"] == "internal_server_error"
    assert body["retrieval_failed"] is True
    assert body["sources_used"] is None


def test_timeout_returns_408(client, monkeypatch):
    monkeypatch.setattr(app_module, "_generate_research_answer", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")))
    response = post_question(client)
    assert response.status_code == 408
    body = response.get_json()
    assert body["status"] == "error"
    assert body["error_code"] == "request_timeout"
    assert body["retrieval_failed"] is False


def test_malformed_synthesis_response(client, monkeypatch):
    monkeypatch.setattr(app_module, "_generate_research_answer", lambda *args, **kwargs: (None, []))
    response = post_question(client)
    assert response.status_code == 503
    body = response.get_json()
    assert body["status"] == "error"
    assert body["error_code"] == "malformed_response"


def test_invalid_question_400(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    body = response.get_json()
    assert body["status"] == "error"
    assert body["error_code"] == "invalid_question"
