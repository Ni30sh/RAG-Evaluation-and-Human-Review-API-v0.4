from fastapi.testclient import TestClient

from main import app
from rag_evaluation import FAQRetriever, calculate_metrics


client = TestClient(app)


def test_root_redirects_to_api_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_favicon_returns_no_content():
    response = client.get("/favicon.ico")
    assert response.status_code == 204


def test_dataset_has_ten_cases():
    retriever = FAQRetriever(collection_name="faq_test_dataset")
    metrics = calculate_metrics(retriever)
    assert set(metrics) == {"Hit@3", "MRR"}
    assert 0.0 <= metrics["Hit@3"] <= 1.0
    assert 0.0 <= metrics["MRR"] <= 1.0


def test_ask_known_question():
    response = client.post("/ask", json={"question": "What is the service interval?"})
    assert response.status_code == 200
    body = response.json()
    assert "10,000" in body["answer"]
    assert "service_faq_01" in body["retrieved_sources"]
    assert body["retrieval_hit"] is True
    assert body["needs_human_review"] is False


def test_ask_unknown_question_requires_human_review():
    response = client.post("/ask", json={"question": "Where should I ship my interplanetary rover?"})
    assert response.status_code == 200
    assert response.json()["needs_human_review"] is True


def test_missing_expected_source_requires_human_review():
    response = client.post(
        "/ask",
        json={
            "question": "What is the service interval?",
            "expected_source_id": "service_faq_99",
        },
    )
    assert response.status_code == 200
    assert response.json()["retrieval_hit"] is False
    assert response.json()["needs_human_review"] is True


def test_empty_question_is_rejected():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422
