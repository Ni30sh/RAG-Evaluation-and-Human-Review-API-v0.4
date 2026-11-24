from fastapi import FastAPI, Response
from pydantic import BaseModel, Field, field_validator
from starlette.responses import RedirectResponse

from rag_evaluation import FAQRetriever, calculate_metrics, load_test_questions


app = FastAPI(title="RAG Evaluation and Human Review API", version="0.4")
retriever = FAQRetriever()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    expected_source_id: str | None = None
    expected_source_ids: list[str] | None = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class RetrievedSource(BaseModel):
    source_id: str
    confidence: float


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_sources: list[str]
    retrieval_hit: bool
    needs_human_review: bool
    confidence: float


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    retrieved = retriever.retrieve(question, top_k=3)
    source_ids = [item["source_id"] for item in retrieved]
    top_confidence = retrieved[0]["confidence"] if retrieved else 0.0

    expected_ids = set(request.expected_source_ids or [])
    if request.expected_source_id:
        expected_ids.add(request.expected_source_id)

    if not expected_ids:
        for case in load_test_questions():
            if case["question"].strip().lower() == question.lower():
                expected_ids.add(case["expected_source_id"])
                break

    retrieval_hit = bool(expected_ids and expected_ids.intersection(source_ids))
    expected_missing = bool(expected_ids) and not retrieval_hit
    confidence_insufficient = top_confidence < retriever.confidence_threshold
    needs_human_review = expected_missing or confidence_insufficient

    answer = retrieved[0]["answer"] if retrieved and not needs_human_review else (
        "I could not find sufficiently relevant evidence to answer this question."
    )
    return AskResponse(
        question=question,
        answer=answer,
        retrieved_sources=source_ids,
        retrieval_hit=retrieval_hit,
        needs_human_review=needs_human_review,
        confidence=top_confidence,
    )


@app.get("/evaluation")
def evaluation() -> dict:
    return calculate_metrics(retriever, load_test_questions(), k=3)
