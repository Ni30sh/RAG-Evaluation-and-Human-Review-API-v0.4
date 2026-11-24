# Project Guide: RAG Evaluation and Human Review API

This guide explains what the project does, how a question moves through it, and
where each part of the code and data lives.

## What this project does

This is a small FAQ question-answering API built with FastAPI and ChromaDB. It:

1. Loads a set of FAQ documents with stable source IDs.
2. Uses ChromaDB to find the three FAQ entries most relevant to a question.
3. Returns the text of the top matching FAQ entry as the answer, along with
   retrieved source IDs and a confidence estimate.
4. Marks the result for human review if expected evidence was not retrieved or
   the top result's confidence is below the configured threshold.
5. Measures retrieval quality on a set of test questions using Hit@3 and MRR.

The answer is taken directly from the highest-ranked FAQ document. The project
does not call a generative language model to compose or paraphrase answers.

## Request flow

```text
POST /ask
   |
   v
Pydantic validates the question
   |
   v
FAQRetriever asks ChromaDB for the top 3 FAQ matches
   |
   +--> compare retrieved IDs with expected IDs, if provided or known
   |
   +--> compare top match confidence with the configured threshold
   |
   v
Return answer, source IDs, retrieval_hit, confidence,
and needs_human_review
```

## Project layout

```text
.
├── app.py                     # ASGI import alias for the FastAPI app
├── main.py                    # FastAPI app, request validation, and endpoints
├── rag_evaluation.py          # ChromaDB retrieval and evaluation metrics
├── data/
│   ├── faq_documents.json     # Ten FAQ source documents
│   └── test_questions.json    # Ten questions and their expected source IDs
├── tests/
│   ├── conftest.py            # Makes the project root importable in pytest
│   └── test_rag_api.py        # API and retrieval behavior tests
├── requirements.txt           # Runtime package dependencies
├── pyproject.toml             # Python package/project metadata
├── README.md                  # Short setup and API introduction
└── README_DETAILED.md         # This detailed project guide
```

### `main.py` — web API

This is the main application module. It creates the FastAPI `app` and one
`FAQRetriever` instance when the module is imported.

- `AskRequest` validates incoming `/ask` JSON. `question` must contain
  non-whitespace text and be no longer than 1,000 characters. Callers may also
  provide `expected_source_id` or `expected_source_ids`.
- `AskResponse` defines the response fields for `/ask`.
- `GET /` redirects a browser to `/docs`.
- `GET /favicon.ico` returns an empty `204 No Content` response.
- `GET /health` reports that the service is running.
- `POST /ask` retrieves up to three sources and decides whether human review is
  needed.
- `GET /evaluation` evaluates the bundled test questions and returns metrics.

For `/ask`, expected IDs are used in this order:

1. Use `expected_source_id` and/or `expected_source_ids` from the request, if
   supplied.
2. Otherwise, if the question exactly matches a bundled test question (ignoring
   letter case and surrounding whitespace), use that test question's expected
   source ID.
3. Otherwise, no expected ID is available; `retrieval_hit` is `false`, but that
   alone does not trigger human review.

Human review is requested when a supplied or recognized expected source is
missing from the retrieved results, or when the top result's confidence is
below `FAQRetriever.confidence_threshold`. If review is needed, the API returns
a message indicating that it could not find sufficiently relevant evidence,
rather than presenting the retrieved text as a confident answer.

### `rag_evaluation.py` — retrieval and metrics

- `load_faq_documents()` and `load_test_questions()` read the JSON datasets.
- `FAQRetriever` creates or accesses a ChromaDB collection. When the collection
  is empty, it adds each FAQ's ID, title, text, and metadata.
- `FAQRetriever.retrieve(question, top_k=3)` queries ChromaDB and returns the
  matching IDs, texts, distances, and confidence estimates. The collection uses
  cosine distance. Confidence is currently calculated as `1 - distance`,
  clamped to the range 0–1; it is a retrieval heuristic, not a calibrated
  probability.
- `hit_at_k()` reports whether the expected source occurs anywhere in the
  retrieved IDs. `hit_at_3()` is a convenience wrapper for that check.
- `reciprocal_rank()` returns `1 / rank` for the expected source, or zero when
  it was not retrieved.
- `mean_reciprocal_rank()` averages reciprocal ranks across query results.
- `calculate_metrics()` retrieves the top `k` documents for every test question
  and returns aggregate `Hit@3` and `MRR` values. The evaluation endpoint calls
  it with `k=3`.

ChromaDB's default embedding function is used locally to embed FAQ text and
questions. No external API key is required by this project.

### Data files

- `data/faq_documents.json` contains ten FAQ entries. Each entry has an `id`
  (the stable source ID returned by the API), a `title`, and answer `text`.
- `data/test_questions.json` contains ten evaluation questions. Each has a
  `question` and the `expected_source_id` that should be found in its results.
  Update this file when adding evaluation questions, and make sure each expected
  ID exists in the FAQ dataset.

### Other files

- `app.py` re-exports `main.app`, for hosts that expect an `app:app` import
  path. The examples below use `main:app`.
- `tests/test_rag_api.py` checks the root redirect, favicon response, known and
  unknown questions, missing expected evidence, validation, and metric output.
- `tests/conftest.py` adds the repository root to the test import path.
- `requirements.txt` lists the packages needed to run the application.
- `pyproject.toml` describes the package name, version, Python version
  requirement, and runtime dependencies.
- `README.md` is the short quick-start page; this file is the expanded
  explanation.

## Setup and run

Use Python 3.11 or newer. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` to be redirected to the interactive API docs, or
go directly to `http://127.0.0.1:8000/docs`.

Run the test suite with:

```bash
pytest -q
```

## API examples

### Ask a question

Request:

```http
POST /ask
Content-Type: application/json
```

```json
{
  "question": "What is the service interval?"
}
```

Example response shape:

```json
{
  "question": "What is the service interval?",
  "answer": "The recommended service interval is every 10,000 kilometres or every 12 months, whichever comes first.",
  "retrieved_sources": [
    "service_faq_01",
    "service_faq_04",
    "service_faq_08"
  ],
  "retrieval_hit": true,
  "needs_human_review": false,
  "confidence": 0.70
}
```

Exact source ordering and confidence can vary with the installed ChromaDB
embedding model/version. `retrieved_sources` contains at most three IDs.

To check a particular expected source, include it in the request:

```json
{
  "question": "What is the service interval?",
  "expected_source_id": "service_faq_01"
}
```

Use `expected_source_ids` instead when more than one source is acceptable.

### Health and evaluation

- `GET /health` returns `{"status": "ok"}`.
- `GET /evaluation` returns an object like
  `{"Hit@3": 1.0, "MRR": 1.0}` for the bundled evaluation dataset.

## Understanding the metrics

- **Hit@3** is the fraction of test questions for which the expected source
  appears anywhere in the top three retrieved results. A score of `1.0` means
  every expected source appeared in the top three.
- **Mean Reciprocal Rank (MRR)** is the average reciprocal position of the
  expected source. A source at rank 1 contributes `1.0`, at rank 2 contributes
  `0.5`, at rank 3 contributes about `0.33`, and a missing source contributes
  `0`.

These metrics measure whether the retriever finds the expected source; they do
not measure generated-answer quality, since the API does not generate answers.
