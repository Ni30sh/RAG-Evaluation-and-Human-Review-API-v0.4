# RAG Evaluation and Human Review API v0.4

FastAPI service with 10 FAQ documents, ChromaDB top-3 retrieval, retrieval metrics,
and human-review routing.

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## API

`POST /ask`

```json
{"question":"What is the service interval?"}
```

Optional `expected_source_id` or `expected_source_ids` fields allow callers to indicate
the evidence that should support an answer. If evidence is absent or top confidence is
below the configured threshold, `needs_human_review` is returned as `true`.

`GET /evaluation` reports `Hit@3` and `MRR` for the 10 bundled test questions.