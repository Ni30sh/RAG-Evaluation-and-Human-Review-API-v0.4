import json
from pathlib import Path
from typing import Iterable, Sequence

import chromadb


BASE_DIR = Path(__file__).resolve().parent
FAQ_PATH = BASE_DIR / "data" / "faq_documents.json"
QUESTIONS_PATH = BASE_DIR / "data" / "test_questions.json"


def load_faq_documents(path: str | Path = FAQ_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_test_questions(path: str | Path = QUESTIONS_PATH) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class FAQRetriever:
    """Top-k retrieval over a ChromaDB collection."""

    def __init__(
        self,
        documents: Sequence[dict] | None = None,
        collection_name: str = "faq_documents",
        confidence_threshold: float = 0.35,
    ) -> None:
        self.documents = list(documents if documents is not None else load_faq_documents())
        self.confidence_threshold = confidence_threshold
        self.client = chromadb.Client()
        existing_names = {c.name for c in self.client.list_collections()}
        if collection_name in existing_names:
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.client.create_collection(
                collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        if not self.collection.count():
            self.collection.add(
                ids=[doc["id"] for doc in self.documents],
                documents=[f'{doc["title"]}. {doc["text"]}' for doc in self.documents],
                metadatas=[{"title": doc["title"], "text": doc["text"]} for doc in self.documents],
            )

    def retrieve(self, question: str, top_k: int = 3) -> list[dict]:
        result = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, len(self.documents)),
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for source_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            confidence = max(0.0, min(1.0, 1.0 - float(distance)))
            retrieved.append({
                "source_id": source_id,
                "text": document,
                "answer": metadata["text"],
                "distance": float(distance),
                "confidence": confidence,
            })
        return retrieved

    def retrieve_source_ids(self, question: str, top_k: int = 3) -> list[str]:
        return [item["source_id"] for item in self.retrieve(question, top_k)]


def hit_at_k(retrieved_ids: Iterable[str], expected_id: str) -> bool:
    return expected_id in set(retrieved_ids)


def reciprocal_rank(retrieved_ids: Iterable[str], expected_id: str) -> float:
    for rank, source_id in enumerate(retrieved_ids, start=1):
        if source_id == expected_id:
            return 1.0 / rank
    return 0.0


def hit_at_3(retrieved_ids: Iterable[str], expected_id: str) -> bool:
    return hit_at_k(retrieved_ids, expected_id)


def mean_reciprocal_rank(
    ranking_sets: Sequence[Iterable[str]],
    expected_ids: Sequence[str],
) -> float:
    if not ranking_sets:
        return 0.0
    ranks = [
        reciprocal_rank(retrieved_ids, expected_id)
        for retrieved_ids, expected_id in zip(ranking_sets, expected_ids)
    ]
    return sum(ranks) / len(ranks)


def calculate_metrics(
    retriever: FAQRetriever,
    questions: Sequence[dict] | None = None,
    k: int = 3,
) -> dict:
    test_cases = list(questions if questions is not None else load_test_questions())
    hits = []
    ranks = []
    for case in test_cases:
        retrieved_ids = retriever.retrieve_source_ids(case["question"], k)
        expected_id = case["expected_source_id"]
        hits.append(hit_at_k(retrieved_ids, expected_id))
        ranks.append(reciprocal_rank(retrieved_ids, expected_id))
    return {
        "Hit@3": sum(hits) / len(hits) if hits else 0.0,
        "MRR": sum(ranks) / len(ranks) if ranks else 0.0,
    }
