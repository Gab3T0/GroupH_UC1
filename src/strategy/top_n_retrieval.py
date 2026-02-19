from typing import List

from src.models.retrieved_context import RetrievedContext
from src.strategy.retrieval_strategy import RetrievalStrategy


class TopNRetrieval(RetrievalStrategy):
    """Return the top N most similar documents from the vector database."""

    def __init__(self, n: int = 10):
        self.n = n

    def retrieve(self, collection, query_text: str) -> List[RetrievedContext]:
        results = collection.query(query_texts=[query_text], n_results=self.n)

        contexts = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            contexts.append(
                RetrievedContext(
                    context_text=doc,
                    source_course_id=meta.get("course_id", ""),
                    relevance_score=dist,
                )
            )
        return contexts
