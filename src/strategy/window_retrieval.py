from typing import List

from src.models.retrieved_context import RetrievedContext
from src.strategy.retrieval_strategy import RetrievalStrategy


class WindowRetrieval(RetrievalStrategy):
    """Retrieve top-N results, then expand each to include neighboring chunks
    from the same course for additional surrounding context."""

    def __init__(self, n: int = 3, window_size: int = 1):
        self.n = n
        self.window_size = window_size

    def retrieve(self, collection, query_text: str) -> List[RetrievedContext]:
        results = collection.query(query_texts=[query_text], n_results=self.n)

        contexts = []
        seen_courses = set()

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            course_id = meta.get("course_id", "")
            chunk_index = meta.get("chunk_index", 0)

            if course_id in seen_courses:
                continue
            seen_courses.add(course_id)

            # Fetch neighboring chunks within the window
            neighbor_indices = list(
                range(
                    max(0, chunk_index - self.window_size),
                    chunk_index + self.window_size + 1,
                )
            )

            window_texts = []
            for idx in neighbor_indices:
                neighbor_id = f"{course_id}_chunk_{idx}"
                try:
                    neighbor = collection.get(ids=[neighbor_id])
                    if neighbor["documents"]:
                        window_texts.append(neighbor["documents"][0])
                except Exception:
                    continue

            combined_text = "\n".join(window_texts) if window_texts else doc
            contexts.append(
                RetrievedContext(
                    context_text=combined_text,
                    source_course_id=course_id,
                    relevance_score=dist,
                )
            )

        return contexts
