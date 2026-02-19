from typing import List

from src.models.retrieved_context import RetrievedContext
from src.strategy.retrieval_strategy import RetrievalStrategy


class DocumentRetrieval(RetrievalStrategy):
    """Return the full document (all chunks) for the top matching courses,
    rather than individual chunks."""

    def __init__(self, n_courses: int = 2):
        self.n_courses = n_courses

    def retrieve(self, collection, query_text: str) -> List[RetrievedContext]:
        # First, find the most relevant chunks to identify top courses
        results = collection.query(query_texts=[query_text], n_results=10)

        # Collect unique course IDs in order of relevance
        seen = set()
        top_course_ids = []
        top_distances = {}
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            cid = meta.get("course_id", "")
            if cid and cid not in seen:
                seen.add(cid)
                top_course_ids.append(cid)
                top_distances[cid] = dist
            if len(top_course_ids) >= self.n_courses:
                break

        # For each top course, fetch all its chunks
        contexts = []
        for cid in top_course_ids:
            course_chunks = collection.get(where={"course_id": cid})
            if course_chunks["documents"]:
                # Sort by chunk_index
                paired = zip(course_chunks["documents"], course_chunks["metadatas"])
                sorted_chunks = sorted(paired, key=lambda x: x[1].get("chunk_index", 0))
                full_text = "\n".join(doc for doc, _ in sorted_chunks)
                contexts.append(
                    RetrievedContext(
                        context_text=full_text,
                        source_course_id=cid,
                        relevance_score=top_distances.get(cid, 0.0),
                    )
                )

        return contexts
